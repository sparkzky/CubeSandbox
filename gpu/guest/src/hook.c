/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA hook layer, guest side (protocol v2).
 *
 * libcube-gpu.so is LD_PRELOADed into the sandbox. It intercepts the CUDA
 * Driver API and forwards every stateful call to cube-gpu-daemon on the host
 * over virtio-vsock, using the IVSHMEM file as the bulk-data channel.
 *
 * Interception points:
 *   - dlsym / dlsym@GLIBC_2.34  (applications that resolve the driver lazily)
 *   - cuGetProcAddress / _v2    (what libcudart actually uses since CUDA 11.3)
 *   - the exported cu* symbols  (applications linked directly against libcuda)
 *
 * All three share one table, so a name never resolves differently depending
 * on how the application looked it up.
 *
 * Threading:
 *   - pthread_once performs session setup lazily on the first CUDA call.
 *   - Every RPC round trip and the shared-memory ring accesses bracketing it
 *     run under the single global RPC lock (see rpc_client.h). This makes
 *     concurrent driver-API calls safe even though there is exactly one
 *     vsock socket and one pair of rings.
 *   - The function-metadata table (kernel param sizes) has its own lock and
 *     is never held across an RPC call, so lock ordering is acyclic.
 *
 * Semantics deliberately approximated:
 *   - cuMemcpyHtoDAsync / cuMemcpyDtoHAsync lose their asynchronous nature:
 *     the daemon consumes/produces the ring during the request, so the copy
 *     is complete when the call returns (the stream handle is still passed
 *     through so ordering against other ops is preserved).
 *   - CUDA IPC handles are rejected: the daemon is multi-tenant and must not
 *     expose device pointers across sandboxes.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cuda_defs.h"
#include "rpc_client.h"
#include "rpc_protocol.h"
/* Local integer parsing: avoid glibc's __isoc23_strtoull redirect (GLIBC_2.38)
 * so the library stays loadable on older guest userspaces (e.g. glibc 2.36). */
static unsigned long long cube_parse_u64(const char *s, int base) {
    unsigned long long v = 0;
    int neg = 0;
    if (!s) return 0;
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;
    if (base == 0) {
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
        else if (s[0] == '0') { base = 8; s++; }
        else base = 10;
    } else if (base == 16 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }
    for (;; s++) {
        int d;
        char c = *s;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else break;
        if (d >= base) break;
        v = v * (unsigned)base + (unsigned)d;
    }
    return neg ? (unsigned long long)(-(long long)v) : v;
}
#define strtoull(s, e, b) cube_parse_u64((s), (b))
#define strtoul(s, e, b) ((unsigned long)cube_parse_u64((s), (b)))

#include "shm_ring.h"

/* ─── Environment ─────────────────────────────────────────────── */

#define ENV_DEBUG        "CUBE_GPU_DEBUG"
#define ENV_SANDBOX_ID   "CUBE_GPU_SANDBOX_ID"
#define ENV_MEMORY_QUOTA "CUBE_GPU_MEMORY_QUOTA"
#define ENV_VSOCK_PORT   "CUBE_GPU_VSOCK_PORT"
#define ENV_VSOCK_CID    "CUBE_GPU_VSOCK_CID"

#define DEFAULT_VSOCK_CID   2u
#define DEFAULT_MEMORY_QUOTA (4ULL * 1024 * 1024 * 1024)
#define MODULE_MAX_IMAGE_SIZE (64ULL * 1024 * 1024)
#define FUNC_MAX_PARAMS 32

/* ─── Logging ─────────────────────────────────────────────────── */

static int g_debug = 0;

#define log_err(fmt, ...) \
    fprintf(stderr, "[cube-gpu] ERROR: " fmt "\n", ##__VA_ARGS__)
#define log_dbg(fmt, ...) \
    do { if (g_debug) fprintf(stderr, "[cube-gpu] DBG: " fmt "\n", ##__VA_ARGS__); } while (0)

/* ─── Global state ────────────────────────────────────────────── */

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static cube_gpu_rpc_client_t g_rpc;

static uint64_t g_sandbox_id = 0;
static uint64_t g_memory_quota = DEFAULT_MEMORY_QUOTA;
static uint32_t g_vsock_cid = DEFAULT_VSOCK_CID;
static uint32_t g_vsock_port = CUBE_GPU_VSOCK_PORT;

/* Kernel metadata, keyed by the host function handle returned by
 * cuModuleGetFunction. Needed to serialize kernelParams without guessing
 * the argument count. */
typedef struct {
    uint64_t func;
    uint32_t nparams;
    uint32_t has_meta;
    uint16_t sizes[FUNC_MAX_PARAMS];
} cube_gpu_func_meta_t;

static cube_gpu_func_meta_t *g_funcs = NULL;
static size_t g_func_count = 0;
static size_t g_func_cap = 0;
static pthread_mutex_t g_table_lock = PTHREAD_MUTEX_INITIALIZER;

static void func_meta_put(uint64_t func, uint32_t nparams, uint32_t has_meta,
                          const uint16_t *sizes) {
    size_t i;

    if (func == 0) return;
    if (nparams > FUNC_MAX_PARAMS) nparams = FUNC_MAX_PARAMS;

    pthread_mutex_lock(&g_table_lock);
    for (i = 0; i < g_func_count; i++) {
        if (g_funcs[i].func == func) break;
    }
    if (i == g_func_count) {
        if (g_func_count == g_func_cap) {
            size_t cap = g_func_cap ? g_func_cap * 2 : 64;
            cube_gpu_func_meta_t *grown = realloc(g_funcs, cap * sizeof(*grown));
            if (!grown) {
                pthread_mutex_unlock(&g_table_lock);
                log_err("out of memory tracking kernel parameters");
                return;
            }
            g_funcs = grown;
            g_func_cap = cap;
        }
        memset(&g_funcs[i], 0, sizeof(g_funcs[i]));
        g_funcs[i].func = func;
        g_func_count++;
    }

    g_funcs[i].nparams = nparams;
    g_funcs[i].has_meta = has_meta ? 1u : 0u;
    memset(g_funcs[i].sizes, 0, sizeof(g_funcs[i].sizes));
    if (has_meta && sizes) {
        uint32_t j;
        for (j = 0; j < nparams; j++) g_funcs[i].sizes[j] = sizes[j];
    }
    pthread_mutex_unlock(&g_table_lock);
}

static int func_meta_get(uint64_t func, cube_gpu_func_meta_t *out) {
    size_t i;
    int found = 0;

    pthread_mutex_lock(&g_table_lock);
    for (i = 0; i < g_func_count; i++) {
        if (g_funcs[i].func == func) {
            *out = g_funcs[i];
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_table_lock);
    return found;
}

/* ─── Initialization ──────────────────────────────────────────── */

static uint64_t env_u64_dec(const char *name, uint64_t fallback) {
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return fallback;
    return strtoull(v, NULL, 10);
}

static uint64_t env_u64_auto(const char *name, uint64_t fallback) {
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return fallback;
    return strtoull(v, NULL, 0);
}

static void initialize(void) {
    const char *dbg = getenv(ENV_DEBUG);
    g_debug = (dbg && dbg[0] == '1');

    g_sandbox_id = env_u64_dec(ENV_SANDBOX_ID, 0);
    g_memory_quota = env_u64_dec(ENV_MEMORY_QUOTA, DEFAULT_MEMORY_QUOTA);
    g_vsock_port = (uint32_t)env_u64_auto(ENV_VSOCK_PORT, CUBE_GPU_VSOCK_PORT);
    g_vsock_cid = (uint32_t)env_u64_auto(ENV_VSOCK_CID, DEFAULT_VSOCK_CID);

    log_dbg("initializing: sandbox=%llu quota=%llu cid=%u port=%u",
            (unsigned long long)g_sandbox_id,
            (unsigned long long)g_memory_quota, g_vsock_cid, g_vsock_port);

    if (cube_gpu_rpc_client_init(&g_rpc, g_vsock_cid, g_vsock_port,
                                 g_sandbox_id, g_memory_quota) < 0) {
        log_err("failed to establish a GPU session with cube-gpu-daemon");
        return;
    }

    log_dbg("session ready: devices=%u ring_size=%zu",
            g_rpc.device_count, g_rpc.shm.g2h.data_size);
}

static void ensure_initialized(void) {
    pthread_once(&g_init_once, initialize);
}

/* ─── RPC helpers ─────────────────────────────────────────────── */

static CUresult transport_failure(void) {
    return CUDA_ERROR_OPERATING_SYSTEM;
}

static CUresult no_data_plane(void) {
    log_err("shared-memory data channel is unavailable");
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* One round trip, no shared-memory payload. */
static int rpc_simple(uint32_t api, const void *req, size_t req_len,
                      void *resp, size_t resp_cap, uint32_t *hdr_status) {
    return cube_gpu_rpc_call(&g_rpc, api, CUBE_GPU_FLAG_NONE, req, req_len,
                             resp, resp_cap, hdr_status);
}

/* ─── Kernel/function metadata helpers ────────────────────────── */

/*
 * Parse cuLaunchKernel's `extra` key/value array into a packed parameter
 * buffer. Only CU_LAUNCH_PARAM_BUFFER_POINTER and _SIZE are understood;
 * other keys are skipped. At most 32 pairs are inspected and a NULL key
 * (CU_LAUNCH_PARAM_END) terminates the scan, so a malformed array can never
 * make us walk off the end.
 * Returns 0 with buf/size filled, or -1 if no usable buffer was found.
 */
static int parse_launch_extra(void **extra, const void **buf, size_t *size) {
    int i;

    *buf = NULL;
    *size = 0;
    if (!extra) return -1;

    for (i = 0; i < 32; i++) {
        void *key = extra[i * 2];
        void *value = extra[i * 2 + 1];
        if (key == CU_LAUNCH_PARAM_END) break;
        if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            *buf = value;
        } else if (key == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            if (!value) return -1;
            *size = *(const size_t *)value;
        }
        /* Unknown keys are ignored. */
    }

    if (!*buf || *size == 0) return -1;
    return 0;
}

/* ─── Module image sizing ─────────────────────────────────────── */

static uint16_t rd_u16le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint64_t rd_u64le(const uint8_t *p) {
    return (uint64_t)p[0]
         | ((uint64_t)p[1] << 8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

/*
 * Determine the byte length of a module image loaded with cuModuleLoadData.
 * cubin is an ELF64 object: its size is the end of the section table. PTX is
 * a NUL-terminated string. No length is passed to the driver API, so this is
 * the same inference CUDA itself has to make.
 */
static CUresult module_image_size(const void *image, size_t *out_size) {
    const uint8_t *p = (const uint8_t *)image;
    uint64_t shoff, total;
    uint16_t shentsize, shnum;
    const uint8_t *nul;
    size_t idx;

    *out_size = 0;

    if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
        if (p[4] != 2) { /* ELFCLASS64 only; cubins are always 64-bit here */
            log_err("cuModuleLoadData: unsupported ELF class %u", p[4]);
            return CUDA_ERROR_INVALID_IMAGE;
        }
        shoff = rd_u64le(p + 0x28);
        shentsize = rd_u16le(p + 0x3a);
        shnum = rd_u16le(p + 0x3c);
        if (shoff == 0 || shnum == 0 || shentsize < 0x40) {
            log_err("cuModuleLoadData: malformed ELF section table");
            return CUDA_ERROR_INVALID_IMAGE;
        }
        total = shoff + (uint64_t)shnum * (uint64_t)shentsize;
        if (total < 0x40 || total > MODULE_MAX_IMAGE_SIZE) {
            log_err("cuModuleLoadData: implausible cubin size %llu",
                    (unsigned long long)total);
            return CUDA_ERROR_INVALID_IMAGE;
        }
        *out_size = (size_t)total;
        return CUDA_SUCCESS;
    }

    /* PTX (or another textual image): find the terminating NUL. */
    if (p[0] < 0x20 || p[0] > 0x7e) {
        log_err("cuModuleLoadData: unrecognized module image");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    nul = memchr(image, '\0', (size_t)MODULE_MAX_IMAGE_SIZE);
    if (!nul) {
        log_err("cuModuleLoadData: unterminated textual image");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    idx = (size_t)(nul - p) + 1; /* include the NUL as CUDA expects */
    if (idx < 8) {
        log_err("cuModuleLoadData: textual image is implausibly short");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    *out_size = idx;
    return CUDA_SUCCESS;
}

/* ─── Error name/string table ─────────────────────────────────── */

typedef struct {
    CUresult code;
    const char *name;
    const char *desc;
} cube_gpu_error_entry_t;

static const cube_gpu_error_entry_t g_errors[] = {
    {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
    {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
    {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
    {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
    {CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "driver shutting down"},
    {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"},
    {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
    {CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid"},
    {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
    {CUDA_ERROR_CONTEXT_ALREADY_CURRENT, "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
     "context already current"},
    {CUDA_ERROR_MAP_FAILED, "CUDA_ERROR_MAP_FAILED", "mapping of buffer object failed"},
    {CUDA_ERROR_UNMAP_FAILED, "CUDA_ERROR_UNMAP_FAILED", "unmapping of buffer object failed"},
    {CUDA_ERROR_ARRAY_IS_MAPPED, "CUDA_ERROR_ARRAY_IS_MAPPED", "array is mapped"},
    {CUDA_ERROR_ALREADY_MAPPED, "CUDA_ERROR_ALREADY_MAPPED", "resource already mapped"},
    {CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU",
     "no kernel image is available for execution on the device"},
    {CUDA_ERROR_ALREADY_ACQUIRED, "CUDA_ERROR_ALREADY_ACQUIRED", "resource already acquired"},
    {CUDA_ERROR_NOT_MAPPED, "CUDA_ERROR_NOT_MAPPED", "resource not mapped"},
    {CUDA_ERROR_ECC_UNCORRECTABLE, "CUDA_ERROR_ECC_UNCORRECTABLE",
     "uncorrectable ECC error encountered"},
    {CUDA_ERROR_UNSUPPORTED_LIMIT, "CUDA_ERROR_UNSUPPORTED_LIMIT",
     "limit is not supported on this architecture"},
    {CUDA_ERROR_CONTEXT_ALREADY_IN_USE, "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
     "exclusive-thread device already in use"},
    {CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
     "peer access is not supported"},
    {CUDA_ERROR_INVALID_PTX, "CUDA_ERROR_INVALID_PTX", "a PTX JIT compilation failed"},
    {CUDA_ERROR_INVALID_SOURCE, "CUDA_ERROR_INVALID_SOURCE", "device kernel source is invalid"},
    {CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND", "file not found"},
    {CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
     "link to a shared object failed to resolve"},
    {CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
     "shared object initialization failed"},
    {CUDA_ERROR_OPERATING_SYSTEM, "CUDA_ERROR_OPERATING_SYSTEM", "OS call failed"},
    {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"},
    {CUDA_ERROR_ILLEGAL_STATE, "CUDA_ERROR_ILLEGAL_STATE", "illegal state"},
    {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
    {CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready"},
    {CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS",
     "an illegal memory access was encountered"},
    {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
     "too many resources requested for launch"},
    {CUDA_ERROR_LAUNCH_TIMEOUT, "CUDA_ERROR_LAUNCH_TIMEOUT", "the launch timed out"},
    {CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
     "peer access is already enabled"},
    {CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
     "peer access has not been enabled"},
    {CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE, "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
     "primary context is active"},
    {CUDA_ERROR_CONTEXT_IS_DESTROYED, "CUDA_ERROR_CONTEXT_IS_DESTROYED",
     "context is destroyed"},
    {CUDA_ERROR_ASSERT, "CUDA_ERROR_ASSERT", "device-side assert triggered"},
    {CUDA_ERROR_TOO_MANY_PEERS, "CUDA_ERROR_TOO_MANY_PEERS", "too many peers"},
    {CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
     "host memory already registered"},
    {CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
     "host memory not registered"},
    {CUDA_ERROR_ILLEGAL_INSTRUCTION, "CUDA_ERROR_ILLEGAL_INSTRUCTION",
     "an illegal instruction was encountered"},
    {CUDA_ERROR_MISALIGNED_ADDRESS, "CUDA_ERROR_MISALIGNED_ADDRESS",
     "misaligned address"},
    {CUDA_ERROR_INVALID_ADDRESS_SPACE, "CUDA_ERROR_INVALID_ADDRESS_SPACE",
     "invalid address space"},
    {CUDA_ERROR_INVALID_PC, "CUDA_ERROR_INVALID_PC", "invalid program counter"},
    {CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure"},
    {CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",
     "too many blocks in cooperative launch"},
    {CUDA_ERROR_NOT_PERMITTED, "CUDA_ERROR_NOT_PERMITTED", "operation not permitted"},
    {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported"},
    {CUDA_ERROR_SYSTEM_NOT_READY, "CUDA_ERROR_SYSTEM_NOT_READY", "system not yet initialized"},
    {CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
     "system has unsupported display driver / cuda driver combination"},
    {CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE, "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE",
     "forward compatibility was attempted on non supported HW"},
    {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error"},
};

static const cube_gpu_error_entry_t *err_lookup(CUresult code) {
    size_t i;
    for (i = 0; i < sizeof(g_errors) / sizeof(g_errors[0]); i++) {
        if (g_errors[i].code == code) return &g_errors[i];
    }
    return NULL;
}

/* ─── Device / query entry points ─────────────────────────────── */

CUBE_GPU_EXPORT CUresult cuInit(unsigned int Flags) {
    (void)Flags;
    ensure_initialized();
    return g_rpc.initialized ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
}

CUBE_GPU_EXPORT CUresult cuDeviceGetCount(int *count) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = (int)g_rpc.device_count;
    return CUDA_SUCCESS;
}

CUBE_GPU_EXPORT CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    cube_gpu_get_device_req_t req;
    cube_gpu_get_device_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!device) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.ordinal = (int32_t)ordinal;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_GET_DEVICE, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *device = (CUdevice)resp.device;
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    cube_gpu_get_device_req_t req;
    cube_gpu_device_name_resp_t resp;
    size_t n;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.ordinal = (int32_t)dev;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_GET_DEVICE_NAME, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status != CUDA_SUCCESS) return (CUresult)resp.status;

    n = (size_t)len - 1;
    if (n > sizeof(resp.name)) n = sizeof(resp.name);
    memcpy(name, resp.name, n);
    name[n] = '\0';
    return CUDA_SUCCESS;
}

CUBE_GPU_EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                              CUdevice dev) {
    cube_gpu_get_dev_prop_req_t req;
    cube_gpu_get_dev_prop_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pi) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.device = (int32_t)dev;
    req.attribute = (int32_t)attrib;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_GET_DEV_PROP, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *pi = resp.value;
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    cube_gpu_get_device_req_t req;
    cube_gpu_device_total_mem_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.ordinal = (int32_t)dev;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_DEVICE_TOTAL_MEM, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *bytes = (size_t)resp.total_memory;
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemGetInfo(size_t *free, size_t *total) {
    cube_gpu_mem_get_info_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!free || !total) return CUDA_ERROR_INVALID_VALUE;

    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_MEM_GET_INFO, NULL, 0,
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *free = (size_t)resp.free_memory;
        *total = (size_t)resp.total_memory;
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuCtxGetDevice(CUdevice *device) {
    cube_gpu_ctx_get_device_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!device) return CUDA_ERROR_INVALID_VALUE;

    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_CTX_GET_DEVICE, NULL, 0,
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *device = (CUdevice)resp.device;
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuGetErrorString(CUresult error, const char **pStr) {
    const cube_gpu_error_entry_t *entry;

    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    entry = err_lookup(error);
    if (!entry) {
        *pStr = NULL;
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pStr = entry->desc;
    return CUDA_SUCCESS;
}

CUBE_GPU_EXPORT CUresult cuGetErrorName(CUresult error, const char **pStr) {
    const cube_gpu_error_entry_t *entry;

    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    entry = err_lookup(error);
    if (!entry) {
        *pStr = NULL;
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pStr = entry->name;
    return CUDA_SUCCESS;
}

/* ─── Context entry points ────────────────────────────────────── */

/* Map a CUdevice ordinal / handle onto a device id the daemon accepts.
 * Applications normally pass an ordinal from cuDeviceGet; if it does not
 * look valid we ask the daemon which device the session is using, and fall
 * back to device 0. */
static int resolve_device(CUdevice dev) {
    cube_gpu_ctx_get_device_resp_t resp;
    int count = (int)g_rpc.device_count;

    if (dev >= 0 && (count <= 0 || dev < count)) return (int)dev;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_CTX_GET_DEVICE, NULL, 0,
                   &resp, sizeof(resp), NULL) == 0 &&
        resp.status == CUDA_SUCCESS && resp.device >= 0) {
        return resp.device;
    }
    return 0;
}

CUBE_GPU_EXPORT CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    cube_gpu_ctx_create_req_t req;
    cube_gpu_ctx_create_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.flags = (uint32_t)flags;
    req.device_id = (uint32_t)resolve_device(dev);
    memset(&resp, 0, sizeof(resp));

    if (rpc_simple(CUBE_GPU_API_CTX_CREATE, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *pctx = (CUcontext)(uintptr_t)resp.ctx_handle;
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuCtxDestroy(CUcontext ctx) {
    cube_gpu_ctx_op_req_t req;
    cube_gpu_ctx_op_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;

    memset(&req, 0, sizeof(req));
    req.ctx_handle = (uint64_t)(uintptr_t)ctx;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_CTX_DESTROY, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
    cube_gpu_ctx_op_req_t req;
    cube_gpu_ctx_op_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    memset(&req, 0, sizeof(req));
    req.ctx_handle = (uint64_t)(uintptr_t)ctx; /* 0 unbinds */
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_CTX_SET_CURRENT, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx) {
    cube_gpu_ctx_get_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;

    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_CTX_GET_CURRENT, NULL, 0,
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *pctx = resp.ctx_handle ? (CUcontext)(uintptr_t)resp.ctx_handle : NULL;
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuCtxSynchronize(void) {
    cube_gpu_device_sync_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_DEVICE_SYNC, NULL, 0,
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    cube_gpu_primary_ctx_req_t req;
    cube_gpu_primary_ctx_retain_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.device = (int32_t)dev;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_PRIMARY_CTX_RETAIN, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *pctx = (CUcontext)(uintptr_t)resp.ctx_handle;
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    cube_gpu_primary_ctx_req_t req;
    cube_gpu_primary_ctx_release_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    memset(&req, 0, sizeof(req));
    req.device = (int32_t)dev;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_PRIMARY_CTX_RELEASE, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

/* ─── Memory entry points ─────────────────────────────────────── */

CUBE_GPU_EXPORT CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    cube_gpu_malloc_req_t req;
    cube_gpu_malloc_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.size = (uint64_t)bytesize;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_MALLOC, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *dptr = (CUdeviceptr)resp.vptr;
        log_dbg("cuMemAlloc(%zu) -> 0x%llx", bytesize, (unsigned long long)resp.vptr);
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemFree(CUdeviceptr dptr) {
    cube_gpu_free_req_t req;
    cube_gpu_free_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dptr == 0) return CUDA_SUCCESS;

    memset(&req, 0, sizeof(req));
    req.vptr = (uint64_t)dptr;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_FREE, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize,
                                         CUstream hStream) {
    cube_gpu_malloc_async_req_t req;
    cube_gpu_malloc_async_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.size = (uint64_t)bytesize;
    req.stream_vptr = (uint64_t)(uintptr_t)hStream; /* 0 == default pool/stream */
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_MALLOC_ASYNC, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *dptr = (CUdeviceptr)resp.vptr;
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    cube_gpu_free_async_req_t req;
    cube_gpu_free_async_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    memset(&req, 0, sizeof(req));
    req.vptr = (uint64_t)dptr;
    req.stream_vptr = (uint64_t)(uintptr_t)hStream;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_FREE_ASYNC, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

/*
 * H2D: the payload is staged in the g2h ring before the request is sent; the
 * daemon consumes exactly `n` bytes. The FLAG_SHM_DATA convention does not
 * use the request's shm_offset field (protocol v2 ping rule).
 */
static CUresult memcpy_h2d(uint32_t api, CUdeviceptr dst, const void *src, size_t n,
                           uint64_t stream) {
    cube_gpu_memcpy_resp_t resp;
    cube_gpu_memcpy_async_req_t areq;
    cube_gpu_memcpy_req_t req;
    uint32_t status = 0;
    int rc;

    if (n == 0) return CUDA_SUCCESS;
    if (!src) return CUDA_ERROR_INVALID_VALUE;
    if (!g_rpc.shm.base) return no_data_plane();

    memset(&resp, 0, sizeof(resp));
    cube_gpu_rpc_lock();

    if (cube_gpu_shm_ring_write(&g_rpc.shm.g2h, src, n) != n) {
        cube_gpu_rpc_unlock();
        log_err("cuMemcpy H2D: %zu bytes do not fit in the g2h ring", n);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    if (api == CUBE_GPU_API_MEMCPY_ASYNC) {
        memset(&areq, 0, sizeof(areq));
        areq.kind = CUBE_GPU_MEMCPY_HOST_TO_DEVICE;
        areq.dst = (uint64_t)dst;
        areq.size = (uint64_t)n;
        areq.stream_vptr = stream;
        rc = cube_gpu_rpc_call_locked(&g_rpc, api, CUBE_GPU_FLAG_SHM_DATA,
                                      &areq, sizeof(areq), &resp, sizeof(resp), &status);
    } else {
        memset(&req, 0, sizeof(req));
        req.kind = CUBE_GPU_MEMCPY_HOST_TO_DEVICE;
        req.dst = (uint64_t)dst;
        req.size = (uint64_t)n;
        rc = cube_gpu_rpc_call_locked(&g_rpc, api, CUBE_GPU_FLAG_SHM_DATA,
                                      &req, sizeof(req), &resp, sizeof(resp), &status);
    }

    cube_gpu_rpc_unlock();
    if (rc < 0) return transport_failure();
    return (CUresult)resp.status;
}

/*
 * D2H: the daemon writes exactly `n` bytes into the h2g ring; the guest
 * consumes them immediately after the response (the ring cannot be drained
 * lazily once the lock is released).
 */
static CUresult memcpy_d2h(uint32_t api, void *dst, CUdeviceptr src, size_t n,
                           uint64_t stream) {
    cube_gpu_memcpy_resp_t resp;
    cube_gpu_memcpy_async_req_t areq;
    cube_gpu_memcpy_req_t req;
    uint32_t status = 0;
    int rc;

    if (n == 0) return CUDA_SUCCESS;
    if (!dst) return CUDA_ERROR_INVALID_VALUE;
    if (!g_rpc.shm.base) return no_data_plane();

    memset(&resp, 0, sizeof(resp));
    cube_gpu_rpc_lock();

    if (api == CUBE_GPU_API_MEMCPY_ASYNC) {
        memset(&areq, 0, sizeof(areq));
        areq.kind = CUBE_GPU_MEMCPY_DEVICE_TO_HOST;
        areq.src = (uint64_t)src;
        areq.size = (uint64_t)n;
        areq.stream_vptr = stream;
        rc = cube_gpu_rpc_call_locked(&g_rpc, api, CUBE_GPU_FLAG_SHM_DATA,
                                      &areq, sizeof(areq), &resp, sizeof(resp), &status);
    } else {
        memset(&req, 0, sizeof(req));
        req.kind = CUBE_GPU_MEMCPY_DEVICE_TO_HOST;
        req.src = (uint64_t)src;
        req.size = (uint64_t)n;
        rc = cube_gpu_rpc_call_locked(&g_rpc, api, CUBE_GPU_FLAG_SHM_DATA,
                                      &req, sizeof(req), &resp, sizeof(resp), &status);
    }

    if (rc == 0 && resp.status == CUDA_SUCCESS) {
        if (cube_gpu_shm_ring_read(&g_rpc.shm.h2g, dst, n) != n) {
            log_err("cuMemcpy D2H: daemon did not deliver %zu bytes", n);
            rc = -1;
        }
    }

    cube_gpu_rpc_unlock();
    if (rc < 0) return transport_failure();
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                                      size_t ByteCount) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memcpy_h2d(CUBE_GPU_API_MEMCPY, dstDevice, srcHost, ByteCount, 0);
}

CUBE_GPU_EXPORT CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memcpy_d2h(CUBE_GPU_API_MEMCPY, dstHost, srcDevice, ByteCount, 0);
}

CUBE_GPU_EXPORT CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                      size_t ByteCount) {
    cube_gpu_memcpy_req_t req;
    cube_gpu_memcpy_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ByteCount == 0) return CUDA_SUCCESS;

    memset(&req, 0, sizeof(req));
    req.kind = CUBE_GPU_MEMCPY_DEVICE_TO_DEVICE;
    req.dst = (uint64_t)dstDevice;
    req.src = (uint64_t)srcDevice;
    req.size = (uint64_t)ByteCount;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_MEMCPY, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                                           size_t ByteCount, CUstream hStream) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memcpy_h2d(CUBE_GPU_API_MEMCPY_ASYNC, dstDevice, srcHost, ByteCount,
                      (uint64_t)(uintptr_t)hStream);
}

CUBE_GPU_EXPORT CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                           size_t ByteCount, CUstream hStream) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memcpy_d2h(CUBE_GPU_API_MEMCPY_ASYNC, dstHost, srcDevice, ByteCount,
                      (uint64_t)(uintptr_t)hStream);
}

/* memset sizes are transmitted in BYTES; element_bits tells the daemon how
 * to convert back to an element count for cuMemsetD8/D16/D32. */
static CUresult memset_common(CUdeviceptr dst, uint32_t value, uint32_t element_bits,
                              size_t count, uint64_t stream) {
    cube_gpu_memset_req_t req;
    cube_gpu_memset_resp_t resp;
    size_t element_bytes = element_bits / 8;

    if (dst == 0) return CUDA_ERROR_INVALID_VALUE;
    if (count == 0) return CUDA_SUCCESS;
    if (element_bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    if (count > SIZE_MAX / element_bytes) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.dst = (uint64_t)dst;
    req.size = (uint64_t)count * (uint64_t)element_bytes;
    req.value = value;
    req.element_bits = element_bits;
    req.stream_vptr = stream;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_MEMSET, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memset_common(dstDevice, (uint32_t)uc, 8, N, 0);
}

CUBE_GPU_EXPORT CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memset_common(dstDevice, (uint32_t)us, 16, N, 0);
}

CUBE_GPU_EXPORT CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return memset_common(dstDevice, ui, 32, N, 0);
}

/* ─── Module / kernel entry points ────────────────────────────── */

static CUresult module_load(CUmodule *module, const void *image, const char *api_name) {
    cube_gpu_module_load_data_req_t req;
    cube_gpu_module_load_data_resp_t resp;
    size_t image_size = 0;
    CUresult err;
    int rc;

    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;

    err = module_image_size(image, &image_size);
    if (err != CUDA_SUCCESS) return err;

    if (!g_rpc.shm.base) return no_data_plane();

    memset(&resp, 0, sizeof(resp));
    cube_gpu_rpc_lock();

    if (cube_gpu_shm_ring_write(&g_rpc.shm.g2h, image, image_size) != image_size) {
        cube_gpu_rpc_unlock();
        log_err("%s: %zu-byte image does not fit in the g2h ring",
                api_name, image_size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    memset(&req, 0, sizeof(req));
    req.image_size = (uint64_t)image_size;
    req.shm_offset = 0; /* v2: data lives at the ring cursor, not an offset */
    rc = cube_gpu_rpc_call_locked(&g_rpc, CUBE_GPU_API_MODULE_LOAD_DATA,
                                  CUBE_GPU_FLAG_SHM_DATA, &req, sizeof(req),
                                  &resp, sizeof(resp), NULL);
    cube_gpu_rpc_unlock();

    if (rc < 0) return transport_failure();
    if (resp.status == CUDA_SUCCESS) {
        *module = (CUmodule)(uintptr_t)resp.module_handle;
        log_dbg("%s: %zu-byte image -> module 0x%llx", api_name, image_size,
                (unsigned long long)resp.module_handle);
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return module_load(module, image, "cuModuleLoadData");
}

CUBE_GPU_EXPORT CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                            unsigned int numOptions, void *options,
                                            void *optionValues) {
    /* JIT options only affect how the daemon compiles PTX; the image itself
     * is loaded exactly like cuModuleLoadData. */
    (void)numOptions;
    (void)options;
    (void)optionValues;
    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return module_load(module, image, "cuModuleLoadDataEx");
}

CUBE_GPU_EXPORT CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                             const char *name) {
    cube_gpu_module_get_func_req_t req;
    cube_gpu_module_get_func_resp_t resp;
    uint16_t param_sizes[FUNC_MAX_PARAMS];

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.module_handle = (uint64_t)(uintptr_t)hmod;
    strncpy(req.func_name, name, sizeof(req.func_name) - 1);

    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_MODULE_GET_FUNC, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *hfunc = (CUfunction)(uintptr_t)resp.func_handle;
        memcpy(param_sizes, resp.param_sizes, sizeof(param_sizes));
        func_meta_put(resp.func_handle, resp.nparams, resp.has_meta, param_sizes);
        log_dbg("cuModuleGetFunction(%s): func=0x%llx nparams=%u has_meta=%u",
                name, (unsigned long long)resp.func_handle, resp.nparams, resp.has_meta);
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuLaunchKernel(CUfunction f,
                                        unsigned int gridDimX, unsigned int gridDimY,
                                        unsigned int gridDimZ,
                                        unsigned int blockDimX, unsigned int blockDimY,
                                        unsigned int blockDimZ,
                                        unsigned int sharedMemBytes, CUstream hStream,
                                        void **kernelParams, void **extra) {
    cube_gpu_launch_kernel_req_t req;
    cube_gpu_launch_kernel_resp_t resp;
    cube_gpu_func_meta_t meta;
    const void *extra_buf = NULL;
    size_t extra_size = 0;
    uint64_t params_total = 0;
    uint64_t func_handle = (uint64_t)(uintptr_t)f;
    uint32_t flags = CUBE_GPU_FLAG_NONE;
    uint32_t i;
    int rc;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (func_handle == 0) return CUDA_ERROR_INVALID_HANDLE;
    if (!g_rpc.shm.base) return no_data_plane();

    memset(&req, 0, sizeof(req));
    req.func_handle = func_handle;
    req.grid_dim_x = gridDimX;
    req.grid_dim_y = gridDimY;
    req.grid_dim_z = gridDimZ;
    req.block_dim_x = blockDimX;
    req.block_dim_y = blockDimY;
    req.block_dim_z = blockDimZ;
    req.shared_mem_bytes = sharedMemBytes;
    req.stream_vptr = (uint64_t)(uintptr_t)hStream;

    if (extra) {
        /* Packed-argument form: honor the explicit buffer and its size. */
        if (parse_launch_extra(extra, &extra_buf, &extra_size) < 0) {
            log_err("cuLaunchKernel: `extra` has no usable "
                    "CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE pair");
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        req.extra_size = (uint64_t)extra_size;
        req.params_size = 0;
    } else {
        /* kernelParams form: the array length is not passed to the driver,
         * so the daemon-provided metadata is the only safe source. */
        if (!func_meta_get(func_handle, &meta)) {
            log_err("cuLaunchKernel: unknown function handle 0x%llx",
                    (unsigned long long)func_handle);
            return CUDA_ERROR_INVALID_HANDLE;
        }
        if (!meta.has_meta) {
            if (kernelParams) {
                log_err("cuLaunchKernel: daemon did not report parameter "
                        "metadata; cannot serialize kernelParams");
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            req.params_size = 0;
        } else {
            for (i = 0; i < meta.nparams; i++) params_total += meta.sizes[i];
            if (params_total > 0 && !kernelParams) return CUDA_ERROR_INVALID_VALUE;
            req.params_size = params_total;
        }
    }

    memset(&resp, 0, sizeof(resp));
    cube_gpu_rpc_lock();

    if (extra_buf) {
        if (cube_gpu_shm_ring_write(&g_rpc.shm.g2h, extra_buf, extra_size) != extra_size) {
            cube_gpu_rpc_unlock();
            log_err("cuLaunchKernel: %zu-byte argument buffer exceeds the g2h ring",
                    extra_size);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        flags |= CUBE_GPU_FLAG_SHM_DATA;
    } else if (params_total > 0) {
        if (params_total > cube_gpu_shm_ring_avail_write(&g_rpc.shm.g2h)) {
            cube_gpu_rpc_unlock();
            log_err("cuLaunchKernel: %llu bytes of parameters exceed the g2h ring",
                    (unsigned long long)params_total);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        for (i = 0; i < meta.nparams; i++) {
            size_t size = meta.sizes[i];
            if (size == 0) continue;
            if (!kernelParams[i]) {
                cube_gpu_rpc_unlock();
                log_err("cuLaunchKernel: kernelParams[%u] is NULL", i);
                return CUDA_ERROR_INVALID_VALUE;
            }
            if (cube_gpu_shm_ring_write(&g_rpc.shm.g2h, kernelParams[i], size) != size) {
                cube_gpu_rpc_unlock();
                log_err("cuLaunchKernel: parameter %u does not fit in the ring", i);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }
        flags |= CUBE_GPU_FLAG_SHM_DATA;
    }

    rc = cube_gpu_rpc_call_locked(&g_rpc, CUBE_GPU_API_LAUNCH_KERNEL, flags,
                                  &req, sizeof(req), &resp, sizeof(resp), NULL);
    cube_gpu_rpc_unlock();

    if (rc < 0) return transport_failure();
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                            CUfunction hfunc) {
    cube_gpu_func_get_attr_req_t req;
    cube_gpu_func_get_attr_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pi || !hfunc) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.func_vptr = (uint64_t)(uintptr_t)hfunc;
    req.attribute = (int32_t)attrib;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_FUNC_GET_ATTR, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *pi = resp.value;
    return (CUresult)resp.status;
}

/* ─── Stream entry points ─────────────────────────────────────── */

CUBE_GPU_EXPORT CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    cube_gpu_stream_create_req_t req;
    cube_gpu_stream_create_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.flags = (uint32_t)Flags;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_STREAM_CREATE, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *phStream = (CUstream)(uintptr_t)resp.stream_vptr;
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuStreamDestroy(CUstream hStream) {
    cube_gpu_stream_destroy_req_t req;
    cube_gpu_stream_destroy_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    memset(&req, 0, sizeof(req));
    req.stream_vptr = (uint64_t)(uintptr_t)hStream;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_STREAM_DESTROY, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuStreamSynchronize(CUstream hStream) {
    cube_gpu_stream_sync_req_t req;
    cube_gpu_stream_sync_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    memset(&req, 0, sizeof(req));
    req.stream_vptr = (uint64_t)(uintptr_t)hStream; /* 0 == legacy default */
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_STREAM_SYNC, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

/* ─── Event entry points ──────────────────────────────────────── */

CUBE_GPU_EXPORT CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    cube_gpu_event_create_req_t req;
    cube_gpu_event_create_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;

    memset(&req, 0, sizeof(req));
    req.flags = (uint32_t)Flags;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_EVENT_CREATE, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) {
        *phEvent = (CUevent)(uintptr_t)resp.event_vptr;
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    cube_gpu_event_record_req_t req;
    cube_gpu_event_record_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;

    memset(&req, 0, sizeof(req));
    req.event_vptr = (uint64_t)(uintptr_t)hEvent;
    req.stream_vptr = (uint64_t)(uintptr_t)hStream;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_EVENT_RECORD, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuEventSynchronize(CUevent hEvent) {
    cube_gpu_event_sync_req_t req;
    cube_gpu_event_sync_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;

    memset(&req, 0, sizeof(req));
    req.event_vptr = (uint64_t)(uintptr_t)hEvent;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_EVENT_SYNC, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                            CUevent hEnd) {
    cube_gpu_event_elapsed_req_t req;
    cube_gpu_event_elapsed_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pMilliseconds || !hStart || !hEnd) return CUDA_ERROR_INVALID_HANDLE;

    memset(&req, 0, sizeof(req));
    req.start_vptr = (uint64_t)(uintptr_t)hStart;
    req.end_vptr = (uint64_t)(uintptr_t)hEnd;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_EVENT_ELAPSED, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    if (resp.status == CUDA_SUCCESS) *pMilliseconds = resp.milliseconds;
    return (CUresult)resp.status;
}

CUBE_GPU_EXPORT CUresult cuEventDestroy(CUevent hEvent) {
    cube_gpu_event_destroy_req_t req;
    cube_gpu_event_destroy_resp_t resp;

    ensure_initialized();
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;

    memset(&req, 0, sizeof(req));
    req.event_vptr = (uint64_t)(uintptr_t)hEvent;
    memset(&resp, 0, sizeof(resp));
    if (rpc_simple(CUBE_GPU_API_EVENT_DESTROY, &req, sizeof(req),
                   &resp, sizeof(resp), NULL) < 0) {
        return transport_failure();
    }
    return (CUresult)resp.status;
}

/* ─── CUDA IPC: blocked by policy ─────────────────────────────── */

/*
 * The daemon multiplexes many sandboxes onto one GPU. Sharing device
 * pointers through CUDA IPC would let one tenant read another's memory, so
 * these entry points fail locally instead of being forwarded.
 */
static CUresult ipc_blocked(const char *name) {
    static int warned = 0;
    if (!warned) {
        fprintf(stderr, "[cube-gpu] WARN: %s is disabled in multi-tenant "
                        "sandboxes\n", name);
        warned = 1;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUBE_GPU_EXPORT CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
    (void)pHandle;
    (void)dptr;
    return ipc_blocked("cuIpcGetMemHandle");
}

CUBE_GPU_EXPORT CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                            unsigned int Flags) {
    (void)pdptr;
    (void)handle;
    (void)Flags;
    return ipc_blocked("cuIpcOpenMemHandle");
}

CUBE_GPU_EXPORT CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    return ipc_blocked("cuIpcCloseMemHandle");
}

/* ─── Versioned aliases (driver ABI uses _v2 for many entry points) ── */

CUBE_GPU_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
    __attribute__((alias("cuMemAlloc")));
CUBE_GPU_EXPORT CUresult cuMemFree_v2(CUdeviceptr dptr)
    __attribute__((alias("cuMemFree")));
CUBE_GPU_EXPORT CUresult cuMemGetInfo_v2(size_t *free, size_t *total)
    __attribute__((alias("cuMemGetInfo")));
CUBE_GPU_EXPORT CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
    __attribute__((alias("cuCtxCreate")));
CUBE_GPU_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx)
    __attribute__((alias("cuCtxDestroy")));
CUBE_GPU_EXPORT CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
    __attribute__((alias("cuDeviceTotalMem")));
CUBE_GPU_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                         size_t ByteCount)
    __attribute__((alias("cuMemcpyHtoD")));
CUBE_GPU_EXPORT CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount)
    __attribute__((alias("cuMemcpyDtoH")));
CUBE_GPU_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                         size_t ByteCount)
    __attribute__((alias("cuMemcpyDtoD")));
CUBE_GPU_EXPORT CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                                              size_t ByteCount, CUstream hStream)
    __attribute__((alias("cuMemcpyHtoDAsync")));
CUBE_GPU_EXPORT CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                              size_t ByteCount, CUstream hStream)
    __attribute__((alias("cuMemcpyDtoHAsync")));
CUBE_GPU_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
    __attribute__((alias("cuMemsetD8")));
CUBE_GPU_EXPORT CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
    __attribute__((alias("cuMemsetD16")));
CUBE_GPU_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
    __attribute__((alias("cuMemsetD32")));
CUBE_GPU_EXPORT CUresult cuModuleLoadDataEx_v2(CUmodule *module, const void *image,
                                               unsigned int numOptions, void *options,
                                               void *optionValues)
    __attribute__((alias("cuModuleLoadDataEx")));
CUBE_GPU_EXPORT CUresult cuStreamDestroy_v2(CUstream hStream)
    __attribute__((alias("cuStreamDestroy")));
CUBE_GPU_EXPORT CUresult cuEventDestroy_v2(CUevent hEvent)
    __attribute__((alias("cuEventDestroy")));
CUBE_GPU_EXPORT CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                                               CUevent hEnd)
    __attribute__((alias("cuEventElapsedTime")));
CUBE_GPU_EXPORT CUresult cuCtxSynchronize_v2(void)
    __attribute__((alias("cuCtxSynchronize")));
CUBE_GPU_EXPORT CUresult cuCtxGetDevice_v2(CUdevice *device)
    __attribute__((alias("cuCtxGetDevice")));
CUBE_GPU_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
    __attribute__((alias("cuDevicePrimaryCtxRelease")));
CUBE_GPU_EXPORT CUresult cuModuleGetFunction_v2(CUfunction *hfunc, CUmodule hmod,
                                                const char *name)
    __attribute__((alias("cuModuleGetFunction")));
CUBE_GPU_EXPORT CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                               unsigned int Flags)
    __attribute__((alias("cuIpcOpenMemHandle")));

/* ─── Symbol resolution (dlsym / cuGetProcAddress) ────────────── */

/* Defined below; declared here so the symbol table can reference them. */
CUBE_GPU_EXPORT CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                          int cudaVersion, uint64_t flags);
CUBE_GPU_EXPORT CUresult cuGetProcAddress_v2(const char *symbol, void **pfn,
                                             int cudaVersion, uint64_t flags,
                                             int *symbolStatus);

typedef struct {
    const char *name;
    void *fn;
} cube_gpu_symbol_t;

static const cube_gpu_symbol_t g_symbols[] = {
    {"cuInit", (void *)&cuInit},
    {"cuDeviceGetCount", (void *)&cuDeviceGetCount},
    {"cuDeviceGet", (void *)&cuDeviceGet},
    {"cuDeviceGetName", (void *)&cuDeviceGetName},
    {"cuDeviceGetAttribute", (void *)&cuDeviceGetAttribute},
    {"cuDeviceTotalMem", (void *)&cuDeviceTotalMem},
    {"cuDeviceTotalMem_v2", (void *)&cuDeviceTotalMem},
    {"cuMemGetInfo", (void *)&cuMemGetInfo},
    {"cuMemGetInfo_v2", (void *)&cuMemGetInfo},
    {"cuCtxGetDevice", (void *)&cuCtxGetDevice},
    {"cuCtxGetDevice_v2", (void *)&cuCtxGetDevice},
    {"cuGetErrorString", (void *)&cuGetErrorString},
    {"cuGetErrorName", (void *)&cuGetErrorName},
    {"cuCtxCreate", (void *)&cuCtxCreate},
    {"cuCtxCreate_v2", (void *)&cuCtxCreate},
    {"cuCtxDestroy", (void *)&cuCtxDestroy},
    {"cuCtxDestroy_v2", (void *)&cuCtxDestroy},
    {"cuCtxSetCurrent", (void *)&cuCtxSetCurrent},
    {"cuCtxGetCurrent", (void *)&cuCtxGetCurrent},
    {"cuCtxSynchronize", (void *)&cuCtxSynchronize},
    {"cuCtxSynchronize_v2", (void *)&cuCtxSynchronize},
    {"cuDevicePrimaryCtxRetain", (void *)&cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxRelease", (void *)&cuDevicePrimaryCtxRelease},
    {"cuDevicePrimaryCtxRelease_v2", (void *)&cuDevicePrimaryCtxRelease},
    {"cuMemAlloc", (void *)&cuMemAlloc},
    {"cuMemAlloc_v2", (void *)&cuMemAlloc},
    {"cuMemFree", (void *)&cuMemFree},
    {"cuMemFree_v2", (void *)&cuMemFree},
    {"cuMemAllocAsync", (void *)&cuMemAllocAsync},
    {"cuMemFreeAsync", (void *)&cuMemFreeAsync},
    {"cuMemcpyHtoD", (void *)&cuMemcpyHtoD},
    {"cuMemcpyHtoD_v2", (void *)&cuMemcpyHtoD},
    {"cuMemcpyDtoH", (void *)&cuMemcpyDtoH},
    {"cuMemcpyDtoH_v2", (void *)&cuMemcpyDtoH},
    {"cuMemcpyDtoD", (void *)&cuMemcpyDtoD},
    {"cuMemcpyDtoD_v2", (void *)&cuMemcpyDtoD},
    {"cuMemcpyHtoDAsync", (void *)&cuMemcpyHtoDAsync},
    {"cuMemcpyHtoDAsync_v2", (void *)&cuMemcpyHtoDAsync},
    {"cuMemcpyDtoHAsync", (void *)&cuMemcpyDtoHAsync},
    {"cuMemcpyDtoHAsync_v2", (void *)&cuMemcpyDtoHAsync},
    {"cuMemsetD8", (void *)&cuMemsetD8},
    {"cuMemsetD8_v2", (void *)&cuMemsetD8},
    {"cuMemsetD16", (void *)&cuMemsetD16},
    {"cuMemsetD16_v2", (void *)&cuMemsetD16},
    {"cuMemsetD32", (void *)&cuMemsetD32},
    {"cuMemsetD32_v2", (void *)&cuMemsetD32},
    {"cuModuleLoadData", (void *)&cuModuleLoadData},
    {"cuModuleLoadDataEx", (void *)&cuModuleLoadDataEx},
    {"cuModuleLoadDataEx_v2", (void *)&cuModuleLoadDataEx},
    {"cuModuleGetFunction", (void *)&cuModuleGetFunction},
    {"cuModuleGetFunction_v2", (void *)&cuModuleGetFunction},
    {"cuLaunchKernel", (void *)&cuLaunchKernel},
    {"cuFuncGetAttribute", (void *)&cuFuncGetAttribute},
    {"cuStreamCreate", (void *)&cuStreamCreate},
    {"cuStreamDestroy", (void *)&cuStreamDestroy},
    {"cuStreamDestroy_v2", (void *)&cuStreamDestroy},
    {"cuStreamSynchronize", (void *)&cuStreamSynchronize},
    {"cuEventCreate", (void *)&cuEventCreate},
    {"cuEventRecord", (void *)&cuEventRecord},
    {"cuEventSynchronize", (void *)&cuEventSynchronize},
    {"cuEventElapsedTime", (void *)&cuEventElapsedTime},
    {"cuEventElapsedTime_v2", (void *)&cuEventElapsedTime},
    {"cuEventDestroy", (void *)&cuEventDestroy},
    {"cuEventDestroy_v2", (void *)&cuEventDestroy},
    {"cuIpcGetMemHandle", (void *)&cuIpcGetMemHandle},
    {"cuIpcOpenMemHandle", (void *)&cuIpcOpenMemHandle},
    {"cuIpcOpenMemHandle_v2", (void *)&cuIpcOpenMemHandle},
    {"cuIpcCloseMemHandle", (void *)&cuIpcCloseMemHandle},
    {"cuGetProcAddress", (void *)&cuGetProcAddress},
    {"cuGetProcAddress_v2", (void *)&cuGetProcAddress_v2},
};

static void *symbol_lookup(const char *name) {
    size_t i;
    if (!name) return NULL;
    for (i = 0; i < sizeof(g_symbols) / sizeof(g_symbols[0]); i++) {
        if (strcmp(g_symbols[i].name, name) == 0) return g_symbols[i].fn;
    }
    return NULL;
}

/* ─── cuGetProcAddress (CUDA >= 11.3) ─────────────────────────── */

CUBE_GPU_EXPORT CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                          int cudaVersion, uint64_t flags) {
    (void)cudaVersion;
    (void)flags;

    if (!pfn || !symbol) return CUDA_ERROR_INVALID_VALUE;

    *pfn = symbol_lookup(symbol);
    if (!*pfn) {
        log_dbg("cuGetProcAddress: %s not intercepted", symbol);
        return CUDA_ERROR_NOT_FOUND;
    }
    return CUDA_SUCCESS;
}

CUBE_GPU_EXPORT CUresult cuGetProcAddress_v2(const char *symbol, void **pfn,
                                             int cudaVersion, uint64_t flags,
                                             int *symbolStatus) {
    CUresult rc = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) *symbolStatus = (rc == CUDA_SUCCESS && pfn && *pfn) ? 0 : 1;
    return rc;
}

/* ─── dlsym interception ──────────────────────────────────────── */

static void *(*g_real_dlsym)(void *, const char *) = NULL;

__attribute__((constructor))
static void cube_gpu_resolve_real_dlsym(void) {
    g_real_dlsym = (void *(*)(void *, const char *))
        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
}

CUBE_GPU_EXPORT void *dlsym_225(void *handle, const char *symbol) {
    void *fn = symbol_lookup(symbol);
    if (fn) return fn;

    if (!g_real_dlsym) {
        g_real_dlsym = (void *(*)(void *, const char *))
            dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    return g_real_dlsym ? g_real_dlsym(handle, symbol) : NULL;
}

__asm__(".symver dlsym_225, dlsym@@GLIBC_2.2.5");

CUBE_GPU_EXPORT void *dlsym_234(void *handle, const char *symbol)
    __attribute__((alias("dlsym_225")));
__asm__(".symver dlsym_234, dlsym@GLIBC_2.34");
