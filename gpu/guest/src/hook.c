/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA hook layer: intercepts CUDA Driver API calls via LD_PRELOAD and
 * forwards them to cube-gpu-daemon on the host via RPC over vsock.
 *
 * Derived from nvshare's hook.c mechanism:
 *   - dlsym interception (glibc 2.2.5 + 2.34)
 *   - cuGetProcAddress / cuGetProcAddress_v2 interception
 *   - pthread_once two-phase initialization
 *
 * Backend is replaced: instead of calling real_cuXxx locally,
 * we serialize the call and send via rpc_client.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>

#include "cuda_defs.h"
#include "rpc_client.h"
#include "shm_ring.h"
#include "rpc_protocol.h"

/* ─── Environment variable names ──────────────────────── */

#define ENV_CUBE_GPU_DEBUG        "CUBE_GPU_DEBUG"
#define ENV_CUBE_GPU_SANDBOX_ID   "CUBE_GPU_SANDBOX_ID"
#define ENV_CUBE_GPU_MEMORY_QUOTA "CUBE_GPU_MEMORY_QUOTA"
#define ENV_CUBE_GPU_VSOCK_PORT   "CUBE_GPU_VSOCK_PORT"
#define ENV_CUBE_GPU_SHM_PATH     "CUBE_GPU_SHM_PATH"

#define VMADDR_CID_HOST 2

/* ─── Global state ────────────────────────────────────── */

static pthread_once_t init_done = PTHREAD_ONCE_INIT;

static cube_gpu_rpc_client_t g_rpc = {0};
static cube_gpu_shm_channel_t g_shm = {0};

static int g_debug = 0;
static uint64_t g_sandbox_id = 0;
static uint64_t g_memory_quota = 0;
static uint32_t g_vsock_port = CUBE_GPU_VSOCK_PORT;

/* Virtual handle mapping: we track every virtual pointer given to the
 * application and keep it valid locally. The host maps virtual→real. */
#define VPTR_TABLE_INIT_CAP 256

typedef struct {
    uint64_t vptr;
    uint64_t host_handle;
    size_t   size;
} vptr_entry_t;

static vptr_entry_t *g_vptr_table = NULL;
static size_t g_vptr_count = 0;
static size_t g_vptr_cap = 0;
static pthread_mutex_t g_vptr_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Same for modules, functions, and streams */
typedef struct {
    uint64_t vhandle;
    uint64_t host_handle;
} handle_entry_t;

static handle_entry_t *g_modules = NULL;
static size_t g_module_count = 0;
static size_t g_module_cap = 0;
static handle_entry_t *g_functions = NULL;
static size_t g_func_count = 0;
static size_t g_func_cap = 0;
static handle_entry_t *g_streams = NULL;
static size_t g_stream_count = 0;
static size_t g_stream_cap = 0;

static uint64_t g_next_vptr = 0x10000000ULL;
static uint64_t g_next_module = 0x20000000ULL;
static uint64_t g_next_func = 0x30000000ULL;
static uint64_t g_next_stream = 0x40000000ULL;

/* ─── Logging ─────────────────────────────────────────── */

#define log_err(fmt, ...) fprintf(stderr, "[cube-gpu] ERROR: " fmt "\n", ##__VA_ARGS__)
#define log_dbg(fmt, ...) do { if (g_debug) fprintf(stderr, "[cube-gpu] DBG: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ─── Handle table helpers ────────────────────────────── */

static uint64_t alloc_vptr(void) { return __sync_fetch_and_add(&g_next_vptr, 1); }
static uint64_t alloc_module(void) { return __sync_fetch_and_add(&g_next_module, 1); }
static uint64_t alloc_func(void) { return __sync_fetch_and_add(&g_next_func, 1); }
static uint64_t alloc_stream(void) { return __sync_fetch_and_add(&g_next_stream, 1); }

static void register_vptr(uint64_t vptr, uint64_t host_handle, size_t size) {
    pthread_mutex_lock(&g_vptr_mutex);
    if (g_vptr_count >= g_vptr_cap) {
        size_t new_cap = g_vptr_cap == 0 ? VPTR_TABLE_INIT_CAP : g_vptr_cap * 2;
        g_vptr_table = realloc(g_vptr_table, new_cap * sizeof(vptr_entry_t));
        g_vptr_cap = new_cap;
    }
    g_vptr_table[g_vptr_count].vptr = vptr;
    g_vptr_table[g_vptr_count].host_handle = host_handle;
    g_vptr_table[g_vptr_count].size = size;
    g_vptr_count++;
    pthread_mutex_unlock(&g_vptr_mutex);
}

static uint64_t find_host_handle(uint64_t vptr) {
    pthread_mutex_lock(&g_vptr_mutex);
    for (size_t i = 0; i < g_vptr_count; i++) {
        if (g_vptr_table[i].vptr == vptr) {
            uint64_t h = g_vptr_table[i].host_handle;
            pthread_mutex_unlock(&g_vptr_mutex);
            return h;
        }
    }
    pthread_mutex_unlock(&g_vptr_mutex);
    return 0;
}

static void unregister_vptr(uint64_t vptr) {
    pthread_mutex_lock(&g_vptr_mutex);
    for (size_t i = 0; i < g_vptr_count; i++) {
        if (g_vptr_table[i].vptr == vptr) {
            g_vptr_table[i] = g_vptr_table[g_vptr_count - 1];
            g_vptr_count--;
            break;
        }
    }
    pthread_mutex_unlock(&g_vptr_mutex);
}

static void register_handle(handle_entry_t **tbl, size_t *count, size_t *cap,
                            uint64_t vhandle, uint64_t host_handle) {
    if (*count >= *cap) {
        size_t new_cap = *cap == 0 ? 64 : *cap * 2;
        *tbl = realloc(*tbl, new_cap * sizeof(handle_entry_t));
        *cap = new_cap;
    }
    (*tbl)[*count].vhandle = vhandle;
    (*tbl)[*count].host_handle = host_handle;
    (*count)++;
}

static uint64_t find_handle(handle_entry_t *tbl, size_t count, uint64_t vhandle) {
    for (size_t i = 0; i < count; i++) {
        if (tbl[i].vhandle == vhandle) return tbl[i].host_handle;
    }
    return 0;
}

/* ─── Initialization ──────────────────────────────────── */

static void initialize(void) {
    const char *dbg = getenv(ENV_CUBE_GPU_DEBUG);
    g_debug = (dbg && dbg[0] == '1');

    const char *sid = getenv(ENV_CUBE_GPU_SANDBOX_ID);
    if (sid) g_sandbox_id = strtoull(sid, NULL, 10);

    const char *quota = getenv(ENV_CUBE_GPU_MEMORY_QUOTA);
    if (quota) g_memory_quota = strtoull(quota, NULL, 10);
    else g_memory_quota = 4ULL * 1024 * 1024 * 1024;

    const char *port = getenv(ENV_CUBE_GPU_VSOCK_PORT);
    if (port) g_vsock_port = (uint32_t)strtoul(port, NULL, 10);

    log_dbg("initializing: sandbox_id=%lu quota=%lu port=%u",
            g_sandbox_id, g_memory_quota, g_vsock_port);

    if (cube_gpu_rpc_client_init(&g_rpc, VMADDR_CID_HOST, g_vsock_port,
                                  g_sandbox_id, g_memory_quota) < 0) {
        log_err("failed to connect to cube-gpu-daemon");
        return;
    }

    const char *shm_path = getenv(ENV_CUBE_GPU_SHM_PATH);
    if (shm_path) {
        int fd = open(shm_path, O_RDWR);
        if (fd >= 0) {
            void *base = mmap(NULL, CUBE_GPU_SHM_RING_SIZE * 2,
                              PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if (base != MAP_FAILED) {
                cube_gpu_shm_channel_init(&g_shm, base, CUBE_GPU_SHM_RING_SIZE * 2);
                log_dbg("SHM channel initialized at %s", shm_path);
            }
        }
    }

    log_dbg("initialization complete");
}

/* ─── RPC helper ──────────────────────────────────────── */

static int rpc_call(cube_gpu_api_id_t api, const void *req, size_t req_len,
                    cube_gpu_rpc_header_t *resp_hdr, void *resp, size_t resp_cap) {
    if (!g_rpc.initialized) return -1;

    cube_gpu_rpc_header_t hdr;
    cube_gpu_init_header(&hdr, api, g_rpc.next_request_id++,
                         g_rpc.session_id, (uint32_t)req_len, CUBE_GPU_FLAG_NONE);

    if (cube_gpu_rpc_send(&g_rpc, &hdr, req) < 0) {
        log_err("rpc_send failed for api=%d", api);
        return -1;
    }

    if (cube_gpu_rpc_recv(&g_rpc, resp_hdr, resp, resp_cap) < 0) {
        log_err("rpc_recv failed for api=%d", api);
        return -1;
    }
    return 0;
}

/* ─── Real dlsym (via dlvsym, same as nvshare) ────────── */

static void *(*real_dlsym_225)(void *, const char *) = NULL;

__attribute__((constructor))
static void get_real_dlsym(void) {
    real_dlsym_225 = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
}

/* ─── CUDA Hook functions ─────────────────────────────── */

CUresult cuInit(unsigned int flags) {
    pthread_once(&init_done, initialize);
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    pthread_once(&init_done, initialize);
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    cube_gpu_malloc_req_t req = {.size = bytesize};
    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_malloc_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_MALLOC, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_OUT_OF_MEMORY;

    if (resp.status == CUDA_SUCCESS && resp.vptr != 0) {
        uint64_t vptr = alloc_vptr();
        register_vptr(vptr, resp.vptr, bytesize);
        *dptr = (CUdeviceptr)vptr;
        log_dbg("cuMemAlloc: vptr=0x%lx -> host=0x%lx size=%zu", vptr, resp.vptr, bytesize);
    }
    return (CUresult)resp.status;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dptr == 0) return CUDA_SUCCESS;

    uint64_t host_handle = find_host_handle((uint64_t)dptr);
    if (host_handle == 0) return CUDA_ERROR_INVALID_VALUE;

    cube_gpu_free_req_t req = {.vptr = host_handle};
    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_free_resp_t resp = {0};

    CUresult ret = CUDA_SUCCESS;
    if (rpc_call(CUBE_GPU_API_FREE, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        ret = CUDA_ERROR_LAUNCH_FAILED;
    else ret = (CUresult)resp.status;

    if (ret == CUDA_SUCCESS) unregister_vptr((uint64_t)dptr);
    return ret;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    uint64_t host_dst = find_host_handle((uint64_t)dstDevice);
    if (host_dst == 0) return CUDA_ERROR_INVALID_VALUE;

    uint32_t flags = CUBE_GPU_FLAG_NONE;
    uint64_t shm_offset = 0;

    if (ByteCount > CUBE_GPU_SHM_THRESHOLD && g_shm.base) {
        size_t written = cube_gpu_shm_ring_write(&g_shm.g2h, srcHost, ByteCount);
        if (written == ByteCount) {
            flags = CUBE_GPU_FLAG_SHM_DATA;
            shm_offset = 0;
        }
    }

    cube_gpu_memcpy_req_t req = {
        .kind = CUBE_GPU_MEMCPY_HOST_TO_DEVICE,
        .dst = host_dst,
        .src = 0,
        .size = ByteCount,
        .shm_offset = shm_offset
    };

    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_memcpy_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_MEMCPY, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;
    return (CUresult)resp.status;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    uint64_t host_src = find_host_handle((uint64_t)srcDevice);
    if (host_src == 0) return CUDA_ERROR_INVALID_VALUE;

    cube_gpu_memcpy_req_t req = {
        .kind = CUBE_GPU_MEMCPY_DEVICE_TO_HOST,
        .dst = 0,
        .src = host_src,
        .size = ByteCount,
        .shm_offset = 0
    };
    if (ByteCount > CUBE_GPU_SHM_THRESHOLD && g_shm.base)
        req.shm_offset = 0;

    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_memcpy_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_MEMCPY, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;

    if (resp.status == CUDA_SUCCESS && ByteCount > 0 && g_shm.base) {
        cube_gpu_shm_ring_read(&g_shm.h2g, dstHost, ByteCount);
    }
    return (CUresult)resp.status;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    uint64_t host_func = find_handle(g_functions, g_func_count, (uint64_t)f);
    uint64_t host_stream = hStream ? find_handle(g_streams, g_stream_count, (uint64_t)hStream) : 0;

    if (host_func == 0) return CUDA_ERROR_INVALID_VALUE;

    /* Serialize kernel params into SHM.
     * kernelParams[i] points to the i-th argument value. On 64-bit, each
     * argument occupies at most sizeof(void*) bytes for scalars/pointers,
     * which covers the common CUDA kernel parameter types (int, float,
     * double, pointers). The host reconstructs the argument array from
     * the serialized blob. */
    uint64_t params_shm_offset = 0;
    uint64_t params_size = 0;
    if (kernelParams && g_shm.base) {
        void **p = kernelParams;
        size_t offset = 0;
        while (*p != NULL) {
            size_t arg_size = sizeof(void *);
            if (offset + arg_size <= g_shm.g2h.data_size) {
                cube_gpu_shm_ring_write(&g_shm.g2h, *p, arg_size);
                offset += arg_size;
            }
            p++;
        }
        params_size = offset;
    }

    /* extra uses CU_LAUNCH_PARAM_* key-value pairs. P0: not supported,
     * pass through as zero-size. Host will reject if extra is non-NULL
     * and the kernel requires launch bounds. */
    uint64_t extra_shm_offset = 0;
    uint64_t extra_size = 0;
    (void)extra;

    cube_gpu_launch_kernel_req_t req = {
        .func_handle = host_func,
        .grid_dim_x = gridDimX, .grid_dim_y = gridDimY, .grid_dim_z = gridDimZ,
        .block_dim_x = blockDimX, .block_dim_y = blockDimY, .block_dim_z = blockDimZ,
        .shared_mem_bytes = sharedMemBytes,
        .stream_vptr = host_stream,
        .params_shm_offset = params_shm_offset,
        .params_size = params_size,
        .extra_shm_offset = extra_shm_offset,
        .extra_size = extra_size
    };

    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_launch_kernel_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_LAUNCH_KERNEL, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;
    return (CUresult)resp.status;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;

    cube_gpu_stream_create_req_t req = {.flags = Flags};
    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_stream_create_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_STREAM_CREATE, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;

    if (resp.status == CUDA_SUCCESS && resp.stream_vptr != 0) {
        uint64_t vstream = alloc_stream();
        register_handle(&g_streams, &g_stream_count, &g_stream_cap, vstream, resp.stream_vptr);
        *phStream = (CUstream)(uintptr_t)vstream;
    }
    return (CUresult)resp.status;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    uint64_t host_stream = hStream ? find_handle(g_streams, g_stream_count, (uint64_t)hStream) : 0;

    cube_gpu_stream_sync_req_t req = {.stream_vptr = host_stream};
    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_stream_sync_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_STREAM_SYNC, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;
    return (CUresult)resp.status;
}

CUresult cuCtxSynchronize(void) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;

    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_device_sync_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_DEVICE_SYNC, NULL, 0, &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;
    return (CUresult)resp.status;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;

    /* Determine image size: PTX is text, cubin has a header. Approximate. */
    size_t image_size = 0;
    const char *p = (const char *)image;
    while (image_size < 64 * 1024 * 1024) {
        if (p[image_size] == '\0' && image_size > 16) break;
        image_size++;
    }
    image_size++;

    uint64_t shm_offset = 0;
    if (g_shm.base) {
        cube_gpu_shm_ring_write(&g_shm.g2h, image, image_size);
    }

    cube_gpu_module_load_data_req_t req = {
        .image_size = image_size,
        .shm_offset = shm_offset
    };
    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_module_load_data_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_MODULE_LOAD_DATA, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;

    if (resp.status == CUDA_SUCCESS && resp.module_handle != 0) {
        uint64_t vmod = alloc_module();
        register_handle(&g_modules, &g_module_count, &g_module_cap, vmod, resp.module_handle);
        *module = (CUmodule)(uintptr_t)vmod;
    }
    return (CUresult)resp.status;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!g_rpc.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;

    uint64_t host_module = find_handle(g_modules, g_module_count, (uint64_t)hmod);
    if (host_module == 0) return CUDA_ERROR_INVALID_VALUE;

    cube_gpu_module_get_func_req_t req;
    memset(&req, 0, sizeof(req));
    req.module_handle = host_module;
    strncpy(req.func_name, name, sizeof(req.func_name) - 1);

    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_module_get_func_resp_t resp = {0};

    if (rpc_call(CUBE_GPU_API_MODULE_GET_FUNC, &req, sizeof(req), &resp_hdr, &resp, sizeof(resp)) < 0)
        return CUDA_ERROR_LAUNCH_FAILED;

    if (resp.status == CUDA_SUCCESS && resp.func_handle != 0) {
        uint64_t vfunc = alloc_func();
        register_handle(&g_functions, &g_func_count, &g_func_cap, vfunc, resp.func_handle);
        *hfunc = (CUfunction)(uintptr_t)vfunc;
    }
    return (CUresult)resp.status;
}

/* ─── dlsym hook (glibc 2.2.5) ────────────────────────── */

void *dlsym_225(void *handle, const char *symbol) {
    if (!real_dlsym_225) return NULL;

    if (!symbol || strncmp(symbol, "cu", 2) != 0)
        return real_dlsym_225(handle, symbol);

    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuInit)) == 0)                  return (void *)&cuInit;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0)              return (void *)&cuMemAlloc;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0)              return (void *)&cuMemFree;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyHtoD)) == 0)           return (void *)&cuMemcpyHtoD;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoH)) == 0)           return (void *)&cuMemcpyDtoH;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuLaunchKernel)) == 0)         return (void *)&cuLaunchKernel;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuStreamCreate)) == 0)         return (void *)&cuStreamCreate;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuStreamSynchronize)) == 0)    return (void *)&cuStreamSynchronize;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxSynchronize)) == 0)       return (void *)&cuCtxSynchronize;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuModuleLoadData)) == 0)       return (void *)&cuModuleLoadData;
    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuModuleGetFunction)) == 0)    return (void *)&cuModuleGetFunction;

    return real_dlsym_225(handle, symbol);
}

__asm__(".symver dlsym_225, dlsym@@GLIBC_2.2.5");

/* ─── dlsym hook (glibc 2.34) ─────────────────────────── */

void *dlsym_234(void *handle, const char *symbol) __attribute__((alias("dlsym_225")));
__asm__(".symver dlsym_234, dlsym@GLIBC_2.34");

/* ─── cuGetProcAddress hook (CUDA >= 11.3) ─────────────── */

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags) {
    pthread_once(&init_done, initialize);

    if (!pfn || !symbol) return CUDA_ERROR_INVALID_VALUE;

    if (strcmp(symbol, "cuInit") == 0)                       { *pfn = (void *)&cuInit; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuMemAlloc") == 0)                   { *pfn = (void *)&cuMemAlloc; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuMemFree") == 0)                   { *pfn = (void *)&cuMemFree; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuMemcpyHtoD") == 0)                { *pfn = (void *)&cuMemcpyHtoD; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuMemcpyDtoH") == 0)                { *pfn = (void *)&cuMemcpyDtoH; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuLaunchKernel") == 0)              { *pfn = (void *)&cuLaunchKernel; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuStreamCreate") == 0)              { *pfn = (void *)&cuStreamCreate; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuStreamSynchronize") == 0)         { *pfn = (void *)&cuStreamSynchronize; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuCtxSynchronize") == 0)            { *pfn = (void *)&cuCtxSynchronize; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuModuleLoadData") == 0)            { *pfn = (void *)&cuModuleLoadData; return CUDA_SUCCESS; }
    if (strcmp(symbol, "cuModuleGetFunction") == 0)         { *pfn = (void *)&cuModuleGetFunction; return CUDA_SUCCESS; }

    /* Not intercepted: return NULL to let CUDA handle it (will likely fail,
     * which is expected for APIs we don't yet support). */
    *pfn = NULL;
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             uint64_t flags, int *symbolStatus) {
    CUresult ret = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) *symbolStatus = (ret == CUDA_SUCCESS && *pfn != NULL) ? 0 : 1;
    return ret;
}
