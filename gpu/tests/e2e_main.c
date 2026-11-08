/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * e2e_main -- end-to-end exercise of a fully hooked CUDA Driver API.
 *
 * The program is intentionally dependency-free: it declares the small
 * subset of CUDA types/constants it needs and resolves every entry point
 * at runtime through dlsym(RTLD_DEFAULT, ...).  Under LD_PRELOAD this hits
 * libcube-gpu.so's dlsym hook for the APIs it remotes; everything the hook
 * does not export falls back to the real libcuda (loaded RTLD_GLOBAL), which
 * is exactly how a quota / interception hole shows up as a test failure.
 *
 * Usage:
 *   e2e_main [--cubin PATH] [--ptx PATH] [--quota-oom]
 *
 * Env:
 *   CUBE_GPU_MEMORY_QUOTA   expected per-session quota (bytes)
 *   E2E_REQUIRE_BIGARG=1    turn a NOT_SUPPORTED on the >8B param test into
 *                           a hard failure instead of a SKIP
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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


/* ─── Minimal CUDA Driver API surface ──────────────────────────────── */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUevent;

#define CUDA_SUCCESS               0
#define CUDA_ERROR_INVALID_VALUE   1
#define CUDA_ERROR_OUT_OF_MEMORY   2
#define CUDA_ERROR_NOT_SUPPORTED   801

typedef struct {
    CUresult (*cuInit)(unsigned int flags);
    CUresult (*cuGetErrorName)(CUresult err, const char **name);
    CUresult (*cuDeviceGetCount)(int *count);
    CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
    CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);
    CUresult (*cuDeviceTotalMem)(size_t *bytes, CUdevice dev);
    CUresult (*cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
    CUresult (*cuCtxDestroy)(CUcontext ctx);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuMemGetInfo)(size_t *freeMem, size_t *totalMem);
    CUresult (*cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
    CUresult (*cuMemFree)(CUdeviceptr dptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr dst, const void *src, size_t n);
    CUresult (*cuMemcpyDtoH)(void *dst, CUdeviceptr src, size_t n);
    CUresult (*cuModuleLoadData)(CUmodule *module, const void *image);
    CUresult (*cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
    CUresult (*cuLaunchKernel)(CUfunction f,
                               unsigned int gx, unsigned int gy, unsigned int gz,
                               unsigned int bx, unsigned int by, unsigned int bz,
                               unsigned int sharedMemBytes, CUstream stream,
                               void **kernelParams, void **extra);
    CUresult (*cuMemsetD32)(CUdeviceptr dst, unsigned int ui, size_t n);
    CUresult (*cuEventCreate)(CUevent *event, unsigned int flags);
    CUresult (*cuEventRecord)(CUevent event, CUstream stream);
    CUresult (*cuEventSynchronize)(CUevent event);
    CUresult (*cuEventElapsedTime)(float *ms, CUevent start, CUevent end);
    CUresult (*cuEventDestroy)(CUevent event);
    CUresult (*cuMemAllocAsync)(CUdeviceptr *dptr, size_t bytesize, CUstream stream);
    CUresult (*cuMemFreeAsync)(CUdeviceptr dptr, CUstream stream);
} cuda_api_t;

static cuda_api_t api;

/* ─── Result bookkeeping ───────────────────────────────────────────── */

static int g_pass;
static int g_fail;
static int g_skip;

static uint64_t g_quota;   /* 0 when CUBE_GPU_MEMORY_QUOTA is unset */

static const char *error_name(CUresult res)
{
    static char fallback[64];
    const char *name = NULL;

    if (api.cuGetErrorName && api.cuGetErrorName(res, &name) == CUDA_SUCCESS && name)
        return name;
    snprintf(fallback, sizeof(fallback), "CUDA error %d", (int)res);
    return fallback;
}

static void pass(const char *what)
{
    g_pass++;
    printf("[PASS] %s\n", what);
    fflush(stdout);
}

static void skip(const char *what, const char *why)
{
    g_skip++;
    printf("[SKIP] %s: %s\n", what, why);
    fflush(stdout);
}

static void fail_cuda(const char *what, CUresult res)
{
    g_fail++;
    printf("[FAIL] %s: %s (%d)\n", what, error_name(res), (int)res);
    fflush(stdout);
}

static void fail_msg(const char *what, const char *fmt, ...)
{
    va_list ap;

    g_fail++;
    printf("[FAIL] %s: ", what);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

/* ─── Symbol resolution ────────────────────────────────────────────── */

static int g_guest_hooked;   /* libcube-gpu.so interposes CUDA symbols */

static int symbol_from_guest(void *fn)
{
    Dl_info info;
    return fn && dladdr(fn, &info) && info.dli_fname &&
           strstr(info.dli_fname, "libcube-gpu") != NULL;
}

/*
 * Pick the ABI spelling that is actually remoted by the guest library.
 *
 * CUDA's headers rename several entry points (cuMemAlloc -> cuMemAlloc_v2),
 * but the set of _v2 aliases differs between CUDA versions and from what
 * libcube-gpu.so exports/handles.  Calling an un-remoted _v2 symbol with a
 * virtual handle dereferences it inside libcuda and crashes, so when the
 * guest library is loaded we prefer whichever candidate resolves into it.
 */
static void *resolve_pref(const char *first, const char *second)
{
    void *a = first ? dlsym(RTLD_DEFAULT, first) : NULL;
    void *b = second ? dlsym(RTLD_DEFAULT, second) : NULL;

    if (g_guest_hooked) {
        if (symbol_from_guest(a)) return a;
        if (symbol_from_guest(b)) return b;
    }
    return a ? a : b;
}

static int resolve_syms(void)
{
#define LOAD(field, sym_v2, sym_plain)                                             \
    do {                                                                           \
        void *p_ = resolve_pref((sym_v2), (sym_plain));                            \
        if (!p_) {                                                                 \
            fprintf(stderr, "FATAL: cannot resolve CUDA symbol '%s'\n",            \
                    (sym_plain) ? (sym_plain) : (sym_v2));                         \
            return -1;                                                             \
        }                                                                          \
        api.field = (void *)p_;                                                    \
    } while (0)

    g_guest_hooked = symbol_from_guest(dlsym(RTLD_DEFAULT, "cuInit"));

    LOAD(cuInit,            NULL,                     "cuInit");
    LOAD(cuDeviceGetCount,  NULL,                     "cuDeviceGetCount");
    LOAD(cuDeviceGet,       NULL,                     "cuDeviceGet");
    LOAD(cuDeviceGetName,   NULL,                     "cuDeviceGetName");
    LOAD(cuDeviceTotalMem,  "cuDeviceTotalMem_v2",    "cuDeviceTotalMem");
    LOAD(cuCtxCreate,       "cuCtxCreate_v2",         "cuCtxCreate");
    LOAD(cuCtxDestroy,      "cuCtxDestroy_v2",        "cuCtxDestroy");
    /*
     * CUDA 13.3 dropped the cuCtxSynchronize -> _v2 alias: modern binaries
     * call the unversioned name, and libcuda's legacy _v2 stub returns
     * CUDA_ERROR_CONTEXT_IS_DESTROYED.  Order the candidates accordingly.
     */
    if (g_guest_hooked)
        LOAD(cuCtxSynchronize, "cuCtxSynchronize_v2", "cuCtxSynchronize");
    else
        LOAD(cuCtxSynchronize, "cuCtxSynchronize", "cuCtxSynchronize_v2");
    LOAD(cuMemGetInfo,      "cuMemGetInfo_v2",        "cuMemGetInfo");
    LOAD(cuMemAlloc,        "cuMemAlloc_v2",          "cuMemAlloc");
    LOAD(cuMemFree,         "cuMemFree_v2",           "cuMemFree");
    LOAD(cuMemcpyHtoD,      "cuMemcpyHtoD_v2",        "cuMemcpyHtoD");
    LOAD(cuMemcpyDtoH,      "cuMemcpyDtoH_v2",        "cuMemcpyDtoH");
    LOAD(cuModuleLoadData,  NULL,                     "cuModuleLoadData");
    LOAD(cuModuleGetFunction, "cuModuleGetFunction_v2", "cuModuleGetFunction");
    LOAD(cuLaunchKernel,    NULL,                     "cuLaunchKernel");
    LOAD(cuMemsetD32,       "cuMemsetD32_v2",         "cuMemsetD32");
    LOAD(cuEventCreate,     NULL,                     "cuEventCreate");
    LOAD(cuEventRecord,     NULL,                     "cuEventRecord");
    LOAD(cuEventSynchronize, NULL,                    "cuEventSynchronize");
    LOAD(cuEventElapsedTime, "cuEventElapsedTime_v2", "cuEventElapsedTime");
    LOAD(cuEventDestroy,    "cuEventDestroy_v2",      "cuEventDestroy");
    LOAD(cuMemAllocAsync,   NULL,                     "cuMemAllocAsync");
    LOAD(cuMemFreeAsync,    NULL,                     "cuMemFreeAsync");
#undef LOAD

    /* Optional diagnostic helper; absence is not fatal. */
    api.cuGetErrorName = (void *)dlsym(RTLD_DEFAULT, "cuGetErrorName");
    return 0;
}

/* ─── Helpers ──────────────────────────────────────────────────────── */

static void *read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    long sz;
    unsigned char *buf;

    if (!f) {
        fprintf(stderr, "FATAL: cannot open module image '%s': %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    sz = ftell(f);
    if (sz <= 0) {
        fprintf(stderr, "FATAL: module image '%s' is empty\n", path);
        fclose(f);
        return NULL;
    }
    rewind(f);
    /* +1 so PTX images are NUL-terminated as cuModuleLoadData expects. */
    buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "FATAL: short read on '%s'\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[sz] = '\0';
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

static void fill_pattern(unsigned char *buf, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++)
        buf[i] = (unsigned char)(seed + i * 31u);
}

static int buffer_matches(const unsigned char *a, const unsigned char *b, size_t n)
{
    return n == 0 || memcmp(a, b, n) == 0;
}

static int env_flag(const char *name)
{
    const char *v = getenv(name);
    return v && v[0] == '1';
}

/* Kernel-parameter metadata for the mixed kernel: [4, 8, 8, 8]. */
static void print_mixed_meta(void)
{
    printf("       mixed param sizes: {4, 8, 8, 8} (int, double, float*, char*)\n");
}

/* ─── Quota OOM mode ───────────────────────────────────────────────── */

static int run_quota_oom(void)
{
    CUresult res;
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUdeviceptr ptr = 0;
    size_t size;
    int count = 0;

    res = api.cuInit(0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuInit", res); goto done; }
    res = api.cuDeviceGetCount(&count);
    if (res != CUDA_SUCCESS || count < 1) { fail_cuda("cuDeviceGetCount", res); goto done; }
    res = api.cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuDeviceGet", res); goto done; }
    if (g_quota == 0) {
        /* Zero-config mode: the quota comes from the shared-memory header
         * via the daemon's quota-adjusted device view. */
        size_t qm = 0;
        res = api.cuDeviceTotalMem(&qm, dev);
        if (res == CUDA_SUCCESS && qm > 0) g_quota = (uint64_t)qm;
    }
    if (g_quota == 0) {
        fail_msg("quota-oom", "no GPU quota in env or device view");
        goto done;
    }
    res = api.cuCtxCreate(&ctx, 0, dev);
    if (res != CUDA_SUCCESS) { fail_cuda("cuCtxCreate", res); goto done; }

    /* Ask for quota + 16 MiB: must be rejected by the quota tracker. */
    size = (size_t)g_quota + (16u * 1024u * 1024u);
    res = api.cuMemAlloc(&ptr, size);
    if (res == CUDA_ERROR_OUT_OF_MEMORY) {
        printf("[PASS] quota-oom: %zu bytes over a %llu byte quota -> CUDA_ERROR_OUT_OF_MEMORY\n",
               size, (unsigned long long)g_quota);
        g_pass++;
    } else if (res == CUDA_SUCCESS) {
        fail_msg("quota-oom", "allocation of %zu bytes succeeded with quota %llu "
                 "(quota not enforced; physical GPU is larger)", size,
                 (unsigned long long)g_quota);
        api.cuMemFree(ptr);
    } else {
        fail_cuda("quota-oom cuMemAlloc", res);
    }

done:
    if (ctx) api.cuCtxDestroy(ctx);
    printf("SUMMARY: %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
    return g_fail == 0 ? 0 : 1;
}

/* ─── Main end-to-end sequence ─────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *cubin_path = "gpu/tests/kernel.cubin";
    const char *ptx_path = NULL;
    const char *module_path;
    int quota_oom = 0;
    void *cuda_lib = NULL;
    unsigned char *image = NULL;
    size_t image_size = 0;
    int i;

    int count = 0;
    CUdevice dev = 0;
    char dev_name[256];
    size_t total_mem = 0;
    size_t free_mem = 0, total_mem_info = 0;
    CUcontext ctx = NULL;
    CUmodule module = NULL;
    CUfunction f_add = NULL, f_mixed = NULL, f_big = NULL;
    CUdeviceptr ptr = 0, mixed_out = 0, pad_dev = 0, big_out = 0;
    CUdeviceptr memset_buf = 0, async_ptr = 0;
    CUevent ev0 = NULL, ev1 = NULL;
    CUresult res;
    int rc = 1;
    const size_t big_sz = 1024 * 1024;
    unsigned char *host_a = NULL, *host_b = NULL;
    float *flt_a = NULL, *flt_b = NULL;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cubin") == 0 && i + 1 < argc) {
            cubin_path = argv[++i];
        } else if (strcmp(argv[i], "--ptx") == 0 && i + 1 < argc) {
            ptx_path = argv[++i];
        } else if (strcmp(argv[i], "--quota-oom") == 0) {
            quota_oom = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("usage: %s [--cubin PATH] [--ptx PATH] [--quota-oom]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    {
        const char *q = getenv("CUBE_GPU_MEMORY_QUOTA");
        if (q && q[0])
            g_quota = strtoull(q, NULL, 10);
    }

    cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda_lib) {
        fprintf(stderr, "FATAL: dlopen(libcuda.so.1) failed: %s\n", dlerror());
        return 2;
    }
    if (resolve_syms() != 0)
        return 2;

    printf("== CubeSandbox GPU E2E (quota=%llu, mode=%s) ==\n",
           (unsigned long long)g_quota, quota_oom ? "quota-oom" : "full");

    if (quota_oom) {
        rc = run_quota_oom();
        dlclose(cuda_lib);
        return rc;
    }

    /* 1) cuInit */
    res = api.cuInit(0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuInit", res); goto done; }
    pass("cuInit(0)");

    /* 2) device enumeration */
    res = api.cuDeviceGetCount(&count);
    if (res != CUDA_SUCCESS) { fail_cuda("cuDeviceGetCount", res); goto done; }
    if (count < 1) { fail_msg("cuDeviceGetCount", "expected >= 1 device, got %d", count); goto done; }
    pass("cuDeviceGetCount >= 1");

    res = api.cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuDeviceGet(0)", res); goto done; }
    memset(dev_name, 0, sizeof(dev_name));
    res = api.cuDeviceGetName(dev_name, (int)sizeof(dev_name), dev);
    if (res != CUDA_SUCCESS) { fail_cuda("cuDeviceGetName", res); goto done; }
    printf("[PASS] cuDeviceGet/cuDeviceGetName (device 0 = \"%s\")\n", dev_name);
    g_pass++;

    /* 3) quota-adjusted total memory */
    res = api.cuDeviceTotalMem(&total_mem, dev);
    if (res != CUDA_SUCCESS) { fail_cuda("cuDeviceTotalMem", res); goto done; }
    printf("       cuDeviceTotalMem = %zu bytes\n", total_mem);
    if (g_quota != 0) {
        if (total_mem != (size_t)g_quota) {
            fail_msg("cuDeviceTotalMem quota view",
                     "got %zu bytes, expected quota %llu (physical GPU leaking through?)",
                     total_mem, (unsigned long long)g_quota);
            goto done;
        }
        printf("[PASS] cuDeviceTotalMem == quota (%llu bytes)\n",
               (unsigned long long)g_quota);
        g_pass++;
    } else {
        pass("cuDeviceTotalMem");
    }

    /* 4) context */
    res = api.cuCtxCreate(&ctx, 0, dev);
    if (res != CUDA_SUCCESS) { fail_cuda("cuCtxCreate", res); goto done; }
    pass("cuCtxCreate");

    /* 5) cuMemGetInfo: free ~= quota right after context creation */
    res = api.cuMemGetInfo(&free_mem, &total_mem_info);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemGetInfo", res); goto done; }
    printf("       cuMemGetInfo = free %zu / total %zu\n", free_mem, total_mem_info);
    if (g_quota != 0) {
        size_t slack = 64u * 1024u * 1024u;
        if (total_mem_info != (size_t)g_quota) {
            fail_msg("cuMemGetInfo total", "got %zu, expected quota %llu",
                     total_mem_info, (unsigned long long)g_quota);
            goto done;
        }
        if (free_mem > total_mem_info ||
            free_mem + slack < (size_t)g_quota) {
            fail_msg("cuMemGetInfo free", "got %zu, expected ~%llu",
                     free_mem, (unsigned long long)g_quota);
            goto done;
        }
    }
    pass("cuMemGetInfo (free ~= quota)");

    /* 6) 1 MiB allocation, H2D through the SHM data path, verify D2H */
    host_a = malloc(big_sz);
    host_b = malloc(big_sz);
    if (!host_a || !host_b) { fail_msg("malloc host buffers", "out of memory"); goto done; }
    fill_pattern(host_a, big_sz, 7);

    res = api.cuMemAlloc(&ptr, big_sz);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemAlloc(1MiB)", res); goto done; }
    res = api.cuMemcpyHtoD(ptr, host_a, big_sz);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyHtoD(1MiB)", res); goto done; }
    memset(host_b, 0, big_sz);
    res = api.cuMemcpyDtoH(host_b, ptr, big_sz);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyDtoH(1MiB)", res); goto done; }
    if (!buffer_matches(host_a, host_b, big_sz)) {
        fail_msg("1MiB SHM round-trip", "data mismatch");
        goto done;
    }
    pass("cuMemAlloc + 1MiB H2D/D2H round-trip");

    /* 7) module load + add_one kernel */
    module_path = ptx_path ? ptx_path : cubin_path;
    image = read_file(module_path, &image_size);
    if (!image) { fail_msg("read module image", "%s", module_path); goto done; }
    printf("       module image: %s (%zu bytes)\n", module_path, image_size);

    res = api.cuModuleLoadData(&module, image);
    if (res != CUDA_SUCCESS) { fail_cuda("cuModuleLoadData", res); goto done; }
    pass("cuModuleLoadData");

    res = api.cuModuleGetFunction(&f_add, module, "add_one");
    if (res != CUDA_SUCCESS) { fail_cuda("cuModuleGetFunction(add_one)", res); goto done; }
    pass("cuModuleGetFunction(add_one)");

    {
        const int n = 1024;
        const size_t bytes = (size_t)n * sizeof(float);
        void *args[2];
        int k;

        flt_a = malloc(bytes);
        flt_b = malloc(bytes);
        if (!flt_a || !flt_b) { fail_msg("malloc float buffers", "out of memory"); goto done; }
        for (k = 0; k < n; k++)
            flt_a[k] = (float)k * 0.5f;

        res = api.cuMemcpyHtoD(ptr, flt_a, bytes);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyHtoD(kernel input)", res); goto done; }

        args[0] = &ptr;
        args[1] = (void *)&n;
        res = api.cuLaunchKernel(f_add, (unsigned)((n + 255) / 256), 1, 1,
                                 256, 1, 1, 0, NULL, args, NULL);
        if (res != CUDA_SUCCESS) { fail_cuda("cuLaunchKernel(add_one)", res); goto done; }
        res = api.cuCtxSynchronize();
        if (res != CUDA_SUCCESS) { fail_cuda("cuCtxSynchronize(add_one)", res); goto done; }
        res = api.cuMemcpyDtoH(flt_b, ptr, bytes);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyDtoH(add_one)", res); goto done; }

        for (k = 0; k < n; k++) {
            if (flt_b[k] != flt_a[k] + 1.0f) {
                fail_msg("add_one result", "element %d: got %f, expected %f",
                         k, flt_b[k], flt_a[k] + 1.0f);
                goto done;
            }
        }
        pass("cuLaunchKernel(add_one) -> +1.0 on all 1024 elements");
    }

    /* 8) mixed scalar/pointer parameters (meta [4,8,8,8]) */
    print_mixed_meta();
    {
        const int n = 8;
        const size_t bytes = (size_t)n * sizeof(float);
        int mx = 3;
        double my = 2.0;
        const char *pad_host = "A";
        void *args[4];

        res = api.cuModuleGetFunction(&f_mixed, module, "mixed");
        if (res != CUDA_SUCCESS) { fail_cuda("cuModuleGetFunction(mixed)", res); goto done; }
        res = api.cuMemAlloc(&mixed_out, bytes);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemAlloc(mixed out)", res); goto done; }
        res = api.cuMemAlloc(&pad_dev, 4);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemAlloc(mixed pad)", res); goto done; }
        res = api.cuMemcpyHtoD(pad_dev, pad_host, 1);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyHtoD(mixed pad)", res); goto done; }

        args[0] = &mx;
        args[1] = &my;
        args[2] = &mixed_out;
        args[3] = &pad_dev;
        res = api.cuLaunchKernel(f_mixed, 1, 1, 1, n, 1, 1, 0, NULL, args, NULL);
        if (res != CUDA_SUCCESS) { fail_cuda("cuLaunchKernel(mixed)", res); goto done; }
        res = api.cuCtxSynchronize();
        if (res != CUDA_SUCCESS) { fail_cuda("cuCtxSynchronize(mixed)", res); goto done; }
        res = api.cuMemcpyDtoH(flt_b, mixed_out, bytes);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyDtoH(mixed)", res); goto done; }

        for (i = 0; i < n; i++) {
            float expected = (float)(mx + i) * (float)my + 65.0f;
            if (flt_b[i] != expected) {
                fail_msg("mixed result", "element %d: got %f, expected %f",
                         i, flt_b[i], expected);
                goto done;
            }
        }
        pass("cuLaunchKernel(mixed) int/double/ptr/ptr parameters");
        api.cuMemFree(mixed_out);
        api.cuMemFree(pad_dev);
        mixed_out = 0;
        pad_dev = 0;
    }

    /* 9) 32-byte by-value struct parameter (>8B path) */
    {
        typedef struct { double a[4]; } BigArg;
        BigArg v = {{1.5, 2.5, 3.5, 4.5}};
        const double expected = 12.0;
        double got = 0.0;
        void *args[2];

        res = api.cuModuleGetFunction(&f_big, module, "bigarg");
        if (res == CUDA_ERROR_NOT_SUPPORTED) {
            if (env_flag("E2E_REQUIRE_BIGARG")) {
                fail_cuda("cuModuleGetFunction(bigarg)", res);
                goto done;
            }
            skip("bigarg (32B by-value struct)", "hook returned CUDA_ERROR_NOT_SUPPORTED");
            goto bigarg_done;
        }
        if (res != CUDA_SUCCESS) { fail_cuda("cuModuleGetFunction(bigarg)", res); goto done; }

        res = api.cuMemAlloc(&big_out, sizeof(double));
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemAlloc(bigarg out)", res); goto done; }

        args[0] = &v;
        args[1] = &big_out;
        res = api.cuLaunchKernel(f_big, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
        if (res == CUDA_ERROR_NOT_SUPPORTED) {
            if (env_flag("E2E_REQUIRE_BIGARG")) {
                fail_cuda("cuLaunchKernel(bigarg)", res);
                goto done;
            }
            skip("bigarg (32B by-value struct)", "hook returned CUDA_ERROR_NOT_SUPPORTED");
            goto bigarg_done;
        }
        if (res != CUDA_SUCCESS) { fail_cuda("cuLaunchKernel(bigarg)", res); goto done; }
        res = api.cuCtxSynchronize();
        if (res != CUDA_SUCCESS) { fail_cuda("cuCtxSynchronize(bigarg)", res); goto done; }
        res = api.cuMemcpyDtoH(&got, big_out, sizeof(double));
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyDtoH(bigarg)", res); goto done; }
        if (got != expected) {
            fail_msg("bigarg result", "got %f, expected %f", got, expected);
            goto done;
        }
        pass("cuLaunchKernel(bigarg) 32-byte struct parameter");
    }
bigarg_done:
    if (big_out) { api.cuMemFree(big_out); big_out = 0; }

    /* 10) cuMemsetD32 */
    {
        const size_t n_words = 1024;
        const size_t bytes = n_words * sizeof(unsigned int);
        const unsigned int value = 0xDEADBEEFu;
        unsigned int *back = malloc(bytes);
        size_t k;

        if (!back) { fail_msg("malloc memset buffer", "out of memory"); goto done; }
        res = api.cuMemAlloc(&memset_buf, bytes);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemAlloc(memset)", res); goto done; }
        res = api.cuMemsetD32(memset_buf, value, n_words);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemsetD32", res); goto done; }
        res = api.cuMemcpyDtoH(back, memset_buf, bytes);
        if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyDtoH(memset)", res); goto done; }
        for (k = 0; k < n_words; k++) {
            if (back[k] != value) {
                fail_msg("cuMemsetD32 result", "word %zu: got 0x%08x, expected 0x%08x",
                         k, back[k], value);
                free(back);
                goto done;
            }
        }
        free(back);
        pass("cuMemsetD32 -> D2H verify");
        api.cuMemFree(memset_buf);
        memset_buf = 0;
    }

    /* 11) events */
    res = api.cuEventCreate(&ev0, 0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventCreate(start)", res); goto done; }
    res = api.cuEventCreate(&ev1, 0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventCreate(end)", res); goto done; }
    res = api.cuEventRecord(ev0, NULL);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventRecord(start)", res); goto done; }
    res = api.cuEventSynchronize(ev0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventSynchronize(start)", res); goto done; }
    res = api.cuEventRecord(ev1, NULL);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventRecord(end)", res); goto done; }
    res = api.cuEventSynchronize(ev1);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventSynchronize(end)", res); goto done; }
    {
        float elapsed = -1.0f;
        res = api.cuEventElapsedTime(&elapsed, ev0, ev1);
        if (res != CUDA_SUCCESS) { fail_cuda("cuEventElapsedTime", res); goto done; }
        printf("       cuEventElapsedTime = %.3f ms\n", elapsed);
        if (elapsed < 0.0f) { fail_msg("cuEventElapsedTime", "negative duration %f", elapsed); goto done; }
    }
    pass("cuEventCreate/Record/Synchronize/ElapsedTime");
    res = api.cuEventDestroy(ev0);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventDestroy(start)", res); goto done; }
    res = api.cuEventDestroy(ev1);
    if (res != CUDA_SUCCESS) { fail_cuda("cuEventDestroy(end)", res); goto done; }
    ev0 = ev1 = NULL;
    pass("cuEventDestroy");

    /* 12) cuMemAllocAsync from the default pool */
    res = api.cuMemAllocAsync(&async_ptr, big_sz, NULL);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemAllocAsync(1MiB)", res); goto done; }
    res = api.cuMemcpyHtoD(async_ptr, host_a, big_sz);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyHtoD(async alloc)", res); goto done; }
    memset(host_b, 0, big_sz);
    res = api.cuMemcpyDtoH(host_b, async_ptr, big_sz);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemcpyDtoH(async alloc)", res); goto done; }
    if (!buffer_matches(host_a, host_b, big_sz)) {
        fail_msg("cuMemAllocAsync round-trip", "data mismatch");
        goto done;
    }
    res = api.cuMemFreeAsync(async_ptr, NULL);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemFreeAsync", res); goto done; }
    async_ptr = 0;
    pass("cuMemAllocAsync/H2D/D2H/cuMemFreeAsync (default pool)");

    /* 13) teardown */
    res = api.cuCtxSynchronize();
    if (res != CUDA_SUCCESS) { fail_cuda("cuCtxSynchronize(final)", res); goto done; }
    res = api.cuMemFree(ptr);
    if (res != CUDA_SUCCESS) { fail_cuda("cuMemFree(main 1MiB)", res); goto done; }
    ptr = 0;
    res = api.cuCtxDestroy(ctx);
    if (res != CUDA_SUCCESS) { fail_cuda("cuCtxDestroy", res); goto done; }
    ctx = NULL;
    pass("cuCtxSynchronize + cuMemFree + cuCtxDestroy");

    rc = (g_fail == 0) ? 0 : 1;

done:
    /* Best-effort cleanup on the failure path. */
    if (ptr) api.cuMemFree(ptr);
    if (mixed_out) api.cuMemFree(mixed_out);
    if (pad_dev) api.cuMemFree(pad_dev);
    if (big_out) api.cuMemFree(big_out);
    if (memset_buf) api.cuMemFree(memset_buf);
    if (async_ptr) api.cuMemFreeAsync(async_ptr, NULL);
    if (ev0) api.cuEventDestroy(ev0);
    if (ev1) api.cuEventDestroy(ev1);
    if (ctx) api.cuCtxDestroy(ctx);
    free(host_a);
    free(host_b);
    free(flt_a);
    free(flt_b);
    free(image);

    if (rc != 0)
        rc = (g_fail == 0) ? 1 : 1;

    printf("SUMMARY: %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
    dlclose(cuda_lib);
    return rc;
}
