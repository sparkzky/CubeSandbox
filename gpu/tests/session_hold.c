/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * session_hold -- long-lived CUDA session used by e2e-multi-session.sh.
 *
 * Unlike e2e_main it stays alive so a peer test can be kill -9'd while this
 * session remains open.  It speaks through libcube-gpu.so exactly like any
 * other CUDA process (runtime dlsym, so the guest library's dlsym hook is
 * exercised), verifies one round-trip, publishes a ready file, then waits for
 * a trigger file.  When the trigger appears it performs a second small
 * allocation/round-trip and writes "ok" to the result file -- proving the
 * surviving session still works after its sibling was killed.
 *
 * Usage:
 *   session_hold --ready-file P --trigger-file P --result-file P [--max-seconds N]
 *
 * Exit status 0 on clean shutdown, 1 on any CUDA/IO failure.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;

#define CUDA_SUCCESS 0

typedef struct {
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
} hold_api_t;

static hold_api_t api;

static int resolve(void)
{
    struct { const char *v2; const char *plain; void **slot; } syms[] = {
        { NULL, "cuInit",           (void **)&api.cuInit },
        { NULL, "cuDeviceGetCount", (void **)&api.cuDeviceGetCount },
        { NULL, "cuDeviceGet",      (void **)&api.cuDeviceGet },
        { "cuCtxCreate_v2",  "cuCtxCreate",  (void **)&api.cuCtxCreate },
        { "cuCtxDestroy_v2", "cuCtxDestroy", (void **)&api.cuCtxDestroy },
        { "cuMemAlloc_v2", "cuMemAlloc",   (void **)&api.cuMemAlloc },
        { "cuMemFree_v2",  "cuMemFree",    (void **)&api.cuMemFree },
        { "cuMemcpyHtoD_v2", "cuMemcpyHtoD", (void **)&api.cuMemcpyHtoD },
        { "cuMemcpyDtoH_v2", "cuMemcpyDtoH", (void **)&api.cuMemcpyDtoH },
    };
    size_t i;

    for (i = 0; i < sizeof(syms) / sizeof(syms[0]); i++) {
        void *p = syms[i].v2 ? dlsym(RTLD_DEFAULT, syms[i].v2) : NULL;
        if (!p)
            p = dlsym(RTLD_DEFAULT, syms[i].plain);
        if (!p) {
            fprintf(stderr, "session_hold: cannot resolve %s\n", syms[i].plain);
            return -1;
        }
        *syms[i].slot = p;
    }
    return 0;
}

static int roundtrip(size_t bytes, unsigned char seed)
{
    CUdeviceptr dev = 0;
    unsigned char *a, *b;
    CUresult res;
    size_t i;
    int rc = -1;

    a = malloc(bytes);
    b = malloc(bytes);
    if (!a || !b)
        goto out;
    for (i = 0; i < bytes; i++)
        a[i] = (unsigned char)(seed + i * 7u);
    memset(b, 0, bytes);

    res = api.cuMemAlloc(&dev, bytes);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "session_hold: cuMemAlloc(%zu) = %d\n", bytes, res);
        goto out;
    }
    res = api.cuMemcpyHtoD(dev, a, bytes);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "session_hold: cuMemcpyHtoD = %d\n", res);
        api.cuMemFree(dev);
        goto out;
    }
    res = api.cuMemcpyDtoH(b, dev, bytes);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "session_hold: cuMemcpyDtoH = %d\n", res);
        api.cuMemFree(dev);
        goto out;
    }
    api.cuMemFree(dev);
    if (memcmp(a, b, bytes) != 0) {
        fprintf(stderr, "session_hold: round-trip data mismatch\n");
        goto out;
    }
    rc = 0;
out:
    free(a);
    free(b);
    return rc;
}

static int write_file(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fputs(text, f);
    fclose(f);
    return 0;
}

static int file_exists(const char *path)
{
    return access(path, F_OK) == 0;
}

int main(int argc, char **argv)
{
    const char *ready_file = NULL, *trigger_file = NULL, *result_file = NULL;
    int max_seconds = 120;
    void *cuda_lib;
    int count = 0;
    int i;
    char msg[128];

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc)
            ready_file = argv[++i];
        else if (strcmp(argv[i], "--trigger-file") == 0 && i + 1 < argc)
            trigger_file = argv[++i];
        else if (strcmp(argv[i], "--result-file") == 0 && i + 1 < argc)
            result_file = argv[++i];
        else if (strcmp(argv[i], "--max-seconds") == 0 && i + 1 < argc)
            max_seconds = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s --ready-file P --trigger-file P --result-file P "
                            "[--max-seconds N]\n", argv[0]);
            return 2;
        }
    }
    if (!ready_file || !trigger_file || !result_file) {
        fprintf(stderr, "session_hold: --ready-file/--trigger-file/--result-file required\n");
        return 2;
    }

    cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda_lib) {
        fprintf(stderr, "session_hold: dlopen(libcuda.so.1): %s\n", dlerror());
        return 1;
    }
    if (resolve() != 0)
        return 1;

    if (api.cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "session_hold: cuInit failed\n");
        return 1;
    }
    /*
     * cuDeviceGetCount may still be un-remoted (falls through to the real
     * driver, which returns NOT_INITIALIZED because cuInit was hooked).
     * It is informational here; the allocation round-trip below is what
     * proves the session is live.
     */
    if (api.cuDeviceGetCount(&count) == CUDA_SUCCESS)
        fprintf(stderr, "session_hold: device count = %d\n", count);

    /* A real CUDA client creates a context before allocating; the daemon
     * needs a current context on its side for the same reason. */
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    if (api.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        fprintf(stderr, "session_hold: cuDeviceGet failed\n");
        return 1;
    }
    if (api.cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
        fprintf(stderr, "session_hold: cuCtxCreate failed\n");
        return 1;
    }

    if (roundtrip(4096, 3) != 0)
        return 1;

    snprintf(msg, sizeof(msg), "ready pid=%d\n", (int)getpid());
    if (write_file(ready_file, msg) != 0) {
        fprintf(stderr, "session_hold: cannot write %s: %s\n", ready_file, strerror(errno));
        return 1;
    }

    for (i = 0; i < max_seconds * 10; i++) {
        if (file_exists(trigger_file)) {
            if (roundtrip(64 * 1024, 11) != 0) {
                write_file(result_file, "fail\n");
                return 1;
            }
            if (write_file(result_file, "ok\n") != 0)
                return 1;
            trigger_file = NULL; /* handle once */
        }
        usleep(100 * 1000);
    }

    (void)api.cuCtxDestroy(ctx);
    return 0;
}
