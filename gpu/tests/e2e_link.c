/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * e2e_link -- direct-link variant of the CUDA smoke test.
 *
 * Unlike e2e_main (which resolves everything through dlsym and therefore
 * exercises the guest library's dlsym hook), this binary links against
 * libcuda.so at build time and calls cuInit()/cuDeviceGetCount() through the
 * normal ELF symbol resolution path.  Under LD_PRELOAD the definitions must
 * still be interposed by libcube-gpu.so: that is the behaviour the shim
 * relies on for unmodified CUDA binaries.
 *
 * Provider detection: dladdr(&cuInit) reports the shared object that actually
 * backs the symbol.  E2E_REQUIRE_LINK_HOOK (default 1) makes the test fail
 * when cuInit is resolved from libcuda.so.1 instead of libcube-gpu.so.
 *
 * Build: cc -o e2e_link e2e_link.c -lcuda -ldl
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGetCount(int *count);

int main(void)
{
    const char *provider = "unknown";
    const char *require = getenv("E2E_REQUIRE_LINK_HOOK");
    int count = -1;
    CUresult res;
    Dl_info info;

    if (dladdr((void *)&cuInit, &info) && info.dli_fname)
        provider = info.dli_fname;

    if (require == NULL || require[0] != '0') {
        if (strstr(provider, "libcube-gpu.so") == NULL) {
            printf("[FAIL] e2e_link: cuInit resolved from '%s', not libcube-gpu.so "
                   "(LD_PRELOAD did not interpose a direct-linked CUDA call)\n", provider);
            return 1;
        }
    }

    res = cuInit(0);
    if (res != 0) {
        printf("[FAIL] e2e_link: cuInit returned %d\n", (int)res);
        return 1;
    }

    res = cuDeviceGetCount(&count);
    if (res != 0) {
        printf("[FAIL] e2e_link: cuDeviceGetCount returned %d\n", (int)res);
        return 1;
    }
    if (count < 1) {
        printf("[FAIL] e2e_link: device count is %d\n", count);
        return 1;
    }

    printf("[PASS] e2e_link: cuInit provider=%s, cuDeviceGetCount=%d\n", provider, count);
    return 0;
}
