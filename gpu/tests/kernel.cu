/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA kernel fixtures for the CubeSandbox GPU RPC end-to-end tests.
 *
 * Build (see Makefile / e2e-test.sh):
 *   nvcc -cubin -arch=sm_86 -o kernel.cubin kernel.cu
 *   nvcc -ptx   -arch=sm_86 -o kernel.ptx   kernel.cu
 *
 * The three kernels deliberately exercise different kernel-parameter shapes:
 *
 *   add_one(float *out, int n)                      (ptr, int)
 *   mixed(int x, double y, float *out, char *pad)   (scalar mix, meta [4,8,8,8])
 *   bigarg(BigArg v, double *out)                   (32-byte by-value struct, >8B)
 */

typedef struct {
    double a[4];
} BigArg;

/*
 * out[i] += 1.0f for i < n, one thread per element.
 */
extern "C" __global__ void add_one(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = out[i] + 1.0f;
    }
}

/*
 * out[i] = (x + i) * y + pad[0]
 *
 * Expected (host side): x = 3, y = 2.0, pad[0] = 'A' (65) ->
 *   out[i] = 2 * (3 + i) + 65.
 */
extern "C" __global__ void mixed(int x, double y, float *out, const char *pad) {
    int i = threadIdx.x;
    float p = (pad != 0) ? (float)pad[0] : 0.0f;
    out[i] = (float)(x + i) * (float)y + p;
}

/*
 * out[0] = sum(v.a[]) -- a single thread, so the test is deterministic.
 * The 32-byte struct exercises the >8-byte kernel-parameter path.
 */
extern "C" __global__ void bigarg(BigArg v, double *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = v.a[0] + v.a[1] + v.a[2] + v.a[3];
    }
}
