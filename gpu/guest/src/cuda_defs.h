/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA Driver API type definitions.
 * Derived from nvshare's cuda_defs.h, adapted for RPC remoting.
 *
 * We define function pointer types for every CUDA Driver API function
 * we need to intercept, plus macros for symbol versioning.
 */

#ifndef CUBESANDBOX_GPU_CUDA_DEFS_H
#define CUBESANDBOX_GPU_CUDA_DEFS_H

#include <stddef.h>
#include <stdint.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUevent;
typedef int CUdevice;
typedef unsigned int CUuint64_t;

#define CUDA_SUCCESS                    0
#define CUDA_ERROR_NOT_INITIALIZED      1
#define CUDA_ERROR_OUT_OF_MEMORY        2
#define CUDA_ERROR_NOT_READY            4
#define CUDA_ERROR_LAUNCH_FAILED        5
#define CUDA_ERROR_INVALID_VALUE        1

#define CU_MEM_ATTACH_GLOBAL            0x01

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

/* Function pointer types for CUDA Driver API */
typedef CUresult (*cuInit_func)(unsigned int);
typedef CUresult (*cuGetErrorName_func)(CUresult, const char **);
typedef CUresult (*cuGetErrorString_func)(CUresult, const char **);
typedef CUresult (*cuCtxGetCurrent_func)(CUcontext *);
typedef CUresult (*cuCtxSetCurrent_func)(CUcontext);
typedef CUresult (*cuCtxSynchronize_func)(void);
typedef CUresult (*cuCtxCreate_func)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_func)(CUcontext);
typedef CUresult (*cuDeviceGet_func)(CUdevice *, int);
typedef CUresult (*cuDeviceGetCount_func)(int *);
typedef CUresult (*cuMemAlloc_func)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemAllocManaged_func)(CUdeviceptr *, size_t, unsigned int);
typedef CUresult (*cuMemFree_func)(CUdeviceptr);
typedef CUresult (*cuMemGetInfo_func)(size_t *, size_t *);
typedef CUresult (*cuMemcpy_func)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyAsync_func)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyHtoD_func)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyHtoDAsync_func)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoH_func)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyDtoHAsync_func)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoD_func)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyDtoDAsync_func)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuLaunchKernel_func)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int,
                                        unsigned int, CUstream, void **, void **);
typedef CUresult (*cuModuleLoadData_func)(CUmodule *, const void *);
typedef CUresult (*cuModuleLoadDataEx_func)(CUmodule *, const void *, unsigned int, void *, void *);
typedef CUresult (*cuModuleGetFunction_func)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuModuleUnload_func)(CUmodule);
typedef CUresult (*cuStreamCreate_func)(CUstream *, unsigned int);
typedef CUresult (*cuStreamDestroy_func)(CUstream);
typedef CUresult (*cuStreamSynchronize_func)(CUstream);
typedef CUresult (*cuEventCreate_func)(CUevent *, unsigned int);
typedef CUresult (*cuEventRecord_func)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_func)(CUevent);

typedef CUresult (*cuGetProcAddress_func)(const char *, void **, int, uint64_t);
typedef CUresult (*cuGetProcAddress_v2_func)(const char *, void **, int, uint64_t, int *);

/* Symbol versioning: CUDA functions have _v2 suffixes in the ABI */
#define cuMemAlloc           cuMemAlloc_v2
#define cuMemAllocManaged    cuMemAllocManaged_v2
#define cuMemFree            cuMemFree_v2
#define cuMemGetInfo         cuMemGetInfo_v2
#define cuMemcpy             cuMemcpy_v2
#define cuMemcpyAsync        cuMemcpyAsync_v2
#define cuMemcpyHtoD         cuMemcpyHtoD_v2
#define cuMemcpyHtoDAsync    cuMemcpyHtoDAsync_v2
#define cuMemcpyDtoH         cuMemcpyDtoH_v2
#define cuMemcpyDtoHAsync    cuMemcpyDtoHAsync_v2
#define cuMemcpyDtoD         cuMemcpyDtoD_v2
#define cuMemcpyDtoDAsync    cuMemcpyDtoDAsync_v2
#define cuModuleLoadDataEx   cuModuleLoadDataEx_v2
#define cuModuleGetFunction  cuModuleGetFunction_v2
#define cuStreamDestroy      cuStreamDestroy_v2
#define cuCtxCreate         cuCtxCreate_v2
#define cuCtxDestroy         cuCtxDestroy_v2
#define cuCtxSetCurrent      cuCtxSetCurrent_v2
#define cuCtxGetCurrent      cuCtxGetCurrent_v2
#define cuCtxSynchronize     cuCtxSynchronize_v2
#define cuDeviceGet          cuDeviceGet_v2

#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

#endif
