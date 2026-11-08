/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal CUDA Driver API type/constant definitions for the guest
 * interposition library. Deliberately self-contained: the guest build must
 * not depend on CUDA headers/SDK being installed in the sandbox image.
 *
 * Types and numeric values mirror cuda.h (CUDA 12/13 ABI) closely enough to
 * interpose the driver API; only what libcube-gpu.so actually uses is listed.
 */

#ifndef CUBESANDBOX_GPU_CUDA_DEFS_H
#define CUBESANDBOX_GPU_CUDA_DEFS_H

#include <stddef.h>
#include <stdint.h>

/* ─── Export marker ─────────────────────────────────────────────
 *
 * The shared object is built with -fvisibility=hidden so that only the
 * interposed CUDA entry points (and dlsym) leak out of the library. The
 * version script intentionally does NOT version the cu* symbols: driver
 * API symbols do not carry ELF versions, so interposition works best when
 * they stay unversioned (and `nm -D` shows the plain names).
 */
#define CUBE_GPU_EXPORT __attribute__((visibility("default")))

/* ─── Basic types ───────────────────────────────────────────────── */

typedef int              CUresult;
typedef int              CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void            *CUcontext;
typedef void            *CUstream;
typedef void            *CUmodule;
typedef void            *CUfunction;
typedef void            *CUevent;
typedef void            *CUgraphicsResource;

/* CUipcMemHandle is an opaque 64-byte blob in the driver ABI. */
typedef struct CUipcMemHandle_st {
    char reserved[64];
} CUipcMemHandle;

/* Enums are passed as plain ints across the driver ABI. */
typedef int CUdevice_attribute;
typedef int CUfunction_attribute;
typedef int CUlimit;
typedef int CUmemAttach_flags;

/* ─── CUresult codes ────────────────────────────────────────────── */

#define CUDA_SUCCESS                                  0
#define CUDA_ERROR_INVALID_VALUE                      1
#define CUDA_ERROR_OUT_OF_MEMORY                      2
#define CUDA_ERROR_NOT_INITIALIZED                    3
#define CUDA_ERROR_DEINITIALIZED                      4
#define CUDA_ERROR_NO_DEVICE                          100
#define CUDA_ERROR_INVALID_DEVICE                     101
#define CUDA_ERROR_INVALID_IMAGE                      200
#define CUDA_ERROR_INVALID_CONTEXT                    201
#define CUDA_ERROR_CONTEXT_ALREADY_CURRENT            202
#define CUDA_ERROR_MAP_FAILED                         205
#define CUDA_ERROR_UNMAP_FAILED                       206
#define CUDA_ERROR_ARRAY_IS_MAPPED                    207
#define CUDA_ERROR_ALREADY_MAPPED                     208
#define CUDA_ERROR_NO_BINARY_FOR_GPU                  209
#define CUDA_ERROR_ALREADY_ACQUIRED                   210
#define CUDA_ERROR_NOT_MAPPED                         211
#define CUDA_ERROR_NOT_MAPPED_AS_ARRAY                212
#define CUDA_ERROR_NOT_MAPPED_AS_POINTER              213
#define CUDA_ERROR_ECC_UNCORRECTABLE                  214
#define CUDA_ERROR_UNSUPPORTED_LIMIT                  215
#define CUDA_ERROR_CONTEXT_ALREADY_IN_USE             216
#define CUDA_ERROR_PEER_ACCESS_UNSUPPORTED            217
#define CUDA_ERROR_INVALID_PTX                        218
#define CUDA_ERROR_INVALID_GRAPHICS_CONTEXT           219
#define CUDA_ERROR_NVLINK_UNCORRECTABLE               220
#define CUDA_ERROR_JIT_COMPILER_NOT_FOUND             221
#define CUDA_ERROR_UNSUPPORTED_PTX_VERSION            222
#define CUDA_ERROR_JIT_COMPILATION_DISABLED           223
#define CUDA_ERROR_INVALID_SOURCE                     300
#define CUDA_ERROR_FILE_NOT_FOUND                     301
#define CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND     302
#define CUDA_ERROR_SHARED_OBJECT_INIT_FAILED          303
#define CUDA_ERROR_OPERATING_SYSTEM                   304
#define CUDA_ERROR_INVALID_HANDLE                     400
#define CUDA_ERROR_ILLEGAL_STATE                      401
#define CUDA_ERROR_NOT_FOUND                          500
#define CUDA_ERROR_NOT_READY                          600
#define CUDA_ERROR_ILLEGAL_ADDRESS                    700
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES            701
#define CUDA_ERROR_LAUNCH_TIMEOUT                     702
#define CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING      703
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED        704
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED            705
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE             708
#define CUDA_ERROR_CONTEXT_IS_DESTROYED               709
#define CUDA_ERROR_ASSERT                             710
#define CUDA_ERROR_TOO_MANY_PEERS                     711
#define CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED     712
#define CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED         713
#define CUDA_ERROR_ILLEGAL_INSTRUCTION                715
#define CUDA_ERROR_MISALIGNED_ADDRESS                 716
#define CUDA_ERROR_INVALID_ADDRESS_SPACE              717
#define CUDA_ERROR_INVALID_PC                         718
#define CUDA_ERROR_LAUNCH_FAILED                      719
#define CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE       720
#define CUDA_ERROR_NOT_PERMITTED                      800
#define CUDA_ERROR_NOT_SUPPORTED                      801
#define CUDA_ERROR_SYSTEM_NOT_READY                   802
#define CUDA_ERROR_SYSTEM_DRIVER_MISMATCH             803
#define CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE     804
#define CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED         900
#define CUDA_ERROR_STREAM_CAPTURE_INVALIDATED         901
#define CUDA_ERROR_STREAM_CAPTURE_MERGE               902
#define CUDA_ERROR_STREAM_CAPTURE_UNMATCHED           903
#define CUDA_ERROR_STREAM_CAPTURE_UNJOINED            904
#define CUDA_ERROR_STREAM_CAPTURE_ISOLATION           905
#define CUDA_ERROR_STREAM_CAPTURE_IMPLICIT            906
#define CUDA_ERROR_CAPTURED_EVENT                     907
#define CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD        908
#define CUDA_ERROR_TIMEOUT                            909
#define CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE          910
#define CUDA_ERROR_UNKNOWN                            999

/* ─── cuLaunchKernel extra parameter keys ────────────────────────
 *
 * `extra` is a NULL-terminated array of (key, value) pairs; the keys are
 * these sentinel pointers, not strings.
 */
#define CU_LAUNCH_PARAM_END            ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE    ((void *)0x02)

/* ─── Context / memory flags (subset) ──────────────────────────── */

#define CU_CTX_SCHED_AUTO         0x00
#define CU_CTX_SCHED_SPIN         0x01
#define CU_CTX_SCHED_YIELD        0x02
#define CU_CTX_SCHED_BLOCKING_SYNC 0x04
#define CU_CTX_MAP_HOST           0x08
#define CU_CTX_LMEM_RESIZE_TO_MAX 0x10

#define CU_MEM_ATTACH_GLOBAL 0x01
#define CU_MEM_ATTACH_HOST   0x02
#define CU_MEM_ATTACH_SINGLE 0x04

/* ─── Function pointer types for the interposed entry points ───── */

typedef CUresult (*cuInit_func)(unsigned int);
typedef CUresult (*cuGetErrorName_func)(CUresult, const char **);
typedef CUresult (*cuGetErrorString_func)(CUresult, const char **);

typedef CUresult (*cuDeviceGetCount_func)(int *);
typedef CUresult (*cuDeviceGet_func)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_func)(char *, int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_func)(int *, CUdevice_attribute, CUdevice);
typedef CUresult (*cuDeviceTotalMem_func)(size_t *, CUdevice);

typedef CUresult (*cuCtxCreate_func)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_func)(CUcontext);
typedef CUresult (*cuCtxSetCurrent_func)(CUcontext);
typedef CUresult (*cuCtxGetCurrent_func)(CUcontext *);
typedef CUresult (*cuCtxGetDevice_func)(CUdevice *);
typedef CUresult (*cuCtxSynchronize_func)(void);
typedef CUresult (*cuDevicePrimaryCtxRetain_func)(CUcontext *, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRelease_func)(CUdevice);

typedef CUresult (*cuMemAlloc_func)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_func)(CUdeviceptr);
typedef CUresult (*cuMemAllocAsync_func)(CUdeviceptr *, size_t, CUstream);
typedef CUresult (*cuMemFreeAsync_func)(CUdeviceptr, CUstream);
typedef CUresult (*cuMemGetInfo_func)(size_t *, size_t *);
typedef CUresult (*cuMemcpyHtoD_func)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_func)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyDtoD_func)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyHtoDAsync_func)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_func)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemsetD8_func)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*cuMemsetD16_func)(CUdeviceptr, unsigned short, size_t);
typedef CUresult (*cuMemsetD32_func)(CUdeviceptr, unsigned int, size_t);

typedef CUresult (*cuModuleLoadData_func)(CUmodule *, const void *);
typedef CUresult (*cuModuleLoadDataEx_func)(CUmodule *, const void *, unsigned int, void *, void *);
typedef CUresult (*cuModuleGetFunction_func)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuLaunchKernel_func)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int,
                                        unsigned int, CUstream, void **, void **);
typedef CUresult (*cuFuncGetAttribute_func)(int *, CUfunction_attribute, CUfunction);

typedef CUresult (*cuStreamCreate_func)(CUstream *, unsigned int);
typedef CUresult (*cuStreamDestroy_func)(CUstream);
typedef CUresult (*cuStreamSynchronize_func)(CUstream);

typedef CUresult (*cuEventCreate_func)(CUevent *, unsigned int);
typedef CUresult (*cuEventRecord_func)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_func)(CUevent);
typedef CUresult (*cuEventElapsedTime_func)(float *, CUevent, CUevent);
typedef CUresult (*cuEventDestroy_func)(CUevent);

typedef CUresult (*cuIpcGetMemHandle_func)(CUipcMemHandle *, CUdeviceptr);
typedef CUresult (*cuIpcOpenMemHandle_func)(CUdeviceptr *, CUipcMemHandle, unsigned int);
typedef CUresult (*cuIpcCloseMemHandle_func)(CUdeviceptr);

typedef CUresult (*cuGetProcAddress_func)(const char *, void **, int, uint64_t);
typedef CUresult (*cuGetProcAddress_v2_func)(const char *, void **, int, uint64_t, int *);

#endif /* CUBESANDBOX_GPU_CUDA_DEFS_H */
