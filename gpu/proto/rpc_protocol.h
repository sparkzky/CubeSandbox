/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CubeSandbox GPU RPC Protocol
 *
 * Shared protocol definitions for CUDA API remoting between
 * the guest-side libcube-gpu.so and the host-side cube-gpu-daemon.
 *
 * Communication model:
 *   - Control channel: virtio-vsock (small metadata / parameters)
 *   - Data channel:    IVSHMEM shared memory (large payloads: tensors, cubin)
 *
 * Wire format on vsock:
 *   [rpc_header (32 bytes)] [payload (variable, header.payload_len bytes)]
 */

#ifndef CUBESANDBOX_GPU_RPC_PROTOCOL_H
#define CUBESANDBOX_GPU_RPC_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Protocol constants ────────────────────────────────────────── */

#define CUBE_GPU_RPC_MAGIC      0x43554441u  /* "CUDA" in ASCII */
#define CUBE_GPU_RPC_VERSION    1

/* Default vsock port for cube-gpu-daemon */
#define CUBE_GPU_VSOCK_PORT     0x5055u      /* "PU" = 0x5055 */

/* Default IVSHMEM ring buffer sizes */
#define CUBE_GPU_SHM_RING_SIZE  (64 * 1024 * 1024)  /* 64 MiB per direction */

/* Threshold: payloads larger than this go through IVSHMEM, not inline */
#define CUBE_GPU_SHM_THRESHOLD  (4 * 1024)          /* 4 KiB */

/* ─── Flags ─────────────────────────────────────────────────────── */

#define CUBE_GPU_FLAG_NONE       0x00u
#define CUBE_GPU_FLAG_SHM_DATA   0x01u   /* Data payload is in shared memory */
#define CUBE_GPU_FLAG_ASYNC      0x02u   /* Asynchronous operation */
#define CUBE_GPU_FLAG_ERROR      0x40u   /* Response indicates an error */
#define CUBE_GPU_FLAG_RESPONSE   0x80u   /* This is a response message */

/* ─── CUDA API identifiers ──────────────────────────────────────── */

typedef enum __attribute__((__packed__)) cube_gpu_api_id {
    /* Session management */
    CUBE_GPU_API_SESSION_INIT     = 0,
    CUBE_GPU_API_SESSION_DESTROY  = 1,

    /* Memory management (P0) */
    CUBE_GPU_API_MALLOC           = 10,
    CUBE_GPU_API_FREE             = 11,
    CUBE_GPU_API_MEMCPY           = 12,
    CUBE_GPU_API_MEMCPY_ASYNC     = 13,

    /* Kernel execution (P0) */
    CUBE_GPU_API_MODULE_LOAD      = 20,
    CUBE_GPU_API_MODULE_LOAD_DATA = 21,
    CUBE_GPU_API_MODULE_GET_FUNC  = 22,
    CUBE_GPU_API_LAUNCH_KERNEL    = 23,

    /* Stream management (P0) */
    CUBE_GPU_API_STREAM_CREATE    = 30,
    CUBE_GPU_API_STREAM_DESTROY   = 31,
    CUBE_GPU_API_STREAM_SYNC      = 32,
    CUBE_GPU_API_DEVICE_SYNC      = 33,

    /* Query / properties (P1) */
    CUBE_GPU_API_GET_DEV_PROP     = 40,
    CUBE_GPU_API_MEM_GET_INFO     = 41,
    CUBE_GPU_API_GET_DEVICE       = 42,
    CUBE_GPU_API_SET_DEVICE       = 43,
    CUBE_GPU_API_DEVICE_COUNT     = 44,

    /* Context management (internal) */
    CUBE_GPU_API_CTX_CREATE       = 50,
    CUBE_GPU_API_CTX_DESTROY      = 51,
    CUBE_GPU_API_CTX_SET_CURRENT  = 52,
    CUBE_GPU_API_CTX_GET_CURRENT  = 53,
} cube_gpu_api_id_t;

/* ─── Wire format: fixed header (32 bytes) ──────────────────────── */

typedef struct __attribute__((__packed__)) cube_gpu_rpc_header {
    uint32_t magic;         /* CUBE_GPU_RPC_MAGIC */
    uint32_t version;       /* Protocol version */
    uint32_t api_id;        /* cube_gpu_api_id_t */
    uint32_t flags;         /* CUBE_GPU_FLAG_* bitmask */
    uint64_t request_id;    /* Monotonically increasing request ID */
    uint64_t session_id;    /* Sandbox session identifier */
    uint32_t payload_len;   /* Length of payload following this header */
    uint32_t status;        /* CUresult (in responses) or 0 (in requests) */
} cube_gpu_rpc_header_t;

/* Total: 40 bytes packed. */

/* ─── SHM region descriptor (used in payloads) ──────────────────── */

typedef struct __attribute__((__packed__)) cube_gpu_shm_desc {
    uint64_t offset;    /* Offset into the IVSHMEM region */
    uint64_t size;      /* Number of bytes at that offset */
} cube_gpu_shm_desc_t;

/* ─── cudaMemcpyKind (mirror of CUDA's enum) ────────────────────── */

typedef enum __attribute__((__packed__)) cube_gpu_memcpy_kind {
    CUBE_GPU_MEMCPY_HOST_TO_HOST     = 0,
    CUBE_GPU_MEMCPY_HOST_TO_DEVICE   = 1,
    CUBE_GPU_MEMCPY_DEVICE_TO_HOST   = 2,
    CUBE_GPU_MEMCPY_DEVICE_TO_DEVICE = 3,
    CUBE_GPU_MEMCPY_DEFAULT          = 4,
} cube_gpu_memcpy_kind_t;

/* ─── Request payloads ──────────────────────────────────────────── */

/* ── Session ───────────── */

typedef struct __attribute__((__packed__)) cube_gpu_session_init_req {
    uint64_t memory_quota;    /* Requested GPU memory quota in bytes */
    uint64_t sandbox_id;      /* Sandbox identifier */
} cube_gpu_session_init_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_session_init_resp {
    uint32_t status;          /* CUresult */
    uint64_t session_handle;  /* Server-assigned session handle */
    uint64_t shm_offset;      /* Offset of this session's SHM region */
    uint64_t shm_size;        /* Size of this session's SHM region */
} cube_gpu_session_init_resp_t;

/* ── cudaMalloc ────────── */

typedef struct __attribute__((__packed__)) cube_gpu_malloc_req {
    uint64_t size;            /* Allocation size in bytes */
} cube_gpu_malloc_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_malloc_resp {
    uint32_t status;          /* CUresult */
    uint64_t vptr;            /* Virtual device pointer (guest-side handle) */
} cube_gpu_malloc_resp_t;

/* ── cudaFree ──────────── */

typedef struct __attribute__((__packed__)) cube_gpu_free_req {
    uint64_t vptr;            /* Virtual device pointer to free */
} cube_gpu_free_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_free_resp {
    uint32_t status;          /* CUresult */
} cube_gpu_free_resp_t;

/* ── cudaMemcpy ────────── */

typedef struct __attribute__((__packed__)) cube_gpu_memcpy_req {
    uint32_t kind;            /* cube_gpu_memcpy_kind_t */
    uint64_t dst;             /* Destination vptr or shm_offset */
    uint64_t src;             /* Source vptr or shm_offset */
    uint64_t size;            /* Number of bytes */
    /* If FLAG_SHM_DATA is set, actual data is at SHM:
     *   H2D: guest writes data to SHM, src = shm_offset
     *   D2H: host writes data to SHM, dst = shm_offset
     *   D2D: both are vptrs, no SHM involved
     */
    uint64_t shm_offset;      /* Valid when FLAG_SHM_DATA is set */
} cube_gpu_memcpy_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_memcpy_resp {
    uint32_t status;          /* CUresult */
} cube_gpu_memcpy_resp_t;

/* ── cudaMemcpyAsync ───── */

typedef struct __attribute__((__packed__)) cube_gpu_memcpy_async_req {
    uint32_t kind;            /* cube_gpu_memcpy_kind_t */
    uint64_t dst;
    uint64_t src;
    uint64_t size;
    uint64_t shm_offset;
    uint64_t stream_vptr;     /* Virtual stream handle */
} cube_gpu_memcpy_async_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_memcpy_async_resp {
    uint32_t status;
} cube_gpu_memcpy_async_resp_t;

/* ── cuModuleLoadData ──── */

typedef struct __attribute__((__packed__)) cube_gpu_module_load_data_req {
    uint64_t image_size;      /* Size of the module image (cubin/PTX/fatbin) */
    uint64_t shm_offset;      /* Module data is in SHM at this offset */
} cube_gpu_module_load_data_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_module_load_data_resp {
    uint32_t status;
    uint64_t module_handle;   /* Virtual module handle */
} cube_gpu_module_load_data_resp_t;

/* ── cuModuleGetFunction ─ */

typedef struct __attribute__((__packed__)) cube_gpu_module_get_func_req {
    uint64_t module_handle;   /* Virtual module handle */
    char     func_name[256];  /* Null-terminated kernel function name */
} cube_gpu_module_get_func_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_module_get_func_resp {
    uint32_t status;
    uint64_t func_handle;     /* Virtual function handle */
} cube_gpu_module_get_func_resp_t;

/* ── cuLaunchKernel ────── */

typedef struct __attribute__((__packed__)) cube_gpu_launch_kernel_req {
    uint64_t func_handle;     /* Virtual function handle (from module_get_func) */
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint64_t stream_vptr;     /* Virtual stream handle */
    uint64_t params_shm_offset; /* Kernel params in SHM (array of pointers → values) */
    uint64_t params_size;     /* Total size of params data in SHM */
    uint64_t extra_shm_offset; /* Extra params ( CU_LAUNCH_PARAM_BUFFER_POINTER etc.) */
    uint64_t extra_size;
} cube_gpu_launch_kernel_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_launch_kernel_resp {
    uint32_t status;
} cube_gpu_launch_kernel_resp_t;

/* ── Stream management ─── */

typedef struct __attribute__((__packed__)) cube_gpu_stream_create_req {
    uint32_t flags;           /* Stream creation flags */
} cube_gpu_stream_create_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_stream_create_resp {
    uint32_t status;
    uint64_t stream_vptr;     /* Virtual stream handle */
} cube_gpu_stream_create_resp_t;

typedef struct __attribute__((__packed__)) cube_gpu_stream_destroy_req {
    uint64_t stream_vptr;
} cube_gpu_stream_destroy_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_stream_destroy_resp {
    uint32_t status;
} cube_gpu_stream_destroy_resp_t;

typedef struct __attribute__((__packed__)) cube_gpu_stream_sync_req {
    uint64_t stream_vptr;
} cube_gpu_stream_sync_req_t;

typedef struct __attribute__((__packed__)) cube_gpu_stream_sync_resp {
    uint32_t status;
} cube_gpu_stream_sync_resp_t;

/* ── cudaDeviceSynchronize ─ */

typedef struct __attribute__((__packed__)) cube_gpu_device_sync_resp {
    uint32_t status;
} cube_gpu_device_sync_resp_t;

/* ── cudaGetDeviceProperties ─ */

typedef struct __attribute__((__packed__)) cube_gpu_dev_prop_resp {
    uint32_t status;
    char     name[256];       /* Device name */
    uint64_t total_global_mem;/* Total GPU memory (quota, not physical) */
    uint32_t major;           /* Compute capability major */
    uint32_t minor;           /* Compute capability minor */
    uint32_t multi_processor_count;
    uint32_t max_threads_per_block;
    uint32_t max_threads_dim[3];
    uint32_t max_grid_size[3];
    uint32_t max_threads_per_multiprocessor;
    uint32_t clock_rate;
    uint32_t warp_size;
    uint32_t memory_clock_rate;
    uint32_t memory_bus_width;
} cube_gpu_dev_prop_resp_t;

/* ── cudaMemGetInfo ────── */

typedef struct __attribute__((__packed__)) cube_gpu_mem_get_info_resp {
    uint32_t status;
    uint64_t free_memory;     /* Remaining quota */
    uint64_t total_memory;    /* Total quota */
} cube_gpu_mem_get_info_resp_t;

/* ── Context management ── */

typedef struct __attribute__((packed)) cube_gpu_ctx_create_req {
    uint32_t flags;       /* CU_CTX_* flags */
    uint32_t device_id;   /* Device index */
} cube_gpu_ctx_create_req_t;

typedef struct __attribute__((packed)) cube_gpu_ctx_create_resp {
    int32_t  status;
    uint64_t ctx_handle;  /* Host CUcontext handle */
} cube_gpu_ctx_create_resp_t;

typedef struct __attribute__((packed)) cube_gpu_ctx_op_req {
    uint64_t ctx_handle;  /* Context to destroy / set current */
} cube_gpu_ctx_op_req_t;

typedef struct __attribute__((packed)) cube_gpu_ctx_op_resp {
    int32_t status;
} cube_gpu_ctx_op_resp_t;

typedef struct __attribute__((packed)) cube_gpu_ctx_get_resp {
    int32_t  status;
    uint64_t ctx_handle;  /* Current context for this session */
} cube_gpu_ctx_get_resp_t;

/* ─── Helper: compute total message size ────────────────────────── */

static inline size_t cube_gpu_msg_total_size(const cube_gpu_rpc_header_t *hdr) {
    return sizeof(cube_gpu_rpc_header_t) + hdr->payload_len;
}

/* ─── Helper: initialize a request header ───────────────────────── */

static inline void cube_gpu_init_header(
    cube_gpu_rpc_header_t *hdr,
    cube_gpu_api_id_t api_id,
    uint64_t request_id,
    uint64_t session_id,
    uint32_t payload_len,
    uint32_t flags)
{
    hdr->magic       = CUBE_GPU_RPC_MAGIC;
    hdr->version     = CUBE_GPU_RPC_VERSION;
    hdr->api_id      = (uint32_t)api_id;
    hdr->flags       = flags;
    hdr->request_id  = request_id;
    hdr->session_id  = session_id;
    hdr->payload_len = payload_len;
    hdr->status      = 0;
}

/* ─── Helper: initialize a response header ──────────────────────── */

static inline void cube_gpu_init_response(
    cube_gpu_rpc_header_t *hdr,
    cube_gpu_api_id_t api_id,
    uint64_t request_id,
    uint64_t session_id,
    uint32_t payload_len,
    uint32_t status)
{
    hdr->magic       = CUBE_GPU_RPC_MAGIC;
    hdr->version     = CUBE_GPU_RPC_VERSION;
    hdr->api_id      = (uint32_t)api_id;
    hdr->flags       = CUBE_GPU_FLAG_RESPONSE;
    hdr->request_id  = request_id;
    hdr->session_id  = session_id;
    hdr->payload_len = payload_len;
    hdr->status      = status;
}

#ifdef __cplusplus
}
#endif

#endif /* CUBESANDBOX_GPU_RPC_PROTOCOL_H */
