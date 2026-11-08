// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

pub const CUBE_GPU_RPC_MAGIC: u32 = 0x43554441;
pub const CUBE_GPU_RPC_VERSION: u32 = 2;

#[repr(C, packed)]
#[derive(Clone, Debug)]
pub struct RpcHeader {
    pub magic: u32,
    pub version: u32,
    pub api_id: u32,
    pub flags: u32,
    pub request_id: u64,
    pub session_id: u64,
    pub payload_len: u32,
    pub status: u32,
}

// ─── Shared-memory layout (mirrors rpc_protocol.h) ───────────────────

pub const CUBE_GPU_SHM_GLOBAL_SIZE: u64 = 4096;
pub const CUBE_GPU_SHM_RING_HEADER_SIZE: u64 = 128;
pub const CUBE_GPU_SHM_MAGIC: u64 = 0x4355424547505531; // "CUBEGPU1"
pub const CUBE_GPU_SHM_VERSION: u32 = 1;
pub const CUBE_GPU_SHM_RING_MAGIC: u32 = 0x52494E47; // "RING"
pub const CUBE_GPU_SHM_TOKEN_LEN: usize = 32;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ShmGlobal {
    pub magic: u64,
    pub version: u32,
    pub flags: u32,
    pub token: [u8; CUBE_GPU_SHM_TOKEN_LEN],
    pub total_size: u64,
    pub ring_size: u64,
    pub g2h_offset: u64,
    pub h2g_offset: u64,
    pub session_id: u64,
    pub memory_quota: u64,
    pub reserved: [u8; 32],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ShmRingHeader {
    pub write_pos: u64,
    pub read_pos: u64,
    pub capacity: u64,
    pub magic: u32,
    pub flags: u32,
    pub epoch: u32,
    pub reserved: [u32; 9],
}

// CUDA API identifiers — must mirror gpu/proto/rpc_protocol.h (cube_gpu_api_id)
pub const CUBE_GPU_API_SESSION_INIT: u32     = 0;
pub const CUBE_GPU_API_SESSION_DESTROY: u32  = 1;
pub const CUBE_GPU_API_MALLOC: u32           = 10;
pub const CUBE_GPU_API_FREE: u32             = 11;
pub const CUBE_GPU_API_MEMCPY: u32           = 12;
pub const CUBE_GPU_API_MEMCPY_ASYNC: u32     = 13;
pub const CUBE_GPU_API_MEMSET: u32           = 14;
pub const CUBE_GPU_API_MEMSET_ASYNC: u32     = 15;
pub const CUBE_GPU_API_MALLOC_ASYNC: u32     = 16;
pub const CUBE_GPU_API_FREE_ASYNC: u32       = 17;
pub const CUBE_GPU_API_MODULE_LOAD: u32      = 20;
pub const CUBE_GPU_API_MODULE_LOAD_DATA: u32 = 21;
pub const CUBE_GPU_API_MODULE_GET_FUNC: u32  = 22;
pub const CUBE_GPU_API_LAUNCH_KERNEL: u32    = 23;
pub const CUBE_GPU_API_STREAM_CREATE: u32    = 30;
pub const CUBE_GPU_API_STREAM_DESTROY: u32   = 31;
pub const CUBE_GPU_API_STREAM_SYNC: u32      = 32;
pub const CUBE_GPU_API_DEVICE_SYNC: u32      = 33;
pub const CUBE_GPU_API_EVENT_CREATE: u32     = 34;
pub const CUBE_GPU_API_EVENT_RECORD: u32     = 35;
pub const CUBE_GPU_API_EVENT_SYNC: u32       = 36;
pub const CUBE_GPU_API_EVENT_ELAPSED: u32    = 37;
pub const CUBE_GPU_API_EVENT_DESTROY: u32    = 38;
pub const CUBE_GPU_API_GET_DEV_PROP: u32     = 40;
pub const CUBE_GPU_API_MEM_GET_INFO: u32     = 41;
pub const CUBE_GPU_API_GET_DEVICE: u32       = 42;
pub const CUBE_GPU_API_SET_DEVICE: u32       = 43;
pub const CUBE_GPU_API_DEVICE_COUNT: u32     = 44;
pub const CUBE_GPU_API_GET_DEVICE_NAME: u32  = 45;
pub const CUBE_GPU_API_DEVICE_TOTAL_MEM: u32 = 46;
pub const CUBE_GPU_API_FUNC_GET_ATTR: u32    = 47;
pub const CUBE_GPU_API_CTX_GET_DEVICE: u32   = 48;
pub const CUBE_GPU_API_GET_LAST_ERROR: u32   = 49;
pub const CUBE_GPU_API_CTX_CREATE: u32      = 50;
pub const CUBE_GPU_API_CTX_DESTROY: u32     = 51;
pub const CUBE_GPU_API_CTX_SET_CURRENT: u32 = 52;
pub const CUBE_GPU_API_CTX_GET_CURRENT: u32 = 53;
pub const CUBE_GPU_API_PRIMARY_CTX_RETAIN: u32  = 54;
pub const CUBE_GPU_API_PRIMARY_CTX_RELEASE: u32 = 55;
pub const CUBE_GPU_API_IPC_GET_HANDLE: u32   = 60;
pub const CUBE_GPU_API_IPC_OPEN_HANDLE: u32  = 61;
pub const CUBE_GPU_API_IPC_CLOSE_HANDLE: u32 = 62;

// Header flags — must mirror gpu/proto/rpc_protocol.h (CUBE_GPU_FLAG_*)
pub const CUBE_GPU_FLAG_NONE: u32     = 0x00;
pub const CUBE_GPU_FLAG_SHM_DATA: u32 = 0x01;
pub const CUBE_GPU_FLAG_ASYNC: u32    = 0x02;
pub const CUBE_GPU_FLAG_ERROR: u32    = 0x40;
pub const CUBE_GPU_FLAG_RESPONSE: u32 = 0x80;

// Session init flags
pub const CUBE_GPU_SESSION_FLAG_NONE: u32 = 0x0;
pub const CUBE_GPU_SESSION_FLAG_TOKEN: u32 = 0x1;

// CUresult codes used across the protocol
pub const CUDA_SUCCESS: u32 = 0;
pub const CUDA_ERROR_INVALID_VALUE: u32 = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: u32 = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: u32 = 3;
pub const CUDA_ERROR_DEINITIALIZED: u32 = 4;
pub const CUDA_ERROR_INVALID_CONTEXT: u32 = 201;
pub const CUDA_ERROR_INVALID_HANDLE: u32 = 400;
pub const CUDA_ERROR_NOT_SUPPORTED: u32 = 801;
