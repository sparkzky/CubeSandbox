// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

pub const CUBE_GPU_RPC_MAGIC: u32 = 0x43554441;

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

// CUDA API identifiers — must mirror gpu/proto/rpc_protocol.h (cube_gpu_api_id)
pub const CUBE_GPU_API_SESSION_INIT: u32     = 0;
pub const CUBE_GPU_API_SESSION_DESTROY: u32  = 1;
pub const CUBE_GPU_API_MALLOC: u32           = 10;
pub const CUBE_GPU_API_FREE: u32             = 11;
pub const CUBE_GPU_API_MEMCPY: u32           = 12;
pub const CUBE_GPU_API_MEMCPY_ASYNC: u32     = 13;
pub const CUBE_GPU_API_MODULE_LOAD: u32      = 20;
pub const CUBE_GPU_API_MODULE_LOAD_DATA: u32 = 21;
pub const CUBE_GPU_API_MODULE_GET_FUNC: u32  = 22;
pub const CUBE_GPU_API_LAUNCH_KERNEL: u32    = 23;
pub const CUBE_GPU_API_STREAM_CREATE: u32    = 30;
pub const CUBE_GPU_API_STREAM_DESTROY: u32   = 31;
pub const CUBE_GPU_API_STREAM_SYNC: u32      = 32;
pub const CUBE_GPU_API_DEVICE_SYNC: u32      = 33;
pub const CUBE_GPU_API_CTX_CREATE: u32      = 50;
pub const CUBE_GPU_API_CTX_DESTROY: u32     = 51;
pub const CUBE_GPU_API_CTX_SET_CURRENT: u32 = 52;
pub const CUBE_GPU_API_CTX_GET_CURRENT: u32 = 53;

// Header flags — must mirror gpu/proto/rpc_protocol.h (CUBE_GPU_FLAG_*)
pub const CUBE_GPU_FLAG_NONE: u32     = 0x00;
pub const CUBE_GPU_FLAG_SHM_DATA: u32 = 0x01;
pub const CUBE_GPU_FLAG_ASYNC: u32    = 0x02;
pub const CUBE_GPU_FLAG_ERROR: u32    = 0x40;
pub const CUBE_GPU_FLAG_RESPONSE: u32 = 0x80;
