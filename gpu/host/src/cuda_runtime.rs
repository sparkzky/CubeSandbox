// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

//! Thin libloading bindings over the real CUDA driver (`libcuda.so.1`).
//!
//! Every wrapper returns `CuResult` (`u32`) instead of panicking: a missing
//! symbol maps to `CUDA_ERROR_NOT_SUPPORTED`, never to a process abort.  All
//! context/handle types are represented as `u64` (pointer width on 64-bit);
//! `CUdevice` is an `int`.

use libloading::{Library, Symbol};
use std::ffi::c_void;
use tracing::info;

use crate::vsock_listener::{CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED, CUDA_SUCCESS};

pub type CuResult = u32;

/// `CUexecAffinityType` used with `cuCtxCreate_v3`.
const CU_EXEC_AFFINITY_TYPE_SM_COUNT: u32 = 0;

/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 0x10;

/// Mirror of CUDA's `CUexecAffinityParam` (enum followed by a union containing
/// a single `unsigned int`), layout-compatible with the driver header.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CuExecAffinityParam {
    pub affinity_type: u32,
    pub sm_count: u32,
}

pub struct CudaRuntime {
    lib: Library,
    device_count: i32,
}

// The driver library is thread-safe; access from the daemon is additionally
// serialized behind a mutex (CUcontext is per-thread state).
unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    pub fn new() -> Result<Self, String> {
        let lib = unsafe {
            Library::new("libcuda.so.1")
                .map_err(|e| format!("Failed to load libcuda.so.1: {}", e))?
        };

        let count = unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> CuResult> = lib
                .get(b"cuInit\0")
                .map_err(|e| format!("cuInit not found: {}", e))?;
            let rc = cu_init(0);
            if rc != CUDA_SUCCESS {
                return Err(format!("cuInit failed: {}", rc));
            }

            let cu_device_get_count: Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> = lib
                .get(b"cuDeviceGetCount\0")
                .map_err(|e| format!("cuDeviceGetCount not found: {}", e))?;
            let mut count: i32 = 0;
            let rc = cu_device_get_count(&mut count);
            if rc != CUDA_SUCCESS {
                return Err(format!("cuDeviceGetCount failed: {}", rc));
            }
            count
        };

        info!("Found {} CUDA device(s)", count);
        if count <= 0 {
            return Err("No CUDA devices found".to_string());
        }

        Ok(Self {
            lib,
            device_count: count,
        })
    }

    pub fn device_count(&self) -> i32 {
        self.device_count
    }

    /// Resolve a driver symbol.  A missing entry point is a protocol-level
    /// "feature unsupported", not a crash.
    unsafe fn sym<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, CuResult> {
        self.lib.get(name).map_err(|_| CUDA_ERROR_NOT_SUPPORTED)
    }

    // ── Memory ──────────────────────────────────────────────────────────

    pub fn mem_alloc(&self, size: u64) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, u64) -> CuResult> =
                self.sym(b"cuMemAlloc_v2\0")?;
            let mut ptr: u64 = 0;
            let rc = f(&mut ptr, size);
            if rc == CUDA_SUCCESS {
                Ok(ptr)
            } else {
                Err(rc)
            }
        }
    }

    pub fn mem_free(&self, devptr: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuMemFree_v2\0") {
                Ok(f) => f(devptr),
                Err(rc) => rc,
            }
        }
    }

    pub fn mem_alloc_async(&self, size: u64, stream: u64) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, u64, u64) -> CuResult> =
                self.sym(b"cuMemAllocAsync\0")?;
            let mut ptr: u64 = 0;
            let rc = f(&mut ptr, size, stream);
            if rc == CUDA_SUCCESS {
                Ok(ptr)
            } else {
                Err(rc)
            }
        }
    }

    pub fn mem_free_async(&self, devptr: u64, stream: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64, u64) -> CuResult>(b"cuMemFreeAsync\0") {
                Ok(f) => f(devptr, stream),
                Err(rc) => rc,
            }
        }
    }

    /// Host→device copy from a host buffer.
    pub fn memcpy_h2d(&self, dst: u64, src: &[u8]) -> CuResult {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(u64, *const c_void, u64) -> CuResult> =
                match self.sym(b"cuMemcpyHtoD_v2\0") {
                    Ok(f) => f,
                    Err(rc) => return rc,
                };
            f(dst, src.as_ptr() as *const c_void, src.len() as u64)
        }
    }

    /// Device→host copy into a host buffer.
    pub fn memcpy_d2h(&self, dst: &mut [u8], src: u64) -> CuResult {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut c_void, u64, u64) -> CuResult> =
                match self.sym(b"cuMemcpyDtoH_v2\0") {
                    Ok(f) => f,
                    Err(rc) => return rc,
                };
            f(dst.as_mut_ptr() as *mut c_void, src, dst.len() as u64)
        }
    }

    pub fn memcpy_d2d(&self, dst: u64, src: u64, size: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64, u64, u64) -> CuResult>(b"cuMemcpyDtoD_v2\0") {
                Ok(f) => f(dst, src, size),
                Err(rc) => rc,
            }
        }
    }

    /// Async driver entry point, exposed for completeness.  The session layer
    /// intentionally stages ring buffers through the synchronous copies above:
    /// a ring slot may be recycled as soon as the response is written, so an
    /// outstanding async DMA must not reference it.
    #[allow(dead_code)]
    pub fn memcpy_h2d_async(&self, dst: u64, src: &[u8], stream: u64) -> CuResult {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(u64, *const c_void, u64, u64) -> CuResult> =
                match self.sym(b"cuMemcpyHtoDAsync_v2\0") {
                    Ok(f) => f,
                    Err(rc) => return rc,
                };
            f(dst, src.as_ptr() as *const c_void, src.len() as u64, stream)
        }
    }

    #[allow(dead_code)]
    pub fn memcpy_d2h_async(&self, dst: &mut [u8], src: u64, stream: u64) -> CuResult {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut c_void, u64, u64, u64) -> CuResult> =
                match self.sym(b"cuMemcpyDtoHAsync_v2\0") {
                    Ok(f) => f,
                    Err(rc) => return rc,
                };
            f(dst.as_mut_ptr() as *mut c_void, src, dst.len() as u64, stream)
        }
    }

    /// `element_bits` is 8, 16 or 32; anything else is INVALID_VALUE.
    pub fn memset(&self, dst: u64, value: u32, element_bits: u32, size: u64) -> CuResult {
        unsafe {
            match element_bits {
                8 => {
                    let f: Symbol<unsafe extern "C" fn(u64, u8, u64) -> CuResult> =
                        match self.sym(b"cuMemsetD8_v2\0") {
                            Ok(f) => f,
                            Err(rc) => return rc,
                        };
                    f(dst, value as u8, size)
                }
                16 => {
                    let f: Symbol<unsafe extern "C" fn(u64, u16, u64) -> CuResult> =
                        match self.sym(b"cuMemsetD16_v2\0") {
                            Ok(f) => f,
                            Err(rc) => return rc,
                        };
                    f(dst, value as u16, size)
                }
                32 => {
                    let f: Symbol<unsafe extern "C" fn(u64, u32, u64) -> CuResult> =
                        match self.sym(b"cuMemsetD32_v2\0") {
                            Ok(f) => f,
                            Err(rc) => return rc,
                        };
                    f(dst, value, size)
                }
                _ => CUDA_ERROR_INVALID_VALUE,
            }
        }
    }

    // ── Kernels / modules ───────────────────────────────────────────────

    pub fn launch_kernel(
        &self,
        func: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: u64,
        kernel_params: *const *const c_void,
        extra: *const *const c_void,
    ) -> CuResult {
        unsafe {
            let f: Symbol<
                unsafe extern "C" fn(
                    u64,
                    u32,
                    u32,
                    u32,
                    u32,
                    u32,
                    u32,
                    u32,
                    u64,
                    *const *const c_void,
                    *const *const c_void,
                ) -> CuResult,
            > = match self.sym(b"cuLaunchKernel\0") {
                Ok(f) => f,
                Err(rc) => return rc,
            };
            f(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem,
                stream,
                kernel_params,
                extra,
            )
        }
    }

    pub fn module_load_data(&self, image: &[u8]) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, *const c_void) -> CuResult> =
                self.sym(b"cuModuleLoadData\0")?;
            let mut module: u64 = 0;
            let rc = f(&mut module, image.as_ptr() as *const c_void);
            if rc == CUDA_SUCCESS {
                Ok(module)
            } else {
                Err(rc)
            }
        }
    }

    pub fn module_get_function(&self, module: u64, name: &str) -> Result<u64, CuResult> {
        let c_name = match std::ffi::CString::new(name) {
            Ok(n) => n,
            Err(_) => return Err(CUDA_ERROR_INVALID_VALUE),
        };
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, u64, *const i8) -> CuResult> =
                self.sym(b"cuModuleGetFunction\0")?;
            let mut func: u64 = 0;
            let rc = f(&mut func, module, c_name.as_ptr());
            if rc == CUDA_SUCCESS {
                Ok(func)
            } else {
                Err(rc)
            }
        }
    }

    pub fn module_unload(&self, module: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuModuleUnload\0") {
                Ok(f) => f(module),
                Err(rc) => rc,
            }
        }
    }

    pub fn func_get_attribute(&self, func: u64, attr: i32) -> Result<i32, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut i32, i32, u64) -> CuResult> =
                self.sym(b"cuFuncGetAttribute\0")?;
            let mut value: i32 = 0;
            let rc = f(&mut value, attr, func);
            if rc == CUDA_SUCCESS {
                Ok(value)
            } else {
                Err(rc)
            }
        }
    }

    // ── Streams / sync ──────────────────────────────────────────────────

    pub fn stream_create(&self, flags: u32) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, u32) -> CuResult> =
                self.sym(b"cuStreamCreate\0")?;
            let mut stream: u64 = 0;
            let rc = f(&mut stream, flags);
            if rc == CUDA_SUCCESS {
                Ok(stream)
            } else {
                Err(rc)
            }
        }
    }

    pub fn stream_synchronize(&self, stream: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuStreamSynchronize\0") {
                Ok(f) => f(stream),
                Err(rc) => rc,
            }
        }
    }

    pub fn stream_destroy(&self, stream: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuStreamDestroy_v2\0") {
                Ok(f) => f(stream),
                Err(rc) => rc,
            }
        }
    }

    pub fn device_synchronize(&self) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn() -> CuResult>(b"cuCtxSynchronize\0") {
                Ok(f) => f(),
                Err(rc) => rc,
            }
        }
    }

    // ── Events ──────────────────────────────────────────────────────────

    pub fn event_create(&self, flags: u32) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, u32) -> CuResult> =
                self.sym(b"cuEventCreate\0")?;
            let mut event: u64 = 0;
            let rc = f(&mut event, flags);
            if rc == CUDA_SUCCESS {
                Ok(event)
            } else {
                Err(rc)
            }
        }
    }

    pub fn event_record(&self, event: u64, stream: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64, u64) -> CuResult>(b"cuEventRecord\0") {
                Ok(f) => f(event, stream),
                Err(rc) => rc,
            }
        }
    }

    pub fn event_synchronize(&self, event: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuEventSynchronize\0") {
                Ok(f) => f(event),
                Err(rc) => rc,
            }
        }
    }

    pub fn event_elapsed_time(&self, start: u64, end: u64) -> Result<f32, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut f32, u64, u64) -> CuResult> =
                self.sym(b"cuEventElapsedTime\0")?;
            let mut ms: f32 = 0.0;
            let rc = f(&mut ms, start, end);
            if rc == CUDA_SUCCESS {
                Ok(ms)
            } else {
                Err(rc)
            }
        }
    }

    pub fn event_destroy(&self, event: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuEventDestroy_v2\0") {
                Ok(f) => f(event),
                Err(rc) => rc,
            }
        }
    }

    // ── Device queries ──────────────────────────────────────────────────

    pub fn device_get(&self, ordinal: i32) -> Result<i32, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut i32, i32) -> CuResult> =
                self.sym(b"cuDeviceGet\0")?;
            let mut device: i32 = 0;
            let rc = f(&mut device, ordinal);
            if rc == CUDA_SUCCESS {
                Ok(device)
            } else {
                Err(rc)
            }
        }
    }

    pub fn device_get_attribute(&self, attr: i32, device: i32) -> Result<i32, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut i32, i32, i32) -> CuResult> =
                self.sym(b"cuDeviceGetAttribute\0")?;
            let mut value: i32 = 0;
            let rc = f(&mut value, attr, device);
            if rc == CUDA_SUCCESS {
                Ok(value)
            } else {
                Err(rc)
            }
        }
    }

    pub fn device_get_name(&self, device: i32) -> Result<String, CuResult> {
        let mut buf = [0i8; 256];
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> CuResult> =
                self.sym(b"cuDeviceGetName\0")?;
            let rc = f(buf.as_mut_ptr(), buf.len() as i32, device);
            if rc != CUDA_SUCCESS {
                return Err(rc);
            }
        }
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn device_total_mem(&self, device: i32) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut usize, i32) -> CuResult> =
                match self.sym(b"cuDeviceTotalMem_v2\0") {
                    Ok(f) => f,
                    Err(_) => match self.sym(b"cuDeviceTotalMem\0") {
                        Ok(f) => f,
                        Err(rc) => return Err(rc),
                    },
                };
            let mut total: usize = 0;
            let rc = f(&mut total, device);
            if rc == CUDA_SUCCESS {
                Ok(total as u64)
            } else {
                Err(rc)
            }
        }
    }

    pub fn ctx_get_device(&self) -> Result<i32, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> =
                self.sym(b"cuCtxGetDevice\0")?;
            let mut device: i32 = 0;
            let rc = f(&mut device);
            if rc == CUDA_SUCCESS {
                Ok(device)
            } else {
                Err(rc)
            }
        }
    }

    // ── Contexts ────────────────────────────────────────────────────────

    pub fn ctx_create(&self, flags: u32, device: i32) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, u32, i32) -> CuResult> =
                self.sym(b"cuCtxCreate_v2\0")?;
            let mut ctx: u64 = 0;
            let rc = f(&mut ctx, flags, device);
            if rc == CUDA_SUCCESS {
                Ok(ctx)
            } else {
                Err(rc)
            }
        }
    }

    /// `cuCtxCreate_v3` with a single SM-count execution-affinity parameter.
    pub fn ctx_create_v3(&self, flags: u32, device: i32, sm_count: u32) -> Result<u64, CuResult> {
        let param = CuExecAffinityParam {
            affinity_type: CU_EXEC_AFFINITY_TYPE_SM_COUNT,
            sm_count,
        };
        unsafe {
            let f: Symbol<
                unsafe extern "C" fn(*mut u64, *const CuExecAffinityParam, i32, u32, i32) -> CuResult,
            > = self.sym(b"cuCtxCreate_v3\0")?;
            let mut ctx: u64 = 0;
            let rc = f(&mut ctx, &param, 1, flags, device);
            if rc == CUDA_SUCCESS {
                Ok(ctx)
            } else {
                Err(rc)
            }
        }
    }

    pub fn ctx_set_current(&self, ctx: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuCtxSetCurrent\0") {
                Ok(f) => f(ctx),
                Err(rc) => rc,
            }
        }
    }

    pub fn ctx_destroy(&self, ctx: u64) -> CuResult {
        unsafe {
            match self.sym::<unsafe extern "C" fn(u64) -> CuResult>(b"cuCtxDestroy_v2\0") {
                Ok(f) => f(ctx),
                Err(rc) => rc,
            }
        }
    }

    // ── Primary context ─────────────────────────────────────────────────

    pub fn primary_ctx_retain(&self, device: i32) -> Result<u64, CuResult> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut u64, i32) -> CuResult> =
                self.sym(b"cuDevicePrimaryCtxRetain\0")?;
            let mut ctx: u64 = 0;
            let rc = f(&mut ctx, device);
            if rc == CUDA_SUCCESS {
                Ok(ctx)
            } else {
                Err(rc)
            }
        }
    }

    pub fn primary_ctx_release(&self, device: i32) -> CuResult {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(i32) -> CuResult> = match self
                .sym(b"cuDevicePrimaryCtxRelease_v2\0")
            {
                Ok(f) => f,
                Err(_) => match self.sym(b"cuDevicePrimaryCtxRelease\0") {
                    Ok(f) => f,
                    Err(rc) => return rc,
                },
            };
            f(device)
        }
    }
}
