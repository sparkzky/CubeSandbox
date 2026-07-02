// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

use libloading::{Library, Symbol};
use tracing::{info, error};
use std::path::Path;

type CuResult = u32;

pub struct CudaRuntime {
    lib: Library,
    device_count: i32,
    shm_base: *mut u8,
    shm_size: usize,
}

unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    pub fn new() -> Result<Self, String> {
        let lib = unsafe {
            Library::new("libcuda.so.1").map_err(|e| format!("Failed to load libcuda.so.1: {}", e))?
        };

        unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> CuResult> =
                lib.get(b"cuInit\0").map_err(|e| format!("cuInit not found: {}", e))?;
            cu_init(0);

            let cu_device_get_count: Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> =
                lib.get(b"cuDeviceGetCount\0").map_err(|e| format!("cuDeviceGetCount not found: {}", e))?;

            let mut count = 0i32;
            cu_device_get_count(&mut count);
            info!("Found {} CUDA device(s)", count);

            if count == 0 {
                return Err("No CUDA devices found".to_string());
            }
        }

        Ok(Self {
            lib,
            device_count: 0,
            shm_base: std::ptr::null_mut(),
            shm_size: 0,
        })
    }

    pub fn device_count(&self) -> i32 {
        self.device_count
    }

    pub fn set_shm_region(&mut self, base: *mut u8, size: usize) {
        self.shm_base = base;
        self.shm_size = size;
    }

    pub fn mem_alloc(&self, size: u64) -> Result<u64, CuResult> {
        unsafe {
            let cu_mem_alloc: Symbol<unsafe extern "C" fn(*mut u64, u64) -> CuResult> =
                self.lib.get(b"cuMemAlloc_v2\0").unwrap();
            let mut devptr: u64 = 0;
            let result = cu_mem_alloc(&mut devptr, size);
            if result == 0 { Ok(devptr) } else { Err(result) }
        }
    }

    pub fn mem_free(&self, devptr: u64) -> CuResult {
        unsafe {
            let cu_mem_free: Symbol<unsafe extern "C" fn(u64) -> CuResult> =
                self.lib.get(b"cuMemFree_v2\0").unwrap();
            cu_mem_free(devptr)
        }
    }

    pub fn memcpy_h2d(&self, dst: u64, _src: u64, size: u64) -> CuResult {
        if self.shm_base.is_null() { return 1; }
        unsafe {
            let src_ptr = self.shm_base.offset(_src as isize);
            let cu_memcpy: Symbol<unsafe extern "C" fn(u64, *const std::ffi::c_void, u64) -> CuResult> =
                self.lib.get(b"cuMemcpyHtoD_v2\0").unwrap();
            cu_memcpy(dst, src_ptr as *const std::ffi::c_void, size)
        }
    }

    pub fn memcpy_h2d_via_shm(&self, dst: u64, shm_offset: u64, size: u64) -> CuResult {
        self.memcpy_h2d(dst, shm_offset, size)
    }

    pub fn memcpy_d2h(&self, _dst: u64, src: u64, size: u64) -> CuResult {
        if self.shm_base.is_null() { return 1; }
        unsafe {
            let dst_ptr = self.shm_base.offset(_dst as isize);
            let cu_memcpy: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, u64, u64) -> CuResult> =
                self.lib.get(b"cuMemcpyDtoH_v2\0").unwrap();
            cu_memcpy(dst_ptr as *mut std::ffi::c_void, src, size)
        }
    }

    pub fn memcpy_d2d(&self, dst: u64, src: u64, size: u64) -> CuResult {
        unsafe {
            let cu_memcpy: Symbol<unsafe extern "C" fn(u64, u64, u64) -> CuResult> =
                self.lib.get(b"cuMemcpy_v2\0").unwrap();
            cu_memcpy(dst, src, size)
        }
    }

    pub fn launch_kernel(
        &self,
        func: u64,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_mem: u32,
        _params_offset: u64, _params_size: u64,
    ) -> CuResult {
        unsafe {
            let cu_launch: Symbol<
                unsafe extern "C" fn(u64, u32, u32, u32, u32, u32, u32, u32, u64, *const *const std::ffi::c_void, *const *const std::ffi::c_void) -> CuResult
            > = self.lib.get(b"cuLaunchKernel\0").unwrap();

            cu_launch(func, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                      shared_mem, 0, std::ptr::null(), std::ptr::null())
        }
    }

    pub fn stream_create(&self) -> Result<u64, CuResult> {
        unsafe {
            let cu_stream_create: Symbol<unsafe extern "C" fn(*mut u64, u32) -> CuResult> =
                self.lib.get(b"cuStreamCreate\0").unwrap();
            let mut stream: u64 = 0;
            let result = cu_stream_create(&mut stream, 0);
            if result == 0 { Ok(stream) } else { Err(result) }
        }
    }

    pub fn stream_synchronize(&self, stream: u64) -> CuResult {
        unsafe {
            let cu_stream_sync: Symbol<unsafe extern "C" fn(u64) -> CuResult> =
                self.lib.get(b"cuStreamSynchronize\0").unwrap();
            cu_stream_sync(stream)
        }
    }

    pub fn stream_destroy(&self, stream: u64) -> CuResult {
        unsafe {
            let cu_stream_destroy: Symbol<unsafe extern "C" fn(u64) -> CuResult> =
                self.lib.get(b"cuStreamDestroy_v2\0").unwrap();
            cu_stream_destroy(stream)
        }
    }

    pub fn device_synchronize(&self) -> CuResult {
        unsafe {
            let cu_ctx_sync: Symbol<unsafe extern "C" fn() -> CuResult> =
                self.lib.get(b"cuCtxSynchronize\0").unwrap();
            cu_ctx_sync()
        }
    }

    pub fn module_load_from_shm(&self, shm_offset: u64, image_size: u64) -> Result<u64, CuResult> {
        if self.shm_base.is_null() { return Err(1); }
        unsafe {
            let image_ptr = self.shm_base.offset(shm_offset as isize);
            let cu_module_load: Symbol<unsafe extern "C" fn(*mut u64, *const std::ffi::c_void) -> CuResult> =
                self.lib.get(b"cuModuleLoadData\0").unwrap();
            let mut module: u64 = 0;
            let result = cu_module_load(&mut module, image_ptr as *const std::ffi::c_void);
            if result == 0 { Ok(module) } else { Err(result) }
        }
    }

    pub fn module_get_function(&self, module: u64, name: &str) -> Result<u64, CuResult> {
        unsafe {
            let c_name = std::ffi::CString::new(name).unwrap();
            let cu_module_get_func: Symbol<unsafe extern "C" fn(*mut u64, u64, *const i8) -> CuResult> =
                self.lib.get(b"cuModuleGetFunction\0").unwrap();
            let mut func: u64 = 0;
            let result = cu_module_get_func(&mut func, module, c_name.as_ptr());
            if result == 0 { Ok(func) } else { Err(result) }
        }
    }

    pub fn module_unload(&self, module: u64) -> CuResult {
        unsafe {
            let cu_module_unload: Symbol<unsafe extern "C" fn(u64) -> CuResult> =
                self.lib.get(b"cuModuleUnload\0").unwrap();
            cu_module_unload(module)
        }
    }
}
