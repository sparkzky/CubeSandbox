// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, error};

use crate::cuda_runtime::CudaRuntime;
use crate::shm_manager::ShmManager;
use crate::vsock_listener::{
    CUBE_GPU_RPC_MAGIC, CUBE_GPU_FLAG_RESPONSE, CUBE_GPU_API_SESSION_INIT,
    CUBE_GPU_API_MALLOC, CUBE_GPU_API_FREE, CUBE_GPU_API_MEMCPY,
    CUBE_GPU_API_MODULE_LOAD, CUBE_GPU_API_MODULE_GET_FUNC,
    CUBE_GPU_API_LAUNCH_KERNEL, CUBE_GPU_API_STREAM_CREATE,
    CUBE_GPU_API_STREAM_SYNC, CUBE_GPU_API_DEVICE_SYNC,
};

pub struct Session {
    pub handle: u64,
    pub sandbox_id: u64,
    pub memory_quota: u64,
    pub memory_used: u64,
    pub allocations: HashMap<u64, u64>,
    pub modules: HashMap<u64, u64>,
    pub functions: HashMap<u64, (String, u64)>,
    pub streams: HashMap<u64, u64>,
    pub shm_offset: u64,
    pub shm_size: u64,
}

impl Session {
    pub fn can_allocate(&self, size: u64) -> bool {
        self.memory_used + size <= self.memory_quota
    }
}

pub struct SessionManager {
    cuda: Arc<Mutex<CudaRuntime>>,
    shm_mgr: Arc<ShmManager>,
    sessions: Mutex<HashMap<u64, Session>>,
    next_handle: Mutex<u64>,
}

impl SessionManager {
    pub fn new(cuda: CudaRuntime, shm_mgr: ShmManager) -> Arc<Self> {
        Arc::new(Self {
            cuda: Arc::new(Mutex::new(cuda)),
            shm_mgr: Arc::new(shm_mgr),
            sessions: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1),
        })
    }

    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        let v = *h;
        *h += 1;
        v
    }

    pub fn dispatch(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        match hdr.api_id {
            CUBE_GPU_API_SESSION_INIT => self.handle_session_init(hdr, payload, session_handle),
            CUBE_GPU_API_MALLOC => self.handle_malloc(hdr, payload, session_handle),
            CUBE_GPU_API_FREE => self.handle_free(hdr, payload, session_handle),
            CUBE_GPU_API_MEMCPY => self.handle_memcpy(hdr, payload, session_handle),
            CUBE_GPU_API_LAUNCH_KERNEL => self.handle_launch_kernel(hdr, payload, session_handle),
            CUBE_GPU_API_STREAM_CREATE => self.handle_stream_create(hdr, payload, session_handle),
            CUBE_GPU_API_STREAM_SYNC => self.handle_stream_sync(hdr, payload, session_handle),
            CUBE_GPU_API_DEVICE_SYNC => self.handle_device_sync(hdr, payload, session_handle),
            CUBE_GPU_API_MODULE_LOAD => self.handle_module_load(hdr, payload, session_handle),
            CUBE_GPU_API_MODULE_GET_FUNC => self.handle_module_get_func(hdr, payload, session_handle),
            _ => self.make_error_response(hdr, 1),
        }
    }

    fn handle_session_init(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        if payload.len() < 16 { return self.make_error_response(hdr, 1); }

        let memory_quota = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let sandbox_id = u64::from_le_bytes(payload[8..16].try_into().unwrap());

        let handle = self.alloc_handle();
        let shm_size = 64 * 1024 * 1024;
        let shm_offset = handle * shm_size;

        let session = Session {
            handle,
            sandbox_id,
            memory_quota,
            memory_used: 0,
            allocations: HashMap::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
            streams: HashMap::new(),
            shm_offset,
            shm_size,
        };

        self.sessions.lock().unwrap().insert(handle, session);
        *session_handle = Some(handle);

        info!("Session init: handle={} sandbox={} quota={}", handle, sandbox_id, memory_quota);

        let mut resp = Vec::with_capacity(32);
        resp.extend_from_slice(&0u32.to_le_bytes());
        resp.extend_from_slice(&handle.to_le_bytes());
        resp.extend_from_slice(&shm_offset.to_le_bytes());
        resp.extend_from_slice(&shm_size.to_le_bytes());

        (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
    }

    fn handle_malloc(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 8 { return self.make_error_response(hdr, 1); }
        let size = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let mut sessions = self.sessions.lock().unwrap();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.make_error_response(hdr, 1),
        };

        if !session.can_allocate(size) {
            let mut resp = Vec::with_capacity(12);
            resp.extend_from_slice(&2u32.to_le_bytes());
            resp.extend_from_slice(&0u64.to_le_bytes());
            return (self.make_response_hdr(hdr, resp.len() as u32, 2), resp);
        }

        let cuda = self.cuda.lock().unwrap();
        match cuda.mem_alloc(size) {
            Ok(devptr) => {
                session.memory_used += size;
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&0u32.to_le_bytes());
                resp.extend_from_slice(&devptr.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
            }
            Err(status) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&status.to_le_bytes());
                resp.extend_from_slice(&0u64.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
            }
        }
    }

    fn handle_free(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 8 { return self.make_error_response(hdr, 1); }
        let vptr = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let cuda = self.cuda.lock().unwrap();
        let status = cuda.mem_free(vptr);

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_memcpy(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let _h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 40 { return self.make_error_response(hdr, 1); }

        let kind = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let dst = u64::from_le_bytes(payload[4..12].try_into().unwrap());
        let src = u64::from_le_bytes(payload[12..20].try_into().unwrap());
        let size = u64::from_le_bytes(payload[20..28].try_into().unwrap());
        let shm_offset = u64::from_le_bytes(payload[28..36].try_into().unwrap());

        let has_shm = (hdr.flags & 0x01) != 0;
        let cuda = self.cuda.lock().unwrap();

        let status = match kind {
            1 => {
                if has_shm {
                    cuda.memcpy_h2d_via_shm(dst, shm_offset, size)
                } else {
                    cuda.memcpy_h2d(dst, src, size)
                }
            }
            2 => {
                let status = cuda.memcpy_d2h(dst, src, size);
                if has_shm && status == 0 {
                    // Host wrote result to SHM — nothing more to do here
                }
                status
            }
            3 => cuda.memcpy_d2d(dst, src, size),
            _ => 1,
        };

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_launch_kernel(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let _h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 64 { return self.make_error_response(hdr, 1); }

        let func_handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let grid_x = u32::from_le_bytes(payload[8..12].try_into().unwrap());
        let grid_y = u32::from_le_bytes(payload[12..16].try_into().unwrap());
        let grid_z = u32::from_le_bytes(payload[16..20].try_into().unwrap());
        let block_x = u32::from_le_bytes(payload[20..24].try_into().unwrap());
        let block_y = u32::from_le_bytes(payload[24..28].try_into().unwrap());
        let block_z = u32::from_le_bytes(payload[28..32].try_into().unwrap());
        let shared_mem = u32::from_le_bytes(payload[32..36].try_into().unwrap());
        let _stream = u64::from_le_bytes(payload[36..44].try_into().unwrap());
        let params_offset = u64::from_le_bytes(payload[44..52].try_into().unwrap());
        let params_size = u64::from_le_bytes(payload[52..60].try_into().unwrap());

        let cuda = self.cuda.lock().unwrap();
        let status = cuda.launch_kernel(
            func_handle, grid_x, grid_y, grid_z,
            block_x, block_y, block_z, shared_mem,
            params_offset, params_size,
        );

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_stream_create(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        _payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let _h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        let cuda = self.cuda.lock().unwrap();
        match cuda.stream_create() {
            Ok(stream_ptr) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&0u32.to_le_bytes());
                resp.extend_from_slice(&stream_ptr.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
            }
            Err(status) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&status.to_le_bytes());
                resp.extend_from_slice(&0u64.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
            }
        }
    }

    fn handle_stream_sync(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let _h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 8 { return self.make_error_response(hdr, 1); }
        let stream = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let cuda = self.cuda.lock().unwrap();
        let status = cuda.stream_synchronize(stream);

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_device_sync(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        _payload: &[u8],
        _session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let cuda = self.cuda.lock().unwrap();
        let status = cuda.device_synchronize();

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_module_load(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let _h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 16 { return self.make_error_response(hdr, 1); }
        let _image_size = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let _shm_offset = u64::from_le_bytes(payload[8..16].try_into().unwrap());

        let cuda = self.cuda.lock().unwrap();
        match cuda.module_load_from_shm(_shm_offset, _image_size) {
            Ok(module) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&0u32.to_le_bytes());
                resp.extend_from_slice(&module.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
            }
            Err(status) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&status.to_le_bytes());
                resp.extend_from_slice(&0u64.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
            }
        }
    }

    fn handle_module_get_func(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let _h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 264 { return self.make_error_response(hdr, 1); }
        let _module = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let name_bytes: Vec<u8> = payload[8..264].to_vec();
        let name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_string();

        let cuda = self.cuda.lock().unwrap();
        match cuda.module_get_function(_module, &name) {
            Ok(func) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&0u32.to_le_bytes());
                resp.extend_from_slice(&func.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
            }
            Err(status) => {
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&status.to_le_bytes());
                resp.extend_from_slice(&0u64.to_le_bytes());
                (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
            }
        }
    }

    pub fn destroy_session(&self, handle: u64) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(mut session) = sessions.remove(&handle) {
            let cuda = self.cuda.lock().unwrap();
            for (_, ptr) in session.allocations.drain() {
                let _ = cuda.mem_free(ptr);
            }
            for (_, ptr) in session.modules.drain() {
                let _ = cuda.module_unload(ptr);
            }
            for (_, ptr) in session.streams.drain() {
                let _ = cuda.stream_destroy(ptr);
            }
            info!("Session {} destroyed, freed {} bytes", handle, session.memory_used);
        }
    }

    fn make_response_hdr(
        &self,
        req: &crate::vsock_listener::RpcHeader,
        payload_len: u32,
        status: u32,
    ) -> crate::vsock_listener::RpcHeader {
        crate::vsock_listener::RpcHeader {
            magic: CUBE_GPU_RPC_MAGIC,
            version: 1,
            api_id: req.api_id,
            flags: CUBE_GPU_FLAG_RESPONSE,
            request_id: req.request_id,
            session_id: req.session_id,
            payload_len,
            status,
        }
    }

    fn make_error_response(
        &self,
        req: &crate::vsock_listener::RpcHeader,
        status: u32,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let resp = vec![0u8; 4];
        (self.make_response_hdr(req, 4, status), resp)
    }
}
