// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::Mutex;
use tracing::{info, error};

use crate::cuda_runtime::CudaRuntime;
use crate::shm_manager::ShmManager;
use crate::vsock_listener::{
    CUBE_GPU_RPC_MAGIC, CUBE_GPU_FLAG_RESPONSE, CUBE_GPU_API_SESSION_INIT,
    CUBE_GPU_API_MALLOC, CUBE_GPU_API_FREE, CUBE_GPU_API_MEMCPY,
    CUBE_GPU_API_MODULE_LOAD, CUBE_GPU_API_MODULE_LOAD_DATA,
    CUBE_GPU_API_MODULE_GET_FUNC,
    CUBE_GPU_API_LAUNCH_KERNEL, CUBE_GPU_API_STREAM_CREATE,
    CUBE_GPU_API_STREAM_SYNC, CUBE_GPU_API_DEVICE_SYNC,
    CUBE_GPU_API_CTX_CREATE, CUBE_GPU_API_CTX_DESTROY,
    CUBE_GPU_API_CTX_SET_CURRENT, CUBE_GPU_API_CTX_GET_CURRENT,
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
    pub context: u64,
    pub contexts: HashSet<u64>,
    pub shm_offset: u64,
    pub shm_size: u64,
    pub shm_base: usize,
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
        let mut h = self.next_handle.lock();
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
            CUBE_GPU_API_MODULE_LOAD_DATA => self.handle_module_load(hdr, payload, session_handle),
            CUBE_GPU_API_MODULE_GET_FUNC => self.handle_module_get_func(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_CREATE => self.handle_ctx_create(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_DESTROY => self.handle_ctx_destroy(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_SET_CURRENT => self.handle_ctx_set_current(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_GET_CURRENT => self.handle_ctx_get_current(hdr, payload, session_handle),
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

        /* Finding 1: create and mmap the IVSHMEM backing file for this session. */
        let shm_path = match self.shm_mgr.create_shm_file(handle, shm_size) {
            Some(p) => p,
            None => return self.make_error_response(hdr, 1),
        };
        let shm_base = match self.shm_mgr.mmap_shm_file(&shm_path, shm_size) {
            Some(ptr) => ptr as usize,
            None => return self.make_error_response(hdr, 1),
        };

        let session = Session {
            handle,
            sandbox_id,
            memory_quota,
            memory_used: 0,
            allocations: HashMap::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
            streams: HashMap::new(),
            context: 0,
            contexts: HashSet::new(),
            shm_offset,
            shm_size,
            shm_base,
        };

        self.sessions.lock().insert(handle, session);
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

        let mut sessions = self.sessions.lock();
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

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.mem_alloc(size) {
            Ok(devptr) => {
                session.memory_used += size;
                session.allocations.insert(devptr, size);
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

        /* Finding 2: enforce ownership — only free allocations this session owns. */
        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.make_error_response(hdr, 1),
        };

        let (size, status) = match session.allocations.remove(&vptr) {
            Some(sz) => {
                session.memory_used = session.memory_used.saturating_sub(sz);
                let cuda = self.cuda.lock();
                if session.context != 0 {
                    cuda.ctx_set_current(session.context);
                }
                let st = cuda.mem_free(vptr);
                (0u64, st)
            }
            None => (0u64, 1),
        };
        let _ = size;

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
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 36 { return self.make_error_response(hdr, 1); }

        let kind = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let dst = u64::from_le_bytes(payload[4..12].try_into().unwrap());
        let src = u64::from_le_bytes(payload[12..20].try_into().unwrap());
        let size = u64::from_le_bytes(payload[20..28].try_into().unwrap());
        let shm_offset = u64::from_le_bytes(payload[28..36].try_into().unwrap());

        let has_shm = (hdr.flags & 0x01) != 0;

        /* Finding 1/D: read per-session shm region for bounds-checked access. */
        let (shm_base, shm_size, ctx) = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => (s.shm_base as *mut u8, s.shm_size as usize, s.context),
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
        let status = match kind {
            1 => {
                let off = if has_shm { shm_offset } else { src };
                cuda.memcpy_h2d(dst, off, size, shm_base, shm_size)
            }
            2 => {
                let off = if has_shm { shm_offset } else { dst };
                cuda.memcpy_d2h(off, src, size, shm_base, shm_size)
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
        let h = match session_handle {
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
        let stream = u64::from_le_bytes(payload[36..44].try_into().unwrap());
        let params_offset = u64::from_le_bytes(payload[44..52].try_into().unwrap());
        let params_size = u64::from_le_bytes(payload[52..60].try_into().unwrap());

        let (shm_base, shm_size, ctx) = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => (s.shm_base as *mut u8, s.shm_size as usize, s.context),
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
        let status = cuda.launch_kernel(
            func_handle, grid_x, grid_y, grid_z,
            block_x, block_y, block_z, shared_mem,
            stream, params_offset, params_size,
            shm_base, shm_size,
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
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        let ctx = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => s.context,
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
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
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 8 { return self.make_error_response(hdr, 1); }
        let stream = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let ctx = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => s.context,
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
        let status = cuda.stream_synchronize(stream);

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_device_sync(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        _payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        let ctx = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => s.context,
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
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
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 16 { return self.make_error_response(hdr, 1); }
        let image_size = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let shm_offset = u64::from_le_bytes(payload[8..16].try_into().unwrap());

        let (shm_base, shm_size, ctx) = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => (s.shm_base as *mut u8, s.shm_size as usize, s.context),
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
        match cuda.module_load_from_shm(shm_offset, image_size, shm_base, shm_size) {
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
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        if payload.len() < 264 { return self.make_error_response(hdr, 1); }
        let _module = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let name_bytes: Vec<u8> = payload[8..264].to_vec();
        let name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_string();

        let ctx = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => s.context,
                None => return self.make_error_response(hdr, 1),
            }
        };

        let cuda = self.cuda.lock();
        if ctx != 0 {
            cuda.ctx_set_current(ctx);
        }
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

    fn handle_ctx_create(
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
        let flags = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let device_id = i32::from_le_bytes(payload[4..8].try_into().unwrap());

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.make_error_response(hdr, 1),
        };

        let cuda = self.cuda.lock();
        match cuda.ctx_create(flags, device_id) {
            Ok(ctx) => {
                session.contexts.insert(ctx);
                session.context = ctx;
                let mut resp = Vec::with_capacity(12);
                resp.extend_from_slice(&0u32.to_le_bytes());
                resp.extend_from_slice(&ctx.to_le_bytes());
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

    fn handle_ctx_destroy(
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
        let ctx_handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.make_error_response(hdr, 1),
        };

        session.contexts.remove(&ctx_handle);
        if session.context == ctx_handle {
            session.context = 0;
        }

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let status = cuda.ctx_destroy(ctx_handle);

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&status.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, status), resp)
    }

    fn handle_ctx_set_current(
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
        let ctx_handle = u64::from_le_bytes(payload[0..8].try_into().unwrap());

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.make_error_response(hdr, 1),
        };

        session.context = ctx_handle;

        let mut resp = Vec::with_capacity(4);
        resp.extend_from_slice(&0u32.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
    }

    fn handle_ctx_get_current(
        &self,
        hdr: &crate::vsock_listener::RpcHeader,
        _payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (crate::vsock_listener::RpcHeader, Vec<u8>) {
        let h = match session_handle {
            Some(h) => *h,
            None => return self.make_error_response(hdr, 1),
        };

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.make_error_response(hdr, 1),
        };
        let ctx = session.context;

        let mut resp = Vec::with_capacity(12);
        resp.extend_from_slice(&0u32.to_le_bytes());
        resp.extend_from_slice(&ctx.to_le_bytes());
        (self.make_response_hdr(hdr, resp.len() as u32, 0), resp)
    }

    pub fn destroy_session(&self, handle: u64) {
        let mut sessions = self.sessions.lock();
        if let Some(mut session) = sessions.remove(&handle) {
            let cuda = self.cuda.lock();
            for (_, ptr) in session.allocations.drain() {
                let _ = cuda.mem_free(ptr);
            }
            for (_, ptr) in session.modules.drain() {
                let _ = cuda.module_unload(ptr);
            }
            for (_, ptr) in session.streams.drain() {
                let _ = cuda.stream_destroy(ptr);
            }
            for ctx in session.contexts.drain() {
                let _ = cuda.ctx_destroy(ctx);
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
