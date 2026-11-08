// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

//! Per-connection CUDA session state and RPC handlers.
//!
//! `SessionManager::dispatch` is the single entry point used by the vsock
//! listener.  All handlers validate payload lengths strictly, enforce
//! per-session ownership of contexts/allocations/modules/functions/streams/
//! events, and treat the daemon-side quota as authoritative.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{error, info, warn};

use crate::cuda_runtime::{CudaRuntime, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT};
use crate::shm_manager::{ShmChannel, ShmManager};
use crate::vsock_listener::{
    RpcHeader, CUBE_GPU_FLAG_RESPONSE, CUBE_GPU_FLAG_SHM_DATA, CUBE_GPU_RPC_MAGIC,
    CUBE_GPU_RPC_VERSION, CUBE_GPU_API_CTX_CREATE, CUBE_GPU_API_CTX_DESTROY,
    CUBE_GPU_API_CTX_GET_CURRENT, CUBE_GPU_API_CTX_GET_DEVICE, CUBE_GPU_API_CTX_SET_CURRENT,
    CUBE_GPU_API_DEVICE_COUNT, CUBE_GPU_API_DEVICE_SYNC, CUBE_GPU_API_DEVICE_TOTAL_MEM,
    CUBE_GPU_API_EVENT_CREATE, CUBE_GPU_API_EVENT_DESTROY, CUBE_GPU_API_EVENT_ELAPSED,
    CUBE_GPU_API_EVENT_RECORD, CUBE_GPU_API_EVENT_SYNC, CUBE_GPU_API_FREE,
    CUBE_GPU_API_FREE_ASYNC, CUBE_GPU_API_FUNC_GET_ATTR, CUBE_GPU_API_GET_DEVICE,
    CUBE_GPU_API_GET_DEVICE_NAME, CUBE_GPU_API_GET_DEV_PROP, CUBE_GPU_API_GET_LAST_ERROR,
    CUBE_GPU_API_IPC_CLOSE_HANDLE, CUBE_GPU_API_IPC_GET_HANDLE, CUBE_GPU_API_IPC_OPEN_HANDLE,
    CUBE_GPU_API_LAUNCH_KERNEL, CUBE_GPU_API_MALLOC, CUBE_GPU_API_MALLOC_ASYNC,
    CUBE_GPU_API_MEMCPY, CUBE_GPU_API_MEMCPY_ASYNC, CUBE_GPU_API_MEM_GET_INFO,
    CUBE_GPU_API_MEMSET, CUBE_GPU_API_MEMSET_ASYNC, CUBE_GPU_API_MODULE_GET_FUNC,
    CUBE_GPU_API_MODULE_LOAD, CUBE_GPU_API_MODULE_LOAD_DATA,
    CUBE_GPU_API_PRIMARY_CTX_RELEASE, CUBE_GPU_API_PRIMARY_CTX_RETAIN,
    CUBE_GPU_API_SESSION_DESTROY, CUBE_GPU_API_SESSION_INIT, CUBE_GPU_API_SET_DEVICE,
    CUBE_GPU_API_STREAM_CREATE, CUBE_GPU_API_STREAM_DESTROY, CUBE_GPU_API_STREAM_SYNC,
    CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
    CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_SUCCESS,
    CUBE_GPU_SHM_TOKEN_LEN,
};

/// Maximum inline vsock payload (advertised in `session_init_resp`).
pub const MAX_PAYLOAD: usize = 16 * 1024 * 1024;
/// Sanity cap for module images read through the G2H ring.
const MAX_MODULE_IMAGE: u64 = 64 * 1024 * 1024;
/// `module_get_func_resp` carries at most this many parameter sizes.
const MAX_KERNEL_PARAMS: usize = 32;

// CU_LAUNCH_PARAM_* sentinels used with the `extra` argument.
const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;
const CU_LAUNCH_PARAM_END: usize = 0x00;

#[derive(Default)]
pub struct Session {
    pub handle: u64,
    pub sandbox_id: u64,
    pub memory_quota: u64,
    pub memory_used: u64,
    pub allocations: HashMap<u64, u64>,
    /// Host module handle -> host module handle (membership registry).
    pub modules: HashMap<u64, u64>,
    /// Host module handle -> (kernel name -> parameter byte sizes).
    pub module_meta: HashMap<u64, HashMap<String, Vec<u16>>>,
    /// Host function handle -> (kernel name, owning host module).
    pub functions: HashMap<u64, (String, u64)>,
    pub streams: HashSet<u64>,
    /// Host event handle -> creation flags.
    pub events: HashMap<u64, u64>,
    pub context: u64,
    pub contexts: HashSet<u64>,
    pub primary_ctxs: HashMap<i32, u64>,
    pub current_device: i32,
    pub sm_percent: u64,
    pub shm: Option<ShmChannel>,
    /// `(total_size, ring_capacity, g2h_offset, h2g_offset)`.
    pub shm_layout: Option<(u64, u64, u64, u64)>,
    pub last_error: u32,
}

impl Session {
    pub fn can_allocate(&self, size: u64) -> bool {
        self.memory_used
            .checked_add(size)
            .map_or(false, |total| total <= self.memory_quota)
    }

    fn owns_stream(&self, stream: u64) -> bool {
        stream == 0 || self.streams.contains(&stream)
    }
}

pub struct SessionManager {
    cuda: Arc<Mutex<CudaRuntime>>,
    shm_mgr: Arc<ShmManager>,
    sessions: Mutex<HashMap<u64, Session>>,
    next_handle: Mutex<u64>,
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn rd_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn push_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
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

    fn response_header(req: &RpcHeader, payload_len: u32, status: u32) -> RpcHeader {
        RpcHeader {
            magic: CUBE_GPU_RPC_MAGIC,
            version: CUBE_GPU_RPC_VERSION,
            api_id: req.api_id,
            flags: CUBE_GPU_FLAG_RESPONSE,
            request_id: req.request_id,
            session_id: req.session_id,
            payload_len,
            status,
        }
    }

    /// Minimal 4-byte status response (extra fields stay zero on the guest).
    fn status_response(&self, req: &RpcHeader, status: u32) -> (RpcHeader, Vec<u8>) {
        let mut payload = Vec::with_capacity(4);
        push_u32(&mut payload, status);
        (Self::response_header(req, 4, status), payload)
    }

    fn ok_response(&self, req: &RpcHeader, payload: Vec<u8>) -> (RpcHeader, Vec<u8>) {
        (Self::response_header(req, payload.len() as u32, 0), payload)
    }

    pub fn dispatch(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        match hdr.api_id {
            CUBE_GPU_API_SESSION_INIT => self.handle_session_init(hdr, payload, session_handle),
            CUBE_GPU_API_SESSION_DESTROY => self.handle_session_destroy(hdr, session_handle),
            CUBE_GPU_API_MALLOC => self.handle_malloc(hdr, payload, session_handle),
            CUBE_GPU_API_FREE => self.handle_free(hdr, payload, session_handle),
            CUBE_GPU_API_MEMCPY => self.handle_memcpy(hdr, payload, session_handle, false),
            CUBE_GPU_API_MEMCPY_ASYNC => self.handle_memcpy(hdr, payload, session_handle, true),
            CUBE_GPU_API_MEMSET => self.handle_memset(hdr, payload, session_handle),
            CUBE_GPU_API_MEMSET_ASYNC => self.handle_memset(hdr, payload, session_handle),
            CUBE_GPU_API_MALLOC_ASYNC => self.handle_malloc_async(hdr, payload, session_handle),
            CUBE_GPU_API_FREE_ASYNC => self.handle_free_async(hdr, payload, session_handle),
            CUBE_GPU_API_MODULE_LOAD | CUBE_GPU_API_MODULE_LOAD_DATA => {
                self.handle_module_load(hdr, payload, session_handle)
            }
            CUBE_GPU_API_MODULE_GET_FUNC => self.handle_module_get_func(hdr, payload, session_handle),
            CUBE_GPU_API_LAUNCH_KERNEL => self.handle_launch_kernel(hdr, payload, session_handle),
            CUBE_GPU_API_STREAM_CREATE => self.handle_stream_create(hdr, payload, session_handle),
            CUBE_GPU_API_STREAM_DESTROY => self.handle_stream_destroy(hdr, payload, session_handle),
            CUBE_GPU_API_STREAM_SYNC => self.handle_stream_sync(hdr, payload, session_handle),
            CUBE_GPU_API_DEVICE_SYNC => self.handle_device_sync(hdr, session_handle),
            CUBE_GPU_API_EVENT_CREATE => self.handle_event_create(hdr, payload, session_handle),
            CUBE_GPU_API_EVENT_RECORD => self.handle_event_record(hdr, payload, session_handle),
            CUBE_GPU_API_EVENT_SYNC => self.handle_event_sync(hdr, payload, session_handle),
            CUBE_GPU_API_EVENT_ELAPSED => self.handle_event_elapsed(hdr, payload, session_handle),
            CUBE_GPU_API_EVENT_DESTROY => self.handle_event_destroy(hdr, payload, session_handle),
            CUBE_GPU_API_GET_DEV_PROP => self.handle_get_dev_prop(hdr, payload, session_handle),
            CUBE_GPU_API_MEM_GET_INFO => self.handle_mem_get_info(hdr, payload, session_handle),
            CUBE_GPU_API_GET_DEVICE => self.handle_get_device(hdr, payload, session_handle),
            CUBE_GPU_API_SET_DEVICE => self.handle_set_device(hdr, payload, session_handle),
            CUBE_GPU_API_DEVICE_COUNT => self.handle_device_count(hdr, session_handle),
            CUBE_GPU_API_GET_DEVICE_NAME => self.handle_get_device_name(hdr, payload, session_handle),
            CUBE_GPU_API_DEVICE_TOTAL_MEM => {
                self.handle_device_total_mem(hdr, payload, session_handle)
            }
            CUBE_GPU_API_FUNC_GET_ATTR => self.handle_func_get_attr(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_GET_DEVICE => self.handle_ctx_get_device(hdr, session_handle),
            CUBE_GPU_API_GET_LAST_ERROR => self.handle_get_last_error(hdr, session_handle),
            CUBE_GPU_API_CTX_CREATE => self.handle_ctx_create(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_DESTROY => self.handle_ctx_destroy(hdr, payload, session_handle),
            CUBE_GPU_API_CTX_SET_CURRENT => {
                self.handle_ctx_set_current(hdr, payload, session_handle)
            }
            CUBE_GPU_API_CTX_GET_CURRENT => self.handle_ctx_get_current(hdr, session_handle),
            CUBE_GPU_API_PRIMARY_CTX_RETAIN => {
                self.handle_primary_retain(hdr, payload, session_handle)
            }
            CUBE_GPU_API_PRIMARY_CTX_RELEASE => {
                self.handle_primary_release(hdr, payload, session_handle)
            }
            CUBE_GPU_API_IPC_GET_HANDLE
            | CUBE_GPU_API_IPC_OPEN_HANDLE
            | CUBE_GPU_API_IPC_CLOSE_HANDLE => {
                self.status_response(hdr, CUDA_ERROR_NOT_SUPPORTED)
            }
            _ => self.status_response(hdr, CUDA_ERROR_NOT_SUPPORTED),
        }
    }

    fn require_handle(&self, handle: &Option<u64>) -> Option<u64> {
        *handle
    }

    // ── Session lifecycle ───────────────────────────────────────────────

    fn handle_session_init(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &mut Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        // cube_gpu_session_init_req_t = 64 bytes.
        if payload.len() < 64 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        if session_handle.is_some() {
            warn!("session_init on an already-initialized connection");
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }

        let memory_quota = rd_u64(payload, 0);
        let sandbox_id = rd_u64(payload, 8);
        let sm_percent = rd_u64(payload, 16);
        let shm_size_mb = rd_u32(payload, 24);
        let _flags = rd_u32(payload, 28);
        let mut token = [0u8; CUBE_GPU_SHM_TOKEN_LEN];
        token.copy_from_slice(&payload[32..64]);

        let shm_size = shm_size_mb as u64 * 1024 * 1024;
        if shm_size == 0 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }

        // Hold the session table across the duplicate check and insertion so
        // two concurrent connections cannot initialize the same sandbox.
        let mut sessions = self.sessions.lock();
        if sessions.values().any(|s| s.sandbox_id == sandbox_id) {
            warn!("duplicate session_init for sandbox {}", sandbox_id);
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }

        let shm = match self.shm_mgr.attach(sandbox_id, &token, shm_size) {
            Ok(ch) => ch,
            Err(e) => {
                error!("session_init attach failed for sandbox {}: {}", sandbox_id, e);
                return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
            }
        };
        let layout = shm.layout();

        let handle = self.alloc_handle();
        let device_count = self.cuda.lock().device_count() as u32;

        let session = Session {
            handle,
            sandbox_id,
            memory_quota,
            sm_percent,
            shm: Some(shm),
            shm_layout: Some(layout),
            ..Default::default()
        };
        sessions.insert(handle, session);
        drop(sessions);
        *session_handle = Some(handle);

        info!(
            "session_init: handle={} sandbox={} quota={} sm={}% shm={} ring={} g2h={} h2g={}",
            handle, sandbox_id, memory_quota, sm_percent, layout.0, layout.1, layout.2, layout.3
        );

        // cube_gpu_session_init_resp_t = 56 bytes.
        let mut resp = Vec::with_capacity(56);
        push_u32(&mut resp, CUDA_SUCCESS);
        push_u32(&mut resp, device_count);
        push_u64(&mut resp, handle);
        push_u64(&mut resp, layout.0);
        push_u64(&mut resp, layout.1);
        push_u64(&mut resp, layout.2);
        push_u64(&mut resp, layout.3);
        push_u32(&mut resp, MAX_PAYLOAD as u32);
        push_u32(&mut resp, 0);
        self.ok_response(hdr, resp)
    }

    fn handle_session_destroy(
        &self,
        hdr: &RpcHeader,
        session_handle: &mut Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let handle = match *session_handle {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        self.destroy_session(handle);
        *session_handle = None;
        self.status_response(hdr, CUDA_SUCCESS)
    }

    /// Release every resource owned by `handle` and unmap its SHM channel.
    /// Shared by explicit destroy and vsock disconnect.
    /// Cleanup failures caused by the client already tearing down its own
    /// context/modules are benign: the resource is gone, not leaked.
    fn cleanup_result_ok(rc: u32) -> bool {
        matches!(
            rc,
            CUDA_SUCCESS | CUDA_ERROR_INVALID_HANDLE | CUDA_ERROR_INVALID_CONTEXT
        )
    }

    pub fn destroy_session(&self, handle: u64) {
        let mut session = match self.sessions.lock().remove(&handle) {
            Some(s) => s,
            None => return,
        };
        let cuda = self.cuda.lock();
        let mut failures = 0usize;

        // 1. allocations
        for (ptr, _size) in session.allocations.drain() {
            let rc = cuda.mem_free(ptr);
            if !Self::cleanup_result_ok(rc) {
                failures += 1;
                error!("session {}: cuMemFree(0x{:x}) failed: {}", handle, ptr, rc);
            }
        }
        // 2. events
        for (event, _flags) in session.events.drain() {
            let rc = cuda.event_destroy(event);
            if !Self::cleanup_result_ok(rc) {
                failures += 1;
                error!("session {}: cuEventDestroy(0x{:x}) failed: {}", handle, event, rc);
            }
        }
        // 3. streams
        for stream in session.streams.drain() {
            let rc = cuda.stream_destroy(stream);
            if !Self::cleanup_result_ok(rc) {
                failures += 1;
                error!("session {}: cuStreamDestroy(0x{:x}) failed: {}", handle, stream, rc);
            }
        }
        // 4. modules
        for (module, _) in session.modules.drain() {
            let rc = cuda.module_unload(module);
            if !Self::cleanup_result_ok(rc) {
                failures += 1;
                error!("session {}: cuModuleUnload(0x{:x}) failed: {}", handle, module, rc);
            }
        }
        // 5. contexts (primary contexts are released through the driver's
        //    primary-context API rather than cuCtxDestroy).
        let primary: HashMap<u64, i32> = session
            .primary_ctxs
            .iter()
            .map(|(dev, ctx)| (*ctx, *dev))
            .collect();
        let contexts: Vec<u64> = session.contexts.drain().collect();
        for ctx in contexts {
            let rc = if let Some(device) = primary.get(&ctx) {
                cuda.primary_ctx_release(*device)
            } else {
                cuda.ctx_destroy(ctx)
            };
            if !Self::cleanup_result_ok(rc) {
                failures += 1;
                error!("session {}: context 0x{:x} release failed: {}", handle, ctx, rc);
            }
        }
        for (device, ctx) in session.primary_ctxs.drain() {
            if !primary.contains_key(&ctx) {
                let rc = cuda.primary_ctx_release(device);
                if !Self::cleanup_result_ok(rc) {
                    failures += 1;
                    error!("session {}: primary ctx release({}) failed: {}", handle, device, rc);
                }
            }
        }
        drop(cuda);

        let used = session.memory_used;
        let layout = session.shm_layout;
        let session_handle = session.handle;
        // Dropping the session unmaps the SHM channel.
        drop(session);

        if failures == 0 {
            info!(
                "session {} destroyed ({} bytes accounted, shm layout {:?})",
                session_handle, used, layout
            );
        } else {
            // No Cubelet control plane yet: record quarantine intent in logs.
            error!(
                "session {} destroyed with {} cleanup failure(s); QUARANTINE requested (no Cubelet hook available)",
                session_handle, failures
            );
        }
    }

    // ── Memory ──────────────────────────────────────────────────────────

    fn handle_malloc(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let size = rd_u64(payload, 0);
        if size == 0 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.can_allocate(size) {
            return self.status_response(hdr, CUDA_ERROR_OUT_OF_MEMORY);
        }

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.mem_alloc(size) {
            Ok(ptr) => {
                // Only account after the driver accepted the allocation.
                session.memory_used += size;
                session.allocations.insert(ptr, size);
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, ptr);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_free(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let vptr = rd_u64(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        // Ownership: only free pointers this session allocated.
        let size = match session.allocations.get(&vptr) {
            Some(size) => *size,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.mem_free(vptr);
        if rc == CUDA_SUCCESS {
            session.allocations.remove(&vptr);
            session.memory_used = session.memory_used.saturating_sub(size);
        }
        self.status_response(hdr, rc)
    }

    fn handle_malloc_async(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 16 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let size = rd_u64(payload, 0);
        let stream = rd_u64(payload, 8);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        if size == 0 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        if !session.can_allocate(size) {
            return self.status_response(hdr, CUDA_ERROR_OUT_OF_MEMORY);
        }

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.mem_alloc_async(size, stream) {
            Ok(ptr) => {
                session.memory_used += size;
                session.allocations.insert(ptr, size);
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, ptr);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_free_async(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 16 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let vptr = rd_u64(payload, 0);
        let stream = rd_u64(payload, 8);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let size = match session.allocations.get(&vptr) {
            Some(size) => *size,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.mem_free_async(vptr, stream);
        if rc == CUDA_SUCCESS {
            session.allocations.remove(&vptr);
            session.memory_used = session.memory_used.saturating_sub(size);
        }
        self.status_response(hdr, rc)
    }

    /// Synchronous and asynchronous memcpy share this handler.  The async
    /// variant deliberately executes a *synchronous* driver copy on the host:
    /// the source/destination staging buffer lives in the ring, and it must not
    /// be released before the driver has consumed it.  Guest-visible async
    /// semantics are preserved only in ordering, not in overlap.
    fn handle_memcpy(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
        _is_async: bool,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let is_async = hdr.api_id == CUBE_GPU_API_MEMCPY_ASYNC;
        let min_len = if is_async { 44 } else { 36 };
        if payload.len() < min_len {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let kind = rd_u32(payload, 0);
        let dst = rd_u64(payload, 4);
        let src = rd_u64(payload, 12);
        let size = rd_u64(payload, 20);
        let _shm_offset = rd_u64(payload, 28);
        let stream = if is_async { rd_u64(payload, 36) } else { 0 };
        let has_shm = (hdr.flags & CUBE_GPU_FLAG_SHM_DATA) != 0;

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }

        let rc = match kind {
            // HOST_TO_DEVICE
            1 => {
                if !has_shm {
                    CUDA_ERROR_INVALID_VALUE
                } else if size == 0 {
                    CUDA_SUCCESS
                } else {
                    let len = match usize::try_from(size) {
                        Ok(v) => v,
                        Err(_) => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
                    };
                    match session.shm.as_ref().and_then(|ch| ch.read_g2h(len)) {
                        Some(buf) => cuda.memcpy_h2d(dst, &buf),
                        None => CUDA_ERROR_INVALID_VALUE,
                    }
                }
            }
            // DEVICE_TO_HOST
            2 => {
                if !has_shm {
                    CUDA_ERROR_INVALID_VALUE
                } else if size == 0 {
                    CUDA_SUCCESS
                } else {
                    let len = match usize::try_from(size) {
                        Ok(v) => v,
                        Err(_) => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
                    };
                    let mut buf = vec![0u8; len];
                    let rc = cuda.memcpy_d2h(&mut buf, src);
                    if rc == CUDA_SUCCESS {
                        match session.shm.as_ref() {
                            Some(ch) if ch.write_h2g(&buf) => CUDA_SUCCESS,
                            _ => CUDA_ERROR_INVALID_VALUE,
                        }
                    } else {
                        rc
                    }
                }
            }
            // DEVICE_TO_DEVICE
            3 => cuda.memcpy_d2d(dst, src, size),
            _ => CUDA_ERROR_NOT_SUPPORTED,
        };
        self.status_response(hdr, rc)
    }

    fn handle_memset(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 32 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let dst = rd_u64(payload, 0);
        let size = rd_u64(payload, 8);
        let value = rd_u32(payload, 16);
        let element_bits = rd_u32(payload, 20);
        let stream = rd_u64(payload, 24);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }

        // The wire field is a byte count; memset APIs take element counts.
        let elements = match element_bits {
            8 => size,
            16 => {
                if size % 2 != 0 {
                    return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
                }
                size / 2
            }
            32 => {
                if size % 4 != 0 {
                    return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
                }
                size / 4
            }
            _ => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.memset(dst, value, element_bits, elements);
        self.status_response(hdr, rc)
    }

    // ── Modules / kernels ───────────────────────────────────────────────

    fn handle_module_load(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 16 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let image_size = rd_u64(payload, 0);
        if image_size == 0 || image_size > MAX_MODULE_IMAGE {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let len = match usize::try_from(image_size) {
            Ok(v) => v,
            Err(_) => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let image = match session.shm.as_ref().and_then(|ch| ch.read_g2h(len)) {
            Some(v) => v,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.module_load_data(&image) {
            Ok(module) => {
                session.modules.insert(module, module);
                if let Some(meta) = crate::cubin_params::parse_kernel_param_sizes(&image) {
                    session.module_meta.insert(module, meta);
                }
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, module);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_module_get_func(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 264 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let module = rd_u64(payload, 0);
        let name_bytes = &payload[8..264];
        let name_end = name_bytes.iter().position(|b| *b == 0).unwrap_or(name_bytes.len());
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        // Ownership: the module must belong to this session.
        if !session.modules.contains_key(&module) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.module_get_function(module, &name) {
            Ok(func) => {
                let sizes = session
                    .module_meta
                    .get(&module)
                    .and_then(|m| m.get(&name))
                    .cloned();
                let (nparams, has_meta) = match &sizes {
                    Some(v) => (v.len().min(MAX_KERNEL_PARAMS) as u32, 1u32),
                    None => (0, 0),
                };
                session.functions.insert(func, (name, module));

                // cube_gpu_module_get_func_resp_t = 88 bytes.
                let mut resp = Vec::with_capacity(88);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u32(&mut resp, nparams);
                push_u32(&mut resp, has_meta);
                push_u32(&mut resp, 0);
                push_u64(&mut resp, func);
                for i in 0..MAX_KERNEL_PARAMS {
                    let v = sizes
                        .as_ref()
                        .and_then(|s| s.get(i).copied())
                        .unwrap_or(0);
                    resp.extend_from_slice(&v.to_le_bytes());
                }
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_launch_kernel(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        // cube_gpu_launch_kernel_req_t = 76 bytes.
        if payload.len() < 76 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let func = rd_u64(payload, 0);
        let grid = (rd_u32(payload, 8), rd_u32(payload, 12), rd_u32(payload, 16));
        let block = (rd_u32(payload, 20), rd_u32(payload, 24), rd_u32(payload, 28));
        let shared_mem = rd_u32(payload, 32);
        let stream = rd_u64(payload, 36);
        let params_size = rd_u64(payload, 52);
        let extra_size = rd_u64(payload, 68);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        // Ownership: function must belong to this session *and* its module
        // must still be registered.
        let (name, module) = match session.functions.get(&func) {
            Some(v) => v.clone(),
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE),
        };
        if !session.modules.contains_key(&module) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }

        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }

        // The guest serializes buffers in `params` then `extra` order.  Read
        // both (if present) so ring cursors stay consistent even when only one
        // of them is used for the actual launch.
        let read_blob = |size: u64| -> Result<Option<Vec<u8>>, ()> {
            if size == 0 {
                return Ok(None);
            }
            let len = usize::try_from(size).map_err(|_| ())?;
            session
                .shm
                .as_ref()
                .and_then(|ch| ch.read_g2h(len))
                .map(Some)
                .ok_or(())
        };
        let params_blob = match read_blob(params_size) {
            Ok(v) => v,
            Err(()) => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let extra_blob = match read_blob(extra_size) {
            Ok(v) => v,
            Err(()) => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let rc = if let Some(blob) = extra_blob {
            let mut blob_size = blob.len() as u64;
            let extra: [*const c_void; 5] = [
                CU_LAUNCH_PARAM_BUFFER_POINTER as *const c_void,
                blob.as_ptr() as *const c_void,
                CU_LAUNCH_PARAM_BUFFER_SIZE as *const c_void,
                &mut blob_size as *mut u64 as *const c_void,
                CU_LAUNCH_PARAM_END as *const c_void,
            ];
            cuda.launch_kernel(
                func,
                grid,
                block,
                shared_mem,
                stream,
                std::ptr::null(),
                extra.as_ptr(),
            )
        } else if let Some(blob) = params_blob {
            let sizes = match session
                .module_meta
                .get(&module)
                .and_then(|m| m.get(&name))
            {
                Some(v) => v,
                // No cubin metadata: cannot reconstruct the argument array.
                None => return self.status_response(hdr, CUDA_ERROR_NOT_SUPPORTED),
            };
            if sizes.is_empty() || sizes.len() > MAX_KERNEL_PARAMS {
                return self.status_response(hdr, CUDA_ERROR_NOT_SUPPORTED);
            }
            let mut ptrs: Vec<*const c_void> = Vec::with_capacity(sizes.len());
            let mut offset = 0usize;
            for size in sizes {
                let size = *size as usize;
                if offset + size > blob.len() {
                    return self.status_response(hdr, CUDA_ERROR_NOT_SUPPORTED);
                }
                ptrs.push(unsafe { blob.as_ptr().add(offset) as *const c_void });
                offset += size;
            }
            cuda.launch_kernel(
                func,
                grid,
                block,
                shared_mem,
                stream,
                ptrs.as_ptr(),
                std::ptr::null(),
            )
        } else {
            cuda.launch_kernel(
                func,
                grid,
                block,
                shared_mem,
                stream,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        self.status_response(hdr, rc)
    }

    // ── Streams ─────────────────────────────────────────────────────────

    fn handle_stream_create(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let flags = rd_u32(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.stream_create(flags) {
            Ok(stream) => {
                session.streams.insert(stream);
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, stream);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_stream_destroy(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let stream = rd_u64(payload, 0);
        if stream == 0 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.streams.contains(&stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.stream_destroy(stream);
        if rc == CUDA_SUCCESS {
            session.streams.remove(&stream);
        }
        self.status_response(hdr, rc)
    }

    fn handle_stream_sync(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let stream = rd_u64(payload, 0);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.stream_synchronize(stream);
        self.status_response(hdr, rc)
    }

    fn handle_device_sync(
        &self,
        hdr: &RpcHeader,
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.device_synchronize();
        self.status_response(hdr, rc)
    }

    // ── Events ──────────────────────────────────────────────────────────

    fn handle_event_create(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let flags = rd_u32(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.event_create(flags) {
            Ok(event) => {
                session.events.insert(event, flags as u64);
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, event);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_event_record(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 16 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let event = rd_u64(payload, 0);
        let stream = rd_u64(payload, 8);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.events.contains_key(&event) || !session.owns_stream(stream) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.event_record(event, stream);
        self.status_response(hdr, rc)
    }

    fn handle_event_sync(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let event = rd_u64(payload, 0);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.events.contains_key(&event) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.event_synchronize(event);
        self.status_response(hdr, rc)
    }

    fn handle_event_elapsed(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 16 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let start = rd_u64(payload, 0);
        let end = rd_u64(payload, 8);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.events.contains_key(&start) || !session.events.contains_key(&end) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.event_elapsed_time(start, end) {
            Ok(ms) => {
                let mut resp = Vec::with_capacity(8);
                push_u32(&mut resp, CUDA_SUCCESS);
                resp.extend_from_slice(&ms.to_le_bytes());
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_event_destroy(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let event = rd_u64(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.events.contains_key(&event) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        let rc = cuda.event_destroy(event);
        if rc == CUDA_SUCCESS {
            session.events.remove(&event);
        }
        self.status_response(hdr, rc)
    }

    // ── Queries ─────────────────────────────────────────────────────────

    fn handle_get_dev_prop(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let _h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let device = rd_i32(payload, 0);
        let attribute = rd_i32(payload, 4);

        let cuda = self.cuda.lock();
        match cuda.device_get_attribute(attribute, device) {
            Ok(value) => {
                let mut resp = Vec::with_capacity(8);
                push_u32(&mut resp, CUDA_SUCCESS);
                resp.extend_from_slice(&value.to_le_bytes());
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_mem_get_info(
        &self,
        hdr: &RpcHeader,
        _payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let free = session.memory_quota.saturating_sub(session.memory_used);
        let mut resp = Vec::with_capacity(20);
        push_u32(&mut resp, CUDA_SUCCESS);
        push_u64(&mut resp, free);
        push_u64(&mut resp, session.memory_quota);
        self.ok_response(hdr, resp)
    }

    fn handle_get_device(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let _h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let ordinal = rd_i32(payload, 0);
        let cuda = self.cuda.lock();
        if ordinal < 0 || ordinal >= cuda.device_count() {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        match cuda.device_get(ordinal) {
            Ok(device) => {
                let mut resp = Vec::with_capacity(8);
                push_u32(&mut resp, CUDA_SUCCESS);
                resp.extend_from_slice(&device.to_le_bytes());
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_set_device(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let device = rd_i32(payload, 0);
        let device_count = self.cuda.lock().device_count();
        if device < 0 || device >= device_count {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let mut sessions = self.sessions.lock();
        if let Some(session) = sessions.get_mut(&h) {
            session.current_device = device;
        }
        self.status_response(hdr, CUDA_SUCCESS)
    }

    fn handle_device_count(
        &self,
        hdr: &RpcHeader,
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let _h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let count = self.cuda.lock().device_count();
        let mut resp = Vec::with_capacity(8);
        push_u32(&mut resp, CUDA_SUCCESS);
        resp.extend_from_slice(&count.to_le_bytes());
        self.ok_response(hdr, resp)
    }

    fn handle_get_device_name(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let _h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let device = rd_i32(payload, 0);
        let cuda = self.cuda.lock();
        match cuda.device_get_name(device) {
            Ok(name) => {
                let mut resp = Vec::with_capacity(260);
                push_u32(&mut resp, CUDA_SUCCESS);
                let bytes = name.as_bytes();
                let n = bytes.len().min(255);
                resp.extend_from_slice(&bytes[..n]);
                resp.resize(4 + 256, 0);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_device_total_mem(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let device = rd_i32(payload, 0);

        // Quota-adjusted view: never advertise more than the session quota.
        let quota = {
            let sessions = self.sessions.lock();
            match sessions.get(&h) {
                Some(s) => s.memory_quota,
                None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
            }
        };
        let cuda = self.cuda.lock();
        match cuda.device_total_mem(device) {
            Ok(physical) => {
                let total = physical.min(quota);
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, total);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_func_get_attr(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 12 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let func = rd_u64(payload, 0);
        let attribute = rd_i32(payload, 8);

        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if !session.functions.contains_key(&func) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_HANDLE);
        }
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.func_get_attribute(func, attribute) {
            Ok(value) => {
                let mut resp = Vec::with_capacity(8);
                push_u32(&mut resp, CUDA_SUCCESS);
                resp.extend_from_slice(&value.to_le_bytes());
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_ctx_get_device(
        &self,
        hdr: &RpcHeader,
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let cuda = self.cuda.lock();
        if session.context != 0 {
            cuda.ctx_set_current(session.context);
        }
        match cuda.ctx_get_device() {
            Ok(device) => {
                let mut resp = Vec::with_capacity(8);
                push_u32(&mut resp, CUDA_SUCCESS);
                resp.extend_from_slice(&device.to_le_bytes());
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_get_last_error(
        &self,
        hdr: &RpcHeader,
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let sessions = self.sessions.lock();
        let status = sessions.get(&h).map(|s| s.last_error).unwrap_or(0);
        self.status_response(hdr, status)
    }

    // ── Contexts ────────────────────────────────────────────────────────

    fn handle_ctx_create(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let flags = rd_u32(payload, 0);
        let device = rd_i32(payload, 4);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };

        let cuda = self.cuda.lock();
        if device < 0 || device >= cuda.device_count() {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }

        let mut created = None;
        if session.sm_percent > 0 {
            let mp_count = cuda
                .device_get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
                .unwrap_or(0);
            if mp_count > 0 {
                let sm = ((session.sm_percent.saturating_mul(mp_count as u64)) / 100)
                    .clamp(1, mp_count as u64) as u32;
                match cuda.ctx_create_v3(flags, device, sm) {
                    Ok(ctx) => created = Some(ctx),
                    Err(rc) => warn!(
                        "cuCtxCreate_v3(sm_count={}) failed: {}; falling back to v2 (is MPS enabled?)",
                        sm, rc
                    ),
                }
            }
        }
        let ctx = match created {
            Some(ctx) => ctx,
            None => match cuda.ctx_create(flags, device) {
                Ok(ctx) => ctx,
                Err(rc) => return self.status_response(hdr, rc),
            },
        };

        session.contexts.insert(ctx);
        session.context = ctx;
        session.current_device = device;

        let mut resp = Vec::with_capacity(12);
        push_u32(&mut resp, CUDA_SUCCESS);
        push_u64(&mut resp, ctx);
        self.ok_response(hdr, resp)
    }

    fn handle_ctx_destroy(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let ctx = rd_u64(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        // Ownership: destroying another session's context is rejected.
        if !session.contexts.contains(&ctx) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_CONTEXT);
        }

        let cuda = self.cuda.lock();
        let primary_device = session
            .primary_ctxs
            .iter()
            .find(|(_, c)| **c == ctx)
            .map(|(dev, _)| *dev);
        let rc = match primary_device {
            Some(device) => {
                session.primary_ctxs.remove(&device);
                cuda.primary_ctx_release(device)
            }
            None => cuda.ctx_destroy(ctx),
        };
        if rc == CUDA_SUCCESS {
            session.contexts.remove(&ctx);
            if session.context == ctx {
                session.context = 0;
            }
        }
        self.status_response(hdr, rc)
    }

    fn handle_ctx_set_current(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 8 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let ctx = rd_u64(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        // NULL is allowed (unbinds); any other context must be ours.
        if ctx != 0 && !session.contexts.contains(&ctx) {
            return self.status_response(hdr, CUDA_ERROR_INVALID_CONTEXT);
        }
        let cuda = self.cuda.lock();
        let rc = cuda.ctx_set_current(ctx);
        if rc == CUDA_SUCCESS {
            session.context = ctx;
        }
        self.status_response(hdr, rc)
    }

    fn handle_ctx_get_current(
        &self,
        hdr: &RpcHeader,
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let sessions = self.sessions.lock();
        let session = match sessions.get(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let mut resp = Vec::with_capacity(12);
        push_u32(&mut resp, CUDA_SUCCESS);
        push_u64(&mut resp, session.context);
        self.ok_response(hdr, resp)
    }

    fn handle_primary_retain(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let device = rd_i32(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if let Some(ctx) = session.primary_ctxs.get(&device) {
            let mut resp = Vec::with_capacity(12);
            push_u32(&mut resp, CUDA_SUCCESS);
            push_u64(&mut resp, *ctx);
            return self.ok_response(hdr, resp);
        }

        let cuda = self.cuda.lock();
        if device < 0 || device >= cuda.device_count() {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        match cuda.primary_ctx_retain(device) {
            Ok(ctx) => {
                session.primary_ctxs.insert(device, ctx);
                session.contexts.insert(ctx);
                let mut resp = Vec::with_capacity(12);
                push_u32(&mut resp, CUDA_SUCCESS);
                push_u64(&mut resp, ctx);
                self.ok_response(hdr, resp)
            }
            Err(rc) => self.status_response(hdr, rc),
        }
    }

    fn handle_primary_release(
        &self,
        hdr: &RpcHeader,
        payload: &[u8],
        session_handle: &Option<u64>,
    ) -> (RpcHeader, Vec<u8>) {
        let h = match self.require_handle(session_handle) {
            Some(h) => h,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        if payload.len() < 4 {
            return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE);
        }
        let device = rd_i32(payload, 0);

        let mut sessions = self.sessions.lock();
        let session = match sessions.get_mut(&h) {
            Some(s) => s,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        let ctx = match session.primary_ctxs.remove(&device) {
            Some(ctx) => ctx,
            None => return self.status_response(hdr, CUDA_ERROR_INVALID_VALUE),
        };
        session.contexts.remove(&ctx);
        if session.context == ctx {
            session.context = 0;
        }
        let cuda = self.cuda.lock();
        let rc = cuda.primary_ctx_release(device);
        self.status_response(hdr, rc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shm_manager::ShmManager;

    fn test_hdr(api_id: u32, flags: u32) -> RpcHeader {
        RpcHeader {
            magic: CUBE_GPU_RPC_MAGIC,
            version: CUBE_GPU_RPC_VERSION,
            api_id,
            flags,
            request_id: 1,
            session_id: 1,
            payload_len: 0,
            status: 0,
        }
    }

    fn status_of(resp: &(RpcHeader, Vec<u8>)) -> u32 {
        let (_, payload) = resp;
        if payload.len() >= 4 {
            u32::from_le_bytes(payload[0..4].try_into().unwrap())
        } else {
            resp.0.status
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "cube-gpu-session-test-{}-{}-{}",
            tag,
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn new_manager(dir: &std::path::Path) -> Option<Arc<SessionManager>> {
        let cuda = match CudaRuntime::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping session test: CUDA unavailable: {}", e);
                return None;
            }
        };
        Some(SessionManager::new(cuda, ShmManager::new(dir.to_str().unwrap())))
    }

    fn make_session(handle: u64, sandbox_id: u64) -> Session {
        Session {
            handle,
            sandbox_id,
            memory_quota: 8 * 1024 * 1024,
            memory_used: 64,
            allocations: HashMap::from([(0xA000u64, 64u64)]),
            modules: HashMap::from([(0x3000u64, 0x3000u64)]),
            functions: HashMap::new(),
            contexts: HashSet::from([0x1000u64]),
            context: 0x1000,
            ..Default::default()
        }
    }

    #[test]
    fn ownership_and_quota_are_enforced_across_sessions() {
        let dir = temp_dir("own");
        let mgr = match new_manager(&dir) {
            Some(m) => m,
            None => {
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };

        mgr.sessions.lock().insert(1, make_session(1, 111));
        let mut other = make_session(2, 222);
        other.allocations.clear();
        other.modules.clear();
        other.contexts.clear();
        other.context = 0x1000;
        mgr.sessions.lock().insert(2, other);

        // Cross-session ctx_set_current -> 201.
        let mut h = Some(2u64);
        let mut payload = Vec::new();
        push_u64(&mut payload, 0x1000);
        let resp = mgr.dispatch(
            &test_hdr(CUBE_GPU_API_CTX_SET_CURRENT, 0),
            &payload,
            &mut h,
        );
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_CONTEXT);
        assert_eq!(h, Some(2));

        // Freeing another session's allocation -> 1.
        let mut payload = Vec::new();
        push_u64(&mut payload, 0xA000);
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_FREE, 0), &payload, &mut h);
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_VALUE);
        // The owning session still holds it.
        assert!(mgr.sessions.lock().get(&1).unwrap().allocations.contains_key(&0xA000));

        // module_get_func with another session's module -> 400.
        let mut payload = vec![0u8; 264];
        payload[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
        payload[8..15].copy_from_slice(b"kernel\0");
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_MODULE_GET_FUNC, 0), &payload, &mut h);
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_HANDLE);

        // Cross-session ctx_destroy -> 201.
        let mut payload = Vec::new();
        push_u64(&mut payload, 0x1000);
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_CTX_DESTROY, 0), &payload, &mut h);
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_CONTEXT);

        // Quota: 16 MiB request against an 8 MiB quota -> OOM.
        let mut payload = Vec::new();
        push_u64(&mut payload, 16 * 1024 * 1024);
        let mut h1 = Some(1u64);
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_MALLOC, 0), &payload, &mut h1);
        assert_eq!(status_of(&resp), CUDA_ERROR_OUT_OF_MEMORY);

        // The quota also applies to the async allocation path (default pool).
        let mut payload = Vec::new();
        push_u64(&mut payload, 16 * 1024 * 1024);
        push_u64(&mut payload, 0); // default stream
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_MALLOC_ASYNC, 0), &payload, &mut h1);
        assert_eq!(status_of(&resp), CUDA_ERROR_OUT_OF_MEMORY);
        assert_eq!(mgr.sessions.lock().get(&1).unwrap().memory_used, 64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memcpy_payload_and_ring_bounds_are_checked() {
        let dir = temp_dir("memcpy");
        let mgr = match new_manager(&dir) {
            Some(m) => m,
            None => {
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };

        // Build a real SHM mapping for session 5.
        let shm_path = dir.join("sandbox-5.shm");
        let file = std::fs::File::create(&shm_path).unwrap();
        file.set_len(2 * 1024 * 1024).unwrap();
        drop(file);
        let ch = mgr
            .shm_mgr
            .attach(5, &[0u8; 32], 2 * 1024 * 1024)
            .expect("attach shm");
        let layout = ch.layout();
        let mut s = make_session(5, 555);
        s.context = 0;
        s.contexts.clear();
        s.shm = Some(ch);
        s.shm_layout = Some(layout);
        mgr.sessions.lock().insert(5, s);
        let mut h = Some(5u64);

        // Truncated memcpy payload -> INVALID_VALUE.
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_MEMCPY, 0), &[0u8; 35], &mut h);
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_VALUE);

        // H2D with SHM_DATA but more bytes than the ring can hold.
        let mut payload = Vec::with_capacity(36);
        push_u32(&mut payload, 1); // H2D
        push_u64(&mut payload, 0xB000); // dst
        push_u64(&mut payload, 0); // src
        push_u64(&mut payload, layout.1 + 4096); // size > capacity
        push_u64(&mut payload, 0); // shm_offset
        let resp = mgr.dispatch(
            &test_hdr(CUBE_GPU_API_MEMCPY, CUBE_GPU_FLAG_SHM_DATA),
            &payload,
            &mut h,
        );
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_VALUE);

        // Non-SHM H2D has no data source -> INVALID_VALUE.
        let mut payload = Vec::with_capacity(36);
        push_u32(&mut payload, 1);
        push_u64(&mut payload, 0xB000);
        push_u64(&mut payload, 0);
        push_u64(&mut payload, 16);
        push_u64(&mut payload, 0);
        let resp = mgr.dispatch(&test_hdr(CUBE_GPU_API_MEMCPY, 0), &payload, &mut h);
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_VALUE);

        // Async variant must validate its stream handle.
        let mut payload = Vec::with_capacity(44);
        push_u32(&mut payload, 1);
        push_u64(&mut payload, 0xB000);
        push_u64(&mut payload, 0);
        push_u64(&mut payload, 16);
        push_u64(&mut payload, 0);
        push_u64(&mut payload, 0xDEAD); // unknown stream
        let resp = mgr.dispatch(
            &test_hdr(CUBE_GPU_API_MEMCPY_ASYNC, CUBE_GPU_FLAG_SHM_DATA),
            &payload,
            &mut h,
        );
        assert_eq!(status_of(&resp), CUDA_ERROR_INVALID_HANDLE);

        drop(mgr);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
