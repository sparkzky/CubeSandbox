// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::sandbox::config::GpuConfig;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::sync::{Mutex, OnceLock};

const CUBE_GPU_VSOCK_PORT: u32 = 0x5055;
const CUBE_GPU_SHM_SIZE_MB: u64 = 128;

// ── Shared-memory global header layout (gpu/proto/rpc_protocol.h) ──────────
//   offset 0  : magic          u64  "CUBEGPU1"
//   offset 8  : version        u32
//   offset 12 : flags          u32  CUBE_GPU_SESSION_FLAG_TOKEN
//   offset 16 : token          [u8; 32]
//   offset 48 : total_size     u64
//   offset 56 : ring_size      u64  (written by the daemon)
//   offset 64 : g2h_offset     u64  (written by the daemon)
//   offset 72 : h2g_offset     u64  (written by the daemon)
//   offset 80 : session_id     u64
//   offset 88 : reserved       [u8; 40]
// The shim owns only the identity fields; the daemon is the sole authority
// for the ring layout and fills those fields in on attach.
const SHM_GLOBAL_HEADER_SIZE: usize = 128;
const SHM_MAGIC: u64 = 0x4355_4245_4750_5531; // "CUBEGPU1"
const SHM_VERSION: u32 = 1;
const SESSION_FLAG_TOKEN: u32 = 1;
const SHM_TOKEN_LEN: usize = 32;

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 8;
const OFF_FLAGS: usize = 12;
const OFF_TOKEN: usize = 16;
const OFF_TOTAL_SIZE: usize = 48;
const OFF_SESSION_ID: usize = 80;

/// Token registry shared by every `GpuIntegration` instance. `sb.rs` creates
/// the backing file through one instance and `container/mod.rs` later injects
/// the env through a freshly constructed one, so the token cannot live on the
/// struct itself.
// std::sync::LazyLock is unavailable on the shim's pinned toolchain (1.77.2).
fn lock_registry() -> std::sync::MutexGuard<'static, HashMap<u64, String>> {
    static TOKENS: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
    TOKENS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn generate_token() -> std::io::Result<[u8; SHM_TOKEN_LEN]> {
    let mut token = [0u8; SHM_TOKEN_LEN];
    let mut urandom = std::fs::File::open("/dev/urandom")?;
    urandom.read_exact(&mut token)?;
    Ok(token)
}

fn token_to_hex(token: &[u8; SHM_TOKEN_LEN]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(SHM_TOKEN_LEN * 2);
    for byte in token {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Read back the token from an existing backing file. Used when the registry
/// was lost (e.g. shim restart) so we do not have to truncate a file the
/// daemon may already be attached to.
fn read_token_from_file(path: &str) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut header = [0u8; SHM_GLOBAL_HEADER_SIZE];
    file.read_exact(&mut header).ok()?;
    let magic = u64::from_le_bytes(header[OFF_MAGIC..OFF_MAGIC + 8].try_into().ok()?);
    if magic != SHM_MAGIC {
        return None;
    }
    let token: [u8; SHM_TOKEN_LEN] = header[OFF_TOKEN..OFF_TOKEN + SHM_TOKEN_LEN]
        .try_into()
        .ok()?;
    if token.iter().all(|b| *b == 0) {
        return None;
    }
    Some(token_to_hex(&token))
}

pub struct GpuIntegration {
    pub enabled: bool,
    pub memory_quota_mb: u64,
    pub vsock_port: u32,
    pub shm_size_mb: u64,
    pub sm_percent: Option<u64>,
}

impl GpuIntegration {
    pub fn from_config(gpu: &Option<GpuConfig>) -> Option<Self> {
        gpu.as_ref().filter(|g| g.enable).map(|g| Self {
            enabled: true,
            memory_quota_mb: g.memory_quota_mb,
            vsock_port: if g.vsock_port > 0 {
                g.vsock_port
            } else {
                CUBE_GPU_VSOCK_PORT
            },
            shm_size_mb: CUBE_GPU_SHM_SIZE_MB,
            sm_percent: g.sm_percent,
        })
    }

    pub fn ld_preload_value(&self) -> String {
        "/usr/lib/cube-gpu/libcube-gpu.so".to_string()
    }

    pub fn env_vars(&self, sandbox_id: u64) -> Vec<(String, String)> {
        let mut env = vec![
            ("LD_PRELOAD".to_string(), self.ld_preload_value()),
            ("CUBE_GPU_SANDBOX_ID".to_string(), format!("{}", sandbox_id)),
            (
                "CUBE_GPU_MEMORY_QUOTA".to_string(),
                format!("{}", self.memory_quota_mb * 1024 * 1024),
            ),
            (
                "CUBE_GPU_VSOCK_PORT".to_string(),
                format!("{}", self.vsock_port),
            ),
            (
                "CUBE_GPU_SHM_PATH".to_string(),
                "/dev/shm/cube-gpu-shm".to_string(),
            ),
        ];
        if let Some(token) = self.token_for(sandbox_id) {
            env.push(("CUBE_GPU_SHM_TOKEN".to_string(), token));
        } else {
            eprintln!(
                "cube-gpu: failed to obtain shm token for sandbox {}",
                sandbox_id
            );
        }
        if let Some(percent) = self.sm_percent {
            env.push(("CUBE_GPU_SM_PERCENT".to_string(), format!("{}", percent)));
        }
        env
    }

    /// Token for `sandbox_id`, creating the backing file if we have no record
    /// of it yet.
    fn token_for(&self, sandbox_id: u64) -> Option<String> {
        if let Some(token) = lock_registry().get(&sandbox_id).cloned() {
            return Some(token);
        }
        if let Some(token) = read_token_from_file(&self.ivshmem_backing_file_path(sandbox_id)) {
            lock_registry().insert(sandbox_id, token.clone());
            return Some(token);
        }
        match self.create_backing_file(sandbox_id) {
            Ok((_, token)) => Some(token),
            Err(e) => {
                eprintln!(
                    "cube-gpu: failed to create backing file for sandbox {}: {}",
                    sandbox_id, e
                );
                None
            }
        }
    }

    pub fn ivshmem_size_bytes(&self) -> u64 {
        self.shm_size_mb * 1024 * 1024
    }

    pub fn ivshmem_backing_file_path(&self, sandbox_id: u64) -> String {
        format!("/var/run/cube-gpu/shm/sandbox-{}.shm", sandbox_id)
    }

    /// Create (or re-create) the IVSHMEM backing file and write a fresh
    /// per-sandbox token into its global header. Every call generates a new
    /// token, so a sandbox can never reuse a stale one.
    ///
    /// Returns `(path, token_hex)`; the token is also cached for `env_vars`.
    pub fn create_backing_file(&self, sandbox_id: u64) -> std::io::Result<(String, String)> {
        let path = self.ivshmem_backing_file_path(sandbox_id);
        let p = std::path::Path::new(&path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(p)?;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;

        let total_size = self.ivshmem_size_bytes();
        file.set_len(total_size)?;

        let token = generate_token()?;
        let mut header = [0u8; SHM_GLOBAL_HEADER_SIZE];
        header[OFF_MAGIC..OFF_MAGIC + 8].copy_from_slice(&SHM_MAGIC.to_le_bytes());
        header[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&SHM_VERSION.to_le_bytes());
        header[OFF_FLAGS..OFF_FLAGS + 4].copy_from_slice(&SESSION_FLAG_TOKEN.to_le_bytes());
        header[OFF_TOKEN..OFF_TOKEN + SHM_TOKEN_LEN].copy_from_slice(&token);
        header[OFF_TOTAL_SIZE..OFF_TOTAL_SIZE + 8].copy_from_slice(&total_size.to_le_bytes());
        // ring_size / g2h_offset / h2g_offset stay 0: the daemon owns layout.
        header[OFF_SESSION_ID..OFF_SESSION_ID + 8].copy_from_slice(&sandbox_id.to_le_bytes());

        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;

        let token_hex = token_to_hex(&token);
        lock_registry().insert(sandbox_id, token_hex.clone());
        Ok((path, token_hex))
    }

    /// Ensure the IVSHMEM backing file exists on disk with the correct size.
    ///
    /// The VMM opens the backing path without O_CREAT, so the file must be
    /// pre-created and sized before the VM is launched; otherwise boot fails
    /// with `MemoryManager(SharedFileCreate(ENOENT))`.
    pub fn ensure_backing_file(&self, sandbox_id: u64) -> std::io::Result<String> {
        self.create_backing_file(sandbox_id).map(|(path, _)| path)
    }

    /// Remove the IVSHMEM backing file created by `ensure_backing_file`.
    pub fn remove_backing_file(&self, sandbox_id: u64) {
        lock_registry().remove(&sandbox_id);
        let _ = std::fs::remove_file(self.ivshmem_backing_file_path(sandbox_id));
    }
}
