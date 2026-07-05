// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::sandbox::config::GpuConfig;

const CUBE_GPU_VSOCK_PORT: u32 = 0x5055;
const CUBE_GPU_SHM_SIZE_MB: u64 = 128;

pub struct GpuIntegration {
    pub enabled: bool,
    pub memory_quota_mb: u64,
    pub vsock_port: u32,
    pub shm_size_mb: u64,
}

impl GpuIntegration {
    pub fn from_config(gpu: &Option<GpuConfig>) -> Option<Self> {
        gpu.as_ref().filter(|g| g.enable).map(|g| Self {
            enabled: true,
            memory_quota_mb: g.memory_quota_mb,
            vsock_port: if g.vsock_port > 0 { g.vsock_port } else { CUBE_GPU_VSOCK_PORT },
            shm_size_mb: CUBE_GPU_SHM_SIZE_MB,
        })
    }

    pub fn ld_preload_value(&self) -> String {
        "/usr/lib/cube-gpu/libcube-gpu.so".to_string()
    }

    pub fn env_vars(&self, sandbox_id: u64) -> Vec<(String, String)> {
        vec![
            ("LD_PRELOAD".to_string(), self.ld_preload_value()),
            ("CUBE_GPU_SANDBOX_ID".to_string(), format!("{}", sandbox_id)),
            ("CUBE_GPU_MEMORY_QUOTA".to_string(), format!("{}", self.memory_quota_mb * 1024 * 1024)),
            ("CUBE_GPU_VSOCK_PORT".to_string(), format!("{}", self.vsock_port)),
            ("CUBE_GPU_SHM_PATH".to_string(), "/dev/shm/cube-gpu-shm".to_string()),
        ]
    }

    pub fn ivshmem_size_bytes(&self) -> u64 {
        self.shm_size_mb * 1024 * 1024
    }

    pub fn ivshmem_backing_file_path(&self, sandbox_id: u64) -> String {
        format!("/var/run/cube-gpu/shm/sandbox-{}.shm", sandbox_id)
    }

    /// Ensure the IVSHMEM backing file exists on disk with the correct size.
    ///
    /// The VMM opens the backing path without O_CREAT, so the file must be
    /// pre-created and sized before the VM is launched; otherwise boot fails
    /// with `MemoryManager(SharedFileCreate(ENOENT))`.
    pub fn ensure_backing_file(&self, sandbox_id: u64) -> std::io::Result<String> {
        let path = self.ivshmem_backing_file_path(sandbox_id);
        let p = std::path::Path::new(&path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(p)?;
        file.set_len(self.ivshmem_size_bytes())?;
        Ok(path)
    }

    /// Remove the IVSHMEM backing file created by `ensure_backing_file`.
    pub fn remove_backing_file(&self, sandbox_id: u64) {
        let _ = std::fs::remove_file(self.ivshmem_backing_file_path(sandbox_id));
    }
}
