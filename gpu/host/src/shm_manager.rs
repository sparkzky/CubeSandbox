// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use tracing::{info, warn};

pub struct ShmManager {
    base_dir: PathBuf,
}

impl ShmManager {
    pub fn new(base_dir: &str) -> Self {
        let base_dir = PathBuf::from(base_dir);
        std::fs::create_dir_all(&base_dir).ok();
        Self { base_dir }
    }

    pub fn create_shm_file(&self, session_handle: u64, size: u64) -> Option<PathBuf> {
        let path = self.base_dir.join(format!("session-{}.shm", session_handle));
        match std::fs::File::create(&path) {
            Ok(file) => {
                if file.set_len(size).is_ok() {
                    info!("Created SHM file: {:?} size={}", path, size);
                    Some(path)
                } else {
                    warn!("Failed to set SHM file size");
                    None
                }
            }
            Err(e) => {
                warn!("Failed to create SHM file: {:?}", e);
                None
            }
        }
    }

    pub fn mmap_shm_file(&self, path: &Path, size: u64) -> Option<*mut u8> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path).ok()?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            warn!("mmap failed");
            None
        } else {
            Some(ptr as *mut u8)
        }
    }

    pub fn cleanup_session(&self, session_handle: u64) {
        let path = self.base_dir.join(format!("session-{}.shm", session_handle));
        std::fs::remove_file(&path).ok();
    }
}

impl Drop for ShmManager {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base_dir);
    }
}
