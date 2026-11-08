// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

//! IVSHMEM shared-memory channel for GPU session data (protocol v2).
//!
//! The backing file is created by the shim as
//! `{base_dir}/sandbox-{sandbox_id}.shm` and mapped by both guest and host.
//! Layout (identical on both sides):
//!
//! ```text
//!   [0, 4096)                          global header
//!   [4096, 4096+128+cap)               G2H ring header + data
//!   [4096+128+cap, ...)                H2G ring header + data
//! ```
//!
//! with `cap = (total - 2*4096 - 2*128) / 2`, `g2h_offset = 4096` and
//! `h2g_offset = 4096 + 128 + cap`.  The daemon is the only initializer of
//! the global header and of the ring headers; when the shim has pre-written a
//! valid magic but left the geometry zero (ring_size == 0), the geometry is
//! filled in here.

use std::path::PathBuf;
use std::sync::atomic::{fence, AtomicU64, Ordering};

use tracing::{info, warn};

use crate::vsock_listener::{
    CUBE_GPU_SHM_GLOBAL_SIZE, CUBE_GPU_SHM_MAGIC, CUBE_GPU_SHM_RING_HEADER_SIZE,
    CUBE_GPU_SHM_RING_MAGIC, CUBE_GPU_SHM_TOKEN_LEN, CUBE_GPU_SHM_VERSION, ShmGlobal,
    ShmRingHeader,
};

pub struct ShmManager {
    base_dir: PathBuf,
}

/// A mapped IVSHMEM channel owned by one session.  Unmapped on drop.
pub struct ShmChannel {
    base: *mut u8,
    total_size: u64,
    ring_size: u64,
    g2h_offset: u64,
    h2g_offset: u64,
}

// The mapping is shared-memory, accessed through explicit atomics/fences.
unsafe impl Send for ShmChannel {}

fn compute_layout(total_size: u64) -> Option<(u64, u64, u64, u64)> {
    const G: u64 = CUBE_GPU_SHM_GLOBAL_SIZE;
    const H: u64 = CUBE_GPU_SHM_RING_HEADER_SIZE;
    if total_size < 2 * G + 2 * H + 2 {
        return None;
    }
    let cap = (total_size - 2 * G - 2 * H) / 2;
    if cap == 0 {
        return None;
    }
    let g2h = G;
    let h2g = G + H + cap;
    Some((total_size, cap, g2h, h2g))
}

fn token_is_zero(token: &[u8; CUBE_GPU_SHM_TOKEN_LEN]) -> bool {
    token.iter().all(|b| *b == 0)
}

/// Constant-time equality over the whole token.
fn token_eq(a: &[u8; CUBE_GPU_SHM_TOKEN_LEN], b: &[u8; CUBE_GPU_SHM_TOKEN_LEN]) -> bool {
    let mut diff = 0u8;
    for i in 0..CUBE_GPU_SHM_TOKEN_LEN {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

impl ShmManager {
    pub fn new(base_dir: &str) -> Self {
        let base_dir = PathBuf::from(base_dir);
        std::fs::create_dir_all(&base_dir).ok();
        Self { base_dir }
    }

    pub fn path_for(&self, sandbox_id: u64) -> PathBuf {
        self.base_dir.join(format!("sandbox-{}.shm", sandbox_id))
    }

    /// Attach to the shim-created backing file, initializing the v2 headers if
    /// this is the first (daemon) attach.
    pub fn attach(
        &self,
        sandbox_id: u64,
        token: &[u8; CUBE_GPU_SHM_TOKEN_LEN],
        expected_size: u64,
    ) -> Result<ShmChannel, String> {
        use std::os::unix::io::AsRawFd;

        let path = self.path_for(sandbox_id);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| format!("open {}: {}", path.display(), e))?;

        let total_size = file
            .metadata()
            .map_err(|e| format!("fstat {}: {}", path.display(), e))?
            .len();
        if total_size != expected_size {
            return Err(format!(
                "shm size mismatch for {}: file={} expected={}",
                path.display(),
                total_size,
                expected_size
            ));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!(
                "mmap {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        let base = ptr as *mut u8;
        drop(file);

        let global = unsafe { std::ptr::read_unaligned(base as *const ShmGlobal) };
        let g_magic = global.magic;
        let g_version = global.version;
        let g_token = global.token;
        let g_total = global.total_size;
        let g_ring = global.ring_size;
        let g_g2h = global.g2h_offset;
        let g_h2g = global.h2g_offset;

        let needs_init = g_magic != CUBE_GPU_SHM_MAGIC;
        if needs_init {
            if let Err(e) = initialize(base, total_size, sandbox_id, token) {
                unsafe { libc::munmap(ptr, total_size as usize) };
                return Err(e);
            }
            let (_, cap, g2h, h2g) = match compute_layout(total_size) {
                Some(v) => v,
                None => {
                    unsafe { libc::munmap(ptr, total_size as usize) };
                    return Err("invalid shm size".to_string());
                }
            };
            info!(
                "initialized SHM v2 for sandbox {} at {} ({} bytes, ring={} h2g_off={})",
                sandbox_id,
                path.display(),
                total_size,
                cap,
                h2g
            );

            return Ok(ShmChannel {
                base,
                total_size,
                ring_size: cap,
                g2h_offset: g2h,
                h2g_offset: h2g,
            });
        }

        // Magic matches: validate version/token/size before trusting geometry.
        if g_version != CUBE_GPU_SHM_VERSION {
            unsafe { libc::munmap(ptr, total_size as usize) };
            return Err(format!(
                "shm version mismatch: file={} expected={}",
                g_version, CUBE_GPU_SHM_VERSION
            ));
        }
        if g_total != total_size {
            unsafe { libc::munmap(ptr, total_size as usize) };
            return Err(format!(
                "shm global total_size mismatch: file={} mapping={}",
                g_total, total_size
            ));
        }
        if !token_eq(&g_token, token) {
            unsafe { libc::munmap(ptr, total_size as usize) };
            return Err("shm token mismatch".to_string());
        }

        let (_, cap, g2h, h2g) = match compute_layout(total_size) {
            Some(v) => v,
            None => {
                unsafe { libc::munmap(ptr, total_size as usize) };
                return Err("invalid shm size".to_string());
            }
        };

        if g_ring == 0 {
            // Shim pre-created the file and wrote only identity fields.
            fill_geometry(base, total_size, cap, g2h, h2g, sandbox_id);
            info!(
                "filled SHM v2 geometry for sandbox {} at {} (ring={} h2g_off={})",
                sandbox_id,
                path.display(),
                cap,
                h2g
            );
            return Ok(ShmChannel {
                base,
                total_size,
                ring_size: cap,
                g2h_offset: g2h,
                h2g_offset: h2g,
            });
        }

        let layout_ok = g_ring == cap && g_g2h == g2h && g_h2g == h2g;
        if !layout_ok {
            unsafe { libc::munmap(ptr, total_size as usize) };
            return Err(format!(
                "shm layout invalid: ring={} g2h={} h2g={} (expected ring={} g2h={} h2g={})",
                g_ring, g_g2h, g_h2g, cap, g2h, h2g
            ));
        }

        for off in [g2h, h2g] {
            let hdr = unsafe { std::ptr::read_unaligned(base.add(off as usize) as *const ShmRingHeader) };
            let hdr_magic = hdr.magic;
            let hdr_capacity = hdr.capacity;
            if hdr_magic != CUBE_GPU_SHM_RING_MAGIC {
                unsafe { libc::munmap(ptr, total_size as usize) };
                return Err(format!("ring header at {} has bad magic", off));
            }
            if hdr_capacity != cap {
                unsafe { libc::munmap(ptr, total_size as usize) };
                return Err(format!(
                    "ring header at {} capacity {} != {}",
                    off, hdr_capacity, cap
                ));
            }
        }

        Ok(ShmChannel {
            base,
            total_size,
            ring_size: cap,
            g2h_offset: g2h,
            h2g_offset: h2g,
        })
    }
}

fn initialize(
    base: *mut u8,
    total_size: u64,
    sandbox_id: u64,
    token: &[u8; CUBE_GPU_SHM_TOKEN_LEN],
) -> Result<(), String> {
    let (_, cap, g2h, h2g) =
        compute_layout(total_size).ok_or_else(|| "invalid shm size".to_string())?;
    if token_is_zero(token) {
        warn!(
            "SHM for sandbox {} initialized with an all-zero token (test/development mode)",
            sandbox_id
        );
    }

    let global = ShmGlobal {
        magic: CUBE_GPU_SHM_MAGIC,
        version: CUBE_GPU_SHM_VERSION,
        flags: 0,
        token: *token,
        total_size,
        ring_size: cap,
        g2h_offset: g2h,
        h2g_offset: h2g,
        session_id: sandbox_id,
        reserved: [0u8; 40],
    };
    unsafe { std::ptr::write_unaligned(base as *mut ShmGlobal, global) };
    fence(Ordering::SeqCst);

    for off in [g2h, h2g] {
        let hdr = ShmRingHeader {
            write_pos: 0,
            read_pos: 0,
            capacity: cap,
            magic: CUBE_GPU_SHM_RING_MAGIC,
            flags: 0,
            epoch: 0,
            reserved: [0u32; 9],
        };
        unsafe {
            std::ptr::write_unaligned(base.add(off as usize) as *mut ShmRingHeader, hdr);
        }
    }
    fence(Ordering::SeqCst);
    Ok(())
}

/// Fill only the geometry fields the shim left zero, and initialize the rings.
fn fill_geometry(base: *mut u8, total_size: u64, cap: u64, g2h: u64, h2g: u64, sandbox_id: u64) {
    // Read-modify-write to preserve the shim-written magic/version/token.
    let mut global = unsafe { std::ptr::read_unaligned(base as *const ShmGlobal) };
    global.total_size = total_size;
    global.ring_size = cap;
    global.g2h_offset = g2h;
    global.h2g_offset = h2g;
    if global.session_id == 0 {
        global.session_id = sandbox_id;
    }
    unsafe { std::ptr::write_unaligned(base as *mut ShmGlobal, global) };

    for off in [g2h, h2g] {
        let hdr = ShmRingHeader {
            write_pos: 0,
            read_pos: 0,
            capacity: cap,
            magic: CUBE_GPU_SHM_RING_MAGIC,
            flags: 0,
            epoch: 0,
            reserved: [0u32; 9],
        };
        unsafe {
            std::ptr::write_unaligned(base.add(off as usize) as *mut ShmRingHeader, hdr);
        }
    }
    fence(Ordering::SeqCst);
}

fn atomic_load_u64(p: *mut u8) -> u64 {
    if (p as usize) % std::mem::align_of::<u64>() == 0 {
        unsafe { (*(p as *const AtomicU64)).load(Ordering::SeqCst) }
    } else {
        let v = unsafe { std::ptr::read_unaligned(p as *const u64) };
        fence(Ordering::SeqCst);
        v
    }
}

fn atomic_store_u64(p: *mut u8, value: u64) {
    if (p as usize) % std::mem::align_of::<u64>() == 0 {
        unsafe { (*(p as *const AtomicU64)).store(value, Ordering::SeqCst) };
    } else {
        fence(Ordering::SeqCst);
        unsafe { std::ptr::write_unaligned(p as *mut u64, value) };
    }
    fence(Ordering::SeqCst);
}

impl ShmChannel {
    /// `(total_size, ring_capacity, g2h_offset, h2g_offset)`.
    pub fn layout(&self) -> (u64, u64, u64, u64) {
        (
            self.total_size,
            self.ring_size,
            self.g2h_offset,
            self.h2g_offset,
        )
    }

    fn header_off(&self, g2h: bool) -> u64 {
        if g2h {
            self.g2h_offset
        } else {
            self.h2g_offset
        }
    }

    fn data_ptr(&self, g2h: bool) -> *mut u8 {
        unsafe {
            self.base
                .add((self.header_off(g2h) + CUBE_GPU_SHM_RING_HEADER_SIZE) as usize)
        }
    }

    fn pos_ptr(&self, g2h: bool, write: bool) -> *mut u8 {
        let off = self.header_off(g2h) + if write { 0 } else { 8 };
        unsafe { self.base.add(off as usize) }
    }

    fn avail_write(&self, g2h: bool) -> u64 {
        let w = atomic_load_u64(self.pos_ptr(g2h, true));
        let r = atomic_load_u64(self.pos_ptr(g2h, false));
        let cap = self.ring_size;
        if cap == 0 {
            0
        } else if w >= r {
            cap.saturating_sub(w - r).saturating_sub(1)
        } else {
            (r - w).saturating_sub(1)
        }
    }

    fn avail_read(&self, g2h: bool) -> u64 {
        let w = atomic_load_u64(self.pos_ptr(g2h, true));
        let r = atomic_load_u64(self.pos_ptr(g2h, false));
        let cap = self.ring_size;
        if cap == 0 {
            0
        } else if w >= r {
            w - r
        } else {
            cap - (r - w)
        }
    }

    fn ring_write(&self, g2h: bool, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        let cap = self.ring_size;
        if data.len() as u64 > self.avail_write(g2h) {
            return false;
        }
        let w = atomic_load_u64(self.pos_ptr(g2h, true));
        let start = (w % cap) as usize;
        let first = std::cmp::min(data.len(), cap as usize - start);
        unsafe {
            let dst = self.data_ptr(g2h);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst.add(start), first);
            if first < data.len() {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(first),
                    dst,
                    data.len() - first,
                );
            }
        }
        atomic_store_u64(self.pos_ptr(g2h, true), w + data.len() as u64);
        true
    }

    fn ring_read(&self, g2h: bool, len: usize) -> Option<Vec<u8>> {
        if len == 0 {
            return Some(Vec::new());
        }
        let cap = self.ring_size;
        if len as u64 > self.avail_read(g2h) {
            return None;
        }
        let r = atomic_load_u64(self.pos_ptr(g2h, false));
        let start = (r % cap) as usize;
        let first = std::cmp::min(len, cap as usize - start);
        let mut out = vec![0u8; len];
        unsafe {
            let src = self.data_ptr(g2h);
            std::ptr::copy_nonoverlapping(src.add(start), out.as_mut_ptr(), first);
            if first < len {
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(first), len - first);
            }
        }
        atomic_store_u64(self.pos_ptr(g2h, false), r + len as u64);
        Some(out)
    }

    /// Host consumes a guest→host message.  `None` means fewer than `len`
    /// bytes are available; the ring is left untouched in that case.
    pub fn read_g2h(&self, len: usize) -> Option<Vec<u8>> {
        self.ring_read(true, len)
    }

    /// Host enqueues a host→guest message.  Returns false if it does not fit.
    pub fn write_h2g(&self, data: &[u8]) -> bool {
        self.ring_write(false, data)
    }

    /// Reset both ring cursors (used on session teardown / test setup).
    #[allow(dead_code)]
    pub fn reset_rings(&self) {
        for g2h in [true, false] {
            atomic_store_u64(self.pos_ptr(g2h, true), 0);
            atomic_store_u64(self.pos_ptr(g2h, false), 0);
        }
    }

    #[cfg(test)]
    pub fn write_g2h(&self, data: &[u8]) -> bool {
        self.ring_write(true, data)
    }

    #[cfg(test)]
    pub fn read_h2g(&self, len: usize) -> Option<Vec<u8>> {
        self.ring_read(false, len)
    }
}

impl Drop for ShmChannel {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(
                self.base as *mut libc::c_void,
                self.total_size as usize,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempShm {
        dir: PathBuf,
        path: PathBuf,
    }

    impl TempShm {
        fn new(sandbox_id: u64, size: u64) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "cube-gpu-shm-test-{}-{}",
                std::process::id(),
                nonce
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("sandbox-{}.shm", sandbox_id));
            let file = std::fs::File::create(&path).unwrap();
            file.set_len(size).unwrap();
            Self { dir, path }
        }
    }

    impl Drop for TempShm {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    const TWO_MIB: u64 = 2 * 1024 * 1024;

    #[test]
    fn attach_initializes_and_validates() {
        let tmp = TempShm::new(7, TWO_MIB);
        let mgr = ShmManager::new(tmp.dir.to_str().unwrap());
        let token = [0xABu8; CUBE_GPU_SHM_TOKEN_LEN];

        let ch = mgr.attach(7, &token, TWO_MIB).expect("first attach");
        let (total, cap, g2h, h2g) = ch.layout();
        assert_eq!(total, TWO_MIB);
        assert_eq!(cap, (TWO_MIB - 2 * 4096 - 2 * 128) / 2);
        assert_eq!(g2h, 4096);
        assert_eq!(h2g, 4096 + 128 + cap);

        // Same token: validation path succeeds.
        let ch2 = mgr.attach(7, &token, TWO_MIB).expect("second attach");
        assert_eq!(ch2.layout(), ch.layout());

        // Wrong token is rejected.
        let bad = [0xCDu8; CUBE_GPU_SHM_TOKEN_LEN];
        assert!(mgr.attach(7, &bad, TWO_MIB).is_err());

        // Wrong expected size is rejected.
        assert!(mgr.attach(7, &token, TWO_MIB * 2).is_err());
    }

    #[test]
    fn shim_partial_geometry_is_filled() {
        let tmp = TempShm::new(9, TWO_MIB);
        let token = [0x42u8; CUBE_GPU_SHM_TOKEN_LEN];
        {
            use std::os::unix::io::AsRawFd;
            let file = std::fs::OpenOptions::new().read(true).write(true).open(&tmp.path).unwrap();
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    TWO_MIB as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(ptr, libc::MAP_FAILED);
            let g = ShmGlobal {
                magic: CUBE_GPU_SHM_MAGIC,
                version: CUBE_GPU_SHM_VERSION,
                flags: 0,
                token,
                total_size: TWO_MIB,
                ring_size: 0,
                g2h_offset: 0,
                h2g_offset: 0,
                session_id: 9,
                reserved: [0u8; 40],
            };
            unsafe { std::ptr::write_unaligned(ptr as *mut ShmGlobal, g) };
            unsafe { libc::munmap(ptr, TWO_MIB as usize) };
        }

        let mgr = ShmManager::new(tmp.dir.to_str().unwrap());
        let ch = mgr.attach(9, &token, TWO_MIB).expect("attach partial");
        let (_, cap, g2h, h2g) = ch.layout();
        assert_eq!(cap, (TWO_MIB - 2 * 4096 - 2 * 128) / 2);
        assert_eq!(h2g, 4096 + 128 + cap);
        assert!(g2h > 0);
    }

    #[test]
    fn ring_roundtrip_with_wraparound() {
        let tmp = TempShm::new(11, TWO_MIB);
        let mgr = ShmManager::new(tmp.dir.to_str().unwrap());
        let token = [1u8; CUBE_GPU_SHM_TOKEN_LEN];
        let ch = mgr.attach(11, &token, TWO_MIB).unwrap();
        let cap = ch.layout().1 as usize;

        // Each round is ~300 KiB; 8 rounds exceed the ~1 MiB capacity.
        let chunk_len = 300_000usize;
        for round in 0..8u8 {
            let data: Vec<u8> = (0..chunk_len).map(|i| (i as u8).wrapping_add(round)).collect();
            assert!(ch.write_g2h(&data), "round {} write", round);
            let got = ch.read_g2h(chunk_len).expect("read exact");
            assert_eq!(got, data, "round {} content", round);
        }
        assert!(cap < chunk_len * 8);

        // A short read must not consume anything.
        assert!(ch.write_g2h(&[7u8; 16]));
        assert!(ch.read_g2h(17).is_none());
        assert_eq!(ch.read_g2h(16).unwrap(), vec![7u8; 16]);

        // Cannot write more than capacity-1.
        let too_big = vec![0u8; cap + 1];
        assert!(!ch.write_g2h(&too_big));
    }

    #[test]
    fn directions_are_independent() {
        let tmp = TempShm::new(13, TWO_MIB);
        let mgr = ShmManager::new(tmp.dir.to_str().unwrap());
        let token = [2u8; CUBE_GPU_SHM_TOKEN_LEN];
        let ch = mgr.attach(13, &token, TWO_MIB).unwrap();

        assert!(ch.write_h2g(b"host-to-guest"));
        assert!(ch.read_g2h(1).is_none(), "g2h must stay empty");
        assert_eq!(ch.read_h2g(13).unwrap(), b"host-to-guest");
        assert!(ch.read_h2g(1).is_none());

        assert!(ch.write_g2h(b"guest-to-host"));
        assert_eq!(ch.read_g2h(13).unwrap(), b"guest-to-host");
        assert!(ch.read_g2h(1).is_none());

        ch.reset_rings();
        assert!(ch.read_g2h(1).is_none());
        assert!(ch.read_h2g(1).is_none());
    }
}
