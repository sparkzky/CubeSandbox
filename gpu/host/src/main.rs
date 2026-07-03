// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

mod cuda_runtime;
mod session;
mod shm_manager;
mod vsock_listener;

use tracing::{info, error};

const DEFAULT_VSOCK_PORT: u32 = 0x5055;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("cube_gpu_daemon=debug")
        .init();

    info!("cube-gpu-daemon starting");

    let cuda = match cuda_runtime::CudaRuntime::new() {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to initialize CUDA: {:?}", e);
            std::process::exit(1);
        }
    };
    info!("CUDA runtime initialized, device count: {}", cuda.device_count());

    let port = std::env::var("CUBE_GPU_VSOCK_PORT")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_VSOCK_PORT);

    let shm_dir = std::env::var("CUBE_GPU_SHM_DIR")
        .unwrap_or_else(|_| "/var/run/cube-gpu/shm".to_string());

    let shm_mgr = shm_manager::ShmManager::new(&shm_dir);

    let session_mgr = session::SessionManager::new(cuda, shm_mgr);

    #[cfg(feature = "vsock")]
    {
        info!("Listening on vsock port {}", port);
        if let Err(e) = vsock_listener::run(port, session_mgr).await {
            error!("Listener error: {:?}", e);
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "vsock"))]
    {
        let _ = session_mgr;
        let _ = port;
        info!("cube-gpu-daemon built without vsock transport; CUDA runtime initialized successfully");
    }
}
