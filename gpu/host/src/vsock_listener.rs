// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "vsock")]
use std::sync::Arc;
#[cfg(feature = "vsock")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(feature = "vsock")]
use tokio_vsock::{VsockAddr, VsockListener, VMADDR_CID_ANY};
#[cfg(feature = "vsock")]
use tracing::{error, info, warn};

#[cfg(feature = "vsock")]
use crate::session::SessionManager;

include!("../../proto/rpc_protocol.rs.inc");

const RPC_HEADER_SIZE: usize = std::mem::size_of::<RpcHeader>();

/// Hard cap on a single inline vsock payload (data itself travels via SHM).
pub const MAX_PAYLOAD: usize = 16 * 1024 * 1024;

#[cfg(feature = "vsock")]
pub async fn run(
    port: u32,
    session_mgr: Arc<SessionManager>,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = VsockListener::bind(VsockAddr::new(VMADDR_CID_ANY, port))?;

    loop {
        let (stream, addr) = listener.accept().await?;
        let cid = addr.cid();
        info!("New connection from CID {}", cid);

        let mgr = session_mgr.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, cid, mgr).await {
                error!("Connection handler error for CID {}: {:?}", cid, e);
            }
        });
    }
}

#[cfg(feature = "vsock")]
async fn handle_connection(
    mut stream: tokio_vsock::VsockStream,
    cid: u32,
    session_mgr: Arc<SessionManager>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut session_handle: Option<u64> = None;
    let mut failure: Option<Box<dyn std::error::Error + Send + Sync>> = None;

    loop {
        let mut hdr_buf = vec![0u8; RPC_HEADER_SIZE];
        match stream.read_exact(&mut hdr_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                failure = Some(e.into());
                break;
            }
        }

        let hdr = parse_header(&hdr_buf);
        let magic = hdr.magic;
        let version = hdr.version;
        let payload_len = hdr.payload_len;
        if magic != CUBE_GPU_RPC_MAGIC {
            warn!("Invalid magic from CID {}: 0x{:08x}", cid, magic);
            break;
        }
        if version != CUBE_GPU_RPC_VERSION {
            warn!(
                "Unsupported protocol version {} from CID {} (expected {})",
                version, cid, CUBE_GPU_RPC_VERSION
            );
            break;
        }
        if payload_len as usize > MAX_PAYLOAD {
            warn!(
                "Payload too large from CID {}: {} bytes (max {})",
                cid, payload_len, MAX_PAYLOAD
            );
            break;
        }

        let mut payload = vec![0u8; payload_len as usize];
        if payload_len > 0 {
            if let Err(e) = stream.read_exact(&mut payload).await {
                failure = Some(e.into());
                break;
            }
        }

        let (resp_hdr, resp_payload) = session_mgr.dispatch(&hdr, &payload, &mut session_handle);

        let mut out = serialize_header(&resp_hdr);
        out.extend_from_slice(&resp_payload);
        if let Err(e) = stream.write_all(&out).await {
            failure = Some(e.into());
            break;
        }
    }

    // Any disconnect path (EOF, protocol error, I/O error) must release the
    // session's host-side resources exactly once.
    if let Some(h) = session_handle {
        session_mgr.destroy_session(h);
    }

    info!("CID {} disconnected", cid);
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn parse_header(buf: &[u8]) -> RpcHeader {
    RpcHeader {
        magic: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
        version: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        api_id: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
        flags: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        request_id: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
        session_id: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        payload_len: u32::from_le_bytes(buf[32..36].try_into().unwrap()),
        status: u32::from_le_bytes(buf[36..40].try_into().unwrap()),
    }
}

fn serialize_header(hdr: &RpcHeader) -> Vec<u8> {
    let mut buf = Vec::with_capacity(RPC_HEADER_SIZE);
    buf.extend_from_slice(&hdr.magic.to_le_bytes());
    buf.extend_from_slice(&hdr.version.to_le_bytes());
    buf.extend_from_slice(&hdr.api_id.to_le_bytes());
    buf.extend_from_slice(&hdr.flags.to_le_bytes());
    buf.extend_from_slice(&hdr.request_id.to_le_bytes());
    buf.extend_from_slice(&hdr.session_id.to_le_bytes());
    buf.extend_from_slice(&hdr.payload_len.to_le_bytes());
    buf.extend_from_slice(&hdr.status.to_le_bytes());
    buf
}
