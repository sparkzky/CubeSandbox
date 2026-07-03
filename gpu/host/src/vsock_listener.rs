// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "vsock")]
use std::sync::Arc;
#[cfg(feature = "vsock")]
use tokio::net::VsockListener;
#[cfg(feature = "vsock")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(feature = "vsock")]
use tracing::{info, error, warn};

#[cfg(feature = "vsock")]
use crate::session::SessionManager;

include!("../../proto/rpc_protocol.rs.inc");

const RPC_HEADER_SIZE: usize = std::mem::size_of::<RpcHeader>();
#[cfg(feature = "vsock")]
pub async fn run(port: u32, session_mgr: Arc<SessionManager>) -> Result<(), Box<dyn std::error::Error>> {
    let listener = VsockListener::bind(nix::sys::socket::VsockAddr::new(
        nix::sys::socket::VMADDR_CID_ANY,
        port,
    ))?;

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
    mut stream: tokio::net::VsockStream,
    cid: u32,
    session_mgr: Arc<SessionManager>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut session_handle: Option<u64> = None;

    loop {
        let mut hdr_buf = vec![0u8; RPC_HEADER_SIZE];
        match stream.read_exact(&mut hdr_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        let hdr = parse_header(&hdr_buf);
        if hdr.magic != CUBE_GPU_RPC_MAGIC {
            warn!("Invalid magic from CID {}: 0x{:08x}", cid, hdr.magic);
            break;
        }

        let mut payload = vec![0u8; hdr.payload_len as usize];
        if hdr.payload_len > 0 {
            stream.read_exact(&mut payload).await?;
        }

        let (resp_hdr, resp_payload) = session_mgr.dispatch(&hdr, &payload, &mut session_handle);

        let mut out = serialize_header(&resp_hdr);
        out.extend_from_slice(&resp_payload);
        stream.write_all(&out).await?;
    }

    if let Some(h) = session_handle {
        session_mgr.destroy_session(h);
    }

    info!("CID {} disconnected", cid);
    Ok(())
}

#[cfg(feature = "vsock")]
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

#[cfg(feature = "vsock")]
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
