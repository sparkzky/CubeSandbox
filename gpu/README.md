# CubeSandbox GPU Sharing

API-level CUDA remoting for **single-machine** CubeSandbox: CUDA calls inside a
sandbox MicroVM are intercepted via `LD_PRELOAD` and forwarded over vsock to a
host daemon that executes them on the **local** GPU (`VMADDR_CID_HOST`). Large
transfers go through IVSHMEM shared memory. Design details: `docs/architecture/gpu-sharing.md`.

## Layout

| Path | Role |
|---|---|
| `proto/rpc_protocol.h` | Shared C protocol header (structs, API IDs, flags) |
| `proto/rpc_protocol.rs.inc` | Rust protocol header, `include!`-ed by `vsock_listener.rs` |
| `guest/src/` | `libcube-gpu.so` — guest-side interception library |
| `host/` | `cube-gpu-daemon` — host-side Rust daemon |

## Build

### Guest library

```sh
cd gpu/guest/src
make            # -> libcube-gpu.so  (pulls rpc_protocol.h via -I../../proto)
```

Requires `gcc` + glibc headers. Install the `.so` into the guest image at
`/usr/lib/cube-gpu/libcube-gpu.so` — that path is what the shim injects as
`LD_PRELOAD` (see `CubeShim/shim/src/sandbox/gpu.rs`).

### Host daemon

```sh
cd gpu/host
cargo build --release
```

This is a **standalone crate** (the repo has no root workspace), so build it on
its own. Linux only — see *Known limitations*.

## Runtime prerequisites (cannot be satisfied by code)

These must be provisioned on the GPU machine before the daemon can serve:

- **NVIDIA driver + `libcuda.so.1`** — loaded at runtime via `libloading`; if
  missing the daemon exits with code 1 (`gpu/host/src/main.rs`).
- **Root** — required to bind the vsock listener and mmap IVSHMEM backing files.
- **MPS (Multi-Process Service)** — consumer GPUs lack MIG, so MPS is used for
  concurrent multi-tenant kernel execution. Start `nvidia-cuda-mps-control`.
- **IVSHMEM backing dir** — default `/var/run/cube-gpu/shm`, writable by the
  daemon. One `.shm` file per sandbox, created/removed with the session.

### Daemon environment variables

| Var | Default | Note |
|---|---|---|
| `CUBE_GPU_VSOCK_PORT` | `0x5055` | vsock port to listen on |
| `CUBE_GPU_SHM_DIR` | `/var/run/cube-gpu/shm` | IVSHMEM backing-file directory |

The tracing filter is **hardcoded** to `cube_gpu_daemon=debug` in `main.rs` and
does **not** read `RUST_LOG`. Change the source to use `EnvFilter::from_default_env()`
if runtime log control is needed.

## Per-sandbox enablement

Set annotation `cube.gpu` on the sandbox spec:

```json
{"enable": true, "memory_quota_mb": 4096, "vsock_port": 0}
```

The shim then injects `LD_PRELOAD` + `CUBE_GPU_*` env vars into the container
process and wires an IVSHMEM device into the MicroVM config
(`CubeShim/shim/src/sandbox/{config,gpu,sb}.rs`, `container/mod.rs`).

## Known limitations

- **macOS cannot build the host or shim crates.** The dependency stack
  (`kvm-ioctls`, `io-uring`, `vmm_sys_util`, `net_gen`, `qcow`, …) is Linux-only.
  Build and test on a Linux machine with KVM.
- **Protocol header is 40 bytes, not 32.** Early drafts of the architecture doc
  said 32; the packed `RpcHeader` is 40 B on both sides — the doc has been
  corrected, keep it that way if you touch `rpc_protocol.h`.
- **Software isolation only.** MPS enables concurrent execution without hardware
  partitioning; `memory_quota_mb` is enforced in software by the daemon. A
  runaway kernel can still monopolize compute units.
- **P0 API surface only.** Only the P0 APIs in `rpc_protocol.h` are dispatched by
  the host (`session.rs`). Query/property/context APIs (P1, ids 40+) and the
  context-management range (50+) are defined in the protocol but not handled.
- **Host daemon process management is manual.** There is no systemd unit yet
  (tracked as a follow-up in `docs/architecture/gpu-sharing.md`).
