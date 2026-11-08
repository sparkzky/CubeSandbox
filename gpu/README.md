# CubeSandbox GPU Sharing (Backend B)

API-level CUDA remoting for CubeSandbox: CUDA Driver API calls inside a
sandbox MicroVM are intercepted by `LD_PRELOAD` (`libcube-gpu.so`) and
forwarded over vsock to a host daemon (`cube-gpu-daemon`) that executes them
on the local GPU. Bulk data (tensors, cubins, kernel parameter buffers) uses a
128 MiB IVSHMEM file that **both sides mmap**, with one SPSC ring per
direction. Design rationale and the roadmap: [`DESIGN_v2.md`](DESIGN_v2.md).

This is the **software-sharing backend** ("backend B"). It provides quotas and
misuse containment, not hardware isolation; prefer MIG or VFIO passthrough for
untrusted multi-tenant use (see *Security model*).

## Layout

| Path | Role |
|---|---|
| `proto/rpc_protocol.h` | Shared C protocol (v2): RPC header, SHM layout, API IDs, structs |
| `proto/rpc_protocol.rs.inc` | Rust mirror of the same constants/structs |
| `guest/src/` | `libcube-gpu.so` — guest interception library |
| `host/src/` | `cube-gpu-daemon` (Rust) + `cubin_params.rs` kernel-parameter metadata parser |
| `host/tests/fixtures/` | cubin/fatbin/PTX fixtures for the metadata parser |
| `tests/` | CUDA E2E fixtures and programs (kernel.cu/cubin/ptx, e2e_main, e2e_link, probe) |
| `scripts/` | `e2e-test.sh`, `e2e-multi-session.sh`, `start-mps.sh`, `install-daemon.sh` |
| `systemd/` | `cube-gpu-daemon.service` |

## Build

```sh
make -C gpu/guest/src                 # -> libcube-gpu.so
cd gpu/host && cargo build --release  # -> target/release/cube-gpu-daemon
make -C gpu/tests kernels programs    # E2E fixtures/binaries (needs nvcc)
```

Install the guest library at `/usr/lib/cube-gpu/libcube-gpu.so` (the path the
shim injects as `LD_PRELOAD`).

## Runtime model

1. The shim creates `/var/run/cube-gpu/shm/sandbox-{id}.shm` (128 MiB), writes a
   global header with a per-sandbox random token, and injects the token plus
   `CUBE_GPU_*` env vars into the sandbox.
2. VMM exposes the file to the guest as an IVSHMEM device; the guest library
   maps it (usually at `/dev/shm/cube-gpu-shm`) and opens a vsock session.
3. The daemon mmaps the same file, validates the token, and serves the session
   against a per-sandbox `CUcontext` (plus an MPS domain when MPS is enabled).

### SHM layout (v2)

```
[0,4096)                 global header (magic/version/token/total/ring/offsets)
[4096, 4096+128)         G2H ring header
[4224, ...)              G2H data  (capacity bytes)
[h2g_offset, +128)       H2G ring header
[...)                    H2G data  (capacity bytes)
capacity = (total - 2*4096 - 2*128) / 2
```

The daemon is the only writer of the layout fields (the shim writes
magic/version/token/total only). The guest attaches read-only w.r.t. the global
header; only ring cursors move. Ring cursors are monotonic and persist across
sessions in the file, so long-running hosts exercise wrap-around naturally.

### Guest environment

| Var | Meaning |
|---|---|
| `LD_PRELOAD` | `/usr/lib/cube-gpu/libcube-gpu.so` |
| `CUBE_GPU_SANDBOX_ID` | Sandbox id (selects the SHM file) |
| `CUBE_GPU_MEMORY_QUOTA` | GPU memory quota in bytes (authoritative in daemon) |
| `CUBE_GPU_VSOCK_PORT` | Daemon port (default `0x5055`) |
| `CUBE_GPU_SHM_PATH` | Guest mapping of the IVSHMEM file |
| `CUBE_GPU_SHM_TOKEN` | 64-hex-char token written by the shim |
| `CUBE_GPU_SM_PERCENT` | Optional SM quota percentage |
| `CUBE_GPU_VSOCK_CID` | Optional override (default 2); use `1` for host-loopback tests |

### Daemon environment

| Var | Default | Note |
|---|---|---|
| `CUBE_GPU_VSOCK_PORT` | `0x5055` (decimal 20565) | Listen port |
| `CUBE_GPU_SHM_DIR` | `/var/run/cube-gpu/shm` | Backing-file directory |

Logging uses a hardcoded `cube_gpu_daemon=debug` filter in `main.rs`.

## Per-sandbox enablement

```json
{"enable": true, "memory_quota_mb": 4096, "vsock_port": 0, "sm_percent": 50}
```

`sm_percent` is optional; the daemon attempts `cuCtxCreate_v3` SM affinity and
falls back to a normal context when the driver does not support it (or MPS is
not enabled). The shim wires the IVSHMEM device and env vars
(`CubeShim/shim/src/sandbox/{config,gpu,sb}.rs`, `container/mod.rs`).

## API coverage (v2)

* **Memory**: `cuMemAlloc/Free`, `cuMemAllocAsync/FreeAsync` (default pool),
  `cuMemcpyHtoD/DtoH/DtoD`, async variants (executed synchronously on the host;
  see limitations), `cuMemsetD8/D16/D32`.
* **Module/kernel**: `cuModuleLoadData(_Ex)`, `cuModuleGetFunction`,
  `cuLaunchKernel` (both `kernelParams` via cubin/fatbin/PTX metadata and
  `extra` buffer pointers), `cuFuncGetAttribute`.
* **Context/stream/event**: create/destroy/set/get/synchronize, primary-context
  retain/release (cudart compatibility), stream create/destroy/sync, event
  create/record/sync/elapsed/destroy.
* **Queries**: device count/get/name/attribute, `cuDeviceTotalMem` and
  `cuMemGetInfo` return the **quota view**.
* **IPC**: `cuIpcGetMemHandle/OpenMemHandle/CloseMemHandle` always return
  `CUDA_ERROR_NOT_SUPPORTED` — CUDA IPC is a cross-tenant leak surface and is
  disabled by design.

Unknown APIs resolve to the real driver only if it is visible; inside a
MicroVM there is no device node, so they fail cleanly.

## Testing

```sh
cd gpu/host && cargo test          # unit tests: SHM layout/token/wrap, ownership,
                                   # quota, cubin/fatbin/PTX param metadata
bash gpu/scripts/e2e-test.sh       # host-loopback E2E over vsock CID 1 with the
                                   # real GPU: cubin + PTX kernels, 32-byte struct
                                   # params, events, async alloc, quota OOM,
                                   # direct-link interception
bash gpu/scripts/e2e-multi-session.sh  # parallel sessions + SIGKILL reclaim
```

The E2E scripts create sparse 128 MiB files, start the daemon on a private
port, and drive `gpu/tests/e2e_main` with `LD_PRELOAD`. `vsock_loopback` must
be loaded (`sudo modprobe vsock_loopback`).

## Security model (honest boundaries)

* The MicroVM boundary is the real security boundary: the guest has no GPU
  device node, so bypassing the hook fails instead of escaping.
* The daemon enforces allocation ownership, quotas (including async paths),
  context/module/stream/event ownership, and blocks CUDA IPC.
* **Shared backend is not hardware isolation**: memory bandwidth, cache, and
  the driver fault domain are shared; a driver-level fault can affect other
  sessions on the same GPU. Use MIG or VFIO passthrough for untrusted tenants.
* MPS is supported as a deployment option (`scripts/start-mps.sh`); it improves
  concurrency but does not partition the fault domain.

## Real-sandbox validation (CubeSandbox deployment, 2026-11)

Backend B was validated end-to-end in **actual KVM sandboxes** on a one-click
deployment (RTX 3060, driver 615.71):

* Sandboxes were created through the E2B-compatible API with
  `metadata={"cube.gpu": '{"enable":true,"memory_quota_mb":512}'}`.
* Inside each sandbox, `e2e_main` ran over envd and passed **17/17** checks:
  quota-sized `cuDeviceTotalMem`/`cuMemGetInfo`, 1 MiB H2D/D2H through the
  IVSHMEM BAR, cubin module load, `add_one`/`mixed`/32-byte-struct kernels,
  `cuMemsetD32`, events, and async allocation.
* Three sandboxes ran concurrently with 512/256/256 MB quotas: each reported
  **its own** quota view, the 256 MB sandbox hit `CUDA_ERROR_OUT_OF_MEMORY` at
  its own limit, and killing one sandbox left the other two fully functional
  (they re-ran the full suite after the kill).

### Deployment changes that were required

1. **Cubelet storage**: `cubecow` needs a reflink-capable filesystem
   (`/data/cubelet/storage` must not be ext4). A loop-mounted XFS image with
   `-m reflink=1` was used.
2. **Annotation plumbing**: CubeAPI lifts `metadata["cube.gpu"]` into the
   sandbox annotations, and CubeMaster must forward the non-`cube.master`
   `cube.gpu` key to Cubelet (both patched in this branch).
3. **Shim/hypervisor** (branch changes):
   * restore path passes the per-sandbox `IvshmemConfig`;
   * device restore tolerates devices absent from the snapshot (IVSHMEM is
     attached after the template snapshot);
   * the IVSHMEM BAR2 KVM memory slot follows guest BAR reprogramming, and the
     guest library triggers `/sys/bus/pci/rescan` when the device is not yet
     enumerated;
   * guest library no longer needs `GLIBC_2.38` (`strtoull` redirect removed),
     so it loads on older guest userspaces such as Debian 12.
4. **Guest image** must contain `libcube-gpu.so`, the `libcuda.so.1` symlink,
   and the test binaries (see `gpu/tests/`). The shim only creates sandbox
   state; it does not install the library into the image.

### Resolved during validation

* **Zero-env bootstrap.** Restored containers do not inherit the
  shim-injected `CUBE_GPU_*` environment. The shim now writes `session_id`,
  the token, and `memory_quota` into the IVSHMEM global header, and the guest
  library falls back to those values when the env is absent. Validated with an
  empty environment: all sandboxes bootstrapped from the header and saw their
  own 512/256 MiB quotas.
* **Native per-sandbox vsock transport.** Guest→host virtio-vsock is proxied
  by the VMM to `/run/vc/vm/<sandbox_id>/cube.sock_<port>`. The daemon now
  watches `CUBE_GPU_VM_DIR` (default `/run/vc/vm`) and binds a listener for
  each sandbox on `CUBE_GPU_GUEST_PORT` (default 20565); no external bridge is
  required.
* Cleanup of resources already torn down by the client is tolerated
  (`INVALID_HANDLE`/`INVALID_CONTEXT`). A Cubelet quarantine hook is still
  absent for genuine cleanup failures.

### GPU-aware scheduling (P2, non-MIG)

* **Cubelet** discovers GPUs (`nvidia-smi`), owns the node-local reservation
  state machine (`Allocatable/Reserved/Attached/Releasing/Quarantined`),
  enforces `memory_quota_mb`, per-card session caps and
  `required_driver_min`/`models` compatibility, and reports inventory via
  node labels `cube.gpu.*` on every heartbeat.
* **CubeMaster** has a `gpu` scheduler filter: requests with
  `cube.gpu` (JSON or bare `true`) are placed only on nodes with enough free
  GPU memory, session slots, matching driver and models. Non-GPU sandboxes are
  unaffected.
* Validated on the real deployment: two 5120 MiB sandboxes fit a 12288 MiB
  card, the third is rejected, a 1536 MiB sandbox still fits, non-GPU
  sandboxes keep scheduling while the GPU is full, incompatible
  driver/model requests are rejected, and capacity is reusable after teardown.

## Known limitations

* Async memory operations (`cuMemcpy*Async`, `cuMemAllocAsync`) complete
  synchronously in the daemon because the staging ring buffer cannot outlive
  the call. Semantics are stronger than required, at a latency cost.
* `cuGetLastError` always reports success; per-call errors are returned
  directly.
* No Cubelet resource state machine or GPU-aware scheduling yet (`cube.gpu`
  annotation is forwarded, but inventory/placement is not implemented).
* No per-session daemon process isolation; all sessions share one daemon
  process and one MPS server (fault domain = GPU).
* Kernels whose module image has no discoverable parameter metadata (e.g.
  hand-written PTX without `.param`, stripped binaries) reject
  `kernelParams` launches with `CUDA_ERROR_NOT_SUPPORTED`; the `extra`
  buffer path and cubin/fatbin/PTX metadata cover the normal cases.
* Host/shim crates are Linux-only.
