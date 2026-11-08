# CubeSandbox GPU end-to-end tests

Test assets for the `libcube-gpu.so` (guest) + `cube-gpu-daemon` (host) CUDA
remoting path. Everything here runs on the host over the vsock loopback
(`vsock_loopback` kernel module, no MicroVM required).

## Files

| File | Purpose |
|---|---|
| `kernel.cu` | Three kernels exercising kernel-parameter shapes: `add_one` (ptr+int), `mixed` (int/double/ptr/ptr, metadata `[4,8,8,8]`), `bigarg` (32-byte by-value struct, `>8B` path). |
| `kernel.cubin`, `kernel.ptx` | Pre-built fixtures (checked in). `make -C gpu/tests kernels` regenerates them with `nvcc -arch=sm_86`; override with `CUDA_ARCH=sm_90` etc. |
| `e2e_main.c` | Full API sequence: context/memory/quota view, 1 MiB SHM round-trip, module load + `add_one`/`mixed`/`bigarg`, `cuMemsetD32`, events, `cuMemAllocAsync`. Resolves everything via `dlopen("libcuda.so.1")` + `dlsym(RTLD_DEFAULT, ...)` so the guest library's `dlsym` hook is exercised. `--quota-oom` mode asserts a quota+16 MiB allocation fails with `CUDA_ERROR_OUT_OF_MEMORY`. |
| `e2e_link.c` | Minimal `-lcuda` binary (`cuInit` + `cuDeviceGetCount`); proves `LD_PRELOAD` interposes direct-linked CUDA calls (`dladdr` verifies the provider is `libcube-gpu.so`). |
| `session_hold.c` | Long-lived session for the multi-session test; waits for a trigger file, then re-runs an alloc/round-trip to prove the survivor still works. |
| `vsock_probe.c` | One-shot vsock connect probe used to wait for the daemon without fixed sleeps. |
| `../scripts/e2e-test.sh` | Single-session orchestration (build, daemon, three test runs, log checks). |
| `../scripts/e2e-multi-session.sh` | Two parallel full suites + SIGKILL one of two live sessions. |

## Build

```sh
make -C gpu/tests kernels     # kernel.cubin / kernel.ptx (committed)
make -C gpu/tests programs    # e2e_main, e2e_link, vsock_probe, session_hold
```

`nvcc` is taken from `$PATH` (default `/usr/local/cuda/bin`); the checked-in
artifacts were built for `sm_86` (RTX 3060). Re-run `make kernels` if the
target GPU differs.

## Run

```sh
# host-loopback end-to-end
gpu/scripts/e2e-test.sh
gpu/scripts/e2e-test.sh --require-bigarg     # make >8B param failure fatal

# multi-session isolation + kill -9
gpu/scripts/e2e-multi-session.sh
```

Both scripts build the guest library (`make -B -C gpu/guest/src` — forced
because the guest Makefile tracks no header dependencies, so a `.so` built
against an older `rpc_protocol.h` would otherwise be reused and speak a stale
wire version), the daemon
(`cargo build --manifest-path gpu/host/Cargo.toml`) and the test programs,
then create sparse 128 MiB SHM files under `/tmp/cube-gpu-e2e*` (the daemon
initializes the global header in test mode; the token is all-zero).

Overrides: `E2E_WORK_DIR`, `E2E_VSOCK_PORT` (20565), `E2E_VSOCK_CID` (1 =
`VMADDR_CID_LOCAL`), `E2E_QUOTA` (512 MiB), `E2E_SANDBOX_ID` (1001); for the
multi-session script use `E2E_MULTI_WORK_DIR`, `E2E_MULTI_VSOCK_PORT` (20566),
`E2E_MULTI_QUOTA` (256 MiB).

## Manual invocation

```sh
CUBE_GPU_VSOCK_CID=1 \
CUBE_GPU_VSOCK_PORT=20565 \
CUBE_GPU_SANDBOX_ID=1001 \
CUBE_GPU_MEMORY_QUOTA=536870912 \
CUBE_GPU_SHM_PATH=/tmp/cube-gpu-e2e/shm/sandbox-1001.shm \
LD_PRELOAD=$PWD/gpu/guest/src/libcube-gpu.so \
    gpu/tests/e2e_main --cubin gpu/tests/kernel.cubin
```

Useful checks when debugging:

- `E2E_REQUIRE_BIGARG=1` turns a `CUDA_ERROR_NOT_SUPPORTED` (801) from the
  `bigarg` kernel into a failure; otherwise it is reported as `SKIP`.
- `E2E_REQUIRE_LINK_HOOK=0` relaxes the `e2e_link` provider assertion.
- Daemon logs: `/tmp/cube-gpu-e2e/logs/daemon.log` (session init/destroy
  records come from the daemon's connection handler).

## Notes

- The set of `cu*_v2` aliases differs between CUDA versions and from what
  `libcube-gpu.so` exports/handles.  When the guest library is preloaded,
  `e2e_main` resolves each entry point to whichever spelling actually lives in
  `libcube-gpu.so` (checked with `dladdr`); otherwise identical-looking calls
  would fall through to `libcuda.so.1` with virtual handles and crash.  When no
  guest library is present it picks the driver's working spelling (CUDA 13.3
  dropped the `cuCtxSynchronize` -> `_v2` alias), so the same binary doubles as
  a sanity check of the test itself.  This also means a missing `_v2` export on
  the guest side (e.g. `cuEventElapsedTime_v2`) is not exercised as a hard
  failure; it is reported to the guest library instead.
- `--quota-oom` expects the quota to be enforced. Against the real driver it
  deliberately fails, because a 528 MiB allocation succeeds on a 12 GiB GPU;
  that is the behaviour the hook must prevent.
