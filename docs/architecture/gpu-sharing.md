# CubeSandbox GPU Sharing Architecture

## Overview

CubeSandbox runs each sandbox as an independent KVM MicroVM with its own Guest OS kernel. The GPU hardware resides on the host machine — the guest has no GPU device, no NVIDIA driver. This document describes how we enable CUDA workloads inside sandboxes through API-level remoting.

The core idea: intercept every CUDA call inside the guest via `LD_PRELOAD`, serialize it into an RPC message, forward it over vsock to a host-side daemon (`cube-gpu-daemon`), execute it on the real GPU, and return the result. For large data transfers (memory copies, kernel parameters), use IVSHMEM shared memory for zero-copy transmission.

## Architecture Diagram

```
┌─────────────────── Guest (KVM MicroVM) ───────────────────┐
│  AI Agent Process                                          │
│    │  cudaMalloc / cuLaunchKernel / ...                     │
│    ▼                                                        │
│  libcube-gpu.so  (LD_PRELOAD interception layer)           │
│    │  Serialize request → vsock RPC                         │
│    │  Large data → IVSHMEM shared memory                    │
│    ▼                                                        │
│  vsock / IVSHMEM endpoints                                 │
└─────────────────────────────────────────────────────────────┘
         │ vsock (control)        │ IVSHMEM (data)
         ▼                       ▼
┌─────────────────── Host (physical machine) ────────────────┐
│  cube-gpu-daemon                                           │
│    │  Receive RPC request                                   │
│    │  Look up sandbox's CUDA context                        │
│    │  Execute on real GPU                                   │
│    ▼                                                        │
│  NVIDIA Driver → GPU hardware                              │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Guest-side: `libcube-gpu.so`

**Files:** `gpu/guest/src/`

| File | Purpose |
|---|---|
| `hook.c` | CUDA Driver API hooks via LD_PRELOAD |
| `rpc_client.c` / `rpc_client.h` | vsock transport layer |
| `shm_ring.c` / `shm_ring.h` | IVSHMEM ring buffer |
| `cuda_defs.h` | CUDA type definitions, symbol versioning macros |
| `libcube_gpu_symbols.ld` | Linker version script for dlsym symbol export |
| `Makefile` | Build libcube-gpu.so |

#### 1.1 Interception Mechanism

CUDA applications obtain function pointers through two paths:

- **dlsym path**: At load time, the dynamic linker resolves symbols like `cuMemAlloc` via `dlsym(RTLD_NEXT, ...)`. This is the primary path before CUDA 11.3.
- **cuGetProcAddress path**: CUDA 11.3+ introduced a new driver API query mechanism where programs explicitly request function pointers via `cuGetProcAddress("cuMemAlloc", ...)`.

We hook both paths simultaneously using LD_PRELOAD:

```c
// Path 1: Hook dlsym itself
// glibc has two versions: GLIBC_2.2.5 and GLIBC_2.34 (the latter moved dlsym from libdl to libc)
// We use .symver assembler directives to export both versions

void *dlsym_225(void *handle, const char *symbol) {
    if (strcmp(symbol, "cuMemAlloc") == 0) return &cuMemAlloc;
    return real_dlsym_225(handle, symbol);
}
__asm__(".symver dlsym_225, dlsym@@GLIBC_2.2.5");

void *dlsym_234(void *handle, const char *symbol)
    __attribute__((alias("dlsym_225")));
__asm__(".symver dlsym_234, dlsym@GLIBC_2.34");
```

```c
// Path 2: Hook cuGetProcAddress
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags) {
    if (strcmp(symbol, "cuMemAlloc") == 0) {
        *pfn = (void *)&cuMemAlloc;
        return CUDA_SUCCESS;
    }
    *pfn = NULL;
    return CUDA_SUCCESS;
}
```

The `__attribute__((constructor))` function `get_real_dlsym` captures the real `dlsym` via `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")` before our hooks take effect. The dual-version approach handles both older glibc (where dlsym lives in libdl.so) and glibc 2.34+ (where it moved into libc.so).

#### 1.2 Virtual Address Space (Dual Address Mapping)

The guest has no real GPU, so `cuMemAlloc` returns a fake device pointer — a virtual pointer. The application will use this pointer for arithmetic, pass it to `cuMemcpyHtoD`, etc., so it must appear real.

Design:

```
Guest address space              Host real GPU address space
─────────────────────           ─────────────────────
vptr 0x10000001 ─────────────→  dptr 0x7f3a20000000 (real VRAM)
vptr 0x10000002 ─────────────→  dptr 0x7f3a21000000
vptr 0x20000001 (module) ────→  CUmodule 0x55a3b1c0
vptr 0x30000001 (function) ──→  CUfunction 0x55a3b2d0
vptr 0x40000001 (stream) ────→  CUstream 0x55a3b3e0
```

Different handle types use distinct prefix ranges:
- `0x1xxx`: Device memory (VRAM allocations)
- `0x2xxx`: Modules (loaded CUDA modules)
- `0x3xxx`: Functions (kernel functions extracted from modules)
- `0x4xxx`: Streams (CUDA streams)

Each hook function follows this pattern:

```
cuMemAlloc(1024) →
  1. Serialize request {size: 1024}
  2. Send via vsock RPC to Host
  3. Host returns {vptr: 0x7f3a20000000, status: CUDA_SUCCESS}
  4. Allocate a local virtual address 0x10000001
  5. Record mapping 0x10000001 → 0x7f3a20000000 in handle table
  6. Return 0x10000001 to the application

cuMemcpyHtoD(0x10000001, host_data, 1024) →
  1. Look up mapping: 0x10000001 → 0x7f3a20000000
  2. Write 1024 bytes of host_data into IVSHMEM shared memory
  3. Send RPC {dst: 0x7f3a20000000, size: 1024, flag: SHM_DATA}
  4. Host reads data from shared memory, calls real cuMemcpyHtoD
```

Handle tables use dynamic arrays with mutex protection. Memory pointers (`vptr_entry_t`) track size for quota management. Modules, functions, and streams use simpler `handle_entry_t` (vhandle + host_handle pairs).

#### 1.3 Supported APIs (P0)

| API | Description |
|---|---|
| `cuInit` | Initialize CUDA (no-op on guest) |
| `cuMemAlloc` | Allocate device memory |
| `cuMemFree` | Free device memory |
| `cuMemcpyHtoD` | Copy host → device |
| `cuMemcpyDtoH` | Copy device → host |
| `cuLaunchKernel` | Launch a CUDA kernel |
| `cuStreamCreate` | Create a CUDA stream |
| `cuStreamSynchronize` | Synchronize a stream |
| `cuCtxSynchronize` | Synchronize device (full barrier) |
| `cuModuleLoadData` | Load a CUDA module (PTX/cubin) |
| `cuModuleGetFunction` | Get a kernel function from a module |

### 2. Host-side: `cube-gpu-daemon`

**Files:** `gpu/host/src/`

| File | Purpose |
|---|---|
| `main.rs` | Daemon entry point, env config, starts vsock listener |
| `vsock_listener.rs` | Async vsock accept loop, header parsing, per-connection dispatch |
| `session.rs` | SessionManager with per-sandbox state, quota tracking, API dispatch |
| `cuda_runtime.rs` | CudaRuntime wrapper over libcuda.so.1 via libloading |
| `shm_manager.rs` | SHM file creation/mmap/cleanup per session |

#### 2.1 Request Flow

```
vsock_listener accepts connection
  │
  ├── Parse 40-byte RPC header
  ├── Extract api_id, session_id, payload_len, flags
  ├── Read payload (or note SHM_DATA flag)
  │
  ▼
SessionManager::dispatch(session_id, api_id, request)
  │
  ├── Look up or create Session for this sandbox_id
  │     Each Session has: CUDA context, memory_used counter, SHM mapping
  │
  ├── Route to handler based on api_id:
  │     MALLOC → check quota → cuMemAlloc → record allocation
  │     FREE → cuMemFree → update memory_used
  │     MEMCPY → if SHM_DATA, read from SHM → cuMemcpyHtoD/DtoH
  │     LAUNCH_KERNEL → reconstruct params → cuLaunchKernel
  │     STREAM_CREATE → cuStreamCreate → record handle
  │     ...
  │
  ▼
CudaRuntime (wraps libcuda.so.1 via libloading)
```

#### 2.2 Concurrency Isolation: MPS

Consumer-grade GPUs (e.g., RTX 4090) do not support MIG (Multi-Instance GPU) hardware partitioning. We use NVIDIA MPS (Multi-Process Service) for concurrent multi-tenant execution:

- Each sandbox gets its own CUDA context
- MPS allows kernels from multiple contexts to execute concurrently on the GPU (true parallelism, not time-slicing)
- Memory isolation is enforced in software: `SessionManager` tracks `memory_used` per session, and `cuMemAlloc` checks `memory_used + size <= memory_quota` before forwarding to the GPU

This trades hardware-level isolation for broader GPU compatibility. The software quota enforcement prevents one sandbox from consuming all GPU memory, but cannot prevent one sandbox's kernel from monopolizing compute units.

#### 2.3 CUDA Runtime Wrapper

`cuda_runtime.rs` uses Rust's `libloading` crate to dynamically load `libcuda.so.1`:

```rust
pub struct CudaRuntime {
    lib: Library,
    cu_init: Symbol<'static, unsafe extern fn(u32) -> CUresult>,
    cu_mem_alloc: Symbol<'static, unsafe extern fn(*mut CUdeviceptr, usize) -> CUresult>,
    // ...
}
```

This design means the daemon does not need to link against CUDA at compile time — if the machine has no GPU, loading simply fails gracefully instead of crashing.

### 3. RPC Protocol

**Files:** `gpu/proto/rpc_protocol.h` (C), `gpu/proto/rpc_protocol.rs.inc` (Rust)

#### 3.1 Wire Format

```
┌─────────────────────── RPC Header (40 bytes) ───────────────────────┐
│ magic (4B) │ version (4B) │ api_id (4B) │ flags (4B)               │
│ request_id (8B)          │ session_id (8B)                          │
│ payload_len (4B)         │ status (4B)                              │
├─────────────────────────────────────────────────────────────────────┤
│ Payload (variable length, up to payload_len bytes)                  │
└─────────────────────────────────────────────────────────────────────┘
```

| Field | Size | Description |
|---|---|---|
| `magic` | 4B | `0x43554441` ("CUDA") — protocol identification |
| `version` | 4B | Protocol version (currently 1) |
| `api_id` | 4B | Which CUDA API is being called |
| `flags` | 4B | Bit flags (e.g., `SHM_DATA = 0x01`) |
| `request_id` | 8B | Monotonically increasing, used for request-response matching |
| `session_id` | 8B | Identifies which sandbox this request belongs to |
| `payload_len` | 4B | Length of the payload following the header |
| `status` | 4B | Response status (CUDA error code) |

#### 3.2 API IDs and Request/Response Structs

| API ID | Request Struct | Response Struct |
|---|---|---|
| `CUBE_GPU_API_INIT` | `InitReq` | `InitResp` |
| `CUBE_GPU_API_MALLOC` | `MallocReq { size }` | `MallocResp { vptr, status }` |
| `CUBE_GPU_API_FREE` | `FreeReq { vptr }` | `FreeResp { status }` |
| `CUBE_GPU_API_MEMCPY` | `MemcpyReq { kind, dst, src, size, shm_offset }` | `MemcpyResp { status }` |
| `CUBE_GPU_API_LAUNCH_KERNEL` | `LaunchKernelReq { func_handle, grid/block dims, shared_mem, stream, params_shm_offset, params_size, ... }` | `LaunchKernelResp { status }` |
| `CUBE_GPU_API_STREAM_CREATE` | `StreamCreateReq { flags }` | `StreamCreateResp { stream_vptr, status }` |
| `CUBE_GPU_API_STREAM_SYNC` | `StreamSyncReq { stream_vptr }` | `StreamSyncResp { status }` |
| `CUBE_GPU_API_DEVICE_SYNC` | (no payload) | `DeviceSyncResp { status }` |
| `CUBE_GPU_API_MODULE_LOAD_DATA` | `ModuleLoadDataReq { image_size, shm_offset }` | `ModuleLoadDataResp { module_handle, status }` |
| `CUBE_GPU_API_MODULE_GET_FUNC` | `ModuleGetFuncReq { module_handle, func_name[256] }` | `ModuleGetFuncResp { func_handle, status }` |

#### 3.3 Why Custom Binary Protocol Instead of gRPC

vsock is a byte-stream transport without HTTP/2 framing. gRPC requires HTTP/2, which would need a bridging layer over vsock. For our scenario — a fixed set of CUDA APIs with simple request/response patterns — a custom binary protocol is lighter weight and lower latency. The 40-byte header parses with a single memcpy into a struct.

### 4. Shared Memory Channel (IVSHMEM)

#### 4.1 Why Shared Memory

RPC over vsock has ~microsecond-level latency, but for `cuMemcpyHtoD` (copying tens of MB from host memory to VRAM), sending data over vsock means two copies (Guest → vsock buffer → Host) plus protocol overhead. For large transfers, this becomes the bottleneck.

IVSHMEM gives the guest and host a shared region of physical memory. The guest writes directly into it; the host reads directly from it — zero-copy.

#### 4.2 Ring Buffer Design

The shared memory region is split into two halves:

```
┌──────────────────── Shared Memory (128MB) ────────────────────┐
│  G2H Ring (64MB)          │  H2G Ring (64MB)                   │
│  Guest writes, Host reads │  Host writes, Guest reads          │
│  ┌──────┐ ┌───────────┐  │  ┌──────┐ ┌───────────┐          │
│  │Header │ │   Data    │  │  │Header │ │   Data    │          │
│  │128B   │ │           │  │  │128B   │ │           │          │
│  └──────┘ └───────────┘  │  └──────┘ └───────────┘          │
└───────────────────────────────────────────────────────────────┘
```

Each ring has a 128-byte header:

```c
typedef struct {
    uint64_t write_pos;    // Updated by writer
    uint64_t read_pos;     // Updated by reader
    uint64_t data_size;    // Total data region size
    uint64_t _reserved;
} cube_gpu_shm_ring_header_t;
```

This is a lock-free single-producer single-consumer (SPSC) model. The guest only writes `write_pos`, the host only writes `read_pos`. Memory visibility is ensured via `__atomic_load` / `__atomic_store`. The ring buffer supports wrap-around — when writing reaches the end, it wraps back to the beginning.

#### 4.3 Data Threshold

Small transfers (< 4KB) go inline in the RPC payload. Large transfers (≥ 4KB) go through the SHM ring buffer, flagged with `CUBE_GPU_FLAG_SHM_DATA` in the RPC header.

#### 4.4 Data Flow Example: cuMemcpyHtoD

```
Guest:
  cuMemcpyHtoD(devPtr, hostData, 10MB)
  ├── Look up mapping: devPtr → real GPU address
  ├── 10MB > 4KB threshold → use SHM path
  ├── shm_ring_write(G2H, hostData, 10MB)  // Write directly into shared memory
  └── Send RPC {dst: real_address, size: 10MB, flags: SHM_DATA}

Host:
  Receive RPC → detect SHM_DATA flag
  ├── shm_ring_read(G2H, buf, 10MB)  // Read from shared memory
  └── Real cuMemcpyHtoD(buf → GPU VRAM)
```

Data is copied only once (Guest writes to SHM → Host reads from SHM → writes to GPU), versus three copies in the vsock path.

### 5. Platform Integration: CubeShim

#### 5.1 Configuration Injection

GPU functionality is triggered via Kubernetes annotations:

```yaml
annotations:
  cube.gpu: '{"enable": true, "memory_quota_mb": 4096, "vsock_port": 20565}'
```

CubeShim's `config.rs` parses this annotation into a `GpuConfig` struct. The downstream flow:

1. **VM creation** (`prepare_resource` in `sb.rs`):
   - Detects GPU enabled
   - Allocates an IVSHMEM backing file path (`/var/run/cube-gpu/shm/sandbox-{hash}.shm`)
   - Configures IVSHMEM device (path + size) via `set_ivshmem()`
   - Hypervisor creates the IVSHMEM PCI device; guest sees `/dev/shm/cube-gpu-shm`

2. **Container creation** (`get_pb_spec` in `container/mod.rs`):
   - Detects GPU enabled
   - Injects into OCI spec process environment variables:
     - `LD_PRELOAD=/usr/lib/cube-gpu/libcube-gpu.so`
     - `CUBE_GPU_SANDBOX_ID=...`
     - `CUBE_GPU_MEMORY_QUOTA=...` (in bytes)
     - `CUBE_GPU_VSOCK_PORT=20565`
     - `CUBE_GPU_SHM_PATH=/dev/shm/cube-gpu-shm`
   - When the container process starts, `libcube-gpu.so` is loaded, reads these env vars, and initializes the vsock connection and SHM channel

#### 5.2 Modified CubeShim Files

| File | Changes |
|---|---|
| `sandbox/config.rs` | Added `GpuConfig` struct, `ANNO_GPU` annotation key, `gpu: Option<GpuConfig>` field on `Config`, parsing logic |
| `sandbox/gpu.rs` | New file — `GpuIntegration` helper: env var generation, IVSHMEM path/size computation |
| `sandbox/mod.rs` | Added `pub mod gpu` |
| `sandbox/sb.rs` | Added `GpuIntegration` import, IVSHMEM setup in `prepare_resource()` |
| `hypervisor/config.rs` | Added `ivshmem: Option<IvshmemConfig>` to `VmConfig`, passthrough in `to_vm_config()`, `set_ivshmem()` builder |
| `container/mod.rs` | Added GPU env var injection in `get_pb_spec()` |

#### 5.3 Why IVSHMEM Works Here

CubeSandbox's underlying Cloud Hypervisor already has a complete IVSHMEM PCI device implementation (`hypervisor/devices/src/ivshmem.rs`). It works as follows:

1. Host creates a backing file (`/var/run/cube-gpu/shm/sandbox-xxx.shm`)
2. Hypervisor maps this file as a PCI device's BAR2 region
3. Guest driver (or raw mmap) maps `/dev/shm/xxx` to the same physical memory

Guest and host now share this memory — when the guest writes a byte, the host can immediately read it.

## Key Design Decisions

### API-level Remoting vs. Full GPU Virtualization

| Approach | Pros | Cons |
|---|---|---|
| **API-level remoting (chosen)** | Works with consumer GPUs, no special hardware, low implementation complexity | Only supports hooked APIs, adds latency per call, requires maintenance as CUDA evolves |
| GPU passthrough (VFIO) | Full GPU access, no latency overhead | One GPU per sandbox, no sharing, requires hardware IOMMU |
| GPU virtualization (vGPU/MIG) | Hardware-level isolation | Requires enterprise GPUs (A100/H100), complex setup |
| PCIe device emulation | Transparent to guest OS | Extremely complex, high latency, not practical for CUDA |

### Why MPS for Concurrency

Consumer GPUs lack MIG. MPS provides:

- True concurrent kernel execution (multiple contexts' kernels run simultaneously)
- Address space isolation per context
- No time-slicing overhead

The tradeoff: no compute unit isolation — a compute-heavy kernel from one sandbox can starve others. This is acceptable for the P0 target of development/testing workloads.

### Why Custom Protocol over vsock

The vsock transport provides reliable, ordered byte streams. gRPC over vsock would require an HTTP/2 framing layer. For a fixed set of ~15 CUDA APIs with simple request/response semantics, a 40-byte header + flat payload is simpler, faster to parse, and has lower per-call overhead.

### Kernel Parameter Serialization in cuLaunchKernel

CUDA kernel parameters are passed as `void *kernelParams[]` — an array of pointers to argument values. The hook layer does not know argument types or sizes (that requires kernel metadata). The P0 approach: serialize each pointer's target as `sizeof(void*)` bytes (8 bytes on 64-bit). This covers all common CUDA kernel parameter types (int, float, double, pointers) on 64-bit platforms. The host reconstructs the `void*[]` array by pointing each entry to the corresponding 8-byte slot in the deserialized buffer.

Limitation: kernel parameters larger than 8 bytes (e.g., `double4`, `uint4`) or the `extra` launch config (`CU_LAUNCH_PARAM_BUFFER_POINTER` format) are not supported in P0.

## Implementation Status

### Completed

- [x] RPC wire protocol design (`gpu/proto/rpc_protocol.h`)
- [x] Rust protocol definitions (`gpu/proto/rpc_protocol.rs.inc`)
- [x] Guest-side `libcube-gpu.so`:
  - [x] dlsym dual-version hook (glibc 2.2.5 + 2.34)
  - [x] cuGetProcAddress hook (CUDA 11.3+)
  - [x] Virtual handle dual-address-space mapping
  - [x] vsock RPC client
  - [x] IVSHMEM ring buffer (lock-free SPSC)
  - [x] All P0 API hooks (cuInit, cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH, cuLaunchKernel, cuStreamCreate, cuStreamSynchronize, cuCtxSynchronize, cuModuleLoadData, cuModuleGetFunction)
  - [x] Linker version script for symbol export
  - [x] Makefile
- [x] Host-side `cube-gpu-daemon`:
  - [x] Async vsock listener (tokio)
  - [x] Session manager with per-sandbox CUDA context and quota tracking
  - [x] CUDA runtime wrapper (libloading)
  - [x] SHM manager (file creation/mmap/cleanup)
  - [x] API dispatch for all P0 APIs
- [x] CubeShim integration:
  - [x] GpuConfig struct and annotation parsing
  - [x] IVSHMEM device config passthrough (VmConfig → Hypervisor)
  - [x] LD_PRELOAD + GPU env var injection into container OCI spec
  - [x] IVSHMEM backing file path generation in prepare_resource()

### Not Yet Implemented

- [ ] Build system: Dockerfile for cross-compiling `libcube-gpu.so` with CUDA headers on Linux
- [ ] Host daemon SHM data read: when `SHM_DATA` flag is received, read from mmap'd IVSHMEM file
- [ ] Host-side `cuLaunchKernel` parameter reconstruction: rebuild `void* kernelParams[]` from serialized 8-byte slots
- [ ] Host daemon systemd service / process management
- [ ] Cubelet integration: GPU resource scheduling, annotation passthrough to CubeShim
- [ ] Graceful shutdown: cleanup SHM files, close CUDA contexts
- [ ] Error recovery: reconnection on vsock disconnect, CUDA context loss handling
- [ ] cuLaunchKernel `extra` launch config support (`CU_LAUNCH_PARAM_BUFFER_POINTER`)
- [ ] Expanded API coverage: cuMemAllocManaged, cuMemcpyAsync, cuEventCreate, cuMemsetD8, etc.
- [ ] Performance optimization: batch RPC for multiple small operations
- [ ] Security: input validation on all RPC payloads, SHM bounds checking
- [ ] End-to-end testing on Linux + KVM + NVIDIA GPU hardware

## File Index

```
gpu/
├── proto/
│   ├── rpc_protocol.h          # C RPC protocol header (all structs, API IDs, flags)
│   └── rpc_protocol.rs.inc     # Rust RPC header struct (included by vsock_listener.rs)
├── guest/
│   └── src/
│       ├── hook.c              # CUDA Driver API hooks (LD_PRELOAD + dlsym + cuGetProcAddress)
│       ├── rpc_client.c        # vsock RPC client implementation
│       ├── rpc_client.h        # vsock RPC client API
│       ├── shm_ring.c          # IVSHMEM ring buffer implementation
│       ├── shm_ring.h          # IVSHMEM ring buffer API
│       ├── cuda_defs.h         # CUDA type definitions, function pointer typedefs, symbol macros
│       ├── libcube_gpu_symbols.ld  # Linker version script
│       └── Makefile            # Build libcube-gpu.so
└── host/
    ├── Cargo.toml              # Rust project manifest
    └── src/
        ├── main.rs             # Daemon entry point
        ├── vsock_listener.rs   # Async vsock accept loop + dispatch
        ├── session.rs          # Per-sandbox session management + API handling
        ├── cuda_runtime.rs     # libcuda.so.1 wrapper via libloading
        └── shm_manager.rs      # SHM file lifecycle management

CubeShim/shim/src/
├── sandbox/
│   ├── config.rs               # +GpuConfig, +ANNO_GPU, +gpu field, +parsing
│   ├── gpu.rs                  # (new) GpuIntegration helper
│   ├── mod.rs                  # +pub mod gpu
│   └── sb.rs                   # +IVSHMEM setup in prepare_resource()
├── hypervisor/
│   └── config.rs               # +ivshmem field, +set_ivshmem(), +passthrough
└── container/
    └── mod.rs                  # +GPU env var injection in get_pb_spec()
```

## Environment Variables

The following environment variables are injected into guest containers when GPU is enabled:

| Variable | Example | Description |
|---|---|---|
| `LD_PRELOAD` | `/usr/lib/cube-gpu/libcube-gpu.so` | Loads the CUDA interception layer |
| `CUBE_GPU_SANDBOX_ID` | `12345` | Unique sandbox identifier for session routing |
| `CUBE_GPU_MEMORY_QUOTA` | `4294967296` | VRAM quota in bytes (default: 4GB) |
| `CUBE_GPU_VSOCK_PORT` | `20565` | vsock port for RPC (default: 0x5055) |
| `CUBE_GPU_SHM_PATH` | `/dev/shm/cube-gpu-shm` | IVSHMEM device path in guest |
| `CUBE_GPU_DEBUG` | `1` | Enable debug logging (optional) |

## Default Configuration

| Parameter | Default Value |
|---|---|
| vsock port | `0x5055` (20565) |
| IVSHMEM total size | 128 MB |
| SHM ring per direction | 64 MB |
| SHM data threshold | 4 KB |
| Memory quota | 4096 MB |
| Handle table initial capacity | 256 (vptr), 64 (other handles) |
