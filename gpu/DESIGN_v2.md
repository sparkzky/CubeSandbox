# CubeSandbox GPU 共享方案 v2 设计

> 基于 `sparkzky/feat/gpu-share` 草稿实现、官方 Discussion #174、四路业界调研综合产出。
> 调研完成时间：2026-11。

## 1. 问题定义

### 1.1 现状

- **草稿实现**（本地分支）：guest 内 `LD_PRELOAD` 拦截 CUDA Driver API → vsock RPC → host `cube-gpu-daemon` 执行；每 sandbox 一个 `CUcontext`；大数据拟走 IVSHMEM（**当前断裂**：shim 建 `sandbox-{id}.shm` 128MB 双 ring，daemon 自建自读 `session-{handle}.shm` 64MB，guest/host 不在同一块内存）。
- **官方 #174**：VFIO 整卡透传为第一参考实现；GPU 共享（mdev/时间片/API forwarding）列为 future direction；要求 Cubelet 拥有资源状态机、模板兼容性契约、调度感知；GPU 沙箱明确不支持 snapshot/pause/resume/live-migration。
- **核心顾虑**（用户）：API 拦截转发路线下，多 sandbox 共享单卡的冲突问题——显存超卖/OOM、SM 争抢/噪声邻居、故障域共享、跨租户泄漏。

### 1.2 调研得出的三条铁律

1. **软件限额 ≠ 安全隔离**。HAMI（CNCF Incubating）的 hook 层显存/算力限制均为 soft，官方承认可被 `CUDA_DISABLE_CONTROL`/DinD/直连驱动/静态链接绕过；其算力节流实测偏差可达 ~2 倍（HAMi-core #350/#360）。**安全边界必须由更低层承担**。
2. **硬件分区是唯一同时解决"显存+带宽+故障域"的官方手段**。MIG 提供物理隔离（独立故障域、显存带宽 QoS），但卡型受限（A100/A30/H100/H200/B200/RTX PRO 6000 Blackwell）、粒度粗（最小 1/7 SM + 1/8 显存）、最多 7 实例。
3. **MicroVM 是 API-forwarding 的天然安全边界**。HAMI 类方案需要防"绕过 hook"（guest 能直接摸到 `/dev/nvidia*`）；而 CubeSandbox 的 guest 是 MicroVM，**物理上没有任何 GPU 设备通道，vsock 是唯一入口**——绕过 hook = 功能失败，不是安全突破。这是容器方案做不到的架构优势，必须显式利用。

## 2. 业界横向结论（支撑决策）

| 路线 | 代表 | 显存隔离 | 算力隔离 | 故障域 | 生命周期 | 对 CUDA | 结论 |
|---|---|---|---|---|---|---|---|
| 纯时间片 | NVIDIA 默认 | ❌ | ❌ | 共享 | — | ✅ | 排除 |
| MPS | NVIDIA 官方 | 弱(软上限) | SM% 配额 | **共享**（一 fault 拖垮同卡全部 client） | — | ✅ | 只可作辅助，不能作主隔离 |
| MIG | NVIDIA 官方 | ✅ 硬件 | ✅ 硬件 | **独立** | ❌ 重配需 reset/idle | ✅ | **多租户硬隔离首选** |
| vGPU | NVIDIA 闭源 | ✅ | ✅ | 独立 | — | ✅ | 排除（license+hypervisor 绑定） |
| 用户态 hook 限额 | HAMI/HAMI-core | ❌（soft，防误用） | ❌（soft） | 无 | — | ✅ | 只当"应用体验层"，不当安全层 |
| 内核模块限额 | cGPU/qGPU（均闭源） | 硬（MB 级） | 时间片 QoS | 弱/声称强 | — | ✅ | 思路参考，不可复用实现 |
| API forwarding | rCUDA(死)/GVirtuS(半死)/Cricket(活)/GTAP(商用) | ❌ | ❌ | 取决于 host 侧组织 | **✅ 与 GPU 生命周期解耦（可 snapshot/迁移）** | ✅(转发) | **共享后端主体，需按 §4 加固** |
| ioctl 层代理 | gVisor nvproxy | 强（参数白名单） | — | 依赖 KMD | — | ✅ | **仅在共享宿主内核时成立**；跨 VM 边界 mmap 不通，等价物即 VFIO |
| VFIO 透传 | Kata/CLH | ✅(IOMMU) | ❌(整卡) | 整卡 | **❌ 不可 snapshot/pause**；VFIO 强制 pin 全部 guest 内存（实测 ~3.3–3.5s/128GiB），Kata 冷启动 ~2min | ✅ 原生 | 独占模式后端，按 #174 节奏 |
| virtio-gpu | crosvm/Venus | — | — | — | — | ❌ 仅图形/Vulkan | 排除 |

关键佐证：
- 腾讯 qGPU 深度文章直接点名批评"MPS context 合并导致故障传播、API 转发后端默认用 MPS 导致一个 VM 越界杀死一堆不相干 VM"——正是本方案 §4.3 要结构性解决的点。
- gVisor nvproxy 的价值在方法论而非直接移植：版本树（version tree）+ driver differ + parity test 是覆盖矩阵可持续维护的标杆；**但它的 mmap 数据直通依赖与宿主共享内核，KVM MicroVM 没有这个条件**——跨 VM 边界的"mmap 直通"恰恰就是 VFIO 本身。这把三条路线统一成一条阶梯：API forwarding（高层、易、生命周期友好）→ VFIO（低层、硬隔离、生命周期代价）。
- Firecracker 至今 VFIO WIP（#5679），印证 MicroVM 生态里 API forwarding 是 GPU 共享的现实主流路线。
- Cricket 证明 CUDA API 状态可 checkpoint/restart——为 §6 的沙箱快照差异化提供可行性先例。

## 3. 方案总览：一控制面 + 三执行后端

```
                          ┌────────────────────────────────────────┐
                          │ Cubelet GPU Manager（节点资源唯一真相）  │
                          │  - 设备/MIG 几何发现(NVML)              │
                          │  - 状态机: allocatable→reserved→       │
                          │    attached→releasing→released/        │
                          │    quarantined（对齐 #174）             │
                          │  - 后端分配: mig | share | exclusive   │
                          └───────┬───────────────┬────────────────┘
                                  │ 分配决策       │ 上报 inventory
                          ┌───────▼──────┐ ┌──────▼─────────┐ ┌──────────────┐
                          │ 后端A: MIG   │ │ 后端B: share   │ │ 后端C: VFIO  │
                          │ 每实例一个    │ │ 加固版          │ │ 独占透传      │
                          │ daemon+MPS   │ │ cube-gpu-daemon│ │ (官方#174主线)│
                          │ 硬件隔离      │ │ +软件限额+加固  │ │ (本文不展开)  │
                          └───────┬──────┘ └──────┬─────────┘ └──────────────┘
                                  │ vsock          │ vsock
                          ┌───────▼────────────────▼─────────┐
                          │ guest: libcube-gpu.so（仅体验层） │
                          └──────────────────────────────────┘
```

- **后端 A（MIG，推荐主力）**：有 MIG 卡时的默认后端。硬件隔离直接消除用户顾虑的四个冲突维度；API-forwarding 只是"把流量引到指定 MIG 实例"的搬运工。
- **后端 B（share，降级兼容）**：无 MIG 卡（消费级/T4/L4/A10 等）时的降级后端，继承现有草稿并按 §4 硬化；诚实标注为"限额与防误用，非安全隔离"，由租户产品层按不可信等级选择。
- **后端 C（exclusive/VFIO）**：整卡独占，走 #174 已明确的 VFIO 透传路径（本设计不重复展开，仅要求控制面状态机兼容其生命周期限制）。
- 三后端共用同一协议（`gpu/proto`）、同一 guest 库、同一 `cube.gpu` 注解入口，通过注解字段选择/回退。

### 3.1 后端 A 设计（MIG）

**核心决策：一个 MIG 实例 = 一个 `cube-gpu-daemon` 进程 = 一个 MPS server 域**（而非单 daemon 服务全部实例）。

理由：
1. 故障域精确到 MIG 实例——MIG 本身故障域独立，daemon 进程级隔离把软件故障也封在实例内；单 daemon 多实例则把故障域重新拉通（qGPU 文章批评的正是这个反模式）。
2. 每实例 daemon 以 `CUDA_VISIBLE_DEVICES=MIG-<uuid>` 启动（或 `cuDeviceGetByPCIBusId` 绑定，实现时 spike 定夺），会话天然只能落在分配的实例上，**不需要任何校验逻辑就能杜绝跨实例访问**。
3. vsock 端口按实例分配（如 `0x5055 + instance_idx`），shim 侧按分配结果注入对应端口。

生命周期：
- Cubelet 用 NVML 管理 MIG 几何（创建/销毁/查询），参考 HAMI 动态 MIG + `mig-parted`、NVIDIA DRA `partitionable devices + shared counters` 的调度语义；重配置需 GPU idle（accept：GPU 池预热阶段统一做，运行中禁止改几何）。
- daemon + MPS 以 systemd 单元按实例部署（`cube-gpu-daemon@mig0.service` …），修复草稿"无进程管理"缺口；client 崩溃 → MPS FAULT 域收敛在单实例 → Cubelet 隔离该实例并重建。

### 3.2 后端 B 设计（share，非 MIG 卡）

单 daemon 服务整卡，隔离强度 = "限额 + 防误用"（MicroVM 边界兜底为安全边界），按 §4 硬化。

### 3.3 后端选择与调度

- `cube.gpu` 注解扩展为结构化请求（向后兼容布尔值）：

```yaml
cube.gpu: |
  backend: mig|share|exclusive|auto   # auto = 按节点能力与模板契约回退
  memory-mib: 8192                     # 显存配额（A/B 后端）
  sm-percent: 50                       # 算力配额（B 后端；A 由 MIG profile 隐含）
  mig-profile: "2g.10gb"               # 可选，A 后端
  allow-fallback: true                 # 调度回退许可
```

- CubeMaster 过滤 = 节点 GPU 库存（数量/型号/MIG profile 空闲计数）+ 模板兼容契约（§5.3）+ binpack/spread 评分（借 HAMI 公式：已用设备/算力/显存三维归一加权）。

## 4. 后端 B 硬化清单（逐一对应现有冲突点）

按"必须"优先级排列；每条注明草稿现状。

### 4.1 数据通路（阻断级）

1. **修复 IVSHMEM 断点**：daemon 废弃自建 `session-{handle}.shm`，改为 mmap shim 预建的 `sandbox-{id}.shm`（路径经 vsock session_init 协商下发）；统一双 ring 布局与 `shm_offset` 语义。
   - 备选：若IVSHMEM 修复成本过高，短期用 vsock 直传 + 16MiB 分块（吞吐损失，先通后快）；长期评估 `virtio-shmem`/memfd 共享。
2. **>8B kernel 参数**：支持 `CU_LAUNCH_PARAM_BUFFER_POINTER` 整包参数缓冲（guest 打包经 SHM 传递，daemon 只透传指针数组）——一条路径覆盖 `double4/uint4` 等全部布局，比逐类型 marshal 划算。

### 4.2 所有权与配额（阻断级）

3. **所有权校验**：`handle_ctx_set_current` 必须校验 ctx ∈ `session.contexts`；`handle_module_get_func` 校验 module ∈ `session.modules`。跨 session 引用一律 `CUDA_ERROR_INVALID_HANDLE`。
4. **配额权威在 daemon**：guest 侧 hook 降级为"体验层"（见 4.4），唯一权威是 `Session.can_allocate`（已有）+ 硬顶（daemon 侧对 `memory_used > quota` 的分配直接 OOM，含 async/pool/managed 分配——对照 HAMI 覆盖清单补齐 `cuMemAllocAsync`/`cuMemAllocFromPoolAsync`/`cuMemCreate`/`cuMemMap`/`cuMemPool*`/`cuGraphAddMemAllocNode`）。
5. **CUDA IPC 封锁**：拦截 `cuIpcGetMemHandle/cuIpcOpenMemHandle/cuIpcCloseMemHandle`；`OpenMemHandle` 仅允许打开**本 session** 注册的 handle，跨 session 一律拒绝。IPC 是 NVIDIA 文档明确的跨进程泄漏面（句柄即能力 + 子分配整块共享），多租户必须默认关。

### 4.3 故障域与 QoS（高优）

6. **接入 MPS**（share 后端的算力隔离）：daemon 作为 MPS client 连接 host 侧 `nvidia-cuda-mps-server`（per-GPU），获得：并发执行（消除草稿"每 session 一个 context → 时间片串行"的隐性退化）、`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 软显存上限、`active thread percentage`/`cuCtxCreate_v3` exec-affinity SM% 配额。
   - 诚实边界：MPS 故障域仍共享（一 tenant fatal fault → 同 server 全部 client 广播退出）。缓解：daemon 捕获 session 失败 → 销毁该 session → 重启 MPS server → Cubelet 短暂 quarantine 该卡（对齐 #174 的 quarantine 语义）；文档明示 share 后端故障域=整卡，租户等级敏感者应选 MIG。
7. **每 session 资源记账 + 回收**：destroy_session 覆盖 allocations/modules/functions/streams/contexts 全量回收（已有），补：异常断连（vsock close）路径的强制回收 + 回收失败的 quarantine 上报。

### 4.4 guest 库体验层（中优）

8. **伪造设备视图**（借 HAMI 成熟做法）：guest hook 拦截 `cuDeviceTotalMem/cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)/cuMemGetInfo` 返回配额视图，`nvidia-smi` 视图经 NVML hook 限制——vLLM/PyTorch 等按"总显存=配额"自适应，把 OOM 前移为规划失败。
9. **API 覆盖补齐按真实负载驱动**：先跑 vLLM/PyTorch/ONNX Runtime probe 抓实际调用集（CUInterposer 证明 driver+runtime 全集仅 ~400 个相对简单调用；HAMI hook.c 登记的 250+ driver API 是现成清单），优先补 `cuMemcpyAsync/cuEvent*/cuMemset*/cuMemAllocManaged/cuStreamWaitValue`；协议层预留 API id 段位即可平滑扩展。
10. **拦截健壮性**：`dlsym_225/_234` 双版本已有，补 `LD_AUDIT` 评估（GTAP 路线：原生 libcuda 根本不加载，杜绝双初始化）；静态链接 CUDA 的应用在 guest 内 `cuInit` 失败即干净报错（MicroVM 无设备可逃逸，这是特性不是缺陷，文档写明）。

### 4.5 协议与工程（中优）

11. P1 查询 API（40-44）落地 handler（vLLM 冷启动路径需要 `cuDeviceGetAttribute` 系列）。
12. `extra` 参数不支持 → 显式返回错误码并文档化（拒绝执行优于静默错误）。
13. daemon systemd 化 + vsock 重连 + 优雅 shutdown（destroy-then-exit）；16MiB payload 上限之外补 per-session 请求速率限制。

## 5. 控制面（对齐 #174 要求，不做则无法上游化）

### 5.1 Cubelet GPU Manager（新增）

- 发现：NVML 枚举物理卡 + MIG 能力/profile；IOMMU/VFIO 就绪检查（C 后端）。
- 库存上报：节点 annotation/心跳携带 `{gpu-uuid, model, mem, mig-capable, mig-geometry, backend-allocatable, health}`。
- 状态机：`initialized → allocatable → reserved → attached → releasing → released → allocatable`，失败路径 `unhealthy/quarantined/disabled`；**只有 Cubelet 能写回 allocatable**；重启后先 reconcile（对照 NVML + 进程存活 + vsock 会话表）再上报。
- 释放保守性：daemon 会话表清零 + NVML 进程清单核验后才 `released`。

### 5.2 调度

- CubeMaster 维护集群 GPU 视图（借 DRA ResourceSlice 属性化模型，而非整数计数）。
- 过滤：后端可用性（mig-profile 空闲计数 / share 剩余配额 / exclusive 空闲卡）+ 模板契约匹配（5.3）+ 显存/算力硬约束。
- 评分：binpack（默认，填充率优先，GPU 沙箱场景碎片敏感）可 spread 覆盖。

### 5.3 模板兼容契约（借 nvproxy 版本窗模型）

GPU 模板显式绑定并入库：`{guest rootfs, guest kernel, hook-lib 版本, CUDA toolkit 版本, host driver 版本区间, 支持卡型列表, 支持后端}`。
- 调度时匹配 host driver ∈ 模板声明区间 + 卡型 ∈ 声明列表（MIG 卡/非 MIG 卡分档）。
- 版本窗策略借鉴 gVisor：Supported（测试过的 driver×CUDA 组合）/ Unsupported（已知不兼容）/ Unknown（默认拒绝，显式 opt-in）三档；配套 `driver_differ` 式 CI（每次 driver 发布跑 probe 回归）。

## 6. 生命周期：与 #174 的差异化机会

#174 因 VFIO 透传而禁掉 GPU 沙箱的 snapshot/pause/resume。**API-forwarding 后端天然没有这个限制**（GPU 状态全在 host daemon，沙箱无设备状态）——这是本方案对上游最有卖点的差异化：

- snapshot：CPU 侧照常快照；GPU 侧 session 冻结（停止接受新 RPC）→ 恢复时 session 原样续用（host daemon 常驻）。
- clone/rollback：新 sandbox 可挂到克隆时刻的 session 状态副本——**CUDA API 状态 checkpoint**（Cricket 已证明可行：分配表/module blob/context 参数可重放重建；vGPU context 硬件状态无需迁移）。
- 诚实限制：kernel 执行中的瞬时状态不可精确捕获（与 CPU 快照的窗口一致性需协议层 barrier：快照前 flush 在途 RPC + `cuStreamSynchronize` 全 stream）；UVM/managed memory 语义受限（标记 unsupported）。

此能力使 GPU 沙箱反而比透传方案更贴合 CubeSandbox 现有 snapshot/rollback/clone 产品线，上游化沟通时应作为首要论点。

## 7. 安全模型（诚实声明）

| 层 | 防什么 | 不防什么 |
|---|---|---|
| MicroVM 边界 | guest 无法触碰任何 GPU 设备/驱动；hook 被绕过 = 功能失败而非安全突破 | vsock 通道滥用（DoS）→ 速率限制 + 配额拒绝 |
| MIG 后端 | 显存容量/带宽硬件隔离、故障域独立、跨 GI 无 IPC | GI 内 CI 细分层的显存共享（CI 共享父 GI 显存） |
| share 后端 | 越权释放/跨 session 句柄（所有权校验）、IPC 泄漏（封锁）、超卖（daemon 硬顶） | driver 自身漏洞（透传路线共有的下限，nvproxy 同样承认）、带宽争抢（无 QoS）、MPS fault 域共享 |
| 控制面 | 双重分配（状态机 + reserved 先于 attach）、失控回收（quarantine） | Cubelet 自身被攻破（信任根，#174 同假设） |

对外文档必须把 share 后端标注为「untrusted-tenant NOT safe；trusted/semi-trusted workload OK」，把 MIG/exclusive 标注为「untrusted-tenant safe」。这与 E2B/agent 沙箱产品分级一致：默认模板走 MIG/exclusive，低成本档走 share。

## 8. 验收清单（由 HAMI 绕过清单反推 + 草稿缺陷）

1. `sandbox-{id}.shm` 与 daemon mmap 同一文件；guest 写 → host 读往返测试通过（含 wrap-around）。
2. 跨 session `ctx_set_current`/`module_get_func`/IPC handle 全部拒绝（单测）。
3. quota 硬顶：async/pool/managed/graph-node 分配路径逐一路径 OOM（对照 HAMI test_alloc_async/pool 测试样式）。
4. `cuDeviceTotalMem` 返回配额；vLLM probe 在 share 后端按配额规划不 OOM host。
5. kill -9 tenant 进程 → vsock 断连 → daemon 10s 内回收全部资源，NVML 无残留进程。
6. MPS client fault → 仅本卡 session 受影响 → server 重启 → Cubelet quarantine → 恢复 allocatable（自动，无需人工）。
7. MIG：`2g.10gb` 实例上跑 probe；另一实例压力测试不互相影响显存带宽（DCGM 观测）。
8. snapshot/restore：含 GPU session 的沙箱快照恢复后继续推理（后端 B）；快照窗口 barrier 生效（恢复后无悬空 async 拷贝）。
9. hook 一致性 CI：hook 表 vs CUDA 头文件自动比对（HAMI `check_cuda_hook_consistency.py` 模式）。

## 9. 路线图

| 阶段 | 内容 | 交付判据 |
|---|---|---|
| P0 修复（1-2 周） | §4.1-4.2 全部 + §4.3.7 + §8.1-5 | 验收 1-5 绿 |
| P1 加固（2-4 周） | MPS 接入 + 体验层 + probe 驱动的 API 补齐 + systemd | 验收 4/6 部分绿；vLLM 实测报告 |
| P2 MIG 后端（3-4 周） | Cubelet MIG manager + 实例级 daemon + 调度 + 兼容契约 | 验收 7 绿；双租户压测 |
| P3 生命周期（3-4 周） | snapshot barrier + session 冻结/续用 + clone 重放 | 验收 8 绿 |
| P4 上游化 | #174 下提交 backend 补充设计 → 按维护者节奏拆 PR | 对齐 #174 状态机/契约术语 |

P0/P1 在现分支迭代；P2 起拆分支；P4 前先在 Discussion #174 评论（教训：PR #1157 因"先实现后讨论"被关）——提案口径：**API forwarding + MIG 是 #174 pluggable backend 框架的第二个 backend 实现**，恰好补齐其 future direction 清单里的 "API forwarding / vendor sharing"，并保留其 VFIO 参考实现计划不变。

## 10. 参考

- NVIDIA：MPS 架构（docs.nvidia.com/deploy/mps/architecture.html）、MIG 概念/部署/支持 profile（docs.nvidia.com/datacenter/tesla/mig-user-guide/）、CUDA IPC（CUDA Programming Guide §Inter-process Communication）、DRA GPU driver（dra-driver-nvidia-gpu.sigs.k8s.io）
- HAMI：原理（project-hami.io/docs/v2.9.0/core-concepts/gpu-virtualization）、HAMi-core（github.com/Project-HAMi/HAMi-core，#350/#360/PR #376）、FAQ/兼容矩阵
- 内核态方案：Alibaba cGPU（alibabacloud.com/help/en/egs）、Tencent qGPU（cloud.tencent.com/developer/article/1831090）
- 拦截/远程：gVisor nvproxy（gvisor.dev/docs/user_guide/gpu、g3doc/proposals/nvidia_driver_proxy.md、version.go、seccomp_filters.go）、rCUDA（hpca.uji.es/rcuda）、GVirtuS（github.com/gvirtus/GVirtuS）、Cricket（github.com/RWTH-ACS/cricket）、GTAP（loopholelabs.io/blog/ollama-remote-gpu）、CUInterposer（EuroSys'24 poster）
- MicroVM：Kata GPU passthrough（kata-containers.github.io/use-cases/NVIDIA-GPU-passthrough-and-Kata*）、Firecracker #4845/#5679/PR #6245、cloud-hypervisor docs/vfio.md、crosvm devices 文档、Kata GPU 启动实测（topofmind.dev 2025-11-19）

---

## 11. 实现状态（2026-11，后端 B）

### 已完成并验证

| 设计项 | 状态 | 验证证据 |
|---|---|---|
| §4.1 数据通路：单一 backing file + v2 布局（全局头/双 ring/单调游标） | ✅ | 单测（init/校验/错 token/错大小、wrap-around 多轮 > 容量、短读不消费、双向独立）；E2E 1MiB 往返 |
| §4.1 >8B 参数：`extra` BUFFER_POINTER/BUFFER_SIZE + cubin/fatbin/PTX 参数元数据 | ✅ | `cubin_params` 6 类单测（含 32B struct/16B int4）；E2E mixed `[4,8,8,8]` 与 32B struct kernel |
| §4.2 所有权校验（ctx/module/func/stream/event/allocation） | ✅ | 跨 session ctx→201、free 他人 vptr→1、module→400 单测 |
| §4.2 daemon 权威配额（同步 + 异步分配） | ✅ | 8MiB 配额下 16MiB malloc/alloc_async→OOM 单测；`--quota-oom` E2E |
| §4.2 CUDA IPC 封锁 | ✅ | `cuIpcGetMemHandle/OpenMemHandle` 均返回 801，且不建立 RPC |
| §4.3 配额视图（`cuDeviceTotalMem`/`cuMemGetInfo` 返回 quota） | ✅ | E2E 断言 512MiB 配额而非 12GiB 物理 |
| §4.5 P1 查询 API / events / memset / async alloc / primary ctx | ✅ | E2E 17/17（cubin 与 PTX 各一遍） |
| 断连/异常回收 | ✅ | `e2e-multi-session.sh`：并行两会话；`kill -9` 后 daemon 存活、另一会话可用、销毁日志出现 |
| 环形回绕（跨 64MiB 容量累积） | ✅ | 同文件 70 个连续会话全部通过（累计流量 > ring capacity） |
| shim token 注入与校验 | ✅ | 正确 token 全流程通过；错误 token 会话建立失败（exit 1 + 拒绝日志） |
| systemd + MPS 脚本 | ✅ | 单元语法校验；`start-mps.sh` 启动 MPS server 后 daemon 跑 E2E 17/17 |

### 尚未实现（后续阶段）

- **Cubelet 资源状态机与 GPU 感知调度**（DESIGN §5，P2）：当前只有 `cube.gpu` 注解转发与 shim 级配额，无节点库存/放置。
- **MIG 后端**（§3.1，P2）：需 Cubelet MIG 几何管理 + 实例级 daemon。
- **snapshot/clone 的 GPU 会话冻结与重放**（§6，P3）。
- **per-session 进程隔离与 MPS 每客户端 SM%**：现为单 daemon 进程 + `cuCtxCreate_v3` 尝试（不支持则回退），故障域仍为整卡。
- per-session 请求速率限制；`cuGetLastError` 的精确跟踪。

---

## 12. 真实沙箱验证（2026-11）

在一体化部署上创建了真实 KVM 沙箱（RTX 3060 / driver 615.71），经 E2B 兼容 API + `metadata["cube.gpu"]` 传入注解，envd 内执行 `e2e_main`：

| 验证项 | 结果 |
|---|---|
| 单沙箱完整 API 序列 | 17/17 通过（配额视图、IVSHMEM 1MiB 往返、cubin/PTX、32B 结构体参数、memset、事件、异步分配） |
| 3 沙箱并发（512/256/256 MiB） | 全部通过；每个沙箱看到自己的配额；256MiB 沙箱在自身上限 OOM |
| 杀死一个沙箱 | 其余两个继续可用并重跑完整测试 17/17 |
| daemon 清理 | 0 错误（客户端先销毁 ctx 的 400 视为 benign） |

为此修复了宿主栈上的真实缺口：CubeAPI/CubeMaster 注解透传、shim restore 路径传递 IVSHMEM、VMM 恢复时允许新增设备、IVSHMEM BAR2 重编程时迁移 KVM slot、guest 库自动 PCI rescan、guest 库消除 GLIBC_2.38 依赖。

缺口修复：共享内存全局头新增 `memory_quota`，guest 库在 env 缺失时从头部读取 `session_id`/token/quota（零配置引导，已用空环境验证）；daemon 监听 `CUBE_GPU_VM_DIR`（默认 `/run/vc/vm`）下每个沙箱的 `cube.sock_<guest_port>`，原生接入 VMM 的 Unix vsock 代理（不再需要桥接）。
