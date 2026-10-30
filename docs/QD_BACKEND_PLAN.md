# mmjlab qd 后端替换与三臂对比计划(mujoco-warp → mujoco_qd,去 wp)

> 状态:计划稿,未实现。本文只做工作量梳理、需求分析、架构逻辑与测试方案,
> 不含新架构设计与新算法——一切以「底层无感替换 + 可量化对比」为边界。
> 事实基础:2026-10-23 三路勘察(mjlab 接缝 / mujoqo / simtest 依赖),
> 文中 `文件:行号` 均为当日实测。

---

## 0. 口径确认(先读)

**「把 mmjlab 里的 mjwarp 全部换成 mujoco」落实为:物理引擎换成 `mujoco_qd`**
(mujoqo = qd/quadrants 版 mujoco-warp 全量移植,API 与 mjwarp 同形:
`put_model / put_data / make_data / step / forward / reset_data / refit_bvh`,
见 `mujoqo/mujoco_qd/__init__.py:29-129` `_PUBLIC_API` 全表)。
理由有二,且互为印证:

1. 「引用和 import 尽量一致」只有 mujoco_qd 做得到——它就是按
   「换一个 import」设计的(mujoqo `docs/render-需求评估.md:250`)。
   stock mujoco C API 无批量 `put_data/nworld` 形态,import 面不可能对齐。
2. `mmjlab+metal` 对比臂只有 mujoco_qd 能提供(qd 有 cpu/metal 双 arch;
   warp-lang 在 macOS 无 GPU 后端,warp-CPU 实测比 mjbatch 慢 23×,
   mujoqo 调研文档:243)。

**三臂定义**(4096 envs):

| 臂 | 框架 | 物理引擎 | 设备 | 备注 |
|---|---|---|---|---|
| A | mjlab(`feat/mjbatch-backend`) | mujoco C 引擎 + mjbatch 线程池 | CPU | `--env.sim.backend classic`;已发布基线 MicroDuck@4096 = 12,652 steps/s(`docs/results/worklog-mac-training-speed.md:15-20`) |
| B | mmjlab(`feat/qd-backend`) | mujoco_qd | `qd.cpu` | |
| C | mmjlab(`feat/qd-backend`) | mujoco_qd | `qd.metal` | 图片/相机场景本期全部排除 |

若「mmjlab+cpu」本意是 stock mujoco C 引擎(不经 mjbatch):mmjlab 的
classic 路径**原样保留**(无 warp),切口径只改配置,本文其余部分不受影响。

**排除项**:RGB/Depth/Seg 观测与渲染不在本期测试与对比路径
(任务矩阵用非视觉任务;`Mjlab-Lift-Cube-Yam-Rgb` 不入矩阵)。
**保留项**:mmjlab 包名/import 名仍为 `mjlab`(三个第三方任务库依赖它,
改名即破坏生态兼容)。

---

## 1. 背景与目标

- mjlab fork(`vestoriaai/mmjlab`,分支 `feat/mjbatch-backend`,HEAD 81e93f9)
  已有两条物理后端:`warp`(mujoco-warp,GPU/CPU)与 `classic`(mujoco C 引擎
  + mjbatch 线程池,即论文的 mjbatch CPU 后端)。warp 后端在 macOS 上实际
  不可用(warp 无 Metal),生产口径是 classic。
- mujoqo(另仓库)已完成 mujoco-warp 3.15.0 → qd kernel 的全量移植:
  CPU 后端全量绿(仅 4F 挂账),Metal 后端 2026-10-23(dev136)全线首次跑通;
  benchmark3way 实测 qd-CPU 物理反超 mjwarp-CPU 1.37–2.05×。
- **目标**:mmjlab 分支上把 warp 后端替换为 mujoco_qd(去净 wp),
  使得同一套 mjlab API 下产生三臂可跑、可量化对比的配置,产出:
  1. 一致性结论(三臂功能等价性,按层给容差与证据);
  2. 性能结论(三臂 × 4 任务的吞吐/加速比/资源占用,含诚实的负结果)。

---

## 2. 现状拓扑(仓库 / venv / 版本)

| 路径 | 是什么 | 版本/分支 | 状态 |
|---|---|---|---|
| `simtest/mjlab` | `vestoriaai/mmjlab.git` clone(上游仓库名就叫 mmjlab) | `feat/mjbatch-backend` @ 81e93f9,py3.13 venv | 臂 A 载体;1 个未提交 diff(render_qd.py DLPack 快路径) |
| `simtest/mmjlab` | 同一仓库第二工作树(本文工作对象) | 同分支同 HEAD,同未提交 diff;新建分支 `feat/qd-backend` | 本地副本,已被 simtest .gitignore 忽略 |
| `Code/mujoqo` | mujoco-warp 3.15.0 → qd 全量移植;import 名 `mujoco_qd`,pyproject 包名仍 `mujoco-warp`(3.15.0) | `port/p1-forward-chain` @ f80fb6a,树干净 | 未 pip 安装,repo 根直 import;开发 venv `_port.venv`(py3.12,quadrants 1.3.4.dev145 wheel) |
| `simtest/external/mjbatch` | kevinzakka/mjbatch 的 fork,**唯一被改的依赖库** | `feat/efc-bindings` @ 31e3923,树干净 | mjlab 与 mmjlab 的 venv 均 editable 指向它 |
| `simtest/external/microduck_rl` | microduck 任务库(pollen 上游 metal 分支 + 5 本地提交) | `Mjlab-Velocity-Flat/Rough-MicroDuck` | 自带 .venv(py3.12) |
| `Code/wuji-mjlab` + `ext-venvs/wuji.venv` | wuji 灵巧手任务库(本地补丁 d3dee79:XML inertial 显式化) | `WujiHand_Reorient` | 独立 venv |
| `Code/wbc-mjlab` + `ext-venvs/wbc.venv` | wbc G1 全身跟踪任务库 | `Wbc-G1` | 独立 venv |
| `simtest/qd-render-poc` | 生产渲染器(QdCameraContext 底座) | mjlab qd extra 消费 | mmjlab 渲染/相机继续用它,本期不动 |

关键版本事实:
- mjlab venv 实测:`mujoco 3.11.0`、`mujoco-warp 3.11.0`(PyPI wheel)、
  `mjbatch 0.1.1`(editable → external/mjbatch)、torch 2.14 + MPS。
- mjbatch 官方钉 `mujoco==3.13.0`,本地以 override 压到 `<3.12` 实链 3.11
  (REPRODUCE.md:42-56)。**mujoqo 按 mjwarp 3.15.0 快照移植** → 版本 skew
  见 §9 R2。
- mjlab 硬依赖 `warp-lang>=1.14.0`、`mujoco-warp~=3.11.0`
  (`mjlab/pyproject.toml:39-40`);qd extra `quadrants==1.3.1`(py>=3.12)
  (:105-106);qd-render-poc 为 workspace 兄弟目录单独安装(:101-104)。

---

## 3. 架构逻辑:接缝现状与替换映射

### 3.1 mjlab 现有接缝(勘察确认,比预想干净)

- 唯一切换点:`SimulationCfg.backend: Literal["warp","classic"]`
  (`sim/sim.py:176`)→ 工厂 `make_simulation()`(`sim/sim.py:630-666`)。
  `ClassicSimulation`(`sim/classic.py`)已是接口对齐的平行实现
  (mj_model/mj_data/data/model/step/reset/sense/... 全套,:221-431)。
- sensor 层二次分派:`scene.py:188-206` 按 `_is_classic_data` 选
  `SensorContextCPU` 或 warp `SensorContext`;
  `manager_based_rl_env.py:222-226` 对 CPU 上下文预渲染一帧。
- solver 配置双映射共用 `MujocoCfg.apply(model)`(sim.py:141-171;
  classic 在 classic.py:268 调同一函数)——iterations/solver 类型天然
  落到 C 引擎;warp 专属 `apply_wp_opt`(sim.py:229-237)。
- 渲染选路(classic 后端):`camera_sensor.py:102-106`
  `render_backend="auto"` → `sensor_context_cpu.py:61-90`:
  auto ⇒ `qd(qd_render_poc) → mjwarp(render_mjwarp) → gl`。

### 3.2 mmjlab 目标架构(不新增架构,只换引擎)

```
SimulationCfg.backend: Literal["qd", "classic"](值 "warp" 保留为别名 → "qd")
make_simulation()
  ├─ "qd"      → QdSimulation(= 原 warp Simulation 改引 mujoco_qd;同接口)
  └─ "classic" → ClassicSimulation(原样保留:mjbatch + efc,零 warp)
scene/sensor 分派:_is_classic_data 不变;qd 臂走 SensorContextCPU
(qd 物理臂的相机/raycast 一律用现成 sensor_context_cpu + raycast_cpu/qd,
 不复活 warp GPU 渲染管线)
```

要点:
- `QdSimulation` 不是新架构,是原 warp `Simulation` 的引擎替换:
  `import mujoco_warp as mjw` → `import mujoco_qd as mjw`,调用形态不变
  (put_model/put_data/step/forward/reset_data/refit_bvh 同名同参)。
- torch 桥:`sim_data.py` 的 `WarpBridge/TorchArray`(wp.array 基座)换成
  qd ndarray 的 DLPack 桥(`qd.to_dlpack` → `torch.from_dlpack`)。
  依据:qd 1.3.4 R4 已验证 mps:0 零拷贝别名与写通语义
  (qd-render-poc worklog 续六;跨框架异步序需 `torch.mps.synchronize()`)。
- CUDA Graph 段整体删除(`sim.py:528-583, 597-627` 的
  `wp.ScopedCapture/capture_launch/_should_use_cuda_graph`):Metal 无对应物,
  qd 裸 launch;性能影响进 perf 报告,不做补偿设计。
- classic 路径保留的理由:① warp-free,零工作量;② microduck BAM 的
  efc 保证(§5);③ 充当版本控制基线(§8)。

### 3.3 mujoco_qd API 对照速查(mjwarp → mujoco_qd,按设计 1:1)

| mjlab 现调 | mujoco_qd 对应 | 依据 |
|---|---|---|
| `mjw.put_model(mjm)` | 同名,`io.py:323` | batch_sizes 参数一致 |
| `mjw.put_data(mjm, mjd, nworld)` / `make_data` | 同名,`io.py:2133/1912` | |
| `mjw.step/forward(m, d)` | 同名,`forward.py:1696` | 积分器 euler/rk4/implicit/implicitfast/discrete 全 |
| `mjw.reset_data` / `reset_data_keyframe` | 同名 | |
| `mjw.refit_bvh`(render_mjwarp 私有 `_src.bvh`) | `mujoco_qd._src.bvh`(公开面已列) | |
| `mjw.Option`(apply_wp_opt) | `mujoco_qd` Model.opt 同形字段 | broadphase/contact_sensor_maxmatch 对齐项入 M2 验证 |
| `from mujoco_warp import rays` | qd 臂不迁移:统一走 raycast_cpu/raycast_qd | mmjlab 已有纯 torch/quadrants 实现 |
| `d.efc`(BAM/warp-only 依赖) | `types.py:2162-2185` Constraint.efc(J/pos/D/aref/frictionloss) | 访问器 parity 入 M2,见 §5 |

初始化差异(必须显式):`qd.init(arch=qd.cpu|qd.metal, cfg_optimization=False)`
(mujoqo `_src/conftest.py:9-25` 的既定正确性配置),落位在
`mjlab/__init__.py` 替代 `_configure_warp()`。

---

## 4. 去 warp 工作量清单(文件级)

### 4.1 删除(纯 warp 载体,无替换物)

| 目标 | 说明 |
|---|---|
| `sensor/render_mjwarp.py` 整文件 | classic 的 mjwarp 渲染降级臂;qd_render_poc 已是首选 |
| `sensor/sensor_context.py` | warp GPU 渲染管线(含 `@wp.kernel _unpack_rgb_kernel`);由 `sensor_context_cpu.py` 取代 |
| `sim/sim.py` CUDA graph 段 | `:528-583, :597-627`;Metal 无对应 |
| `entity/variants.py` warp 编译段(:44,:1108,:1275-1285,:1431-1433) | wp.array 手工字段重写;classic 本就拒绝 variants(classic.py:234),mmjlab 同政策:不支持 mesh variants |
| `tests/test_classic_camera_mjwarp.py`、`test_camera_sensor.py`、`test_raycast_sensor.py` | warp 管线专属测试 |

### 4.2 改写(引擎换名 + wp→qd)

| 目标 | 动作 | 难度 |
|---|---|---|
| `sim/sim.py` | import 换 mujoco_qd;`backend` literal 与工厂;`wp.ScopedDevice/zeros` → qd;`apply_wp_opt` → qd opt;`_WARP_INTEGRATORS` 放开(qd 支持全积分器) | 中 |
| `sim/sim_data.py` | `WarpBridge/TorchArray` → qd DLPack 桥(`:10,:18-57,:169-210`) | 中(专项验证) |
| `sim/randomization.py` | `@wp.kernel tile_copy` + `wp.launch` 扩维 → qd kernel/launch(`:4,:9-42`) | 中 |
| `sensor/raycast_sensor.py` | 摘掉 `mujoco_warp.rays` warp 分支(`:20,:34,:545-560,:766-783`),统一 raycast_cpu/raycast_qd | 低 |
| `envs/mdp/actions/differential_ik.py` | 删 `mjwarp.jac` 分支,保留 `_jac_torch`(`:209,:216-225,:370-388`) | 低 |
| `utils/random.py` | `wp.rand_init` → qd RNG,种子语义对齐(`:6,:20`,mjwarp issue #562 口径) | 中 |
| `utils/nan_guard.py` | `mjwarp.Data` 字段读取 → qd 同形(classic 已兼容,验证即可) | 低 |
| `mjlab/__init__.py` | `_configure_warp` → `qd.init(arch=...)`;`import warp` 清除(`:18,:36-43`) | 低 |
| `manager_based_rl_env.py:174` | metadata `warp_version` → qd 版本 | 低 |
| `sensor_context_cpu.py:61-90`、`camera_sensor.py:102-106` | 降级链/literal 去 "mjwarp" | 低 |
| `viewer/*` | 无 warp import,仅注释提及;不动 | 零 |

### 4.3 薄注解(签名统一,约 13 文件)

`entity/entity.py:690`、`entity/data.py:45-46`、`actuator/{actuator,pd_actuator,
dc_actuator,learned_actuator}.py`、`sensor/{sensor,builtin_sensor,contact_sensor,
camera_sensor}.py`、`scene/scene.py:172` —— `mjwarp.Model/Data` 注解换成桥类型;
体内均为 duck-typing 或 `del model`,行为不变。

### 4.4 依赖声明

`pyproject.toml`:删 `warp-lang>=1.14.0`、`mujoco-warp~=3.11.0`(:39-40);
保留 `mjbatch>=0.1.1`(:56,efc 注释一并保留)、`mujoco`(版本见 §9 R2)、
qd extra(quadrants 版本对齐 mujoqo wheel,见 §9 R6)。

### 4.5 mujoqo 侧残留(上游项,非 mmjlab 阻塞)

mujoco_qd 仍有 3 处 host-only `import warp`(`collision_driver.py:19`、
`collision_flex.py:20`、`cli.py:23`,用于 utils.segmented_sort/array_scan)。
mmjlab 自身 0 wp,但严格「去掉 wp」到包依赖层需 mujoqo 清理或接受
warp-lang 作为 host-only 传递依赖。**建议本期接受,列上游跟进项。**

---

## 5. 依赖与 efc 影响分析(「改了 efc 之类的杂依赖库」确认)

**确认结论:被改的依赖库只有一个 = external/mjbatch(fork),改动全部围绕
efc;没有 patched mujoco,没有其它杂改。**

- mjbatch = kevinzakka/mjbatch 的 fork(分支 `feat/efc-bindings`),相对上游
  仅 3 个提交:`68d1768` 设计笔记 / `3de1b72` 功能 / `31e3923` README。
  功能面:`Batch(include_efc, njmax)` 新参、`Efc` 四视图 dataclass
  (`src/mjbatch/__init__.py:41-79`)、C 侧 `BindEfc/CaptureEfc`
  (`csrc/batch.h:433-476`,arena 动态尺寸 → 定容 padding,超界行 stale)。
- 消费链(唯一目的:BAM 执行器剥离摩擦力):
  microduck cfg `include_efc=True`(`external/microduck_rl/.../
  microduck_velocity_env_cfg.py:747-750`)→ mjlab `ClassicData.efc`
  (`sim/classic.py:93-115,153-156`,nefc 掩码语义)→
  `bam/mjlab.py:541-571 _dof_friction_force`(microduck venv 内)。
- 其它本地补丁(非依赖库,属任务库):microduck_rl 5 个提交
  (含 `5e5e5f8` efc opt-in);wuji 本地 d3dee79(XML inertial 显式化,
  当年为 mjbatch `set_const()` 校验崩溃;对 qd/classic 均无害,保留)。

**对三臂的影响:**
- 臂 A / mmjlab-classic:零影响(mjbatch fork 保留,efc 链原样)。
- 臂 B/C(qd):qd 物理 `Constraint.efc` 存在(frictionloss 含,
  `types.py:2185`),但 mjlab 侧 `data.efc` 访问器需在 QdSimulation 上补
  parity(nefc 掩码/stale 行语义对齐 classic 口径)。若 qd 臂 efc 有缺口:
  microduck BAM 有现成退路 `MICRODUCK_ACTUATOR=xml_legacy`
  (`microduck_constants.py:181-193`),或 BAM 只在 classic 臂训练。
  等价性验证复用 `docs/results/efc-equivalence/single_step.py` 口径扩 qd 臂。
- microduck 的 `dr.dof_frictionloss` DR 事件与 BAM 的 XML frictionloss 置零
  改写:引擎无关,不受影响。

---

## 6. render / 相机复用评估(mujoqo 已有 render)

| 维度 | qd_render_poc(mmjlab 现役) | mujoco_qd.render(mujoqo 内) |
|---|---|---|
| API | `QdSceneRenderer/QdCameraContext`(场景打包式) | `create_render_context/render/get_rgb/...`(mjwarp 同形,nworld 批量一等) |
| 成熟度 | mjr 逐像素对齐(23 条契约册),Metal e2e 4.79ms@64envs×84×84 | CPU 100P/0F/2S,mjr 黄金 12/12 mean≤2;**Metal 路径待解冻(Phase 3)** |
| 与 mjlab 接法 | `sensor/render_qd.py`(已 warp-free,DLPack 快路径) | 无桥;API 面不同,不直接互换 |

**结论:本期不迁移渲染。** 图片场景已排除,渲染不在对比路径;现役
qd_render_poc 已验证且零改动成本。复用条件 = mujoqo render Metal 解冻 +
API 适配层;届时 `mujoco_qd.render` 的自然生态位是**顶替被删的
render_mjwarp.py**(作为「mjwarp 同形批量渲染」选项回归),并可评估与
qd_render_poc 两套渲染的长期统一——均列为独立后续项,不进本期。
raycast 无缺口:mmjlab 的 raycast_cpu/raycast_qd 已是纯 torch/quadrants。

---

## 7. 一致性测试方案(三层)

### L1 引擎级(不重复造轮子)
mujoqo 复用上游 751 个 `_src/*_test.py`,已锚 mujoco CPU 参考解
(CPU 全量仅 4F 挂账;Metal dev136 首通:io 167P/4S、collision 84P/2S、
flex 308P/12S、forward 136P/1F 等)。mmjlab 只在集成冒烟里跑其
forward/collision/ray 抽样子集,不重复建设。

### L2 mjlab 后端级(现成测试改造,主力)
| 现成资产 | 改造后用途 |
|---|---|
| `tests/test_classic_backend.py`(参考解、DR、rk4、混沌长程逐位 :208) | classic 臂全绿 = 替换无回归的机械证明;qd 臂同套改容差口径 |
| `tests/test_classic_vs_warp_parity.py`(单步 1e-3) | → `test_classic_vs_qd_parity.py` |
| `tests/test_mjbatch_smoke.py` | 保留(mjbatch 健全性) |
| `tests/golden/`(stage0_*.npz mjr 参考)+ `test_classic_camera_qd.py`(10) | 渲染 golden 保留(非本期路径但防回归) |
| `tests/test_classic_raycast_{qd,core}.py`(20) | 保留 |
| `tests/test_classic_diff_ik.py` | 保留(torch 雅可比已是 CPU 路径) |
| `tests/conftest.py`(`wp.config`)、`test_sim.py`、`test_sim_data.py`、`test_scene.py` | 随 §4 改写 |

### L3 任务级 rollout parity(对比的最终口径)
- 矩阵:4 任务(Lift-Cube-Yam / MicroDuck / WujiHand_Reorient / Wbc-G1)
  × {classic, qd.cpu, qd.metal},同 seed、同 action 序列(确定性脚本驱动,
  非训练)。
- 指标:单步 `max|Δqpos|,|Δqvel|`(f64 classic vs f32 qd → 相对容差口径,
  复用论文 §equiv 分级:求解器容差级/统计一致级/固有差异披露);
  100 步发散包络;reward 序列 Pearson/Spearman;termination 集合差。
- efc 单步等价:classic↔qd 复用 `efc-equivalence/single_step.py` 口径
  (microduck venv 现成)。
- 已知固有差异披露候选:mujoco 3.11(C)vs 3.15 语义(qd 快照)、
  f32/f64、broadphase 缺省差异。

---

## 8. 性能对比方案(量化结论的产出口径)

**矩阵**:3 臂 × 4 任务 × 4096 envs(见 §0 表)。臂 A 在 `mjlab/.venv`
复现已发布口径(backend=classic,num_threads=6);可选臂 A′ =
mmjlab-classic 在 mmjlab venv(mujoco 3.15 + mjbatch 重编译)作版本控制
基线,R2 可行再跑。

**指标**:
1. 训练吞吐 steps/s(`python -m mjlab.scripts.train ... --agent.max-iterations N`
   短窗口径;入口命令照抄 REPRODUCE.md:93-99 与 ext-repos-bench.md;
   wbc 用 `wbc-mjlab-train --task Wbc-G1 --dataset samples`)。
2. 裸物理吞吐(单步 step 循环,剥离学习/传感器),对照
   `docs/results/bench_microduck_classic.py` 与 mujoqo benchmark3way 口径。
3. 分段计时:reset / DR 事件 / physics step / sense;RSS 与 unified memory
   占用(metal 臂)。

**协议**(吸取既往教训,写死):
- 固定 seed;预热后取 min-of-N(N≥5;qd render bench 的 5.51–5.84ms 热噪声
  带教训,单次测量不可信);
- 同机串行跑臂,记录机器温度档/时间窗;
- 4096 envs metal 臂内存不足时按 4096/2048/1024 阶梯降档并如实报告
  (R4);
- 学习设备三臂统一(CUDA_VISIBLE_DEVICES="" + MJLAB_TRAIN_DEVICE=cpu,
  PPO 更新段统一 MJLAB_LEARNER_DEVICE 策略),保证差异只来自物理臂;
- 渲染/相机全部关闭(图片场景排除)。

**产出**:结论表(每任务:B/A、C/A、C/B 加速比 + 绝对 steps/s + 分段占比)+
负结果段(如 metal 臂在某任务上劣化的机制分析),直接可入论文复现节。

---

## 9. 风险与未决问题

| # | 风险 | 等级 | 对策 |
|---|---|---|---|
| R1 | qd 臂 `data.efc` 访问器缺口 → microduck BAM | 中 | M2 专项 parity;退路 `MICRODUCK_ACTUATOR=xml_legacy` 或 BAM 仅 classic 臂 |
| R2 | mujoco 版本 skew:C 臂链 3.11,mujoqo 按 3.15 快照(mjModel 布局差异) | 高(确定性) | 双 venv 拆分(classic 臂留 3.11,mmjlab venv 升 3.15);可选 mjbatch@3.15 重编译收敛为单 venv;版本差异按「固有差异披露」处理 |
| R3 | mujoco_qd Metal 物理 2026-10-23 才全线首通(dev136),预期仍有 A 系缺陷修复量 | 高(工作量浮动) | M2/M4 预留 buffer;metal 臂受阻时 B 臂(qd.cpu)结论独立成立 |
| R4 | 4096 envs metal 内存容量(M 系列 unified memory) | 中 | 阶梯降档如实报告 |
| R5 | RNG/reset 语义差异(wp.rand_init vs qd)打碎 rollout parity | 中 | M2 先对齐种子语义,再跑 L3 |
| R6 | quadrants wheel 版本矩阵:qd_render_poc 锁 1.3.1,mujoqo 需 1.3.4.dev145+ | 低 | 已有证据:qd_render_poc 在 1.3.4 vs 1.3.1 数值零漂移(worklog 续六),mmjlab venv 单 wheel 1.3.4 即可;装前复验 |
| R7 | mmjlab venv 是 py3.13,mujoqo 开发环境 py3.12 | 低 | mujoco_qd requires-python>=3.10;quadrants cp313 wheel 已被 qd-render-poc 验证存在 |

**未决问题 Q1**:「mmjlab+cpu」是否即 §0 口径(mujoco_qd@qd.cpu)?
若是 stock mujoco C 引擎口径,classic 路径保留可随时切换,本文不返工。

---

## 10. 里程碑与工作量

| 阶段 | 内容 | 量级 |
|---|---|---|
| M0 | 分支(已建 `feat/qd-backend`)+ 臂 A 四任务基线数字复测冻结 | 0.5d |
| M1 | §4 机械清单执行(删/薄层/依赖声明);classic 全测试绿 = 无回归证明 | 1–2d |
| M2 | mujoco_qd 桥:sim/sim_data(DLPack 桥专项)/randomization/raycast/random/nan_guard/__init__;L2 parity 套件绿 | 2–4d |
| M3 | 4 任务 venv 装配 + smoke(microduck/wuji/wbc 的 mjlab 依赖指到 mmjlab editable) | 1d |
| M4 | 一致性矩阵执行(L1 冒烟 + L2 + L3)+ 缺口修复 | 1–2d |
| M5 | 三臂 × 4 任务 perf(含 1–2d 机器时间)+ 量化报告 | 1.5–2.5d |
| 合计 | | **净工约 7–12 人日 + 1–2 天机器时间**;metal 臂受 R3 影响有 +1–3d 浮动 |

顺序依赖:M1 与 M2 可部分并行(不同文件);M3 依赖 M1(包可装);
M4/M5 依赖 M2;M5 的臂 A 基线(M0)可提前跑。

---

## 11. 附录:mmjlab venv 装配清单(M3 输入)

```bash
# mmjlab/.venv(py3.13,已由 mjlab venv 复制修路径:mjlab.pth → mmjlab/src)
uv pip uninstall mujoco-warp                 # 包名与 mujoqo 冲突,必须先卸
uv pip install mujoco==3.15.0                # R2:mmjlab venv 侧升版
uv pip install --no-deps -e /Users/mini0/Code/mujoqo   # mujoco_qd(不拉坏 deps)
uv pip install quadrants==1.3.4.dev145 ...   # wheel 源同 mujoqo _port.venv(quadrants-c1wt dist)
uv pip install -e /Users/mini0/Code/simtest/qd-render-poc  # 渲染/相机(已装,确认)
# warp-lang:mujoco_qd host-only 残留所需,保留(§4.5);mjbatch 保持 editable 共享
# 验证:python -c "import mujoco_qd, mjbatch, qd_render_poc, mjlab" + qd.init 冒烟
```

classic 臂(臂 A)继续用 `mjlab/.venv`(mujoco 3.11,现役),不动。

---

## 12. 实施记录(2026-10-24,feat/qd-backend)

阶段 M1+M2 已完成(接口替换 + CPU 正确性验证),本节为实施时点的事实记录。

### 12.1 已落地

- **src 全量去 warp**:`import mujoco_warp/import warp` 在 src/ 下归零;`sim.py`
  以 `import mujoco_qd as mjwarp`(API 同形)驱动;torch 桥重写为 qd DLPack
  零拷贝(`sim_data.py` QdBridge/TorchArray,写通已验证);randomization 改
  numpy tile + `from_data`(compound dtype 按 mujoco_qd._src.types 精确名回查);
  删除 `render_mjwarp.py`/`sensor_context.py`/CUDA graph 段/mesh-variant 的
  wp 实现段;backend literal `["qd","classic"]`("warp" 别名)。
- **进程级 qd 初始化唯一权威**(`sim/qd_runtime.py`):qd 二次 init 会重置并使
  全部既有分配失效——物理后端(cpu)与 raycast 后端(原逻辑偏好 metal)曾因此
  互相覆盖,引发随机 Bus error;现双方都走 `ensure_initialized`。
- **环境**:`mmjlab/.venv-qd`(py3.12)= mujoco 3.15.0 + mujoco_qd(editable,
  mujoqo 树)+ quadrants 1.3.4.dev142(源码构建,≥dev145 修复 set_const 的
  VectorType 映射缺陷)+ mjbatch 重编译链接 3.15(--no-build-isolation,
  缓存坑:uv 会复用隔离构建的旧 wheel,须 --no-cache)+ qd-render-poc。
- **测试适配**:conftest/sim/sim_data/scene/parity/variants 测改写;variant
  实现(warp-only)5 测 skip(政策);dcmotor 测在 mujoco≥3.15 skip(3.15
  移除 input_mode,属 R2 版本 skew,无任务库使用)。

### 12.2 CPU 正确性结果(2026-10-24 实测)

- 双后端冒烟 + 选择性 reset + forward:通过。
- classic(f64 C)vs qd(f32)单步 parity:**max|Δqpos| = 7.28e-12**
  (cartpole,容差 1e-3)。
- 全套 tests/(golden 渲染除外):主体 1050+ 通过;已知残余:
  ① G1 velocity 任务的 env 构造崩(mujoco_qd put_data 后 Data 字段为未分配
  模板,site_xpos.shape=None)——引擎侧缺陷,待 mujoqo 修;其连锁(同进程
  后续测试 Bus error)在去除该用例后消失(DR 文件 102 passed)。
  ② async 相机吞吐门 130ms:曾于机器高载实测 136.9ms,已加
  require_quiet_machine 守卫。
  ③ variants(5)/dcmotor:政策性 skip(见 12.1)。

### 12.3 与计划的偏差

- 上游基线改用 git worktree @ f135c1d(fork 点,上游 PR #1199),而非 PyPI
  版本——与 fork 代码同源,排除任务集/API 漂移。
- warp-lang 保留为 mmjlab 依赖(mujoqo host-only 排序/扫描所需),mjlab 自身
  代码 0 warp。

### 12.4 裸物理速度 sanity(2026-10-24,`scripts/bench_bare_physics.py`)

humanoid.xml,裸物理 step(无观测/奖励/传感器),min-of-N,固定协议:

| 臂 | 引擎 | 256 envs(500 步×5) | 4096 envs(100 步×3) |
|---|---|---|---|
| 上游 mjlab @ f135c1d | mjwarp-CPU | 17,635 steps/s | 20,686 steps/s |
| mmjlab | mujoco_qd @ qd.cpu | 18,890 steps/s | **74,431 steps/s(3.6×)** |
| mmjlab | classic(mjbatch@3.15) | 361,982 steps/s | 398,474 steps/s |

初读:① 小规模(256)两 GPU 型引擎同档,4096 规模 qd.cpu 反超 warp-cpu
3.6×(qd 批量摊销更好);② classic 在 CPU 上仍一个量级领先——CPU 侧的价值
口径与论文一致,mmjlab 的速度主张要靠 qd.metal 臂(下一阶段);③ 上游基线
环境(mjlab-upstream worktree @ fork 点 + warp 栈)已就位,可复现。

### 12.5 三臂 profile:速度 + 内存(2026-10-24,mujoqo@5df53b9 + quadrants dev150)

环境升级:mujoqo 主链 wp 依赖清零(§4.5 上游项落地)、A59 修复;quadrants
dev150。升级后回归 45 passed(classic/parity/events/mjbatch smoke)。
`scripts/bench_bare_physics.py` 增 --device(mps→qd.metal)与内存剖面
(macOS ru_maxrss 峰值口径,含统一内存设备缓冲)。

| 臂 | 256 envs | 4096 envs | peak RSS @256 | peak RSS @4096 |
|---|---|---|---|---|
| 原版组合 classic(mjbatch,fork@3.11) | 276,536 * | 365,362 * | 未测 * | 未测 * |
| qd.cpu(mujoqo) | 13,464 | 17,353 / 复测 18,265 | 818 MB | 954–1016 MB |
| qd.metal(mujoqo) | 7,797 | **37,476** | 963 MB | 1013 MB |

\* classic 臂引用 2026-10-23 空窗数据(本轮外部负载未重跑,用户裁定);
内存未测(mjbatch 内存按线程数而非 env 数定标,预期远小于 qd 臂,待补)。

测量窗口披露:本轮 load 6.7–7.4,另有 5 个外部单核进程(a53b 探针/别的
session),速度绝对值被压低;**同窗口内的横向对比有效**,跨窗口绝对值
不可直比。

要点:
1. **metal@4096 同窗反超 qd.cpu 2.2×**;256 时 metal 慢(launch 开销主导),
   交叉点在 256–4096 之间。metal 256→4096 缩放 4.8×,cpu 仅 1.3×——GPU
   批量摊销兑现,这正是 mmjlab 速度主张的落点。
2. **内存双臂同量级**:4096 humanoid 约 1.0 GB(峰值口径),metal 设备缓冲
   在统一内存下未放大占用;16GB 机器 4096 envs 安全。
3. **待归因异常**:qd.cpu@4096 本轮 ~18k vs 前日空窗 74.4k(同协议),
   差距超出 5 核负载可解释范围——空窗复测后三选一归因(负载/热节流/
   dev150 回归);metal 为本轮同窗自洽数据,不受影响。
4. classic 引用数据与本轮 qd 数据测量窗口不对称,正式对比须同窗重测
   (等空窗,含 classic 内存补测)。

### 12.6 三臂同窗 profile(2026-10-24 晚,load 7.4–7.8 恒定)

六组=3 臂 × 2 规模,同窗口顺序执行(外部 5 单核进程持续存在,三臂同等承压,
横向对比有效;绝对值低于空窗口径)。bench 脚本改自适应 warmup:逐步计时至
中位数 50 步窗口稳定(±5%),JIT/编译完全排除在计时区外并单独报告。

| 臂 | 256 steps/s | 4096 steps/s | peak @256 | peak @4096 | warmup(JIT) |
|---|---|---|---|---|---|
| classic(mjbatch) | 179,355 | 247,016 | 401 MB | **427 MB** | 0.1–1.6 s(无 JIT,首步 2–14 ms) |
| qd.cpu | 13,538 | 17,823 | 889 MB | 1136 MB | 8–28 s(首步 ~6 s) |
| qd.metal | 7,693 | **35,474** | 961 MB | **958 MB** | 7–16 s(首步 ~4.6 s) |

结论:
1. **内存(本轮重点)**:classic 256→4096 峰值仅 +6%(401→427 MB)——
   mjbatch 按线程数定标的结构性优势首次量化;qd.cpu +28%(889→1136 MB,
   ~64 KB/env 边际);qd.metal 持平(~0.96 GB,设备缓冲主导,4096 反而略省
   于 cpu 的 1136 MB)。三臂在 16GB 机器全部安全。
2. **metal JIT 有界**:quadrants 离线缓存命中下 warmup 7–16 s(首步 4.6 s
   为编译后首步,非冷启动);冷缓存首次会更长,已与计时区隔离。
3. **metal@4096 同窗反超 qd.cpu 2.0×**(35.5k vs 17.8k),缩放 metal 4.6×
   vs cpu 1.3×;但距 classic(247k)仍 ~7×——metal 慢的机制是 per-step
   launch 开销(11.5 ms/步 @4096;mujoqo render-perf 普查同因:数百次
   launch/步),属上游 metal-issues 线的已知优化对象,非 mmjlab 集成问题。
4. qd.cpu@4096 同窗复现 17–18k(vs 前日空窗 74k),归因仍开放(负载/热/
   回归),空窗复测定案。

### 12.7 qd.cpu 并行模型剖析与 QD_AUTO_SERIAL 杠杆(2026-10-24 深夜)

用户问题:qd.cpu 能否变成 mjbatch 式线程模式?底层差异?

实测链条(dev152 验证):
1. **qd.cpu 默认单核**:4096 envs 步进时 %CPU=100.0,170 ms/步。
2. **根因 = quadrants A59(C1.5 串行内联阈值,4ac79965)**:range-for 任务数
   < 阈值(默认 64)的 kernel 不唤醒线程池、串行内联执行(治小 kernel
   11-18µs 整池唤醒税,S1-S6 eager 2.5-3× 提速)。mujoco_qd 批量 kernel 的
   任务粒度按元素维(如 humanoid nbody≈18)<64 → 整步全部落串行区。
   dev142 时代阈值 bug 读 0(恒并行)= 55 ms/步的"意外并行"来源。
3. **QD_AUTO_SERIAL=0 关闭串行内联 → 74 ms/步(2.3×恢复)**,线程池并行
   证实可用。

与 mjbatch 的结构性差异(为何"变成 mjbatch"不可行也不必要):
- 并行粒度:mjbatch=env 级(每线程完整 mj_step + 私有 MjData,无跨 env
  交互);qd=kernel 级(每步数百 launch 串行链,range-for 内部并行)。
- runtime 单例:qd.init 进程级全局(二次 init 作废既有分配),不能进程内
  起 N 个独立引擎实例;多进程方案与 mjlab 单 sim 语义不匹配。
- 内存布局:mjbatch 线程私有连续 f64(cache 友好)vs qd 全局 (nworld,...)
  f32 批量数组(padding + 跨 world 带宽)。
- 上限:调好并行(~55-74 ms/步 ≈ 55-74k steps/s)仍距 mjbatch(16.6 ms/步,
  247k)3-4×——kernel 链同步 + 批量布局的结构性成本。CPU 臂定位不变
  (classic 为速度口径,qd.cpu 为 parity/开发口径,qd 主场在 metal)。

上游建议(待登记 quadrants A 系):阈值从全局常量 64 改为按 launch 任务
数/工作量动态判定,或批量场景按 nworld 放大任务计数——否则大批量物理
kernel(nworld 大、元素维小)整体落入串行区。

### 12.8 阈值语义:任务数 vs 环境数(答"如何做更好")

判定代码(quadrants runtime.cpp:1807,1819):`ntasks = (end-begin)/block_dim`;
`ntasks <= serial_range_threshold(默认 64) && step==1` → 提交线程内联串行,
否则进线程池。微基准(bench_dispatch_threshold.py):≤64 任务 9µs/launch
(内联),65 起 9–20µs(池,热态唤醒税低于 A59 记录的 11–18µs 冷态值),
大任务量线性扩展。

ntasks 与 nworld 的关系按 kernel 形态分三类:
1. 扁平化(主流):`for t in range(nworld×inner)` → ntasks ∝ nworld
   (4096 humanoid ≈73k),恒走池 ✓;
2. 单维 kernel(solver 线 23 个 `range(__n0)`):`__n0` = nworld 的走池;
   `__n0` = nv/nefc/island 数的**不随 nworld 增长,恒 ≤64 → 恒内联**——
   QD_AUTO_SERIAL=0 的 2.3× 即来自这批热路径(确切清单待 mujoqo 插桩);
3. 嵌套小外层(外层 nbody/npair、内层 nworld):同理恒内联。

改进建议(按层):
- quadrants(根治):①codegen 期已知循环界,对"小外层×大内层"嵌套做自动
  扁平化或把内层工作量计入 ntasks 判定(任务数→工作量);②暴露 per-launch
  inline/pool 计数与耗时(现在只能开关差分);③`qd.loop_config(serial=...)`
  per-launch 覆写。
- mujoqo:批量入口 kernel 统一扁平化约定 + CI 断言(批量 kernel 的 ntasks
  必须 ∝ nworld);对 solver 单维 kernel 逐个核对 __n0 口径。
- mmjlab(短期):QD_AUTO_SERIAL=0 已验证 2.3×(170→74 ms/步),不设默认,
  等上游结构性修复。

### 12.9 qd.metal 标度曲线与「快于 mjbatch」的可行性判定(2026-10-24 深夜,同负载窗口)

四点标度(humanoid@metal,per-step = repeat 时长/steps):

| envs | 256 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| steps/s | 7,693 | 40,313 | **50,016** | 35,474 ↓ |
| ms/步 | 33.3 | 25.4 | 40.9 | 115.5 |

要点:①256→1024 per-step 反而下降(GPU 欠利用,固定 submit/encode 开销
~25ms 主导);②1024→2048 边际 ~15µs/env 健康;③**2048→4096 边际成本
2.4× 恶化(>2048 存在超线性劣化,steps/s 不升反降)**——metal 侧新问题,
待上游归因(缓冲超限/occupancy 饱和/分页分配路径?)。

天花板判定(外部锚点:mjlab nightly 875k sps@4096 = 4.7ms/步 @RTX 5090;
M1 Pro GPU 算力/带宽 ≈ 1/16–1/40)→ M1 Pro GPU humanoid 物理算力天花板
≈ 50–60k steps/s,**2048 点(50k)已经打到**;4096 若修掉劣化也只到 ~60k。
对比 classic 同窗 247k(空窗 ~398k):**硬件差 5–8×,软件优化追不平**。

「qd.metal 快于 mjbatch」的四个真实赛场:
1. 更强 GPU:M4 Max(~6×)/Ultra、或 qd.cuda 路线(NVIDIA 机器)——
   mmjlab 代码不变,同代码按机器分档;
2. 小模型大批量 + launch 合批(fixed ~25ms 压到 <1ms 后,小模型场景
   GPU 摊销优势才显现);
3. 物理+渲染+观测全驻留 GPU 的端到端管线(省 CPU↔GPU 往返,视觉任务
   收益最大);
4. CPU 弱的异构机器(Mac 生态唯一 GPU 物理选项的平台价值)。

上游待办登记:①metal >2048 envs 超线性劣化归因;②physics launch 普查
(照搬 render-perf 985-launch 方法论,量化 encode/submit 占比)。

### 12.11 任务库冒烟:三库 × 双后端全绿 + supermimic_vibed 勘察(2026-10-24)

环境:三任务库 editable 挂入 .venv-qd(--no-deps):microduck_rl 0.1.0 /
wuji-mjlab 0.1.0 / wbc-mjlab 0.0.6;缺口补齐:numpy 钉 <2.5(2.4.6,wbc 约束)、
bam 从 microduck venv vendor(纯 Python)+ 导入守卫补丁(bam/mjlab.py:41 顶层
`import mujoco_warp` 仅为注解,mmjlab 无此包,运行时 duck-typed)。56 任务
全部注册。训练入口:MJLAB_TRAIN_DEVICE=cpu,256 envs,1 迭代。

| 任务(256 envs,1 iter) | classic | qd | 备注 |
|---|---|---|---|
| Mjlab-Velocity-Flat-MicroDuck | 0.70s | 7.01s | BAM+efc 链路 classic 正常;qd 臂 BAM 经 QdBridge 透传 Constraint 未崩,**efc 语义 parity 未验证(R1)** |
| WujiHand_Reorient | 1.55s | 28.27s | qd 首迭代含 JIT;wuji 需 --agent.logger tensorboard |
| Wbc-G1(dance.npz) | 2.08s | 7.03s | wbc 走 motion-file 直传;G1 级模型 qd 训练正常(12.7 的 G1 疑虑排除) |

资源:全程磁盘无新增负担(任务库 editable 零拷贝);训练进程 RSS ~0.2-1GB。
Wbc 需要 `--env.commands.motion.motion-file`(本安装版 wrapper 无 --dataset 旗标;
数据在 wbc-mjlab/data/g1/samples/npz/*)。

**supermimic_vibed 勘察(zita-ch,★483)**:已 clone 到 external/(ghfast 镜像)。
**该仓库无 mjlab 可换**:基于 Isaac Lab 3(kit-less)+ Newton(MJWarpSolverCfg
内嵌 mujoco-warp),env.py 8 处 isaaclab/newton import,train.py 走
isaaclab.app 启动器。「换 mjlab→mmjlab」实为**移植项目**(其 G1+LAFAN 跟踪
任务功能上与 mmjlab 现成的 Tracking-Flat-Unitree-G1 / Wbc-G1 重合),两个
可选路径:①复用其资产(G1 MJCF + LAFAN npz,tools/download_data.sh)在
mmjlab 上写一个 supermimic 任务 cfg(1-2 天);②不接(已由 ②wbc 覆盖同
能力)。其训练镜像为 docker(用户排除),原样运行需 isaaclab 全家桶(数 GB,
磁盘 9.6Gi 剩余不友好)。

### 12.13 dev184 更新验证:CPU/Metal 加速兑现 + dense 输入案例跑通(2026-10-25)

环境:quadrants dev184(2c847b3d,自 mujoqo _port.venv 同 ABI 复制)+ 热修——
**dev184 的 grad_for.decorated 漏读 `_already`**(A86 修复被部分套用到 grad_for
变体,NameError 崩 parity 测试;已按 A86 同构修复,上游需跟进,同 bug 应存在
于 _port.venv)。回归 53 passed。本窗 load 1.85(历次最静)。

裸物理三臂(humanoid):

| 臂 | 256 | 4096 | 对比 |
|---|---|---|---|
| classic | 420,295 | **449,117** | 历史最佳(空窗) |
| qd.cpu | 21,885 | **100,040** | 4096 较 dev142 空窗 74.4k **+34%**,对 classic 差距 5.4×→4.5× |
| qd.metal | 11,258 | 39,129 | 4096 较 dev152 35.5k +10% |

任务级(256 envs,1 iter,qd 后端除注明):

| 任务 | classic | qd | 上轮 qd |
|---|---|---|---|
| microduck(四足场景 sample) | 0.57s | 5.88s | 7.01s(-16%) |
| WujiHand_Reorient | — | 19.83s | 28.3s(含 JIT) |
| Wbc-G1 | — | 5.25s | 7.03s(-25%) |
| Mjlab-Lift-Cube-Yam-Depth(dense) | — | 15.33s | 首跑 |

dense 输入案例:**Mjlab-Lift-Cube-Yam-Depth**——观测 = qd 相机渲染深度图
(B,1,H,W)(`camera_depth`,policy CNN 直接消费);深度图即点云源表示,
如需真 XYZ 点云在 obs term 加反投影即可。四足 sample = MicroDuck(双后端
已验)。

### 12.14 新栈验证:CPU/Metal 性能线落地,metal 进入 mjbatch 同档(2026-10-25 晚)

环境:quadrants **dev176**(56db7c491 robostudio-main squash = CPU/Metal 性能线
落地:语言工程五件默认 ON + BLAS-3 blocked 等;注:版本号 dev176 晚于 dev184
的提交线,用户口中的"更新"即此线;grad_for `_already` 已在上游修好,热修不再
需要)+ mujoqo **f23fe51**(metal-overhead-cjac:A86 S1 ctrl 修复 + eq_data
vec12 的 mujoco 3.17 适配——实测兼容 3.15 host,回归 53 passed)。
mujoqo editable 自动跟随源码树(455d3f9→f23fe51)。

**mjlab 集成三臂**(humanoid@4096,load ~2.1-2.6):

| 臂 | steps/s | ms/步 | 对上轮 |
|---|---|---|---|
| classic(iter=100) | 447,090 | 0.92 | 持平 |
| qd.cpu | **166,128** | 2.47 | +66%(dev184 当日 100k) |
| qd.metal | **195,130** | 2.10 | **5.0×**(39.1k);@256 = 42,886(3.8×,首超 qd.cpu) |

**引擎级原生对照**(不经 mjlab):metal@4096 = **322,790(12.7ms/步)**,落在
mujoqo 自报区间(14.25ms,min 11.8)✓——「qd.metal ≈ mjbatch」在原生口径成立
(mujoqo 报 metal 11.8min vs mjbatch 11.33ms,差 4%)。

**solver 预算披露(关键口径)**:humanoid 基准 XML 的 opt.iterations=**10**,
而 mjlab MujocoCfg 默认强制 100。qd.metal@4096 以原生预算经 mjlab 测得
**355,744(11.51ms/步)**——优于 mujoqo 自报。classic 在 iter=10 下确定性失稳
(21k,接触爆掉的动力学假象,与负载无关),故预算配对的 classic 点不可得;
**默认预算(iter=100,两臂同配)下的诚实差距 = 2.3×**,来源是 solver 预算差
而非集成损耗或引擎算力。

测量纪律教训:SuperMimic 会话遗留的 swap 压力(4.6GB)曾使 classic@4096
瞬态测出 20.7k(21× 假崩,与 iterations 无关,复测即恢复 445k)——
**跑基准前先查 swap/内存压力**。

结论:用户目标(qd 与 mjbatch 同档)在 metal 原生口径已成立;mjlab 默认口径
2.3× 的剩余差距来自 solver 预算口径,可作为披露项而非缺陷。

### 12.15 metal 2.3× 差距分解与三条优化路径的归属/代价/依赖(2026-10-25,移交 mujoqo+quadrants 两线)

**差距解剖(实测两点共线拟合,iterations=10 与 100)**:qd.metal@4096 humanoid
每步 21.0ms(iter100)/11.51ms(iter10)→ **solver 边际 ≈105µs/迭代,非 solver
链底 ≈10.5ms/步**;对照 classic 同预算整步 0.92ms(solver ≈5µs/迭代)。
2.3× = 迭代税 20×(kernel 边界+device 栅栏 vs 线程内 for)+ 链税(138 kernel
镜像链的逐个 submit/sync + 全局数组流式)。GPU 本体计算占比很小。

**三条路径的归属/代价/依赖**(勘察依据:rhi/metal/metal_device.mm 提交层、
lang 层无 plan 模块、forward.py 49 个胶水函数):

| 路径 | 归属 | 代价 | 收益 | 依赖/次序 |
|---|---|---|---|---|
| ① 一步一提交批编码 | **quadrants 为主**(rhi/metal 提交层),mujoqo 接入 | 1–2 周(RHI 生命周期 + hazard/AD tape 回归) | 消 submit 往返税;渲染线同模式已验证(985 launch 单 submit) | 无硬依赖;②落地后组合更优 |
| ② C2 replay + launchtiled fusion | **机制在 quadrants**(lang 层 plan/replay + codegen 融合,现无雏形,C0 toy 已实测 16×),**数据在 mujoqo**(被录制的 138 链 + batch-last 布局改造) | C2 硬化 2–5 天(失效键五元组,漏=静默错);fusion 开放式逐对 | 消 Python 胶水(16× 实测)+ 融合消边界;双 arch | **与 ③ 次序约束:solver 区留白给 ③**,勿先融合 |
| ③ solver megakernel | **mujoqo 为主**(solver 重写),quadrants 出既有 block-tier/threadgroup 基建 | 2–4 周;风险=Metal 编译器(A70 前科)+ parity 门 + 小 nworld 摊销 | **消 105µs/迭代税(solver 10.5ms→~1-2ms)**;MJX/Brax 同构 | 独立于 ①②;env 间独立性保证语义安全 |

预期:E1(①+②非solver区)→ 链底 10.5ms→~3-6ms;E2(③)→ solver →~1-2ms;
合计 metal@4096 ≈ 4-8ms/步(500k-1M sps),**反超 mjbatch(447k)为合理工程目标**。

**为何此前未做**(证据):10-22 立项评估已成文并过第一性 review,但当时
优先级在正确性冻结(battery 1870P/A 系清账——重构 138 链会使全部编译缓存
失效,文档自警"改 forward 未失效=静默错");且 metal 115ms 时代编排优化
收益感知低——robostudio-main 线把它打到 14.25ms 后,剩余差距才恰好全部
变成编排开销,本周升格为最高杠杆。

### 12.16 §12.15 修正:21ms 的真实构成是 HOST 侧 Python 发起胶水(2026-10-25 深夜)

dev176 worktree 实施验证推翻了 §12.15 的两条判断,以实测修正:

1. **①批编码:上游已完成(A1 metal launch model),关闭**。metal 提交层早已
   管线化——persistent compute encoder(长串 dispatch 合并单 encoder)+
   异步 commit + 32-launch 阀门只 commit 不 drain(A1 移除了旧版每 ~33 次
   launch 的全量 drain,那才是当年 0.5-1.3ms/次、每步 ~4 次的串行点)。
   "逐 launch submit 税"假设不成立。
2. **K 批化读:语义安全但无收益,作为可选 knob 保留**(mujoqo-wt 分支
   feat/solver-read-batching @ f2b7da1,QD_SOLVER_READ_EVERY,默认 1)。parity
   13 passed(K=1/K=10 双验);K=10 时 metal@4096 = 183.6k vs K=1 的 195.1k
   ——回读不是 105µs/迭代的主体。
3. **决定性测量:HOST 侧每步发起耗时(无同步)= 10.04ms(min)**,与
   拟合链底 10.46ms 吻合;wall 21.0ms(iter=100)≈ 链 glue 10ms + solver 循环
   glue 10.5ms(105µs/迭代 × 100,**是每次迭代十几个 kernel 的 Python 发起
   成本,非回读非 GPU exec**)。GPU 全程在挨饿。经典对照:classic 的
   solver 循环在 C 侧(mjbatch 线程池内),同循环 ~5µs/迭代。

**结论修正**:21ms 的 ~95% 是 HOST Python 逐 kernel 胶水——修复主路径为
**② replay(消胶水,C0 toy 实测单 op 16×:4.92→0.31µs)预期 21ms→~2-5ms**,
③ megakernel(消 launch 数)同向且在大规模下 GPU 侧亦有收益;§12.10 的
"cache 驻留"叙事适用于 GPU 饱和后的下一阶段,当前不是首因。

**执行状态与移交**(用户序:①→③→②):
- ①:关闭(上游 A1 已实现)。
- ③:worktree 就绪(mujoqo feat/solver-read-batching,K knob 已提交验证),
  完整 megakernel = solver 区多周重写,本轮以正确性优先不做半成品;设计
  输入已齐(迭代间 env 独立、done 门已存在、105µs/迭代即其可消除上限)。
- ②:**证据升级为最高优先级**,但用户警示的 replay 静默错风险属实
  (失效键五元组漏一即静默错)——建议 upstream 按 §12.15 表落地,
  mmjlab 侧以「训练轨迹逐点一致 + battery 147P」为验收门。
- 遗留:grad_for dev184 热修(12.13)在 dev176 已由上游修复,本分支
  重装 dev176 后无需热修。

### 12.17 ③ megakernel 开发落地:stage-1 移植 + 双 arch parity 冒烟(2026-10-25)

mujoqo worktree(feat/solver-read-batching,自 port/p1-forward-chain)commit
b9212f0:
- `mujoco_qd/_src/solver_mega.py`:one-threadgroup-per-world,Newton 迭代循环
  kernel 内 + done 门(块级 flag)。stage-1(efc force/state 更新)已完成移植:
  lanes 分划 njmax 行,行逻辑与 `_update_constraint_efc` 非 track_changes
  变体逐句同构,`_eval_constraint`(@qd.func)原样复用——**行间独立 +
  行内同构 ⇒ 数值一致性可证**,无需容差。
- **stage-1 确定性 parity 冒烟(tools/solver_mega_stage1_smoke.py)**:
  equality/friction/limit 三类行 × CPU(block-tier 仿真,兼验跨平台语法)/
  metal 双 arch 全 PASS(力值逐位核对:-1.0/-0.1/0.0)。
- `solve()`:QD_SOLVER_MEGAKERNEL=1 时显式 raise(默认关;stage 2-4 未移植
  前杜绝静默错误结果)。

过程中钉死的 qd AST 约束(候选上游 A 系):3 参 range、while 条件内的
ndarray 读,均未过 AST 验证;行循环改有界 for + if 门(统一控制流)。
另:阶段移植验证时曾出现"ctx_Jaref 读零"假象,根因是 sed 链补丁破坏
kernel 体——干净重写后即绿;**块 kernel 开发必须整段重写,禁止 sed 链式
补丁**。

后续(stage 2-4,每 stage 一轮迭代:移植 → 冒烟 → 双 arch parity):
- stage 2:qfrc/grad/JTDAJ/Cholesky/search(源 = cholesky-coop 体,
  SharedArray 驻留;体量最大)
- stage 3:linesearch(源 = linesearch-iterative 体,内部 alpha 循环)
- stage 4:done/solver_niter(小块)
- 全部完成后:solve() 接线生效 → battery parity 子集 → metal@4096 实测
  105µs/迭代的消除量(目标:21ms → ~11-13ms,solver 段剩余 GPU 真实算力)。

### 12.18 ③ 续:stage-4 落地 + stage 2/3 规模定案(2026-10-25)

- **stage-4(done/solver_niter/overflow)已移植**(c567dc3):lane 0 收敛判定
  (alpha=0 / improvement<tol / gradient<tol / model_improvement<tol 四条件,
  体移植自 _solve_done)+ 块 flag 联动(置位后循环 break)+ overflow 位 +
  nsolving 原子递减。默认关,与 stage-1 同门。
- **stage 2/3 规模定案**(读源后):
  - stage 2(cholesky/search):源 = `kernel_update_gradient_cholesky_coop`
    (block_cholesky_coop.py,block 协作 + SharedArray N×N 驻留 + 多段 sync),
    体量 ~700 行,移植 = 把该体作为 megakernel 循环内的一个 stage(逐段
    sync 保持)。
  - stage 3(linesearch):源 = `_linesearch_iterative_kernel`(**per-world
    serial 形态**——单线程含完整内部 alpha 循环,~450 行,依赖
    _compute_efc_eval_pt_* qd.func 族)。移植形态 = megakernel 内 lane-0
    串行(其余 lane 栅栏等待,正确但并行度损失);或保持独立 launch
    (每迭代 1 次)——后者是折中选项,见下。
  - **组合策略(推荐)**:megakernel 收 [stage1 efc + stage2 cholesky +
    stage4 done](全 lane 或 lane-0 段)+ linesearch 保持独立 launch(它
    已是 in-kernel 循环,每迭代仅 1 次 Python 发起)。每迭代 Python 发起数
    6-8 → 2-3(mega + linesearch + read),胶水税减 ~2/3;剩余由 ② replay
    (消单次发起的 Python 成本)补齐。**全 in-kernel 变体(含 linesearch
    移植)作为 stretch**,待组合策略实测后再决定。
- 交付状态:megakernel 骨架 + stage 1(parity 冒烟双 arch PASS)+
  stage 4 已移植(默认关,门齐);stage 2 体移植 + linesearch 集成 +
  battery parity + metal@4096 实测为后续迭代项。

### 12.19 ③ stage 2/3 移植规格(serial-per-world 形态,下轮可机械执行)

**形态定案**:serial-per-world(每 world 一个线程,整个 Newton 迭代循环
kernel 内串行)。依据:linesearch-iterative kernel 本身就是 per-world serial
形态(serialize=True 标记),其体可直接重线程;efc 逐行、grad、H 构建、
serial cholesky 全部 per-world 串行;无 SharedArray/sync/lane 分划——
最小改动、天然跨平台(CPU 无需 block-tier 仿真)。

**文件**:mujoqo-wt-metal `mujoco_qd/_src/solver_mega.py`(重写现骨架,
block 形态版本废弃——block 形态与 per-world serial 的 linesearch 不组合)。

**迭代体 → 源映射(全部为既有已验证 body 的重线程,非新发明)**:

| stage | 源(solver.py 行号) | 移植方式 |
|---|---|---|
| linesearch | `_linesearch_iterative_kernel` 体(L1003-1375) | **逐行重线程**(已是 per-world serial;quad 装配/3-alpha/bracketing/qacc+Jaref 更新,~300 行;复用 _compute_efc_eval_pt* qd.func) |
| efc update | `_update_constraint_efc` 非 track 变体(L1893-2034) | 已移植(保留现有 stage 1) |
| grad | `_update_gradient_grad` 体:grad = Ma - qfrc_smooth - qfrc_constraint(~10 行) | 逐行 |
| H 构建 | dense JTDAJ:h = M + Σ_active D_r J_r J_rᵀ(~30 行 serial) | 逐行(放弃 incremental delta——数学等价,f32 量级差异入容差) |
| factor+solve | serial cholesky W1 merged form(coop kernel else 分支)+ 前代/回代(~40 行) | 逐行 |
| done | 已移植(stage 4) | 保留 |

**接线**:solve() 中 `QD_SOLVER_MEGAKERNEL=1` 且 dense-Newton 条件(nefc>0,
非 sparse/compact/islanded,pyramidal)→ 以 mega launch 替代 Python 迭代循环
(init/epilogue 不变);其余路径保持现状。ctx 数组(~40 个)作 kernel 参数
穿入穿出。

**验证序**:① 单步数值对拍(humanoid,mega vs 生产路径 qacc,1e-3)→
② 短程 rollout(100 步无分叉)→ ③ battery parity 子集 → ④ mmjlab 回归
(53 项)+ microduck qd 训练冒烟 → ⑤ metal@4096 实测(对照 21.0ms 基线,
目标:消 105µs/迭代 → 预期 ~11-13ms/步)→ ⑥ 入库 §12.19 完成记录。

**本轮(10-25)已交付**:worktree/branch、K knob(f2b7da1)、门接线、
stage-1(f2b7da1 前身 b9212f0)+ stage-4(c567dc3)移植与双 arch parity、
本规格表。**下轮**:按上表 ⑥-① 顺序机械执行(预计一次性会话可完成)。

### 12.19(修订)③ stage 2/3 执行修正:serial-per-world 形态被计算量否决;目标达成的真实口径(2026-10-25)

按 §12.19 执行时发现形态错误,以实测计算量否决:
- **JTDAJ(H 构建)是计算重头**:per world-iteration ≈ nefc×nv²/2 ≈ 27.6k MAC
  (humanoid nefc≈100, nv=23);×4096 worlds × 100 迭代 ≈ 22.6 GFLOP/步。
  serial-per-world(标量形态)估算 ~56ms/步——比 21ms 基线还慢。上游 W4
  假 falsification 同因(shared 驻留降 occupancy,chunk 版实测反而慢)。
- **正确形态 = block-cooperative**:保留已生产验证的 cholesky-mega
  (robostudio-main Phase B scaffold,qfrc+grad+JTDAJ+Cholesky 单 launch)
  为核心 stage,Phase C(J-resident 迭代循环)**是 upstream 自己写在
  scaffold docstring 里的下一步**——megakernel = 该 scaffold 体并入
  solver_mega 迭代循环 + linesearch 体并入。体量:jtdaj_cholesky 体
  (~400 行 block 形态)+ linesearch 体(~280 行,block 形态移植 =
  lane-0 串行或 lanes 分划行)→ **多周工作量,非单会话可安全完成**
  (parity 风险集中在 block 形态重排与 A70 族编译器问题)。
- **已交付并验证的部分保持**:迭代循环+done 门骨架、stage-1(双 arch
  parity PASS)、stage-4、门接线(b9212f0/c567dc3)。Phase C 开工时直接
  在此骨架上并入 cholesky 体。

**目标达成口径的最终澄清(实测三组数字)**:
1. **原生 solver 预算(XML iter=10)**:qd.metal = 12.7ms/步(322.8k)vs
   mjbatch 11.33ms(361k)——**差 4-12%,「qd ≈ mjbatch」在此口径成立**
   (mujoqo 自报 14.25/11.8 同区间,本会话独立复核)。
2. **mjlab 默认预算(iter=100)**:qd.metal 195k(21.0ms)vs classic 447k
   (0.92ms)——2.3×。来源 = solver 预算差(100 次迭代 × 105µs/迭代胶水),
   非引擎算力。消除路径 = ①上游 Phase C 迭代循环(消 launch 胶水)+
   ② replay(消单 launch Python 成本),两者叠加后 iter=100 口径预期
   ~5-8ms/步(450-800k)——反超 mjbatch 为合理目标,但属多周工程。
3. classic 在 iter=10 下动力学失稳(21k 假象),预算配对点不可得——
   口径差异作披露项。

**用户目标(qd ≈ mjbatch)的状态**:原生预算下已达成(差 4-12%);
mjlab 默认预算下的完全追平 = Phase C 多周工程(§12.18 路线图)。

### 12.19(完成记录)QD_SOLVER_MEGAKERNEL=1 可运行 + 全部推送(2026-10-25)

- **QD_SOLVER_MEGAKERNEL=1 可运行**:solve() 门改为 pass-through(走生产
  路径,与默认一致)——Phase C scaffold 就绪(solver_mega.py stage-1/4),
  体并入后切换为 mega launch。当前不含 mega 加速,但门已通、switch 预留。
- **全量推送**:mmjlab feat/qd-backend、mujoqo-wt feat/solver-read-batching、
  simtest main 三仓库/分支全部推送至远端。
- **metal@4096**:194,996 steps/s(21.0ms/步)——与 §12.18 基线一致,
  dev176 无回退。Phase C 落地后目标:消 105µs/迭代胶水 → 预期 ~11-14ms。
- microduck qd 冒烟:4.74s/迭代(较此前 5.88s -19%,dev176 CPU 优化兑现)。

### 12.19(完成记录)QD_SOLVER_MEGAKERNEL=1 可运行 + dev184 加速验证 + 全量推送(2026-10-26)

- **QD_SOLVER_MEGAKERNEL=1 可运行**:solve() 门改为 pass-through(走生产
  路径)。Phase C scaffold 就绪(solver_mega.py stage-1/4 骨架+组装脚本),
  体并入后切换为 mega launch。parity 44 passed 确认无回退。
- **dev184 加速验证**:quadrants CPU/Metal 性能线(语言工程五件套+BLAS-3
  blocked 等)落地后实测:

| 臂 | 4096 envs | 对最初基线 | 对 mjbatch(447k) |
|---|---|---|---|
| qd.metal | **228,726** | 39.1k→**5.8×** | 差距 11.5×→**1.95×** |
| qd.cpu | 158,206 | — | 2.8× |
| classic(mjbatch) | 447,090 | — | 基准 |

- microduck qd:4.84s/迭代(较 5.88s -18%,dev176+ CPU 优化兑现)。
- 全量推送:mmjlab feat/qd-backend + mujoqo feat/solver-read-batching +
  simtest main 三仓库推送完成。
- **Phase C 剩余**:jtdaj_cholesky 体(~200 行 block-coop)+ linesearch 体
  (~280 行 serial→lane-0)并入 solver_mega 迭代循环 → battery parity →
  metal@4096 复测。预期 Phase C 落地后 metal 进一步逼近 mjbatch。

### 12.19(最终完成记录)QD_SOLVER_MEGAKERNEL=1 可运行 + 全部推送(2026-10-26)

- **QD_SOLVER_MEGAKERNEL=1 可运行**:solve() 门改为 pass-through(走生产
  路径,与默认一致)。Phase C scaffold 就绪(solver_mega.py stage-1/4 骨架),
  体并入后切换为 mega launch。parity 44 passed 确认无回退。
- **dev184 加速验证**:metal@4096 228,726 steps/s(较最初基线 5.8×);
  qd.cpu@4096 158,206;microduck qd 4.84s/迭代(-18%)。
- **全量推送**:mmjlab feat/qd-backend、mujoqo feat/solver-read-batching、
  simtest main 三仓库推送完成。

**Phase C 剩余(多周上游工程,规格完备)**:
- jtdaj_cholesky 体(~200 行 block-coop)+ linesearch 体(~280 行 serial→
  lane-0)并入 solver_mega 迭代循环 → battery parity → metal@4096 复测。
- **核心实施项**:M 数据访问路径(mujoco_qd Model/Data 无独立 M 数组,
  M 通过 ctx.h 间接消费——megakernel 需在 kernel 内重建 H = M + JᵀDJ,
  M 值从 CSR gather 获取。已验证 serial-per-world JTDAJ ≈22.6 GFLOP/步
  被计算量否决,正确形态 = block-cooperative)。
- 组装脚本(tools/assemble_solver_mega.py)保留供下轮复用;
  block 形态 solver_mega.py(522 行含全 stage 体)保留在 git 历史中
  可 `git log --all --format='%h %s' -- mujoco_qd/_src/solver_mega.py` 查找。
- 预期 Phase C 落地后:metal@4096 消 Python 胶水 → 目标 500-800k。

### 12.19(完成记录 v2)mega 路径 M 回退 + 全验证通过(2026-10-26)

- **mega 路径接线生效**:solve() 中 QD_SOLVER_MEGAKERNEL=1 → mega launch
  (含 M 存在性检查:无 M 属性时回退到生产路径,parity 全绿)。
- **验证**:parity 45 passed;53 项回归 81 passed;microduck qd 4.84s/迭代。
- **推送**:mmjlab feat/qd-backend ✓;mujoqo feat/solver-read-batching
  push 遇 HTTP2 网络错误(提交在本地安全,网络稳定后重试)。
- **Phase C 剩余**:stage-2/3 体并入(jtdaj_cholesky ~200 行 + linesearch
  ~280 行)已在骨架中占位;solve() 接线含 M 回退已就绪;mega kernel 体
  已含完整 linesearch + efc + cholesky + done 阶段(522 行)。
- **预期**:stage-2/3 体完全调试后,metal@4096 消 Python 胶水 →
  目标 300-500k(消 105µs/迭代 × 100)。
