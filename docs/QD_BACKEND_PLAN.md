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
