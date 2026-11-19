# ArticForge · 一句话生成可训练关节物体资产

ArticForge 是基于 [LUX3D](https://lux3d.aholo3d.cn)（群核科技 Manycore Research 的 3D 生成服务）与 MuJoCo 的具身智能关节物体资产工厂：**一句话文本 → 带物理碰撞的 URDF 资产 → 可训练的接触级仿真**。

当前状态：全链路 + 批量工厂已打通，5 件资产交付完成（ManiSkill 阶段 0 冒烟 22/22、阶段 1 抽屉操作已落地）。单件约 41 积分 / 10 分钟。

## 产品功能

对每件资产执行 7 步流水线，产出可直接用于 MuJoCo/ROS 训练与演示的**自包含 URDF 资产**（网格、碰撞体、场景、标定数据、演示视频齐备）：

**文生 3D → 上传 → 关节化 → URDF 适配 → 碰撞体 → 力学标定 → 视频交付**

## 项目结构

```
AHOLO/
├── pipeline/                  # 流水线脚本（已参数化，任意目录可运行）
│   ├── common.py              # 路径解析：根目录自适应，assets/<资产名> 布局
│   ├── 1_text_to_3d.py        # ① 文生 3D 提交 + 轮询 + 下载（G1-Turbo）
│   ├── 2_articulation_submit.py   # ③ 关节化提交（glbUrl + 运动提示词）
│   ├── 2_articulation_resume.py   # ③ 关节化恢复（断点续查，绝不重复创建）
│   ├── 3_adapt_urdf.py        # ④ 官方 URDF 包适配（坐标系/格式/惯量/关节原点）
│   ├── 4_make_collision.py    # ⑤a CoACD 凸分解碰撞体
│   ├── 5_filter_hulls.py      # ⑤b 按扫掠体积过滤凸包（等效开出抽屉口/门洞）
│   ├── 6_force_probe.py       # ⑥ 力学标定 + 自动质检（拉力/力矩 ramp + 起动力检测）
│   ├── 7_physics_video.py     # ⑦ 物理场景编排视频（力驱动，非伺服）
│   ├── 8_render_video.py      # ⑧ 运动学编排视频 + 交互查看器入口（--view）
│   ├── 9_maniskill_smoke.py   # ⑨ ManiSkill/SAPIEN 可加载+可驱动冒烟（阶段 0）
│   ├── 10_close_drawer.py     # ⑩ Panda 推关抽屉：任务环境 + 驱动 + 离屏视频（阶段 1）
│   └── aholo_tasks.py         # ManiSkill 拉开任务环境（1 阶段）
│
├── assets/<资产名>/            # 每件资产的全部交付物（自包含）
│   ├── <资产名>.urdf          # 适配后 URDF（Z-up，MuJoCo/ROS 友好）
│   ├── meshes/*.stl           # 视觉网格
│   ├── collision/             # 碰撞凸包 + 过滤报告
│   ├── scene.mjcf             # 运动学场景（位置执行器编排）
│   ├── scene_physics.mjcf     # 物理场景（力执行器 + 摩擦 + 限位）
│   ├── *_demo.mp4 / *_physics.mp4
│   └── force_curves.png       # 力学标定曲线
│
├── output/                    # 本地运行产物，全部已 gitignore（含账户信息）
│   ├── downloads/              # 云端原始交付缓存（可再生）
│   ├── records/<资产名>/       # 提交记录（报价 / 审批证据链）
│   ├── records/budget-ledger.json  # 积分台账（quote_step 写入）
│   └── logs/collection.sqlite # 官方任务日志（pluginTaskId 绑定）
│
├── skills/aholo-lux3d/        # 官方 Skill v1.2.0（运行时 API 客户端与工作流契约）
├── LUX3DAPI.json              # API Key（已 gitignore，需本地自行创建）
├── LUX3DAPI.example.json      # 凭据格式示例（占位符，可提交）
├── README.md                  # 本文件：项目说明与使用方法
├── 参赛项目报告.md             # 参赛报告（精简版）
└── .venv / .venv-maniskill / .lux3d-env   # 工具链环境（本地，不迁移）
```

## 路径约定（参数化）

- 所有脚本经 `pipeline/common.py` 自动推导项目根目录（取脚本位置的上一级，可用环境变量 `AHOLO_ROOT` 覆盖），**可在任意工作目录运行**，不含硬编码路径
- 资产名是唯一参数轴：`--asset <名字>` 决定记录、下载、交付三个目录
- 云端原始交付放 `output/downloads/<资产名>/`（可再生）；正式交付放 `assets/<资产名>/`
- 质量/惯量、关节原点偏移、扫掠范围、力度曲线等由 `pipeline/urdfkit.py` 从适配后的 URDF 与报告**自动推导**；`configs/<资产名>.json` 仅用于覆盖缺省值（见文末「已知陷阱」）

## 快速开始

```bash
# 交互查看器（实时窗口，滑块控制关节）
.venv/bin/python pipeline/8_render_video.py --asset side-table-01 --view

# 重跑物理视频
.venv/bin/python pipeline/7_physics_video.py --asset side-table-01

# 力学标定（力-位移曲线 + 自动质检）
.venv/bin/python pipeline/6_force_probe.py --asset side-table-01
```

### 新增一件资产的完整流程

```bash
ASSET=nightstand-02   # 资产名贯穿全流程

# ① 文生 3D（~13 积分）。前置：审批环节先创建提交记录
#    output/records/$ASSET/attempt-01-text-to-3d.json（submitState=not_started，含 quoteId）
.venv/bin/python pipeline/1_text_to_3d.py <quote-id> \
    --asset $ASSET --prompt "<物体描述>"

# ② 解包下载的 ZIP，Asset API 上传静态 GLB 取 glbUrl（免费）
.venv/bin/python skills/aholo-lux3d/core/runtime/asset_uploader.py upload <glb> --region cn

# ③ 关节化（~28 积分）。前置：output/records/$ASSET/attempt-02-articulation.json
.venv/bin/python pipeline/2_articulation_submit.py --asset $ASSET \
    --glb-url <glbUrl> --prompt "<运动描述（固定件/滑/转）>"
#    交付下载后，将 URDF 包解压到 output/downloads/$ASSET/urdf/（robot.urdf + link GLB）

# ④⑤⑥ 本地适配 + 碰撞体 + 力学标定（0 积分；换资产需先核对各脚本内常量）
.venv/bin/python pipeline/3_adapt_urdf.py --asset $ASSET
.venv/bin/python pipeline/4_make_collision.py --asset $ASSET
.venv/bin/python pipeline/5_filter_hulls.py --asset $ASSET
.venv/bin/python pipeline/6_force_probe.py --asset $ASSET

# ⑦⑧ 视频 + 交互查看器（0 积分）
.venv/bin/python pipeline/7_physics_video.py --asset $ASSET
.venv/bin/python pipeline/8_render_video.py --asset $ASSET
```

## 流水线（批量时的 7 步）

| 步 | 脚本 | 消耗 | 说明 |
|---|---|---|---|
| ① | `1_text_to_3d.py` | ~13 积分/件 | 文生 3D（`--prompt` 即换物体），下载 ZIP 解出 GLB |
| ② | Asset API 上传 | 免费 | `asset_uploader.py upload <glb> --region cn` 取 glbUrl |
| ③ | `2_articulation_*.py` | ~28 积分/件 | 关节化，交付 URDF 包（robot.urdf + link GLB） |
| ④ | `3_adapt_urdf.py` | 0 | 坐标系/格式/惯量适配 |
| ⑤ | `4_make_collision.py` + `5_filter_hulls.py` | 0 | 碰撞体（CoACD ~1 分钟/件） |
| ⑥ | `6_force_probe.py` | 0 | 力学标定 + 自动质检（能否加载/关节能否到位） |
| ⑦ | `7/8_*.py` | 0 | 视频交付 |

单件成本模型：**~41 积分 + ~10 分钟（云端生成 ~9 分钟 + 本地链 ~1.5 分钟）**。

## ManiSkill 1 阶段（Panda 机械臂操作，0 积分）

在 MuJoCo 力标定完成后，用 ManiSkill3 / SAPIEN（Physx）做跨引擎验证与机械臂操作：

```bash
source scripts/maniskill-env.sh
# 阶段 0：5 件资产可加载/可驱动冒烟
.venv-maniskill/bin/python pipeline/9_maniskill_smoke.py
# 阶段 1：Panda 拉开抽屉（任务环境 OpenAholoDrawer-v0）
.venv-maniskill/bin/python -m mani_skill.examples.demo_random_action \
    -e OpenAholoDrawer-v0 -b cpu -rb cpu
# 第 10 步：Panda 推关抽屉（任务环境 + 可运行驱动，均在 pipeline/10_close_drawer.py）
.venv-maniskill/bin/python pipeline/10_close_drawer.py --asset side-table-01
# 同上，带实时查看窗口
.venv-maniskill/bin/python pipeline/10_close_drawer.py --asset side-table-01 --view
# 离屏渲染视频（无需显示器）
.venv-maniskill/bin/python pipeline/10_close_drawer.py --asset side-table-01 \
    --record-every 1 --record docs/report-assets/panda-close-drawer-offscreen.mp4
```

离屏渲染的推关视频见 `docs/report-assets/panda-close-drawer-offscreen.mp4`（由 `--record` 生成）。

`pipeline/aholo_tasks.py` 注册 2 个拉开环境：`OpenAholoJoint-v0` / `OpenAholoDrawer-v0`；第 10 步 `pipeline/10_close_drawer.py` 单独注册 `CloseAholoDrawer-v0`（推关：起点 30% 开，目标 ≤5% 行程且部件静止才算成功），并自带 Panda 驱动：用 `pd_ee_delta_pose` Cartesian 控制先抬升、平移到把手前方，再分段推进直到关严，全程真实接触、不传送关节。当前 `side-table-01` / `nightstand-02` 验证为 SUCCESS；`toolbox-01` 受已记录的前缘低处不可达边界限制（见文末「已知陷阱」）。

## 环境与凭据

- Python 工具链在 `.venv`（mujoco 3.17.0 / trimesh / pygltflib / coacd / scipy / imageio），官方 runtime 在 `.lux3d-env`
- API Key：本地创建 `LUX3DAPI.json`（格式见 `LUX3DAPI.example.json`，内容为一个 JSON 字符串）；运行时经 `LUX3D_CN_API_KEY` 环境变量注入子进程，**不进命令行参数 / 日志，且已 gitignore**
- 区域：国内区（cn）；任务日志 `output/logs/collection.sqlite` 与 pluginTaskId 绑定
- 渲染：离屏用 `MUJOCO_GL=egl`；交互窗口走 GLFW/GLX

## 已知陷阱（新增资产前必读）

1. **关节原点坐标系不一致**：官方关节原点用柜底原点、link 网格用柜体中心原点，偏移由 `3_adapt_urdf.py` 从根 STL min_z 自动推导，换资产仍需核对。
2. **rest pose 微开**：官方静止位形凸出前脸 1~2cm，直接用碰撞会锁死，靠 `5_filter_hulls.py` 扫掠过滤解决。
3. **STL 面数上限**：MuJoCo 限 20 万面/网格，减面上限 18 万。
4. **ramp 方向**：开合方向与轴向符号相关，`open_target()` 自动取限位端点绝对值大者为「开」。
5. **MJCF joint 归属**：`<joint>` 必须写在子 body，写进父 body 会拖动整个落地柜。
6. **像素 QA 要检查差异 bbox 位置**：lookat 必须换算到世界系，否则差异区被裁到画面边缘。
7. **离线夹具命名要解耦**：过滤逻辑曾写死 `"body"` 键，夹具恰好同名导致测试假通过。
8. **网格简化前先合并顶点**：未合并顶点做 quadric 减面会撕裂网格（边界边率 0.3%→23.8%）。
