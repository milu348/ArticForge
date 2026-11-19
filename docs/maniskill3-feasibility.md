# ManiSkill3 接入可行性调研（ArticForge 关节物体资产 × ManiSkill 评测）

> **历史调研文档（2026-11-16）**：第一步已落地——阶段 0 冒烟 22/22，阶段 1 抽屉拉开/推关均已实现（见 `pipeline/10_close_drawer.py`）。第二步（GPU 并行训练）按需启动。

日期：2026-11-16 ｜ 结论先行：**可行，推荐分两步走（CPU 单环境冒烟 → GPU 并行训练）**

## 1. ManiSkill3 是什么

- SAPIEN 驱动的机器人操作训练/评测框架（RSS 2025 论文），pip 一装即用
- 关键特性：GPU 并行仿真（4090 上视觉数据采集 30k+ FPS）、内置 RL/IL/VLA 基线（PPO、Diffusion Policy、RDT-1B 等）
- 本身就有**关节物体任务**：`OpenCabinetDoor-v1` / `OpenCabinetDrawer-v1`（Fetch 移动操作臂开柜门/抽屉，资产来自 PartNet-Mobility）——与 ArticForge 资产完全同构的任务范式

## 2. ArticForge 资产接入路径（官方支持三种，我们已有其中两种的现成产物）

| 路径 | 官方 API | ArticForge 现状 |
|---|---|---|
| **URDF loader** | `scene.create_urdf_loader()` → `loader.parse(urdf)["articulation_builders"]` | ✅ `assets/<name>/<name>.urdf` 即是；且支持 `fix_root_link`、`load_multiple_collisions_from_file`（我们的 CoACD 多凸包碰撞正需要） |
| MJCF loader | `create_mjcf_loader()` | ✅ 有 `scene.mjcf`，但官方注明 MJCF 属性非全支持，**首选 URDF 路线** |
| ArticulationBuilder | 代码手写 links/joints | 不需要 |

### 已知限制（官方明示）
- Physx 后端单 articulation ≤ **64 links**——我们的资产 2-3 links，无碍
- GPU 仿真下 articulation 初始位姿要防交叠（我们有 body 落地推导，天然满足）
- 碰撞凸包多会影响 GPU 仿真速度——每件 13 hulls 在正常范围

### 我们的差异化优势：**力标定元数据**
- `force_probe_report.json`（onset 力/力矩、行程、QA 线）是 PartNet-Mobility 资产**没有**的
- 直接可用于：任务难度分层（轻拉抽屉 vs 重门）、reward shaping（克服静摩擦的力塑形）、sim2real 参数迁移参考

## 3. 本机硬件核查（RTX 2060 6GB）

| 项 | 状态 |
|---|---|
| NVIDIA 驱动 591.86 | ✅ |
| Vulkan ICD | ⚠️ 只有 intel/radeon/lvp 等通用 ICD，**缺 nvidia_icd.json** → 渲染需先补（`sudo apt install libvulkan1` + 手动建 nvidia ICD 文件，官方文档给了模板） |
| Python 3.10 | ✅ 官方支持 |
| 磁盘 882G 可用 | ✅ |
| GPU 仿真 | RTX 2060 属官方支持区间，但 6GB 显存 → 并行环境数开小（32-64 env），冒烟足够 |

注：CPU 仿真不需要 Vulkan，可以先跑通逻辑；渲染/GPU 并行再补 ICD。

## 4. 两步走计划

### 第一步：冒烟验证（~0.5 天，零积分）
1. `pip install mani_skill torch`（装进独立 venv 或现有 .venv）
2. 写 `pipeline/9_maniskill_smoke.py`：用 `create_urdf_loader` 逐件加载 5 个 URDF
   - 验证点：URDF 解析通过、关节方向/限位正确、`fix_root_link=True` 落地稳定、CoACD 碰撞加载
3. 用内置 Panda 机器人写一个最小"拉开抽屉"任务脚本（状态观测 + 运动规划即可，不必训练）
4. 产出：5 件资产在 SAPIEN/Physx 下"可加载 + 可驱动"的冒烟报告

### 第二步：训练评测（可选，~2-3 天）
1. 仿照 `OpenCabinetDrawer-v1` 写自定义任务类（官方 Custom Tasks 教程有完整范式）
2. BC/PPO 基线在 5 件资产上训练，指标 = 成功率 + 需要的样本数
3. 对标实验：同任务在 PartNet-Mobility 资产上训练 → 对比（这就是"生成资产可训练性"的证据）

## 5. 风险清单

| 风险 | 概率 | 缓解 |
|---|---|---|
| SAPIEN 的 URDF 解析对惯性/关节写法挑剔 | 中 | 我们 URDF 已是 MuJoCo/ROS 双验证格式；冒烟即知 |
| 减面后视觉网格在 Physx 渲染管线里的兼容性 | 低 | 碰撞用凸包、视觉用 STL，分工明确 |
| 6GB 显存限制并行规模 | 确定 | 冒烟阶段 CPU 单环境；训练阶段 32-64 env 起步 |
| Panda 臂开抽屉任务对力参数敏感 | 中 | 我们有 onset 标定，可按资产调 reward/控制器增益 |

## 6. 结论

- **技术可行性：高**——URDF 是官方一等公民，资产结构（2-3 links + 凸包碰撞）完全在支持范围内
- **独特卖点**：力标定元数据 + 一句话生成的边际成本，学术界无直接同类
- **建议先做第一步冒烟**（半天，零积分，不装新东西进主 venv 之外）
