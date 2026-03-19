# OpenPerceptionLab

一个长期演进的 **多传感器感知与 SLAM 平台**（camera-first），按“升级打怪”路线逐阶段实现：感知 → 多视图几何 → 视觉 SLAM → LiDAR SLAM → 多传感器融合 → 3D 重建 → 端侧部署。

## ✨ 现在能做什么

- **视觉感知（Demo）**：目标检测 / 语义分割 / 单目深度估计
- **多视图几何（Demo）**：相机模型、极几何、位姿估计、三角化
- **视觉 SLAM（实时摄像头）**：轨迹可视化 + 稀疏点云（Open3D）

## 🚀 统一入口（推荐）

先在项目根目录安装一次（可编辑安装，后续改代码无需反复安装）：

```bash
python -m pip install -e .
```

然后你只需要记住一个入口：

```bash
python -m openperceptionlab --help
```

### 实时 SLAM（摄像头）

```bash
python -m openperceptionlab slam --camera 0
# 退出时保存地图、下次加载继续
python -m openperceptionlab slam --camera 0 --save-map map.npz
python -m openperceptionlab slam --camera 0 --load-map map.npz --save-map map2.npz
```

### 跑 Demo

单图 demo 可用 `--image` 指定输入，双图 demo 用 `--image1` / `--image2`（不指定则用默认文件名如 test.jpg、image1.jpg）：

```bash
python -m openperceptionlab demo detect
python -m openperceptionlab demo depth --image path/to/photo.jpg
python -m openperceptionlab demo triangulation --image1 left.jpg --image2 right.jpg
python -m openperceptionlab demo segment
python -m openperceptionlab demo perception
python -m openperceptionlab demo camera
python -m openperceptionlab demo epipolar
python -m openperceptionlab demo pose
python -m openperceptionlab demo feature_match
```

也可以用更短的命令（安装后自动提供）：

```bash
opl --help
opl slam --camera 0
```

## 🧭 路线图（升级打怪）

- **Stage 0（已开始）**：视觉感知任务（检测 / 分割 / 深度）
- **Stage 1（进行中）**：多视图几何（相机模型 / 三角化 / BA）
- **Stage 2（进行中）**：视觉 SLAM（特征、位姿、稀疏地图、优化）
  - **当前能力**：实时 VO + 局部 BA + 轨迹/点云可视化 + 地图保存与加载（`--save-map` / `--load-map`）。
  - **限制**：单目无绝对尺度、无回环、长时间会漂移。
- **Stage 3**：LiDAR SLAM 与点云建图
- **Stage 4**：多传感器融合（VIO、LiDAR-Vision，GTSAM/Ceres）
- **Stage 5**：3D 重建（SFM、NeRF、3D Gaussian Splatting）
- **Stage 6**：模型压缩与端侧部署（ONNX / TensorRT）

## 🧱 目录结构（当前）

```text
demos/                # 小实验 & 练手 demo
geometry/             # 多视图几何模块
perception/           # 感知模块（检测/分割/深度）
slam/                 # SLAM 系统（frontend/backend/core/visualization）
openperceptionlab/    # 统一入口（python -m openperceptionlab / opl）
```

## 🧪 开发与测试

- **安装开发依赖（含 pytest）**：`pip install -e ".[dev]"`
- **运行测试**：在项目根目录执行 `pytest tests/ -v`
- **配置文件**：复制 `config.example.yaml` 为 `config.yaml` 后按需修改，SLAM 可通过 `opl slam --config config.yaml` 加载。

推送/PR 到 `main` 或 `master` 时会自动跑 CI（pytest，Python 3.10/3.11）。

## 📌 说明

- 本项目优先保证 **实时摄像头可跑通**，然后逐步补齐工程化（配置、日志、测试、CI、数据集支持）。
- 后续允许引入 **C++**（性能关键模块 / 优化器 / 部署），但会保持 Python 入口与工程体验一致。