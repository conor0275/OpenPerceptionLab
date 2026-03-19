# OpenPerceptionLab

一个长期演进的 **多传感器感知与 SLAM 平台**（camera-first），按“升级打怪”路线逐阶段实现：感知 → 多视图几何 → 视觉 SLAM → LiDAR SLAM → 多传感器融合 → 3D 重建 → 端侧部署。

## ✨ 现在能做什么

- **视觉感知（Demo）**：目标检测 / 语义分割 / 单目深度估计
- **多视图几何（Demo）**：相机模型、极几何、位姿估计、三角化
- **视觉 SLAM（实时摄像头）**：轨迹可视化 + 稀疏点云（Open3D）
- **LiDAR SLAM（点云序列）**：PCD 序列 → ICP 里程计 → 全局点云地图（Stage 3）
- **多传感器融合（Stage 4）**：视觉 + LiDAR 位姿的 2D pose graph 融合，可选 VIO 接口占位
- **3D 重建（Stage 5）**：增量式 SfM（稀疏点云 + 相机位姿），NeRF/3DGS 占位

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

### LiDAR SLAM（点云建图，Stage 3）

对目录下的一串 PCD 做 ICP 建图并保存全局点云：

```bash
# 用自带的合成数据试跑（生成 sample_pcds/ 并建图）
opl lidar-slam --sample -o lidar_map.pcd

# 用自己的 PCD 序列（目录内按文件名排序的 *.pcd）
opl lidar-slam /path/to/pcd_dir --output-map my_map.pcd --voxel-size 0.05 --show
```

生成测试用 PCD 序列（不跑 SLAM）：

```bash
python -m lidar.sample_data sample_pcds -n 5
```

### 多传感器融合（Stage 4）

用 2D 位姿图融合视觉与 LiDAR 轨迹（需先有视觉地图与 LiDAR 轨迹）：

```bash
# 合成数据演示（无需真实数据）
opl fusion --demo -o fused_trajectory.npz

# 使用真实数据：视觉地图（slam --save-map）+ LiDAR 轨迹（lidar-slam --save-trajectory）
opl lidar-slam --sample -o lidar_map.pcd --save-trajectory lidar_traj.npz
opl fusion --vo-map map.npz --lidar-trajectory lidar_traj.npz -o fused_trajectory.npz
```

### 3D 重建 SfM（Stage 5）

从多张图像恢复稀疏点云与相机位姿：

```bash
# 合成图像演示（生成 sample_sfm_images/ 并跑 SfM）
opl sfm --sample -o sfm_pointcloud.ply --poses sfm_poses.npz

# 使用自己的图像目录
opl sfm /path/to/images -o my_pointcloud.ply --poses my_poses.npz
```

## 🧭 路线图（升级打怪）

- **Stage 0（已开始）**：视觉感知任务（检测 / 分割 / 深度）
- **Stage 1（进行中）**：多视图几何（相机模型 / 三角化 / BA）
- **Stage 2（进行中）**：视觉 SLAM（特征、位姿、稀疏地图、优化）
  - **当前能力**：实时 VO + 局部 BA + 轨迹/点云可视化 + 地图保存与加载（`--save-map` / `--load-map`）。
  - **限制**：单目无绝对尺度、无回环、长时间会漂移。
- **Stage 3（已开始）**：LiDAR SLAM 与点云建图
  - **当前能力**：PCD 序列输入、ICP 帧间配准、全局点云地图保存（`.pcd`）。
  - **限制**：暂无回环、未用 IMU；适合离线/录制的点云序列。
- **Stage 4（已开始）**：多传感器融合（LiDAR-Vision pose graph、VIO 占位）
  - **当前能力**：2D pose graph 融合视觉与 LiDAR 轨迹；LiDAR 支持 `--save-trajectory`；VIO 接口占位。
  - **限制**：2D 平面、无 IMU 融合、未接入 GTSAM/Ceres。
- **Stage 5（已开始）**：3D 重建（SfM、NeRF/3DGS 占位）
  - **当前能力**：增量式 SfM（两视图初始化 + PnP 加视图）、PLY/位姿输出；NeRF/3DGS 为占位接口。
  - **限制**：稀疏重建、无稠密/网格、NeRF/3DGS 待接入。
- **Stage 6**：模型压缩与端侧部署（ONNX / TensorRT）

## 🧱 目录结构（当前）

```text
demos/                # 小实验 & 练手 demo
geometry/             # 多视图几何模块
perception/           # 感知模块（检测/分割/深度）
slam/                 # 视觉 SLAM（frontend/backend/core/visualization）
lidar/                # LiDAR SLAM（io/registration/map/odometry）
fusion/               # 多传感器融合（pose graph、LiDAR-Vision、VIO 占位）
reconstruction/       # 3D 重建（SfM、NeRF/3DGS 占位）
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