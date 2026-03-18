🚀 从零实现的迷你单目 SLAM（Mini-SLAM-from-Scratch）
一个轻量级的单目 SLAM 系统，从零开始实现，基于经典计算机视觉与几何方法构建。

✨ 功能特性
ORB 特征提取与匹配
极几何（基础矩阵 F 与本质矩阵 E）
相机位姿估计（E + PnP）
用三角化进行 3D 重建
稀疏地图与关键帧管理
Bundle Adjustment（捆绑调整）优化（SciPy）
实时相机跟踪
3D 点云可视化（Open3D）
🧠 流水线
图像 → 特征 → 匹配 → 位姿（E）→ 三角化 → 地图 → BA → 可视化

🎬 演示
实时轨迹可视化
稀疏 3D 点云重建
🛠️ 技术栈
Python
OpenCV
NumPy
SciPy
Open3D
📦 项目结构
slam/
├── frontend/ # 特征与跟踪
├── backend/ # 地图与优化
├── core/ # 帧与相机
├── visualization/ # 可视化
├── run_slam.py

🚀 如何运行
pip install -r requirements.txt
python slam/run_slam.py
🎯 亮点
从零构建完整 SLAM 流水线
实现 Bundle Adjustment 以减少漂移
设计模块化系统架构（前端/后端分离）
📌 未来工作
回环检测（Loop Closure）
全局 BA（Ceres / g2o）
视觉-惯性 SLAM（Visual-Inertial SLAM）
📎 参考
灵感来源于：

ORB-SLAM
COLMAP