# GPU / ONNX / TensorRT 高性能分支说明

分支名建议：**`feature/gpu-onnx-trt`**（或你本地等价分支）。

本页说明：**全项目还有哪些优化空间**、**本分支已落地的部署能力**、**如何在带 NVIDIA GPU 的机器上复现**。

---

## 1. 全项目仍可优化的方向（按模块）

| 模块 | 现状 | 可做的「硬核」优化 |
|------|------|-------------------|
| **感知** `perception/` | MiDaS **固定 CPU**（`midas_depth.py`）；YOLO 走 ultralytics（可用 GPU 但未与 ONNX/TRT 统一） | 深度：ONNX 导出 → ORT GPU / TRT EP；检测：导出 YOLO 或换轻量 ONNX 检测头；批处理、异步流水线 |
| **部署** `deployment/` | 本分支前仅 CPU ORT + TensorRT **占位** | ✅ 本分支：**CUDA / TensorRT EP**、**图优化**、**trtexec 辅助**、**infer 计时** |
| **视觉 SLAM** `slam/` | OpenCV ORB + Python 后端 | C++ ORB/G2O、GPU 特征（可选）、降采样分辨率、关键帧间隔 |
| **SfM** `reconstruction/` | OpenCV ORB + Python | GPU SIFT/学习特征、多线程、COLMAP 接口（中长期） |
| **LiDAR** `lidar/` | Open3D ICP | CUDA ICP / 第三方 GPU 配准（需额外依赖） |
| **融合** `fusion/` | 小规模 NumPy / 图优化 | 大规模时用 GTSAM/Ceres（C++） |

**结论**：与 **ONNX/TensorRT** 最直接相关的是 **感知 + deployment**；SLAM/SfM 的瓶颈多在 **特征与几何**，需另列路线图。

---

## 2. 本分支已实现的 ONNX / TensorRT 相关能力

### 2.1 ONNX Runtime（推荐主路径）

- **`deployment/ort_gpu.py`**：按优先级构造 `providers`（TensorRT EP → CUDA → CPU），并打开 **ORT 图优化**。
- **`deployment/onnx_inference.py`**：`load_onnx_session(..., prefer_gpu=..., prefer_tensorrt=...)`。
- **CLI**：

```bash
# 查看当前 ORT 编译进了哪些 Provider
opl infer dummy.onnx --list-providers

# 优先 CUDA（需安装 onnxruntime-gpu，且与 CUDA 版本匹配）
opl export -o model.onnx --model tiny
opl infer model.onnx --gpu --warmup 10 --runs 50

# 优先 TensorRT EP（需 ORT 的 TensorRT 集成 + TensorRT 库在环境中可见）
opl infer model.onnx --tensorrt --gpu --warmup 10 --runs 50
```

> **说明**：`--tensorrt` 使用的是 **ONNX Runtime 的 TensorrtExecutionProvider**，在运行期把部分子图交给 TensorRT，并可用 **engine cache**（见 `ort_gpu.py` 中 `trt_engine_cache_path`）。

### 2.2 独立 TensorRT Engine（`trtexec`）

- **`deployment/trtexec_helper.py`**：若系统 PATH 上有 NVIDIA 的 **`trtexec`**，可从 ONNX 生成 **`.engine`**（常用于 C++ 部署或对照实验）。

```bash
python -c "from deployment.trtexec_helper import build_engine_trtexec; build_engine_trtexec('depth.onnx','depth_fp16.engine')"
```

### 2.3 与主分支的差异小结

| 能力 | 主分支 / CPU | 本分支 |
|------|----------------|--------|
| ORT Provider | 仅 CPU | 可选 CUDA / TensorRT EP |
| SessionOptions | 默认 | `ORT_ENABLE_ALL` |
| 占位 `tensorrt_stub` | 仅日志 | 仍保留；**实际加速走 ORT+TRT EP 或 trtexec** |

---

## 3. 环境安装（带 GPU 的机器）

1. **驱动**：安装与 GPU 匹配的 NVIDIA 驱动。  
2. **CUDA / cuDNN**：与 **`onnxruntime-gpu`** 官方说明表一致（见 [ORT 发布页](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)）。  
3. **Python 依赖**（与 `onnxruntime` **二选一**，勿混装）：

```bash
pip install -e ".[deploy-gpu]"
```

`pyproject.toml` 中 **`deploy-gpu`** 额外依赖 `onnxruntime-gpu`（见文件内注释）。

4. **TensorRT EP**：需使用 **带 TensorRT 的 ONNX Runtime 构建** 或按微软/NVIDIA 文档配置 `TensorrtExecutionProvider` 所需动态库；否则 `--tensorrt` 会回退并打 warning。

---

## 4. 如何向别人证明「确实会用 ONNX + TensorRT」

1. 在同一台机器、同一 ONNX、固定输入尺寸下，贴出 **`opl infer ... --runs 100`** 在 **CPU / CUDA / TensorRT** 下的 **mean ms** 对比表。  
2. 保留 **`trtexec` 生成的 .engine** 或 **ORT 的 trt cache 目录** 说明（可 `.gitignore`）。  
3. 在 PR/简历中写清：**ORT TensorRT EP** vs **独立 engine + trtexec** 的分工（本仓库两种都涉及）。

---

## 5. 建议的后续提交（本分支可继续迭代）

- [ ] `MiDaSDepth` 增加 **ONNXRuntime 后端**（与现有 Torch 二选一，环境变量切换）。  
- [ ] 检测 demo：**导出 + ORT GPU** 路径与 ultralytics 对比延迟。  
- [ ] `scripts/benchmark_onnx_providers.py` 一键出表（若未合并可本地运行）。  
- [ ] CI：CPU 路径保持 pytest；GPU 仅在自测机或带 GPU 的 Action runner 跑。

---

*文档随分支 `feature/gpu-onnx-trt` 维护。*
