"""
Export perception or other models to ONNX (Stage 6).

Requires optional deps: pip install torch onnx.
For MiDaS/DeepLab export, torch is needed; for simple models, see export_tiny_demo.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("opl.deployment.export")


def export_tiny_demo(output_path: str | Path) -> Path:
    """
    Export a tiny placeholder model to ONNX (no torch needed for the model itself).
    Uses a single identity-like op so the pipeline works without heavy deps.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError as e:
        raise ImportError("Install onnx: pip install onnx") from e
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal graph: input [1,3,H,W] -> output [1,1,H,W] (depth-like)
    input_name = "input"
    output_name = "output"
    node = helper.make_node(
        "Identity",
        inputs=[input_name],
        outputs=[output_name],
    )
    graph = helper.make_graph(
        [node],
        "tiny_demo",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, "H", "W"])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, "H", "W"])],
    )
    model = helper.make_model(graph, producer_name="OpenPerceptionLab")
    onnx.save(model, str(output_path))
    logger.info("Exported tiny demo ONNX to %s", output_path)
    return output_path


def export_depth_onnx(output_path: str | Path, model_name: str = "MiDaS_small") -> Path:
    """
    Export MiDaS depth model to ONNX. Requires: pip install torch onnx.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("Install torch for depth export: pip install torch") from e
    try:
        import onnx
    except ImportError as e:
        raise ImportError("Install onnx: pip install onnx") from e
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = torch.hub.load("intel-isl/MiDaS", model_name)
    model.eval()
    # Dummy input [1, 3, H, W]
    dummy = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "H", 3: "W"}, "output": {0: "batch", 2: "H", 3: "W"}},
        opset_version=14,
    )
    logger.info("Exported depth ONNX to %s", output_path)
    return output_path
