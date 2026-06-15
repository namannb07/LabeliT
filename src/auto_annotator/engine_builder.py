from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


@dataclass
class OnnxModelInfo:
    resolution: str
    batch_dynamic: bool
    input_shape: Tuple[int, ...]


def _require_trt() -> None:
    if not TRT_AVAILABLE:
        raise RuntimeError(
            "tensorrt is not available.\n"
            "Ensure JetPack TRT system packages are on PYTHONPATH.\n"
            "Typical path: /usr/lib/python3/dist-packages/"
        )


def inspect_onnx(onnx_path: Path) -> OnnxModelInfo:
    _require_trt()

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError(
            "Failed to parse ONNX model:\n" + "\n".join(errors[:3])
        )

    if network.num_inputs < 1:
        raise RuntimeError("ONNX model has no input tensors.")

    inp = network.get_input(0)
    shape = tuple(inp.shape)

    if len(shape) < 4:
        raise RuntimeError(
            f"Unexpected input shape {shape} — expected (batch, channels, H, W)."
        )

    batch_dynamic = shape[0] == -1
    h, w = int(shape[2]), int(shape[3])

    if h <= 0 or w <= 0:
        raise RuntimeError(
            f"Model has dynamic spatial dimensions (H={h}, W={w}). "
            "Only fixed-resolution models are supported."
        )

    return OnnxModelInfo(
        resolution=f"{w}×{h}",
        batch_dynamic=batch_dynamic,
        input_shape=shape,
    )


def build_engine(
    onnx_path: Path,
    output_path: Path,
    fp16: bool = True,
    workspace_gb: int = 1,
    on_status: Optional[Callable[[str], None]] = None,
) -> None:
    _require_trt()

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    _status("Preparing model…")

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError(
            "Failed to parse ONNX model:\n" + "\n".join(errors[:3])
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
    )

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    inp = network.get_input(0)
    shape = tuple(inp.shape)
    if shape[0] == -1:
        profile = builder.create_optimization_profile()
        static_shape = (1, *shape[1:])
        profile.set_shape(inp.name, static_shape, static_shape, static_shape)
        config.add_optimization_profile(profile)

    _status("Building engine (this may take several minutes)…")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "Engine build failed. Check GPU memory and ONNX model compatibility."
        )

    _status("Saving engine…")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)
