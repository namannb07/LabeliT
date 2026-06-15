from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from auto_annotator.models import BoundingBox, TRTBuffer
from auto_annotator.parsers import OutputParser, detect_parser

try:
    import pycuda.autoinit  # noqa: F401 — initializes CUDA context on import
    import pycuda.driver as cuda
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class TRTDetector:
    def __init__(self):
        self.engine_path: Optional[Path] = None
        self.labels: List[str] = []
        self.num_model_classes: int = 0
        self.net_w: int = 0
        self.net_h: int = 0
        self._engine = None
        self._context = None
        self._stream = None
        self._input_buf: Optional[TRTBuffer] = None
        self._output_bufs: List[TRTBuffer] = []
        self._parser: Optional[OutputParser] = None

    @property
    def model_type(self) -> Optional[str]:
        return self._parser.name if self._parser else None

    def load(self, engine_path: Path, labels: List[str]) -> None:
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "tensorrt or pycuda not available.\n"
                "Ensure JetPack TRT system packages are on PYTHONPATH.\n"
                "Typical path: /usr/lib/python3/dist-packages/"
            )
        try:
            self._load_impl(engine_path, labels)
        except Exception as e:
            raise RuntimeError(f"Engine load failed: {e}") from e

    def _load_impl(self, engine_path: Path, labels: List[str]) -> None:
        self.engine_path = engine_path
        self.labels = labels

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self._engine.get_tensor_shape(name)
                if len(shape) < 4:
                    raise RuntimeError(
                        f"Unexpected input tensor shape {tuple(shape)} — "
                        "expected (batch, channels, H, W)"
                    )
                h, w = int(shape[2]), int(shape[3])
                if h <= 0 or w <= 0:
                    raise RuntimeError(
                        f"Engine has dynamic spatial dimensions (H={h}, W={w}). "
                        "Rebuild the engine with a fixed input shape."
                    )
                self.net_h = h
                self.net_w = w
                break

        output_names, output_shapes = self._collect_outputs()
        self._parser = detect_parser(output_names, output_shapes,
                                     num_classes_hint=len(labels))
        self.num_model_classes = self._parser.num_model_classes

        if len(self.labels) < self.num_model_classes:
            for i in range(len(self.labels), self.num_model_classes):
                self.labels.append(f"class_{i}")
        elif len(self.labels) > self.num_model_classes:
            self.labels = self.labels[: self.num_model_classes]

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._allocate_buffers()

    def _collect_outputs(self) -> Tuple[List[str], Dict[str, Tuple]]:
        output_names: List[str] = []
        output_shapes: Dict[str, Tuple] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
                output_shapes[name] = tuple(self._engine.get_tensor_shape(name))
        return output_names, output_shapes

    def _allocate_buffers(self) -> None:
        dtype_map = {}
        if TRT_AVAILABLE:
            dtype_map = {
                trt.DataType.FLOAT: np.float32,
                trt.DataType.HALF:  np.float16,
                trt.DataType.INT32: np.int32,
                trt.DataType.INT8:  np.int8,
                trt.DataType.BOOL:  np.bool_,
            }

        self._output_bufs = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            trt_dtype = self._engine.get_tensor_dtype(name)
            np_dtype = dtype_map.get(trt_dtype, np.float32)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self._context.set_tensor_address(name, int(dev_mem))

            buf = TRTBuffer(name=name, host=host_mem, device=dev_mem,
                            shape=shape, dtype=np_dtype)

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_buf = buf
            else:
                self._output_bufs.append(buf)

    def run(self, image_path: Path, conf_thresh: float) -> List[BoundingBox]:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return []
        orig_h, orig_w = img_bgr.shape[:2]

        tensor, meta = self._parser.preprocess(img_bgr, self.net_w, self.net_h)

        inp = tensor.astype(self._input_buf.dtype)
        np.copyto(self._input_buf.host, inp.ravel())
        cuda.memcpy_htod_async(self._input_buf.device, self._input_buf.host, self._stream)

        self._context.execute_async_v3(self._stream.handle)

        for buf in self._output_bufs:
            cuda.memcpy_dtoh_async(buf.host, buf.device, self._stream)
        self._stream.synchronize()

        return self._parser.parse(self._output_bufs, meta, orig_w, orig_h, conf_thresh)

    def close(self) -> None:
        try:
            del self._context
            del self._engine
            if self._input_buf is not None:
                self._input_buf.device.free()
            for buf in self._output_bufs:
                buf.device.free()
        except Exception:
            pass
