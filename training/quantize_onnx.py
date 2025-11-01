import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize FP16 decoder to INT8 and validate parity.")
    parser.add_argument("--fp16-model", type=Path, required=True, help="Path to FP16 ONNX decoder.")
    parser.add_argument("--output-model", type=Path, default=Path("models/decoder.int8.onnx"), help="INT8 output path.")
    parser.add_argument("--meta", type=Path, default=Path("models/meta.json"), help="Metadata JSON to update.")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel weight quantization.")
    parser.add_argument("--samples", type=int, default=32, help="Number of random latent samples for evaluation.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for evaluation noise.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"], help="ONNX Runtime providers.")
    return parser.parse_args()


def _tensor_to_fp32(tensor: onnx.TensorProto) -> onnx.TensorProto:
    if tensor.data_type != onnx.TensorProto.FLOAT16:
        return tensor
    array = numpy_helper.to_array(tensor).astype(np.float32)
    return numpy_helper.from_array(array, tensor.name)


def _convert_value_infos(infos) -> None:
    for info in infos:
        tensor_type = info.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            tensor_type.elem_type = onnx.TensorProto.FLOAT


def _convert_graph(graph: onnx.GraphProto) -> None:
    for initializer in graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            initializer.CopyFrom(_tensor_to_fp32(initializer))

    _convert_value_infos(graph.input)
    _convert_value_infos(graph.output)
    _convert_value_infos(graph.value_info)

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.FLOAT16:
                attr.t.CopyFrom(_tensor_to_fp32(attr.t))
            elif attr.type == AttributeProto.GRAPH:
                _convert_graph(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for sub_graph in attr.graphs:
                    _convert_graph(sub_graph)

        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT


def convert_to_fp32_model(fp16_path: Path) -> Path:
    model = onnx.load(fp16_path)
    _convert_graph(model.graph)
    tmp_path = fp16_path.with_suffix(".fp32.onnx")
    onnx.save(model, tmp_path)
    return tmp_path


class RandomLatentDataReader(CalibrationDataReader):
    def __init__(self, latent_shape: Sequence[int], samples: int, seed: int) -> None:
        self.latent_shape = tuple(latent_shape)
        rng = np.random.default_rng(seed)
        count = max(samples, 32)
        self._data = [
            {"latent": rng.standard_normal(size=(1, *self.latent_shape)).astype(np.float32)}
            for _ in range(count)
        ]
        self._index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._index >= len(self._data):
            return None
        value = self._data[self._index]
        self._index += 1
        return value

    def rewind(self) -> None:
        self._index = 0


def quantize_model(
    fp16_model: Path,
    output_model: Path,
    per_channel: bool,
    latent_shape: Sequence[int],
    samples: int,
    seed: int,
) -> None:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = convert_to_fp32_model(fp16_model)
    try:
        data_reader = RandomLatentDataReader(latent_shape, samples, seed)
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(output_model),
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=[],
            op_types_to_quantize=["Conv"],
        )
    finally:
        fp32_path.unlink(missing_ok=True)


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 2.0) -> float:
    mse = np.mean((reference - test) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def evaluate_parity(
    fp16_model: Path,
    int8_model: Path,
    latent_shape: List[int],
    providers: List[str],
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    session_fp16 = ort.InferenceSession(str(fp16_model), providers=providers)
    session_int8 = ort.InferenceSession(str(int8_model), providers=providers)

    preds_fp16 = []
    preds_int8 = []
    for _ in range(max(samples, 8)):
        latent = rng.standard_normal(size=(1, *latent_shape)).astype(np.float32)
        preds_fp16.append(session_fp16.run(None, {"latent": latent})[0])
        preds_int8.append(session_int8.run(None, {"latent": latent})[0])

    outputs_fp16 = np.concatenate(preds_fp16, axis=0)
    outputs_int8 = np.concatenate(preds_int8, axis=0)

    return psnr(outputs_fp16, outputs_int8)


def update_metadata(meta_path: Path, int8_path: Path, psnr_value: float, per_channel: bool, samples: int) -> None:
    payload = json.loads(meta_path.read_text())
    payload["int8_decoder"] = str(int8_path)
    payload["quantization"] = {
        "method": "static_qdq",
        "per_channel": per_channel,
        "psnr_db": psnr_value,
        "samples": samples,
    }
    meta_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta.read_text())
    latent_shape = meta.get("latent_shape")
    if latent_shape is None:
        raise ValueError(f"'latent_shape' missing from metadata: {args.meta}")

    quantize_model(
        fp16_model=args.fp16_model,
        output_model=args.output_model,
        per_channel=args.per_channel,
        latent_shape=latent_shape,
        samples=args.samples,
        seed=args.seed,
    )

    psnr_value = evaluate_parity(
        args.fp16_model,
        args.output_model,
        latent_shape,
        args.providers,
        args.samples,
        args.seed,
    )

    update_metadata(args.meta, args.output_model, psnr_value, args.per_channel, args.samples)

    report = {
        "fp16_model": str(args.fp16_model),
        "int8_model": str(args.output_model),
        "psnr_db": psnr_value,
        "per_channel": args.per_channel,
        "samples": args.samples,
        "providers": args.providers,
    }
    report_path = args.output_model.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
