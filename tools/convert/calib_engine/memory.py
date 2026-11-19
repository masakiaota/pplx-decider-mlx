"""Safetensors header-only bounds, before asking MLX to load large tensors."""
import json
import math
import struct
from pathlib import Path


def tensor_headers(path):
    with Path(path).open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Incomplete safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 32 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large safetensors header: {path}")
        header = json.loads(stream.read(length))
    return {name: item for name, item in header.items() if name != "__metadata__"}


def inspect_weights(directory, dtype="bfloat16", *, include_readout=True):
    directory = Path(directory)
    scalar_size = {"bfloat16": 2, "float16": 2, "float32": 4}.get(dtype)
    if scalar_size is None:
        raise ValueError("Weight memory estimation supports bfloat16/float16/float32")
    index_path = directory / "model.safetensors.index.json"
    files = (sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists()
             else sorted(path.name for path in directory.glob("model*.safetensors")))
    if not files:
        raise FileNotFoundError("No model weight shards found")
    if include_readout:
        files.append("readout.safetensors")
    resident, maximum_cast_source, tensor_count = 0, 0, 0
    target = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}[dtype]
    for filename in files:
        for name, item in tensor_headers(directory / filename).items():
            stored = item["data_offsets"][1] - item["data_offsets"][0]
            if item["dtype"] in {"F16", "BF16", "F32"}:
                resident += math.prod(item["shape"]) * scalar_size
                if item["dtype"] != target:
                    maximum_cast_source = max(maximum_cast_source, stored)
            else:
                resident += stored
            tensor_count += 1
    return {"resident_bytes": resident, "maximum_cast_source_bytes": maximum_cast_source,
            "tensor_count": tensor_count}


def preflight_weights(directory, dtype, limit, *, active_bytes=0, sidecar=None):
    result = inspect_weights(directory, dtype)
    if sidecar is not None:
        metadata = json.loads((Path(sidecar) / "mlx_config.json").read_text())
        base_file = Path(directory) / "mlx_config.json"
        if base_file.exists():
            revision = json.loads(base_file.read_text()).get("source_revision")
        else:
            revision = json.loads((Path(directory) / "provenance.json").read_text()).get("revision")
        if (metadata.get("format") != "decider-locate-merged-mlx-v1" or metadata.get("lora_applied") is not True
                or not revision or metadata.get("base_revision") != revision):
            raise ValueError("Merged Locate sidecar format or base revision differs before loading weights")
        merged = inspect_weights(sidecar, dtype, include_readout=False)
        result["sidecar_bytes"] = merged["resident_bytes"]
        result["resident_bytes"] += merged["resident_bytes"]
        result["maximum_cast_source_bytes"] = max(result["maximum_cast_source_bytes"], merged["maximum_cast_source_bytes"])
    result["estimated_load_peak_bytes"] = active_bytes + result["resident_bytes"] + result["maximum_cast_source_bytes"] + 1_000_000_000
    if result["estimated_load_peak_bytes"] > limit:
        raise MemoryError(f"Estimated model load {result['estimated_load_peak_bytes'] / 1e9:.2f} GB exceeds "
                          f"{limit / 1e9:.2f} GB before loading weights; use quantized weights or a safe explicit budget")
    return result
