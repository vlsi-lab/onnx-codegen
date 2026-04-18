#!/usr/bin/env python3
"""
Generate a self-contained C inference library from an ONNX model.

Outputs:
- <prefix>_model.h      public API and metadata
- <prefix>_model.c      pure C inference implementation
- <prefix>_weights.h    model constants/weights

Design goals:
- No external runtime dependency in generated C.
- Static-shape ONNX graphs (recommended for MCU deployment).
- Float32 inference path.
- Graph execution follows ONNX node order.

Supported ONNX ops in generated code:
- Identity, Relu, LeakyRelu, Sigmoid, Tanh, Clip
- Add, Mul (numpy-style broadcast)
- MatMul, Gemm
- Conv (1D NCW and 2D NCHW, group supported)
- MaxPool, AveragePool (2D NCHW)
- GlobalAveragePool (NCHW)
- Flatten, Reshape, Transpose, Squeeze, Unsqueeze
- Softmax
- BatchNormalization
- Concat

Notes:
- Dynamic dimensions are not supported; all runtime tensor shapes must be known.
- Control-flow ops (Loop/If/Scan) are not supported.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from .renderer import render_template


AttrValue = Union[int, float, str, List[int], List[float], np.ndarray]


# ---------------------------------------------------------------------------
# Quantisation configuration
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Post-training type-override configuration.

    Parsed from strings like ``8w8a``, ``8w16a``, ``16w16a``.
    ``weight_bits`` controls the C type used for constant weight arrays.
    ``act_bits`` controls the storage type used for requantized activations.
    Integer accumulators and final non-requantized logits may still use
    ``int32_t`` internally when needed to preserve semantics. A value of 0
    means "keep original" (float).
    """

    weight_bits: int = 0
    act_bits: int = 0

    # Mapping from bit-width → (C type name, numpy dtype, element size)
    _TYPE_MAP: Dict[int, Tuple[str, type, int]] = field(
        default_factory=lambda: {
            8: ("int8_t", np.int8, 1),
            16: ("int16_t", np.int16, 2),
            32: ("int32_t", np.int32, 4),
        },
        repr=False,
    )

    @staticmethod
    def parse(spec: str) -> "QuantConfig":
        """Parse a quantisation spec such as ``8w8a`` or ``8w16a``.

        Format: ``<weight_bits>w<act_bits>a`` (case-insensitive).
        """
        spec = spec.strip().lower()
        m = re.fullmatch(r"(\d+)w(\d+)a", spec)
        if m is None:
            raise ValueError(
                f"Invalid --quant format '{spec}'. Expected e.g. 8w8a, 8w16a, 16w16a."
            )
        wb = int(m.group(1))
        ab = int(m.group(2))
        valid = {8, 16, 32}
        if wb not in valid or ab not in valid:
            raise ValueError(
                f"Unsupported bit widths in '{spec}'. Supported: 8, 16, 32."
            )
        return QuantConfig(weight_bits=wb, act_bits=ab)

    @property
    def weight_ctype(self) -> str:
        if self.weight_bits == 0:
            return "float"
        return self._TYPE_MAP[self.weight_bits][0]

    @property
    def act_ctype(self) -> str:
        if self.act_bits == 0:
            return "float"
        if self.act_bits == 8:
            return "uint8_t"
        return self._TYPE_MAP[self.act_bits][0]

    @property
    def weight_np_dtype(self) -> np.dtype:
        if self.weight_bits == 0:
            return np.dtype(np.float32)
        return np.dtype(self._TYPE_MAP[self.weight_bits][1])

    @property
    def act_elem_size(self) -> int:
        if self.act_bits == 0:
            return 4
        return 1 if self.act_bits == 8 else self._TYPE_MAP[self.act_bits][2]

    @property
    def weight_elem_size(self) -> int:
        if self.weight_bits == 0:
            return 4
        return self._TYPE_MAP[self.weight_bits][2]

    @property
    def enabled(self) -> bool:
        return self.weight_bits != 0 or self.act_bits != 0


SUPPORTED_OPS = {
    "Identity",
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Tanh",
    "Clip",
    "Add",
    "Div",
    "Mul",
    "MatMul",
    "Gemm",
    "Conv",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "Flatten",
    "Reshape",
    "Slice",
    "Transpose",
    "Pad",
    "Squeeze",
    "Unsqueeze",
    "Softmax",
    "BatchNormalization",
    "Concat",
    "Floor",
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinearConv",
    "RequantShift",
}


FLOAT_TENSOR_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.DOUBLE,
}

RUNTIME_TENSOR_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.DOUBLE,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.UINT8,
    TensorProto.INT32,
}

QUANT_BACKEND_ATTRS: Dict[str, Set[str]] = {
    "Conv": {"weight_bits", "bias_bits"},
    "Mul": {"mult_bits"},
    "Add": {"add_bits"},
    "Clip": {"out_bits"},
}


@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    elem_type: int
    is_const: bool

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class NodeOp:
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, AttrValue]


@dataclass
class CompareCaseResult:
    name: str
    matches: bool
    max_abs_diff: float


@dataclass
class CompareResult:
    matches: bool
    cases: List[CompareCaseResult]


@dataclass
class MemoryBreakdown:
    weights_bytes: int
    scratch_bytes: int
    input_bytes: int
    output_bytes: int
    inlined_const_bytes: int
    num_scratch_buffers: int

    @property
    def total_const_bytes(self) -> int:
        return self.weights_bytes + self.inlined_const_bytes

    @property
    def total_runtime_bytes(self) -> int:
        return self.scratch_bytes + self.input_bytes + self.output_bytes

    @property
    def total_bytes(self) -> int:
        return self.total_const_bytes + self.total_runtime_bytes


@dataclass
class GenerationResult:
    model_h_path: Path
    model_c_path: Path
    weights_h_path: Path
    kernels_h_path: Path
    kernels_c_path: Path
    n_inputs: int
    n_outputs: int
    n_nodes: int
    memory: MemoryBreakdown


class CodegenError(RuntimeError):
    pass


def sanitize_symbol(name: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not out:
        out = "tensor"
    if out[0].isdigit():
        out = "t_" + out
    return out


def c_float_literal(v: float) -> str:
    if np.isnan(v):
        return "NAN"
    if np.isposinf(v):
        return "INFINITY"
    if np.isneginf(v):
        return "-INFINITY"
    s = f"{float(v):.9g}"
    if "e" not in s and "." not in s:
        s += ".0"
    return s + "f"


def parse_attribute(attr: onnx.AttributeProto) -> AttrValue:
    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if attr.type == onnx.AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == onnx.AttributeProto.FLOATS:
        return [float(v) for v in attr.floats]
    if attr.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    raise CodegenError(f"Unsupported attribute type in {attr.name}")


def quantized_onnx_elem_type(quant: Optional[QuantConfig]) -> Optional[int]:
    if quant is None or quant.act_bits == 0:
        return None
    mapping = {
        8: TensorProto.UINT8,
        16: TensorProto.INT16,
        32: TensorProto.INT32,
    }
    return mapping[quant.act_bits]


def runtime_elem_type(elem_type: int, quant: Optional[QuantConfig] = None) -> int:
    return elem_type


def collect_value_info(model: onnx.ModelProto) -> Dict[str, Tuple[List[int], int]]:
    out: Dict[str, Tuple[List[int], int]] = {}

    def read_vi(vi: onnx.ValueInfoProto) -> Optional[Tuple[List[int], int]]:
        if not vi.type.HasField("tensor_type"):
            return None
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None
        shape: List[int] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                shape.append(int(d.dim_value))
            else:
                return None
        return shape, int(tt.elem_type)

    for vi in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        parsed = read_vi(vi)
        if parsed is not None:
            out[vi.name] = parsed
    return out


def read_model(onnx_path: Path, skip_shape_inference: bool) -> onnx.ModelProto:
    model = onnx.load(str(onnx_path))
    if not skip_shape_inference:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            raise CodegenError(
                "ONNX shape inference failed; pass --skip-shape-inference if your model "
                "already has complete static shapes"
            ) from exc
    return model


def sanitize_quantized_model(
    model: onnx.ModelProto,
    quant: Optional[QuantConfig],
    *,
    strip_backend_attrs: bool = True,
) -> onnx.ModelProto:
    if strip_backend_attrs:
        for node in model.graph.node:
            bad_attrs = QUANT_BACKEND_ATTRS.get(node.op_type)
            if not bad_attrs:
                continue
            keep = [attr for attr in node.attribute if attr.name not in bad_attrs]
            if len(keep) == len(node.attribute):
                continue
            del node.attribute[:]
            node.attribute.extend(keep)

    return model


def tensor_from_initializer(init: onnx.TensorProto) -> TensorInfo:
    arr = numpy_helper.to_array(init)
    return TensorInfo(
        name=init.name,
        shape=list(arr.shape),
        elem_type=int(init.data_type),
        is_const=True,
    )


def build_graph(model: onnx.ModelProto) -> Tuple[
    Dict[str, TensorInfo],
    Dict[str, np.ndarray],
    List[NodeOp],
    List[str],
    List[str],
]:
    graph = model.graph
    vi_map = collect_value_info(model)

    const_arrays: Dict[str, np.ndarray] = {}
    tensors: Dict[str, TensorInfo] = {}

    initializer_names = {i.name for i in graph.initializer}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        const_arrays[init.name] = arr
        tensors[init.name] = tensor_from_initializer(init)

    for n in graph.node:
        if n.op_type == "Constant":
            if len(n.output) != 1:
                raise CodegenError("Constant node with unexpected outputs")
            value_attr = None
            for a in n.attribute:
                if a.name == "value":
                    value_attr = a
                    break
            if value_attr is None:
                raise CodegenError("Constant node without 'value' attribute")
            arr = numpy_helper.to_array(value_attr.t)
            const_name = n.output[0]
            const_arrays[const_name] = arr
            tensors[const_name] = TensorInfo(
                name=const_name,
                shape=list(arr.shape),
                elem_type=int(value_attr.t.data_type),
                is_const=True,
            )

    runtime_inputs: List[str] = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        runtime_inputs.append(inp.name)

    graph_outputs = [o.name for o in graph.output]

    nodes: List[NodeOp] = []
    for i, node in enumerate(graph.node):
        if node.op_type == "Constant":
            continue
        attrs: Dict[str, AttrValue] = {
            a.name: parse_attribute(a) for a in node.attribute
        }
        name = node.name if node.name else f"node_{i}_{node.op_type}"
        nodes.append(
            NodeOp(
                op_type=node.op_type,
                name=name,
                inputs=list(node.input),
                outputs=list(node.output),
                attrs=attrs,
            )
        )

    for name in set(runtime_inputs + graph_outputs):
        if name in tensors:
            continue
        if name not in vi_map:
            raise CodegenError(
                f"Missing static shape/type for tensor '{name}'. "
                "Export ONNX with shapes or enable shape inference."
            )
        shape, elem_type = vi_map[name]
        tensors[name] = TensorInfo(
            name=name, shape=shape, elem_type=elem_type, is_const=False
        )

    produced_outputs = {t for n in nodes for t in n.outputs if t}
    referenced_tensors = set(runtime_inputs + graph_outputs) | produced_outputs
    for t in referenced_tensors:
        if not t:
            continue
        if t in tensors:
            continue
        if t in vi_map:
            shape, elem_type = vi_map[t]
            tensors[t] = TensorInfo(
                name=t, shape=shape, elem_type=elem_type, is_const=False
            )
            continue
        raise CodegenError(
            f"Tensor '{t}' has no static shape/type information. "
            "Only static-shape models are supported."
        )

    # Validate all non-const tensors are supported runtime types.
    for tname, info in tensors.items():
        if info.is_const:
            continue
        if info.elem_type not in RUNTIME_TENSOR_TYPES:
            raise CodegenError(
                f"Runtime tensor '{tname}' has unsupported type {info.elem_type}; "
                "generated runtime currently supports float/int8/uint8/int32 tensors"
            )

    for n in nodes:
        if n.op_type not in SUPPORTED_OPS:
            raise CodegenError(f"Unsupported op '{n.op_type}' in node '{n.name}'")

    return tensors, const_arrays, nodes, runtime_inputs, graph_outputs


def _clip_bounds(
    node: NodeOp, const_arrays: Dict[str, np.ndarray]
) -> Tuple[float, float]:
    clip_lo = 0.0
    clip_hi = 255.0
    clip_ins = [x for x in node.inputs if x]
    if len(clip_ins) >= 3:
        if clip_ins[1] in const_arrays:
            clip_lo = float(const_arrays[clip_ins[1]].flat[0])
        if clip_ins[2] in const_arrays:
            clip_hi = float(const_arrays[clip_ins[2]].flat[0])
    else:
        clip_lo = float(node.attrs.get("min", 0.0))
        clip_hi = float(node.attrs.get("max", 255.0))
    return clip_lo, clip_hi


def _quant_range_for_elem_type(elem_type: int) -> Optional[Tuple[int, int]]:
    if elem_type == TensorProto.UINT8:
        return (0, 255)
    if elem_type == TensorProto.INT16:
        return (-32768, 32767)
    if elem_type == TensorProto.INT32:
        return (-2147483648, 2147483647)
    return None


def _assign_quantized_runtime_tensor_types(
    tensors: Dict[str, TensorInfo],
    const_arrays: Dict[str, np.ndarray],
    nodes: List[NodeOp],
    inputs: Sequence[str],
    quant: Optional[QuantConfig],
) -> None:
    """Assign runtime tensor storage types for quantized codegen.

    Quantized activations use the requested activation storage type only after a
    requant/clip step. Accumulators, residual-add outputs and final conv logits
    remain int32.
    """

    quant_elem_type = quantized_onnx_elem_type(quant)
    if quant_elem_type is None:
        return

    for name in inputs:
        info = tensors[name]
        if not info.is_const and info.elem_type in FLOAT_TENSOR_TYPES:
            info.elem_type = TensorProto.INT32

    passthrough_ops = {
        "Identity",
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
        "Pad",
        "Slice",
        "Reshape",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
        "Concat",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Softmax",
        "BatchNormalization",
    }

    quant_range = _quant_range_for_elem_type(quant_elem_type)

    for node in nodes:
        outs = [o for o in node.outputs if o]
        ins = [i for i in node.inputs if i]
        if not outs:
            continue
        out_name = outs[0]
        out_info = tensors[out_name]
        if out_info.is_const:
            continue

        op = node.op_type
        if op in passthrough_ops and ins:
            out_info.elem_type = tensors[ins[0]].elem_type
            continue

        if op == "Add":
            out_info.elem_type = TensorProto.INT32
            continue

        if op in {"Mul", "Div", "Floor", "MatMul", "Gemm"}:
            out_info.elem_type = TensorProto.INT32
            continue

        if op == "Conv":
            out_info.elem_type = TensorProto.INT32
            continue

        if op == "RequantShift":
            out_info.elem_type = quant_elem_type
            continue

        if op == "Clip" and ins:
            in_type = tensors[ins[0]].elem_type
            clip_lo, clip_hi = _clip_bounds(node, const_arrays)
            if (
                in_type == TensorProto.INT32
                and quant_range is not None
                and int(clip_lo) == quant_range[0]
                and int(clip_hi) == quant_range[1]
            ):
                out_info.elem_type = quant_elem_type
            else:
                out_info.elem_type = in_type


# ---------------------------------------------------------------------------
# Requant fusion: Mul → Add → Div → Floor → Clip  →  RequantShift
# ---------------------------------------------------------------------------


def _fuse_requant(
    nodes: List[NodeOp],
    const_arrays: Dict[str, np.ndarray],
    tensors: Dict[str, TensorInfo],
) -> List[NodeOp]:
    """Replace Mul→Add→Div→Floor→Clip sequences with a RequantShift pseudo-op.

    Pattern:
        y = clip(floor((x * scale + bias) / 2^shift), lo, hi)

    RequantShift attrs:
        shift (int)   – right shift amount
        lo, hi (int)  – clip bounds
    RequantShift inputs:
        [activation, scale_array, bias_array]

    When scale==1 and shift==0, the whole block is an identity clip (no
    scale/bias arrays needed); we store a plain Clip instead.
    """
    output_of: Dict[str, int] = {}
    for idx, n in enumerate(nodes):
        for o in n.outputs:
            if o:
                output_of[o] = idx

    fused_set: Set[int] = set()
    result: List[NodeOp] = []

    i = 0
    while i < len(nodes):
        matched = _try_match_requant(nodes, i, const_arrays, tensors, output_of)
        if matched is None:
            matched = _try_match_requant_no_bias(
                nodes, i, const_arrays, tensors, output_of
            )
        if matched is not None:
            new_node, consumed, removed_consts = matched
            result.append(new_node)
            fused_set.update(range(i, i + consumed))
            for cname in removed_consts:
                const_arrays.pop(cname, None)
                tensors.pop(cname, None)
            i += consumed
        else:
            result.append(nodes[i])
            i += 1

    return result


def _fuse_zero_pad_into_conv1d(
    nodes: List[NodeOp],
    const_arrays: Dict[str, np.ndarray],
    tensors: Dict[str, TensorInfo],
    outputs: Sequence[str],
) -> List[NodeOp]:
    """Fold Pad(constant=0) into a following Conv1D when safe."""
    if not nodes:
        return nodes

    consumers: Dict[str, int] = {}
    for node in nodes:
        for inp in node.inputs:
            if inp:
                consumers[inp] = consumers.get(inp, 0) + 1

    graph_outputs = set(outputs)
    result: List[NodeOp] = []
    i = 0
    while i < len(nodes):
        pad = nodes[i]
        conv = nodes[i + 1] if i + 1 < len(nodes) else None

        can_fuse = (
            pad.op_type == "Pad"
            and conv is not None
            and conv.op_type == "Conv"
            and len(pad.outputs) >= 1
            and len(pad.inputs) >= 1
            and len(conv.inputs) >= 1
            and pad.outputs[0]
            and conv.inputs[0] == pad.outputs[0]
        )
        if not can_fuse:
            result.append(pad)
            i += 1
            continue

        pad_out = pad.outputs[0]
        pad_in = pad.inputs[0]
        x = tensors.get(pad_in)
        if (
            x is None
            or x.rank != 3
            or pad_out in graph_outputs
            or consumers.get(pad_out, 0) != 1
        ):
            result.append(pad)
            i += 1
            continue

        rank = x.rank
        if len(pad.inputs) >= 2 and pad.inputs[1] in const_arrays:
            pads_raw = const_arrays[pad.inputs[1]].reshape(-1)
            pads = [int(v) for v in pads_raw]
        else:
            pads = get_attr_ints(pad.attrs, "pads", [0] * (2 * rank))
        if len(pads) != 2 * rank:
            result.append(pad)
            i += 1
            continue

        mode = str(pad.attrs.get("mode", "constant"))
        pad_value = 0.0
        if len(pad.inputs) >= 3 and pad.inputs[2] in const_arrays:
            pad_value = float(const_arrays[pad.inputs[2]].reshape(-1)[0])
        if mode != "constant" or pad_value != 0.0:
            result.append(pad)
            i += 1
            continue

        pad_begin = pads[:rank]
        pad_end = pads[rank:]
        if any(v != 0 for v in pad_begin[:-1]) or any(v != 0 for v in pad_end[:-1]):
            result.append(pad)
            i += 1
            continue

        conv_pads = get_attr_ints(conv.attrs, "pads", [0, 0])
        if len(conv_pads) != 2:
            result.append(pad)
            i += 1
            continue

        fused_conv = NodeOp(
            op_type=conv.op_type,
            name=conv.name,
            inputs=[pad_in] + conv.inputs[1:],
            outputs=list(conv.outputs),
            attrs=dict(conv.attrs),
        )
        fused_conv.attrs["pads"] = [
            conv_pads[0] + pad_begin[-1],
            conv_pads[1] + pad_end[-1],
        ]
        result.append(fused_conv)
        i += 2

    return result


def _try_match_requant(
    nodes: List[NodeOp],
    start: int,
    const_arrays: Dict[str, np.ndarray],
    tensors: Dict[str, TensorInfo],
    output_of: Dict[str, int],
) -> Optional[Tuple[NodeOp, int, List[str]]]:
    """Try to match a 5-node requant pattern starting at *start*.

    Returns (new_node, num_nodes_consumed, list_of_removed_const_names) or
    None if the pattern doesn't match.
    """
    if start + 4 >= len(nodes):
        return None

    n_mul = nodes[start]
    n_add = nodes[start + 1]
    n_div = nodes[start + 2]
    n_floor = nodes[start + 3]
    n_clip = nodes[start + 4]

    if (
        n_mul.op_type != "Mul"
        or n_add.op_type != "Add"
        or n_div.op_type != "Div"
        or n_floor.op_type != "Floor"
        or n_clip.op_type != "Clip"
    ):
        return None

    # Check chain connectivity: Mul→Add→Div→Floor→Clip
    if n_mul.outputs[0] not in n_add.inputs:
        return None
    if n_add.outputs[0] not in n_div.inputs:
        return None
    if n_div.outputs[0] not in n_floor.inputs:
        return None
    if n_floor.outputs[0] not in n_clip.inputs:
        return None

    # Identify scale / bias / divisor constants.
    mul_ins = [x for x in n_mul.inputs if x]
    scale_name = None
    act_name = None
    for inp in mul_ins:
        if inp in const_arrays:
            scale_name = inp
        else:
            act_name = inp
    if scale_name is None or act_name is None:
        return None

    add_ins = [x for x in n_add.inputs if x]
    bias_name = None
    for inp in add_ins:
        if inp in const_arrays and inp != n_mul.outputs[0]:
            bias_name = inp
    if bias_name is None:
        return None

    div_ins = [x for x in n_div.inputs if x]
    div_const_name = None
    for inp in div_ins:
        if inp in const_arrays and inp != n_add.outputs[0]:
            div_const_name = inp
    if div_const_name is None:
        return None

    # Validate divisor is a power of 2.
    div_arr = const_arrays[div_const_name].flatten()
    if div_arr.size != 1 or div_arr[0] <= 0:
        return None
    div_val = float(div_arr[0])
    log2_val = np.log2(div_val)
    if log2_val != int(log2_val):
        return None
    shift = int(log2_val)

    scale_arr = const_arrays[scale_name]
    bias_arr = const_arrays[bias_name]

    # Validate scale and bias are integer-valued.
    if not np.allclose(scale_arr, np.round(scale_arr)):
        return None
    if not np.allclose(bias_arr, np.round(bias_arr)):
        return None

    # Get clip bounds.
    clip_lo = 0.0
    clip_hi = 255.0
    clip_ins = [x for x in n_clip.inputs if x]
    if len(clip_ins) >= 3:
        if clip_ins[1] in const_arrays:
            clip_lo = float(const_arrays[clip_ins[1]].flat[0])
        if clip_ins[2] in const_arrays:
            clip_hi = float(const_arrays[clip_ins[2]].flat[0])
    else:
        clip_lo = float(n_clip.attrs.get("min", 0.0))
        clip_hi = float(n_clip.attrs.get("max", 255.0))

    removed_consts: List[str] = [div_const_name]
    if len(clip_ins) >= 3:
        if clip_ins[1] in const_arrays:
            removed_consts.append(clip_ins[1])
        if clip_ins[2] in const_arrays:
            removed_consts.append(clip_ins[2])

    # Check for identity requant (scale=1, shift=0, bias=0).
    is_identity = (
        shift == 0
        and np.all(np.round(scale_arr) == 1)
        and np.all(np.round(bias_arr) == 0)
    )

    if is_identity:
        removed_consts.extend([scale_name, bias_name])
        new_node = NodeOp(
            op_type="Clip",
            name=f"{n_mul.name}..{n_clip.name} (identity requant)",
            inputs=[act_name],
            outputs=list(n_clip.outputs),
            attrs={"min": clip_lo, "max": clip_hi},
        )
        return new_node, 5, removed_consts

    # Replace scale/bias arrays with compact integer types.
    int_scale = np.round(scale_arr).astype(np.int32)
    int_bias = np.round(bias_arr).astype(np.int32)
    const_arrays[scale_name] = int_scale
    const_arrays[bias_name] = int_bias
    tensors[scale_name] = TensorInfo(
        name=scale_name,
        shape=list(int_scale.shape),
        elem_type=TensorProto.INT32,
        is_const=True,
    )
    tensors[bias_name] = TensorInfo(
        name=bias_name,
        shape=list(int_bias.shape),
        elem_type=TensorProto.INT32,
        is_const=True,
    )

    new_node = NodeOp(
        op_type="RequantShift",
        name=f"{n_mul.name}..{n_clip.name} (fused requant)",
        inputs=[act_name, scale_name, bias_name],
        outputs=list(n_clip.outputs),
        attrs={"shift": shift, "lo": int(clip_lo), "hi": int(clip_hi)},
    )
    return new_node, 5, removed_consts


def _try_match_requant_no_bias(
    nodes: List[NodeOp],
    start: int,
    const_arrays: Dict[str, np.ndarray],
    tensors: Dict[str, TensorInfo],
    output_of: Dict[str, int],
) -> Optional[Tuple[NodeOp, int, List[str]]]:
    """Try to match a 4-node requant pattern Mul→Div→Floor→Clip (no bias Add).

    This handles the post-residual-add requant where the ONNX emits
    ``clip(floor(x * kappa / 2^shift), lo, hi)`` without an additive bias.
    A synthetic zero-bias array is created so the existing RequantShift /
    ``requant_channel_ncw`` kernel can be reused unchanged.
    """
    if start + 3 >= len(nodes):
        return None

    n_mul = nodes[start]
    n_div = nodes[start + 1]
    n_floor = nodes[start + 2]
    n_clip = nodes[start + 3]

    if (
        n_mul.op_type != "Mul"
        or n_div.op_type != "Div"
        or n_floor.op_type != "Floor"
        or n_clip.op_type != "Clip"
    ):
        return None

    # Check chain connectivity: Mul→Div→Floor→Clip
    if n_mul.outputs[0] not in n_div.inputs:
        return None
    if n_div.outputs[0] not in n_floor.inputs:
        return None
    if n_floor.outputs[0] not in n_clip.inputs:
        return None

    # Identify scale constant and activation input.
    mul_ins = [x for x in n_mul.inputs if x]
    scale_name = None
    act_name = None
    for inp in mul_ins:
        if inp in const_arrays:
            scale_name = inp
        else:
            act_name = inp
    if scale_name is None or act_name is None:
        return None

    # Identify divisor constant.
    div_ins = [x for x in n_div.inputs if x]
    div_const_name = None
    for inp in div_ins:
        if inp in const_arrays and inp != n_mul.outputs[0]:
            div_const_name = inp
    if div_const_name is None:
        return None

    # Validate divisor is a power of 2.
    div_arr = const_arrays[div_const_name].flatten()
    if div_arr.size != 1 or div_arr[0] <= 0:
        return None
    div_val = float(div_arr[0])
    log2_val = np.log2(div_val)
    if log2_val != int(log2_val):
        return None
    shift = int(log2_val)

    scale_arr = const_arrays[scale_name]
    if not np.allclose(scale_arr, np.round(scale_arr)):
        return None

    # Get clip bounds.
    clip_lo = 0.0
    clip_hi = 255.0
    clip_ins = [x for x in n_clip.inputs if x]
    if len(clip_ins) >= 3:
        if clip_ins[1] in const_arrays:
            clip_lo = float(const_arrays[clip_ins[1]].flat[0])
        if clip_ins[2] in const_arrays:
            clip_hi = float(const_arrays[clip_ins[2]].flat[0])
    else:
        clip_lo = float(n_clip.attrs.get("min", 0.0))
        clip_hi = float(n_clip.attrs.get("max", 255.0))

    removed_consts: List[str] = [div_const_name]
    if len(clip_ins) >= 3:
        if clip_ins[1] in const_arrays:
            removed_consts.append(clip_ins[1])
        if clip_ins[2] in const_arrays:
            removed_consts.append(clip_ins[2])

    # Broadcast scalar kappa to per-channel when activation is 3-D NCW.
    act_info = tensors.get(act_name)
    if act_info is not None and act_info.rank == 3 and scale_arr.size == 1:
        c = act_info.shape[1]  # NCW: channels axis
        scale_arr = np.full((c,), np.round(scale_arr.flat[0]), dtype=np.float32)

    # Check for identity requant (scale==1, shift==0).
    int_scale = np.round(scale_arr).astype(np.int32)
    is_identity = shift == 0 and np.all(int_scale == 1)

    if is_identity:
        removed_consts.append(scale_name)
        new_node = NodeOp(
            op_type="Clip",
            name=f"{n_mul.name}..{n_clip.name} (identity requant)",
            inputs=[act_name],
            outputs=list(n_clip.outputs),
            attrs={"min": clip_lo, "max": clip_hi},
        )
        return new_node, 4, removed_consts

    # Store the (possibly broadcast) integer scale back.
    const_arrays[scale_name] = int_scale
    tensors[scale_name] = TensorInfo(
        name=scale_name,
        shape=list(int_scale.shape),
        elem_type=TensorProto.INT32,
        is_const=True,
    )

    # Synthesise a zero-bias array matching the scale shape.
    bias_name = f"{scale_name}_zero_bias"
    int_bias = np.zeros_like(int_scale)
    const_arrays[bias_name] = int_bias
    tensors[bias_name] = TensorInfo(
        name=bias_name,
        shape=list(int_bias.shape),
        elem_type=TensorProto.INT32,
        is_const=True,
    )

    new_node = NodeOp(
        op_type="RequantShift",
        name=f"{n_mul.name}..{n_clip.name} (fused requant, no bias)",
        inputs=[act_name, scale_name, bias_name],
        outputs=list(n_clip.outputs),
        attrs={"shift": shift, "lo": int(clip_lo), "hi": int(clip_hi)},
    )
    return new_node, 4, removed_consts


def shape_c_array(name: str, dims: Sequence[int]) -> str:
    if not dims:
        return f"static const int {name}[1] = {{1}};"
    vals = ", ".join(str(int(d)) for d in dims)
    return f"static const int {name}[{len(dims)}] = {{{vals}}};"


def safe_tensor_ref(name: str) -> str:
    return sanitize_symbol(name)


def _float_array_as_int8(flat: np.ndarray) -> bool:
    """Return True if a float32 array's values are all integers in [-128, 127]."""
    return bool(
        np.all(flat == np.floor(flat))
        and float(np.min(flat)) >= -128.0
        and float(np.max(flat)) <= 127.0
    )


def _float_array_is_integer_valued(arr: np.ndarray) -> bool:
    """Return True if every float element is an integer value."""
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    return bool(np.all(flat == np.floor(flat)))


def _check_weight_range(name: str, flat: np.ndarray, target_bits: int) -> None:
    """Warn if float weight values will be truncated by the target bit-width."""
    if target_bits == 0:
        return
    limits = {8: (-128, 127), 16: (-32768, 32767), 32: (-2147483648, 2147483647)}
    lo, hi = limits[target_bits]
    vmin, vmax = float(np.min(flat)), float(np.max(flat))
    if vmin < lo or vmax > hi:
        warnings.warn(
            f"Weight '{name}': value range [{vmin:.4g}, {vmax:.4g}] exceeds "
            f"int{target_bits} range [{lo}, {hi}]. Values will be truncated — "
            f"this may destroy model accuracy.",
            stacklevel=3,
        )


def _cast_integer_float_array(
    name: str, arr: np.ndarray, target_dtype: np.dtype, target_bits: int
) -> np.ndarray:
    """Convert an integer-valued float array to an integer dtype with range checks."""
    if not _float_array_is_integer_valued(arr):
        raise CodegenError(
            f"Tensor '{name}' uses float storage but contains non-integer values; "
            f"cannot cast it to int{target_bits} safely."
        )

    flat = arr.reshape(-1).astype(np.float64, copy=False)
    info = np.iinfo(target_dtype)
    vmin = float(np.min(flat))
    vmax = float(np.max(flat))
    if vmin < info.min or vmax > info.max:
        raise CodegenError(
            f"Tensor '{name}' range [{vmin:.4g}, {vmax:.4g}] does not fit in "
            f"int{target_bits}."
        )
    return np.round(arr).astype(target_dtype)


def _retype_integer_valued_float_constants(
    tensors: Dict[str, TensorInfo],
    const_arrays: Dict[str, np.ndarray],
    nodes: List[NodeOp],
) -> None:
    """Retype integer-valued float constants when their consumer requires integers.

    This is mainly used for models such as GraphModule_float.onnx where ONNX stores
    convolution parameters as FLOAT even though their values are integral.
    """

    consumer_type_map: Dict[Tuple[str, int], Tuple[np.dtype, int]] = {
        ("Conv", 1): (np.dtype(np.int8), TensorProto.INT8),
        ("Conv", 2): (np.dtype(np.int32), TensorProto.INT32),
    }

    targets: Dict[str, Set[Tuple[np.dtype, int]]] = {}
    blocked: Set[str] = set()

    for node in nodes:
        for input_idx, name in enumerate(node.inputs):
            if not name or name not in const_arrays:
                continue
            target = consumer_type_map.get((node.op_type, input_idx))
            if target is None:
                blocked.add(name)
                continue
            targets.setdefault(name, set()).add(target)

    for name, target_set in targets.items():
        if name in blocked or len(target_set) != 1:
            continue
        arr = const_arrays[name]
        if arr.dtype not in (np.float16, np.float32, np.float64):
            continue

        target_dtype, elem_type = next(iter(target_set))
        casted = _cast_integer_float_array(
            name,
            arr,
            target_dtype,
            8 if elem_type == TensorProto.INT8 else 32,
        )
        const_arrays[name] = casted

        info = tensors[name]
        tensors[name] = TensorInfo(
            name=info.name,
            shape=list(casted.shape),
            elem_type=elem_type,
            is_const=True,
        )


def _build_weight_definitions(
    prefix: str,
    const_arrays: Dict[str, np.ndarray],
    tensor_elem_types: Optional[Dict[str, int]] = None,
    quant: Optional[QuantConfig] = None,
) -> Tuple[List[str], Set[str]]:
    """Build C weight array definitions.

    Returns (defs, int8_weight_names) where int8_weight_names is the set of
    tensor names stored as int8_t.
    """
    defs: List[str] = []
    int8_names: Set[str] = set()
    forced_weight_ctype = quant.weight_ctype if quant and quant.weight_bits else None

    for name in sorted(const_arrays.keys()):
        arr = const_arrays[name]
        sym = f"{prefix}_w_{safe_tensor_ref(name)}"
        flat = arr.reshape(-1)

        if arr.dtype in (np.float16, np.float32, np.float64):
            f32 = flat.astype(np.float32)

            if forced_weight_ctype and forced_weight_ctype != "float":
                # Quant override: cast float weights to the requested integer type.
                _check_weight_range(name, f32, quant.weight_bits)
                np_dt = quant.weight_np_dtype
                clamped = np.clip(
                    np.round(f32),
                    np.iinfo(np_dt).min,
                    np.iinfo(np_dt).max,
                ).astype(np_dt)
                vals = ", ".join(str(int(v)) for v in clamped)
                defs.append(
                    f"static const {forced_weight_ctype} {sym}[{flat.size}] = {{{vals}}};"
                )
                if quant.weight_bits == 8:
                    int8_names.add(name)
            elif (
                tensor_elem_types is not None
                and tensor_elem_types.get(name) == TensorProto.INT8
            ):
                # Preserve semantic int8 tensors retyped earlier in the pipeline.
                if not _float_array_as_int8(f32):
                    raise CodegenError(
                        f"Tensor '{name}' is marked as int8 but contains out-of-range values."
                    )
                vals = ", ".join(str(int(v)) for v in f32)
                defs.append(f"static const int8_t {sym}[{flat.size}] = {{{vals}}};")
                int8_names.add(name)
            else:
                vals = ", ".join(c_float_literal(float(v)) for v in f32)
                defs.append(f"static const float {sym}[{flat.size}] = {{{vals}}};")
        elif arr.dtype in (np.int8, np.int16, np.int32, np.int64):
            ctype = {
                np.dtype(np.int8): "int8_t",
                np.dtype(np.int16): "int16_t",
                np.dtype(np.int32): "int32_t",
                np.dtype(np.int64): "int64_t",
            }[arr.dtype]
            vals = ", ".join(str(int(v)) for v in flat)
            defs.append(f"static const {ctype} {sym}[{flat.size}] = {{{vals}}};")
            if arr.dtype == np.dtype(np.int8):
                int8_names.add(name)
        elif arr.dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            ctype = {
                np.dtype(np.uint8): "uint8_t",
                np.dtype(np.uint16): "uint16_t",
                np.dtype(np.uint32): "uint32_t",
                np.dtype(np.uint64): "uint64_t",
            }[arr.dtype]
            vals = ", ".join(str(int(v)) for v in flat)
            defs.append(f"static const {ctype} {sym}[{flat.size}] = {{{vals}}};")
        else:
            raise CodegenError(
                f"Unsupported constant dtype {arr.dtype} for tensor '{name}'"
            )
    return defs, int8_names


def render_weights_header(
    prefix: str,
    const_arrays: Dict[str, np.ndarray],
    tensors: Optional[Dict[str, TensorInfo]] = None,
    quant: Optional[QuantConfig] = None,
) -> Tuple[str, Set[str]]:
    """Return (rendered_header, int8_weight_names)."""
    tensor_elem_types = None
    if tensors is not None:
        tensor_elem_types = {name: info.elem_type for name, info in tensors.items()}
    defs, int8_names = _build_weight_definitions(
        prefix, const_arrays, tensor_elem_types, quant
    )
    return (
        render_template(
            "weights_h.mako",
            guard=f"{prefix.upper()}_WEIGHTS_H",
            weight_defs=defs,
        ),
        int8_names,
    )


def _compute_buffer_assignments(
    tensors: Dict[str, TensorInfo],
    nodes: List[NodeOp],
    inputs: List[str],
    outputs: List[str],
    const_arrays: Dict[str, np.ndarray],
    quant: Optional[QuantConfig] = None,
) -> Tuple[Dict[str, int], Dict[int, Tuple[int, str]]]:
    """Assign intermediate (scratch) tensors to a minimal pool of reused buffers.

    Uses linear scan live-range analysis: tensors whose live ranges do not
    overlap are allowed to share the same backing buffer.

    Returns:
        assignments  – tensor_name -> buffer_id
        pool         – buffer_id -> (max_numel, ctype)
    """
    # Only include tensors actually referenced by remaining nodes.
    referenced_by_nodes = {t for n in nodes for t in n.inputs + n.outputs if t}
    scratch: Set[str] = {
        name
        for name, info in tensors.items()
        if not info.is_const
        and name not in inputs
        and name not in const_arrays
        and name in referenced_by_nodes
    }

    # birth[t] = index of the node that first writes t
    birth: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        for out in node.outputs:
            if out and out in scratch and out not in birth:
                birth[out] = i

    # death[t] = index of the last node that reads t
    # Graph outputs survive until output-copy (just past the last node)
    death: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        for inp in node.inputs:
            if inp and inp in scratch:
                death[inp] = i
    for name in outputs:
        if name in scratch:
            death[name] = len(nodes)
    for name in scratch:
        if name not in death:
            death[name] = birth.get(name, 0)

    fused_conv_outputs: Set[str] = set()
    fused_add_outputs: Set[str] = set()
    for i, node in enumerate(nodes):
        if node.op_type != "Conv":
            if node.op_type != "Add":
                continue
            nxt = nodes[i + 1] if i + 1 < len(nodes) else None
            if (
                nxt is None
                or nxt.op_type != "Clip"
                or quant is None
                or not quant.act_bits
            ):
                continue
            add_out = node.outputs[0] if node.outputs else None
            clip_out = nxt.outputs[0] if nxt.outputs else None
            if (
                not add_out
                or add_out not in nxt.inputs
                or not clip_out
                or tensors[add_out].elem_type != TensorProto.INT32
                or tensors[node.inputs[0]].elem_type != quantized_onnx_elem_type(quant)
                or tensors[node.inputs[1]].elem_type != quantized_onnx_elem_type(quant)
                or tensors[clip_out].elem_type != quantized_onnx_elem_type(quant)
            ):
                continue
            for add_in in node.inputs[:2]:
                if add_in and add_in in death:
                    death[add_in] = max(death[add_in], i + 1)
            fused_add_outputs.add(add_out)
            continue
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None
        if nxt is None or nxt.op_type != "RequantShift":
            continue
        conv_out = node.outputs[0] if node.outputs else None
        if not conv_out or conv_out not in nxt.inputs:
            continue
        # Adjustment 1: extend Conv input life.
        conv_in = node.inputs[0] if node.inputs else None
        if conv_in and conv_in in death:
            death[conv_in] = max(death[conv_in], i + 1)
        # Adjustment 2: mark Conv output as not needing its own scratch buffer.
        if conv_out:
            fused_conv_outputs.add(conv_out)

    scratch -= fused_conv_outputs
    scratch -= fused_add_outputs

    # Group by C element type so we never alias across types.
    # When quant overrides activations, float tensors become act_ctype.
    def _resolve_ctype(tname: str) -> str:
        return c_type_for_elem_type(runtime_elem_type(tensors[tname].elem_type, quant))

    by_ctype: Dict[str, List[str]] = {}
    for name in scratch:
        ctype = _resolve_ctype(name)
        by_ctype.setdefault(ctype, []).append(name)

    assignments: Dict[str, int] = {}
    pool: Dict[int, Tuple[int, str]] = {}  # buffer_id -> (max_numel, ctype)
    next_id = 0

    for ctype, group in by_ctype.items():
        group.sort(key=lambda n: birth.get(n, 0))
        # free_slots: list of buffer_ids whose last tenant has died
        free_slots: List[int] = []
        # active: list of (death, buffer_id) for still-live tensors, sorted by death
        active: List[Tuple[int, int]] = []

        for name in group:
            b = birth.get(name, 0)
            d = death.get(name, b)
            numel = tensors[name].numel

            # Expire buffers whose last tenant died strictly before this birth
            still_active = []
            for item in active:
                if item[0] < b:
                    free_slots.append(item[1])
                else:
                    still_active.append(item)
            active = still_active

            if free_slots:
                bid = free_slots.pop()
                pool[bid] = (max(pool[bid][0], numel), ctype)
            else:
                bid = next_id
                next_id += 1
                pool[bid] = (numel, ctype)

            assignments[name] = bid
            active.append((d, bid))

    return assignments, pool


def render_model_header(
    prefix: str,
    inputs: List[str],
    outputs: List[str],
    tensors: Dict[str, TensorInfo],
    quant: Optional[QuantConfig] = None,
) -> str:
    io_size_macros: List[str] = []
    io_type_macros: List[str] = []
    for i, n in enumerate(inputs):
        elem_type = runtime_elem_type(tensors[n].elem_type, quant)
        io_size_macros.append(
            f"#define {prefix.upper()}_INPUT_{i}_SIZE {tensors[n].numel}"
        )
        io_size_macros.append(
            f"#define {prefix.upper()}_INPUT_{i}_ELEM_SIZE {elem_size_for_elem_type(elem_type)}"
        )
        io_type_macros.append(
            f"#define {prefix.upper()}_INPUT_{i}_ONNX_TYPE {elem_type}"
        )
    for i, n in enumerate(outputs):
        elem_type = runtime_elem_type(tensors[n].elem_type, quant)
        io_size_macros.append(
            f"#define {prefix.upper()}_OUTPUT_{i}_SIZE {tensors[n].numel}"
        )
        io_size_macros.append(
            f"#define {prefix.upper()}_OUTPUT_{i}_ELEM_SIZE {elem_size_for_elem_type(elem_type)}"
        )
        io_type_macros.append(
            f"#define {prefix.upper()}_OUTPUT_{i}_ONNX_TYPE {elem_type}"
        )

    return render_template(
        "model_h.mako",
        guard=f"{prefix.upper()}_MODEL_H",
        prefix=prefix,
        prefix_upper=prefix.upper(),
        num_inputs=len(inputs),
        num_outputs=len(outputs),
        io_size_macros=io_size_macros,
        io_type_macros=io_type_macros,
    )


def render_kernels_header(prefix: str, rogue: int, quant: Optional[QuantConfig] = None) -> str:
    act_ctype = quant.act_ctype if quant and quant.act_bits else "float"
    quant_act_elem_type = quantized_onnx_elem_type(quant)
    return render_template(
        "kernels_h.mako",
        guard=f"{prefix.upper()}_KERNELS_H",
        prefix=prefix,
        rogue=rogue,
        act_ctype=act_ctype,
        quant_enabled=bool(quant and quant.enabled),
    )


def render_kernels_source(
    prefix: str,
    rogue: int,
    custom_kernels_header: Optional[str],
    quant: Optional[QuantConfig] = None,
) -> str:
    act_ctype = quant.act_ctype if quant and quant.act_bits else "float"
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return render_template(
        "kernels_c.mako",
        prefix=prefix,
        rogue=rogue,
        timestamp=timestamp,
        custom_kernels_header=custom_kernels_header,
        act_ctype=act_ctype,
        include_math=not (quant and quant.enabled),
        quant_enabled=bool(quant and quant.enabled),
    )


def get_attr_int(attrs: Dict[str, AttrValue], key: str, default: int) -> int:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, (int, float)):
        return int(value)
    raise CodegenError(f"Attribute '{key}' is not numeric")


def get_attr_ints(
    attrs: Dict[str, AttrValue], key: str, default: List[int]
) -> List[int]:
    if key not in attrs:
        return list(default)
    value = attrs[key]
    if isinstance(value, list):
        return [int(v) for v in value]
    raise CodegenError(f"Attribute '{key}' is not a list")


def resolve_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise CodegenError(f"Axis {axis} out of range for rank {rank}")
    return axis


def c_type_for_elem_type(elem_type: int) -> str:
    mapping = {
        TensorProto.FLOAT: "float",
        TensorProto.FLOAT16: "float",
        TensorProto.DOUBLE: "float",
        TensorProto.INT8: "int8_t",
        TensorProto.INT16: "int16_t",
        TensorProto.UINT8: "uint8_t",
        TensorProto.INT32: "int32_t",
    }
    if elem_type not in mapping:
        raise CodegenError(f"Unsupported runtime tensor type {elem_type}")
    return mapping[elem_type]


def elem_size_for_elem_type(elem_type: int) -> int:
    mapping = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 4,
        TensorProto.DOUBLE: 4,
        TensorProto.INT8: 1,
        TensorProto.INT16: 2,
        TensorProto.UINT8: 1,
        TensorProto.INT32: 4,
    }
    if elem_type not in mapping:
        raise CodegenError(f"Unsupported runtime tensor type {elem_type}")
    return mapping[elem_type]


def numpy_dtype_for_elem_type(elem_type: int) -> np.dtype:
    mapping = {
        TensorProto.FLOAT: np.dtype(np.float32),
        TensorProto.FLOAT16: np.dtype(np.float32),
        TensorProto.DOUBLE: np.dtype(np.float32),
        TensorProto.INT8: np.dtype(np.int8),
        TensorProto.INT16: np.dtype(np.int16),
        TensorProto.UINT8: np.dtype(np.uint8),
        TensorProto.INT32: np.dtype(np.int32),
    }
    if elem_type not in mapping:
        raise CodegenError(f"Unsupported runtime tensor type {elem_type}")
    return mapping[elem_type]


def copy_stmt(
    src: str, dst: str, info: TensorInfo, quant: Optional[QuantConfig] = None
) -> str:
    elem_type = runtime_elem_type(info.elem_type, quant)
    ctype = c_type_for_elem_type(elem_type)
    if ctype == "float":
        return f"    tensor_copy((const float*)({src}), (float*)({dst}), (size_t){info.numel});"
    nbytes = info.numel * elem_size_for_elem_type(elem_type)
    return f"    tensor_copy_bytes((const void*)({src}), (void*)({dst}), (size_t){nbytes});"


def render_runtime_helpers(quant: Optional[QuantConfig] = None) -> str:
    template_name = (
        "runtime_helpers_quant.c.inc"
        if quant and quant.enabled
        else "runtime_helpers.c.inc"
    )
    helpers_path = Path(__file__).resolve().parent / "templates" / template_name
    return helpers_path.read_text(encoding="utf-8").rstrip()


# ---------------------------------------------------------------------------
# Layer grouping – derive per-layer #if guards from ONNX node names
# ---------------------------------------------------------------------------


def _derive_layer_key(node_name: str) -> str:
    """Derive a layer grouping key from an ONNX node path name.

    Groups consecutive ops that share the same key into one layer block.
    Pad ops are merged with the following conv by returning the conv's key.
    """
    parts = [p for p in node_name.strip("/").split("/") if p]
    if not parts:
        return "misc"
    first = parts[0]

    # pad/Pad → "stem";  pad_N/Pad → "conv_N"
    m = re.match(r"^pad(?:_(\d+))?$", first, re.IGNORECASE)
    if m:
        n = m.group(1)
        return "stem" if n is None else f"conv_{n}"

    # conv/Op → "stem";  conv_N/Op → "conv_N"
    if first.lower() == "conv":
        return "stem"
    m = re.match(r"^conv_(\d+)$", first, re.IGNORECASE)
    if m:
        return first.lower()

    # Add → "block_0_add";  Add_N → "block_N_add"
    if first == "Add":
        return "block_0_add"
    m = re.match(r"^Add_(\d+)$", first)
    if m:
        return f"block_{m.group(1)}_add"

    # add_blocks_N_... → merge into "block_N_add"
    m = re.match(r"^add_blocks_(\d+)", first)
    if m:
        return f"block_{m.group(1)}_add"

    # Slice / Squeeze → "head"
    if first in ("Slice", "Squeeze"):
        return "head"

    # Fallback: sanitise the first component.
    return re.sub(r"[^a-z0-9]+", "_", first.lower()).strip("_")


def _layer_macro_name(prefix: str, layer_key: str) -> str:
    return f"{prefix.upper()}_LAYER_{layer_key.upper()}"


def render_layer_config_header(prefix: str, layer_keys: List[str]) -> str:
    """Generate a header with one ``#define`` per layer (all enabled)."""
    guard = f"{prefix.upper()}_LAYER_CFG_H"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "/*",
        " * Layer enable/disable switches.",
        " * Set a macro to 0 to skip the corresponding layer at compile time.",
        " * Generated automatically — safe to hand-edit.",
        " */",
        "",
    ]
    for key in layer_keys:
        macro = _layer_macro_name(prefix, key)
        lines.append(f"#define {macro} 1")
    lines += ["", f"#endif /* {guard} */", ""]
    return "\n".join(lines)


def render_model_source(
    prefix: str,
    tensors: Dict[str, TensorInfo],
    const_arrays: Dict[str, np.ndarray],
    nodes: List[NodeOp],
    inputs: List[str],
    outputs: List[str],
    custom_kernels_header: Optional[str],
    int8_weight_names: Optional[Set[str]] = None,
    quant: Optional[QuantConfig] = None,
    rogue: int = 0,
) -> str:
    if int8_weight_names is None:
        int8_weight_names = set()

    # Resolve the C type for activation buffers (float unless quant overrides)
    act_ctype = quant.act_ctype if quant and quant.act_bits else "float"
    quant_act_elem_type = quantized_onnx_elem_type(quant)
    weight_ctype = quant.weight_ctype if quant and quant.weight_bits else None

    shape_lines: List[str] = []
    for tname, info in sorted(tensors.items(), key=lambda kv: kv[0]):
        sym = safe_tensor_ref(tname)
        shape_lines.append(shape_c_array(f"{prefix}_shape_{sym}", info.shape))
        shape_lines.append(f"static const int {prefix}_rank_{sym} = {info.rank};")

    # Compute liveness-based buffer assignments to minimise scratch memory.
    buf_assignments, buf_pool = _compute_buffer_assignments(
        tensors, nodes, inputs, outputs, const_arrays, quant
    )

    # Emit one scratch buffer per alias group
    buffer_lines: List[str] = []
    for bid in sorted(buf_pool.keys()):
        numel, ctype = buf_pool[bid]
        buffer_lines.append(f"static {ctype} {prefix}_scratch_{bid}[{numel}];")

    lines: List[str] = []

    def tensor_expr(name: str) -> str:
        if name in const_arrays:
            return f"{prefix}_w_{safe_tensor_ref(name)}"
        if name in inputs:
            idx = inputs.index(name)
            ctype = c_type_for_elem_type(
                runtime_elem_type(tensors[name].elem_type, quant)
            )
            return f"((const {ctype}*)inputs[{idx}])"
        bid = buf_assignments[name]
        return f"{prefix}_scratch_{bid}"

    def shape_expr(name: str) -> str:
        return f"{prefix}_shape_{safe_tensor_ref(name)}"

    def rank_expr(name: str) -> str:
        return f"{prefix}_rank_{safe_tensor_ref(name)}"

    current_layer: Optional[str] = None
    layer_keys: List[str] = []

    node_idx = 0
    while node_idx < len(nodes):
        node = nodes[node_idx]
        op = node.op_type
        ins = [i for i in node.inputs if i]
        outs = [o for o in node.outputs if o]
        if len(outs) == 0:
            node_idx += 1
            continue

        out0 = outs[0]
        out_info = tensors[out0]

        # Layer grouping: insert #if/#endif at group boundaries.
        layer_key = _derive_layer_key(node.name)
        if layer_key != current_layer:
            if current_layer is not None:
                lines.append(f"#endif /* {_layer_macro_name(prefix, current_layer)} */")
                lines.append("")
            macro = _layer_macro_name(prefix, layer_key)
            lines.append(f"#if {macro}")
            current_layer = layer_key
            if layer_key not in layer_keys:
                layer_keys.append(layer_key)

        lines.append(f"    /* {node.name}: {op} */")

        if op == "Identity":
            lines.append(
                copy_stmt(tensor_expr(ins[0]), tensor_expr(out0), out_info, quant)
            )

        elif op == "Relu":
            lines.append(
                f"    tensor_relu({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel});"
            )

        elif op == "LeakyRelu":
            alpha_val = node.attrs.get("alpha", 0.01)
            alpha = float(cast(Union[int, float], alpha_val))
            lines.append(
                f"    tensor_leaky_relu({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel}, {c_float_literal(alpha)});"
            )

        elif op == "Sigmoid":
            lines.append(
                f"    tensor_sigmoid({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel});"
            )

        elif op == "Tanh":
            lines.append(
                f"    tensor_tanh({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel});"
            )

        elif op == "Clip":
            in_type = tensors[ins[0]].elem_type
            out_type = out_info.elem_type
            if len(ins) >= 3 and ins[1] in const_arrays and ins[2] in const_arrays:
                min_v = float(const_arrays[ins[1]].reshape(-1)[0])
                max_v = float(const_arrays[ins[2]].reshape(-1)[0])
            else:
                min_raw = node.attrs.get("min", -np.inf)
                max_raw = node.attrs.get("max", np.inf)
                min_v = float(cast(Union[int, float], min_raw))
                max_v = float(cast(Union[int, float], max_raw))
            if quant and quant.act_bits and in_type == TensorProto.INT32:
                lo_lit = str(int(min_v))
                hi_lit = str(int(max_v))
                clip_func = (
                    "tensor_clip_i32_to_act"
                    if out_type == quant_act_elem_type
                    else "tensor_clip_i32"
                )
            elif quant and quant.act_bits and out_type == quant_act_elem_type:
                lo_lit = f"ONNXCG_ACT({int(min_v)})"
                hi_lit = f"ONNXCG_ACT({int(max_v)})"
                clip_func = "tensor_clip_act"
            else:
                lo_lit = c_float_literal(min_v)
                hi_lit = c_float_literal(max_v)
                clip_func = "tensor_clip"
            lines.append(
                f"    {clip_func}({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel}, {lo_lit}, {hi_lit});"
            )

        elif op == "Add":
            next_nd = nodes[node_idx + 1] if node_idx + 1 < len(nodes) else None
            fuse_clip = (
                quant is not None
                and quant.act_bits
                and out_info.elem_type == TensorProto.INT32
                and next_nd is not None
                and next_nd.op_type == "Clip"
                and out0 in next_nd.inputs
                and tensors[ins[0]].elem_type == quant_act_elem_type
                and tensors[ins[1]].elem_type == quant_act_elem_type
                and tensors[next_nd.outputs[0]].elem_type == quant_act_elem_type
            )
            if fuse_clip:
                clip_out = next_nd.outputs[0]
                if tensor_expr(ins[0]) == tensor_expr(clip_out) or tensor_expr(
                    ins[1]
                ) == tensor_expr(clip_out):
                    fuse_clip = False

            if fuse_clip:
                min_v = 0.0
                max_v = 255.0
                clip_ins = [x for x in next_nd.inputs if x]
                if len(clip_ins) >= 3:
                    if clip_ins[1] in const_arrays:
                        min_v = float(const_arrays[clip_ins[1]].reshape(-1)[0])
                    if clip_ins[2] in const_arrays:
                        max_v = float(const_arrays[clip_ins[2]].reshape(-1)[0])
                else:
                    min_raw = next_nd.attrs.get("min", 0.0)
                    max_raw = next_nd.attrs.get("max", 255.0)
                    min_v = float(cast(Union[int, float], min_raw))
                    max_v = float(cast(Union[int, float], max_raw))

                lines.append(
                    "    tensor_add_broadcast_act_act_clip_to_act("
                    f"{tensor_expr(ins[0])}, {shape_expr(ins[0])}, {rank_expr(ins[0])}, "
                    f"{tensor_expr(ins[1])}, {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                    f"{tensor_expr(clip_out)}, {shape_expr(clip_out)}, {rank_expr(clip_out)}, "
                    f"{int(min_v)}, {int(max_v)}"
                    ");"
                )
                node_idx += 1
            else:
                if quant and quant.act_bits and out_info.elem_type == TensorProto.INT32:
                    a_type = tensors[ins[0]].elem_type
                    b_type = tensors[ins[1]].elem_type
                    if a_type == quant_act_elem_type and b_type == quant_act_elem_type:
                        add_func = "tensor_add_broadcast_act_act_i32"
                    elif a_type == TensorProto.INT32 and b_type == TensorProto.INT32:
                        add_func = "tensor_add_broadcast_i32"
                    else:
                        raise CodegenError(
                            f"Quantized Add type combination ({a_type}, {b_type}) "
                            f"is unsupported in node '{node.name}'"
                        )
                else:
                    add_func = "tensor_add_broadcast"
                lines.append(
                    f"    {add_func}("
                    f"{tensor_expr(ins[0])}, {shape_expr(ins[0])}, {rank_expr(ins[0])}, "
                    f"{tensor_expr(ins[1])}, {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                    f"{tensor_expr(out0)}, {shape_expr(out0)}, {rank_expr(out0)}"
                    ");"
                )

        elif op == "Mul":
            lines.append(
                "    tensor_mul_broadcast("
                f"{tensor_expr(ins[0])}, {shape_expr(ins[0])}, {rank_expr(ins[0])}, "
                f"{tensor_expr(ins[1])}, {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                f"{tensor_expr(out0)}, {shape_expr(out0)}, {rank_expr(out0)}"
                ");"
            )

        elif op == "Div":
            lines.append(
                "    tensor_div_broadcast("
                f"{tensor_expr(ins[0])}, {shape_expr(ins[0])}, {rank_expr(ins[0])}, "
                f"{tensor_expr(ins[1])}, {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                f"{tensor_expr(out0)}, {shape_expr(out0)}, {rank_expr(out0)}"
                ");"
            )

        elif op == "Floor":
            lines.append(
                f"    tensor_floor({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel});"
            )

        elif op == "RequantShift":
            # Standalone per-channel requant (fallback when not fused with Conv).
            act = tensors[ins[0]]
            kappa_name = ins[1]
            lambda_name = ins[2]
            shift = int(node.attrs["shift"])
            lo = int(node.attrs["lo"])
            hi = int(node.attrs["hi"])
            if act.rank == 3:
                n, c, w = act.shape
                lines.append(
                    f"    requant_channel_ncw("
                    f"{tensor_expr(ins[0])}, {tensor_expr(out0)}, "
                    f"{tensor_expr(kappa_name)}, {tensor_expr(lambda_name)}, "
                    f"{shift}, {lo}, {hi}, {n}, {c}, {w});"
                )
            else:
                raise CodegenError(
                    f"RequantShift only supports rank-3 NCW tensors in '{node.name}'"
                )

        elif op == "MatMul":
            a = tensors[ins[0]]
            b = tensors[ins[1]]
            if a.rank != 2 or b.rank != 2 or out_info.rank != 2:
                raise CodegenError(
                    f"MatMul currently requires rank-2 tensors in node '{node.name}'"
                )
            m, k = a.shape
            kb, n = b.shape
            if k != kb:
                raise CodegenError(
                    f"MatMul incompatible inner dims in node '{node.name}'"
                )
            lines.append(
                f"    ONNXCG_MATMUL2D_FUNC({tensor_expr(ins[0])}, {tensor_expr(ins[1])}, {tensor_expr(out0)}, {m}, {k}, {n});"
            )

        elif op == "Gemm":
            a = tensors[ins[0]]
            b = tensors[ins[1]]
            if a.rank != 2 or b.rank != 2 or out_info.rank != 2:
                raise CodegenError(
                    f"Gemm currently requires rank-2 tensors in node '{node.name}'"
                )
            trans_a = get_attr_int(node.attrs, "transA", 0)
            trans_b = get_attr_int(node.attrs, "transB", 0)
            alpha_raw = node.attrs.get("alpha", 1.0)
            beta_raw = node.attrs.get("beta", 1.0)
            alpha = float(cast(Union[int, float], alpha_raw))
            beta = float(cast(Union[int, float], beta_raw))

            a_m = a.shape[1] if trans_a else a.shape[0]
            a_k = a.shape[0] if trans_a else a.shape[1]
            b_k = b.shape[1] if trans_b else b.shape[0]
            b_n = b.shape[0] if trans_b else b.shape[1]
            if a_k != b_k:
                raise CodegenError(f"Gemm inner dim mismatch in node '{node.name}'")

            c_expr = "NULL"
            c_rank = 0
            c_dims = "NULL"
            if len(ins) >= 3 and ins[2]:
                c_expr = tensor_expr(ins[2])
                c_rank = tensors[ins[2]].rank
                c_dims = shape_expr(ins[2])

            lines.append(
                f"    ONNXCG_GEMM2D_FUNC({tensor_expr(ins[0])}, {tensor_expr(ins[1])}, {c_expr}, {tensor_expr(out0)}, "
                f"{a_m}, {a_k}, {b_n}, {trans_a}, {trans_b}, {c_float_literal(alpha)}, {c_float_literal(beta)}, {c_rank}, {c_dims});"
            )

        elif op == "Conv":
            x = tensors[ins[0]]
            w = tensors[ins[1]]
            b = tensor_expr(ins[2]) if len(ins) >= 3 and ins[2] else "NULL"
            groups = get_attr_int(node.attrs, "group", 1)
            dil = get_attr_ints(node.attrs, "dilations", [1] * (x.rank - 2))
            strides = get_attr_ints(node.attrs, "strides", [1] * (x.rank - 2))
            pads = get_attr_ints(node.attrs, "pads", [0] * (2 * (x.rank - 2)))

            # Look ahead: fuse Conv → RequantShift into a single kernel call
            # (pulp-nn style: accumulate int32 → requant → uint8 in one pass).
            next_nd = nodes[node_idx + 1] if node_idx + 1 < len(nodes) else None
            fuse_rq = (
                next_nd is not None
                and next_nd.op_type == "RequantShift"
                and out0 in next_nd.inputs
                and x.rank == 3
                and ins[1] in int8_weight_names
                and (len(ins) < 3 or not ins[2])  # no conv bias
            )
            if fuse_rq:
                rq_out = next_nd.outputs[0]
                if tensor_expr(ins[0]) == tensor_expr(rq_out):
                    fuse_rq = False

            if fuse_rq and x.rank == 3:
                n, cin, lin = x.shape
                cout, _, k = w.shape
                lout = out_info.shape[2]
                if len(pads) != 2:
                    raise CodegenError(
                        f"Conv1D expects 2 pad values in node '{node.name}'"
                    )
                rq = next_nd
                rq_out = rq.outputs[0]
                kappa_name = rq.inputs[1]
                lambda_name = rq.inputs[2]
                shift = int(rq.attrs["shift"])
                if rogue == 1:
                    conv_func = (
                        "ONNXCG_CONV1D_REQUANT_ROGUE_FUNC"
                    )

                    if x.elem_type == TensorProto.INT32:
                        lines.append(
                            f"    {conv_func}("
                            "(uint32_t)"f"{tensor_expr(ins[0])}, " "(uint32_t)"f"{tensor_expr(ins[1])}, "
                            "(uint32_t)"f"{tensor_expr(rq_out)}, "
                            f"{tensor_expr(kappa_name)}, {tensor_expr(lambda_name)}, {shift}, "
                            f"{cin}, {cout}, {lin}, {k}, {dil[0]}, " "DMA_DATA_TYPE_WORD, DMA_DATA_TYPE_BYTE);"
                        )
                    else:
                        lines.append(
                            f"    {conv_func}("
                            "(uint32_t)"f"{tensor_expr(ins[0])}, " "(uint32_t)"f"{tensor_expr(ins[1])}, "
                            "(uint32_t)"f"{tensor_expr(rq_out)}, "
                            f"{tensor_expr(kappa_name)}, {tensor_expr(lambda_name)}, {shift}, "
                            f"{cin}, {cout}, {lin}, {k}, {dil[0]}, " "DMA_DATA_TYPE_BYTE, DMA_DATA_TYPE_BYTE);"
                        )
                else:
                    conv_func = (
                        "ONNXCG_CONV1D_I32X_I8W_REQUANT_FUNC"
                        if x.elem_type == TensorProto.INT32
                        else "ONNXCG_CONV1D_I8W_REQUANT_FUNC"
                    )
                    lines.append(
                        f"    {conv_func}("
                        f"{tensor_expr(ins[0])}, {tensor_expr(ins[1])}, "
                        f"{tensor_expr(rq_out)}, "
                        f"{tensor_expr(kappa_name)}, {tensor_expr(lambda_name)}, {shift}, "
                        f"{n}, {cin}, {lin}, {cout}, {k}, "
                        f"{strides[0]}, {pads[0]}, {pads[1]}, {dil[0]}, {groups}, {lout});"
                    )
                node_idx += 1  # skip the RequantShift node

            elif x.rank == 3:
                n, cin, lin = x.shape
                cout, _, k = w.shape
                lout = out_info.shape[2]
                if len(pads) != 2:
                    raise CodegenError(
                        f"Conv1D expects 2 pad values in node '{node.name}'"
                    )
                if quant and ins[1] in int8_weight_names:
                    if len(ins) >= 3 and ins[2]:
                        bias_type = tensors[ins[2]].elem_type
                        if bias_type != TensorProto.INT32:
                            raise CodegenError(
                                "Quantized Conv1D currently requires int32 bias or no bias "
                                f"in node '{node.name}'"
                            )
                        b = f"(const int32_t*){tensor_expr(ins[2])}"
                    conv_func = (
                        "ONNXCG_CONV1D_I32X_I8W_FUNC"
                        if x.elem_type == TensorProto.INT32
                        else "ONNXCG_CONV1D_I8W_FUNC"
                    )
                    lines.append(
                        f"    {conv_func}({tensor_expr(ins[0])}, "
                        f"{tensor_expr(ins[1])}, {b}, (int32_t*){tensor_expr(out0)}, "
                        f"{n}, {cin}, {lin}, {cout}, {k}, {strides[0]}, {pads[0]}, {pads[1]}, {dil[0]}, {groups}, {lout});"
                    )
                else:
                    # Use int8 weight variant when the weight was compressed to int8_t
                    conv_func = "ONNXCG_CONV1D_FUNC"
                    if ins[1] in int8_weight_names:
                        bias_type = (
                            tensors[ins[2]].elem_type
                            if len(ins) >= 3 and ins[2]
                            else None
                        )
                        if bias_type == TensorProto.INT32:
                            conv_func = "ONNXCG_CONV1D_I8W_I32B_FUNC"
                            b = f"(const int32_t*){tensor_expr(ins[2])}"
                        else:
                            conv_func = "ONNXCG_CONV1D_I8W_FUNC"
                    lines.append(
                        f"    {conv_func}({tensor_expr(ins[0])}, {tensor_expr(ins[1])}, {b}, {tensor_expr(out0)}, "
                        f"{n}, {cin}, {lin}, {cout}, {k}, {strides[0]}, {pads[0]}, {pads[1]}, {dil[0]}, {groups}, {lout});"
                    )
            elif x.rank == 4:
                n, cin, hin, win = x.shape
                cout, _, kh, kw = w.shape
                hout, wout = out_info.shape[2], out_info.shape[3]
                if len(pads) != 4:
                    raise CodegenError(
                        f"Conv2D expects 4 pad values in node '{node.name}'"
                    )
                lines.append(
                    f"    conv2d_nchw({tensor_expr(ins[0])}, {tensor_expr(ins[1])}, {b}, {tensor_expr(out0)}, "
                    f"{n}, {cin}, {hin}, {win}, {cout}, {kh}, {kw}, {strides[0]}, {strides[1]}, "
                    f"{pads[0]}, {pads[1]}, {pads[2]}, {pads[3]}, {dil[0]}, {dil[1]}, {groups}, {hout}, {wout});"
                )
            else:
                raise CodegenError(
                    f"Conv currently supports rank-3/4 inputs only in node '{node.name}'"
                )

        elif op in ("MaxPool", "AveragePool"):
            x = tensors[ins[0]]
            if x.rank != 4:
                raise CodegenError(
                    f"{op} currently supports rank-4 NCHW inputs only in '{node.name}'"
                )
            n, c, hin, win = x.shape
            hout, wout = out_info.shape[2], out_info.shape[3]
            kernel = get_attr_ints(node.attrs, "kernel_shape", [1, 1])
            strides = get_attr_ints(node.attrs, "strides", [1, 1])
            dil = get_attr_ints(node.attrs, "dilations", [1, 1])
            pads = get_attr_ints(node.attrs, "pads", [0, 0, 0, 0])
            if len(kernel) != 2 or len(strides) != 2 or len(dil) != 2 or len(pads) != 4:
                raise CodegenError(f"{op} has unexpected attrs in node '{node.name}'")

            if op == "MaxPool":
                lines.append(
                    f"    maxpool2d_nchw({tensor_expr(ins[0])}, {tensor_expr(out0)}, {n}, {c}, {hin}, {win}, "
                    f"{kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pads[0]}, {pads[1]}, {dil[0]}, {dil[1]}, {hout}, {wout});"
                )
            else:
                count_include_pad = get_attr_int(node.attrs, "count_include_pad", 0)
                lines.append(
                    f"    avgpool2d_nchw({tensor_expr(ins[0])}, {tensor_expr(out0)}, {n}, {c}, {hin}, {win}, "
                    f"{kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pads[0]}, {pads[1]}, {dil[0]}, {dil[1]}, {hout}, {wout}, {count_include_pad});"
                )

        elif op == "GlobalAveragePool":
            x = tensors[ins[0]]
            if x.rank != 4 or out_info.rank != 4:
                raise CodegenError(
                    f"GlobalAveragePool expects rank-4 NCHW in '{node.name}'"
                )
            n, c, h, w = x.shape
            lines.append(
                f"    global_avg_pool_nchw({tensor_expr(ins[0])}, {tensor_expr(out0)}, {n}, {c}, {h}, {w});"
            )

        elif op == "Flatten":
            lines.append(
                copy_stmt(tensor_expr(ins[0]), tensor_expr(out0), out_info, quant)
            )

        elif op in ("Reshape", "Squeeze", "Unsqueeze"):
            lines.append(
                copy_stmt(tensor_expr(ins[0]), tensor_expr(out0), out_info, quant)
            )

        elif op == "QuantizeLinear":
            if len(ins) < 3:
                raise CodegenError(
                    f"QuantizeLinear requires x/scale/zero_point in node '{node.name}'"
                )
            qtype = tensors[out0].elem_type
            if qtype == TensorProto.UINT8:
                lines.append(
                    f"    quantize_linear_u8((const float*)({tensor_expr(ins[0])}), (uint8_t*)({tensor_expr(out0)}), "
                    f"(size_t){out_info.numel}, (const float*)({tensor_expr(ins[1])}), {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                    f"(const uint8_t*)({tensor_expr(ins[2])}), {shape_expr(ins[2])}, {rank_expr(ins[2])});"
                )
            elif qtype == TensorProto.INT8:
                lines.append(
                    f"    quantize_linear_s8((const float*)({tensor_expr(ins[0])}), (int8_t*)({tensor_expr(out0)}), "
                    f"(size_t){out_info.numel}, (const float*)({tensor_expr(ins[1])}), {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                    f"(const int8_t*)({tensor_expr(ins[2])}), {shape_expr(ins[2])}, {rank_expr(ins[2])});"
                )
            else:
                raise CodegenError(
                    f"QuantizeLinear output type {qtype} unsupported in node '{node.name}'"
                )

        elif op == "DequantizeLinear":
            if len(ins) < 3:
                raise CodegenError(
                    f"DequantizeLinear requires x/scale/zero_point in node '{node.name}'"
                )
            qtype = tensors[ins[0]].elem_type
            if qtype == TensorProto.UINT8:
                lines.append(
                    f"    dequantize_linear_u8((const uint8_t*)({tensor_expr(ins[0])}), (float*)({tensor_expr(out0)}), "
                    f"(size_t){out_info.numel}, (const float*)({tensor_expr(ins[1])}), {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                    f"(const uint8_t*)({tensor_expr(ins[2])}), {shape_expr(ins[2])}, {rank_expr(ins[2])});"
                )
            elif qtype == TensorProto.INT8:
                lines.append(
                    f"    dequantize_linear_s8((const int8_t*)({tensor_expr(ins[0])}), (float*)({tensor_expr(out0)}), "
                    f"(size_t){out_info.numel}, (const float*)({tensor_expr(ins[1])}), {shape_expr(ins[1])}, {rank_expr(ins[1])}, "
                    f"(const int8_t*)({tensor_expr(ins[2])}), {shape_expr(ins[2])}, {rank_expr(ins[2])});"
                )
            else:
                raise CodegenError(
                    f"DequantizeLinear input type {qtype} unsupported in node '{node.name}'"
                )

        elif op == "QLinearConv":
            if len(ins) < 8:
                raise CodegenError(
                    f"QLinearConv expects at least 8 inputs in node '{node.name}'"
                )
            x = tensors[ins[0]]
            w = tensors[ins[3]]
            y = tensors[out0]
            if x.rank != 3 or w.rank != 3 or y.rank != 3:
                raise CodegenError(
                    f"QLinearConv currently supports 1D NCW only in node '{node.name}'"
                )
            if (
                x.elem_type != TensorProto.UINT8
                or w.elem_type != TensorProto.INT8
                or y.elem_type != TensorProto.UINT8
            ):
                raise CodegenError(
                    f"QLinearConv currently supports uint8 activations + int8 weights + uint8 outputs in node '{node.name}'"
                )
            groups = get_attr_int(node.attrs, "group", 1)
            dil = get_attr_ints(node.attrs, "dilations", [1])
            strides = get_attr_ints(node.attrs, "strides", [1])
            pads = get_attr_ints(node.attrs, "pads", [0, 0])
            n, cin, lin = x.shape
            cout, _, k = w.shape
            lout = y.shape[2]
            b = "NULL"
            if len(ins) >= 9 and ins[8]:
                b = tensor_expr(ins[8])
            lines.append(
                f"    ONNXCG_QLINEAR_CONV1D_FUNC((const uint8_t*)({tensor_expr(ins[0])}), (const float*)({tensor_expr(ins[1])}), (const uint8_t*)({tensor_expr(ins[2])}), "
                f"(const int8_t*)({tensor_expr(ins[3])}), (const float*)({tensor_expr(ins[4])}), (const int8_t*)({tensor_expr(ins[5])}), "
                f"(const float*)({tensor_expr(ins[6])}), (const uint8_t*)({tensor_expr(ins[7])}), (const int32_t*)({b}), (uint8_t*)({tensor_expr(out0)}), "
                f"{n}, {cin}, {lin}, {cout}, {k}, {strides[0]}, {pads[0]}, {pads[1]}, {dil[0]}, {groups}, {lout});"
            )

        elif op == "Pad":
            x = tensors[ins[0]]
            rank = x.rank
            if len(ins) >= 2 and ins[1] in const_arrays:
                pads_raw = const_arrays[ins[1]].reshape(-1)
                pads = [int(v) for v in pads_raw]
            else:
                pads = get_attr_ints(node.attrs, "pads", [0] * (2 * rank))
            if len(pads) != 2 * rank:
                raise CodegenError(
                    f"Pad expects {2 * rank} pad values in node '{node.name}'"
                )

            mode = str(node.attrs.get("mode", "constant"))
            if mode != "constant":
                raise CodegenError(
                    f"Pad mode '{mode}' is not supported in node '{node.name}'"
                )

            pad_value = 0.0
            if len(ins) >= 3 and ins[2] in const_arrays:
                pad_value = float(const_arrays[ins[2]].reshape(-1)[0])

            name_s = safe_tensor_ref(out0)
            pb = pads[:rank]
            pe = pads[rank:]
            pb_vals = ", ".join(str(v) for v in pb)
            pe_vals = ", ".join(str(v) for v in pe)
            lines.append(
                f"    static const int {prefix}_pad_begin_{name_s}[{rank}] = {{{pb_vals}}};"
            )
            lines.append(
                f"    static const int {prefix}_pad_end_{name_s}[{rank}] = {{{pe_vals}}};"
            )
            if quant and quant.act_bits and x.elem_type == TensorProto.INT32:
                pad_lit = str(int(pad_value))
                pad_func = "pad_tensor_constant_i32"
            elif quant and quant.act_bits and x.elem_type == quant_act_elem_type:
                pad_lit = f"ONNXCG_ACT({int(pad_value)})"
                pad_func = "pad_tensor_constant_act"
            else:
                pad_lit = c_float_literal(pad_value)
                pad_func = "pad_tensor_constant"
            lines.append(
                f"    {pad_func}({tensor_expr(ins[0])}, {tensor_expr(out0)}, {shape_expr(ins[0])}, {shape_expr(out0)}, {rank}, "
                f"{prefix}_pad_begin_{name_s}, {prefix}_pad_end_{name_s}, {pad_lit});"
            )

        elif op == "Slice":
            x = tensors[ins[0]]
            rank = x.rank
            if len(ins) < 3 or ins[1] not in const_arrays or ins[2] not in const_arrays:
                raise CodegenError(
                    f"Slice in node '{node.name}' requires constant starts/ends"
                )

            starts_raw = [int(v) for v in const_arrays[ins[1]].reshape(-1)]
            ends_raw = [int(v) for v in const_arrays[ins[2]].reshape(-1)]
            if len(starts_raw) != len(ends_raw):
                raise CodegenError(f"Slice starts/ends mismatch in node '{node.name}'")

            if len(ins) >= 4 and ins[3] in const_arrays:
                axes_raw = [int(v) for v in const_arrays[ins[3]].reshape(-1)]
            else:
                axes_raw = list(range(len(starts_raw)))

            if len(ins) >= 5 and ins[4] in const_arrays:
                steps_raw = [int(v) for v in const_arrays[ins[4]].reshape(-1)]
            else:
                steps_raw = [1] * len(starts_raw)

            starts = [0] * rank
            steps = [1] * rank

            for i, axis_raw in enumerate(axes_raw):
                axis = resolve_axis(axis_raw, rank)
                step = steps_raw[i]
                if step <= 0:
                    raise CodegenError(
                        f"Slice only supports positive steps in node '{node.name}'"
                    )
                dim = x.shape[axis]
                start = starts_raw[i]
                end = ends_raw[i]

                if start < 0:
                    start += dim
                if end < 0:
                    end += dim
                if start < 0:
                    start = 0
                if start > dim:
                    start = dim
                if end < 0:
                    end = 0
                if end > dim:
                    end = dim

                starts[axis] = start
                steps[axis] = step

            name_s = safe_tensor_ref(out0)
            s_vals = ", ".join(str(v) for v in starts)
            st_vals = ", ".join(str(v) for v in steps)
            lines.append(
                f"    static const int {prefix}_slice_starts_{name_s}[{rank}] = {{{s_vals}}};"
            )
            lines.append(
                f"    static const int {prefix}_slice_steps_{name_s}[{rank}] = {{{st_vals}}};"
            )
            if quant and quant.act_bits and x.elem_type == TensorProto.INT32:
                slice_func = "slice_tensor_i32"
            elif quant and quant.act_bits and x.elem_type == quant_act_elem_type:
                slice_func = "slice_tensor_act"
            else:
                slice_func = "slice_tensor"
            lines.append(
                f"    {slice_func}({tensor_expr(ins[0])}, {tensor_expr(out0)}, {shape_expr(ins[0])}, {shape_expr(out0)}, {rank}, "
                f"{prefix}_slice_starts_{name_s}, {prefix}_slice_steps_{name_s});"
            )

        elif op == "Transpose":
            in_rank = tensors[ins[0]].rank
            perm_raw = node.attrs.get("perm", list(reversed(range(in_rank))))
            perm = cast(List[int], perm_raw)
            if len(perm) != in_rank:
                raise CodegenError(
                    f"Transpose perm length mismatch in node '{node.name}'"
                )
            perm_name = f"{prefix}_perm_{safe_tensor_ref(out0)}"
            perm_vals = ", ".join(str(int(v)) for v in perm)
            lines.append(
                f"    static const int {perm_name}[{len(perm)}] = {{{perm_vals}}};"
            )
            lines.append(
                f"    transpose_tensor({tensor_expr(ins[0])}, {tensor_expr(out0)}, {shape_expr(ins[0])}, {perm_name}, {in_rank});"
            )

        elif op == "Softmax":
            axis = get_attr_int(node.attrs, "axis", -1)
            axis = resolve_axis(axis, tensors[ins[0]].rank)
            lines.append(
                f"    softmax_axis({tensor_expr(ins[0])}, {tensor_expr(out0)}, {shape_expr(ins[0])}, {rank_expr(ins[0])}, {axis});"
            )

        elif op == "BatchNormalization":
            x = tensors[ins[0]]
            if x.rank < 2:
                raise CodegenError(
                    f"BatchNormalization expects rank>=2 in node '{node.name}'"
                )
            n = x.shape[0]
            c = x.shape[1]
            spatial = 1
            for d in x.shape[2:]:
                spatial *= d
            eps_raw = node.attrs.get("epsilon", 1e-5)
            eps = float(cast(Union[int, float], eps_raw))
            lines.append(
                f"    batch_norm_nchw({tensor_expr(ins[0])}, {tensor_expr(out0)}, {tensor_expr(ins[1])}, {tensor_expr(ins[2])}, "
                f"{tensor_expr(ins[3])}, {tensor_expr(ins[4])}, {n}, {c}, {spatial}, {c_float_literal(eps)});"
            )

        elif op == "Concat":
            axis = get_attr_int(node.attrs, "axis", 0)
            axis = resolve_axis(axis, out_info.rank)
            arr_inputs = ", ".join(tensor_expr(i) for i in ins)
            arr_dims = ", ".join(shape_expr(i) for i in ins)
            arr_ranks = ", ".join(rank_expr(i) for i in ins)
            name_s = safe_tensor_ref(out0)
            lines.append(
                f"    static const ONNXCG_ACT_T* {prefix}_cat_inputs_{name_s}[{len(ins)}] = {{{arr_inputs}}};"
            )
            lines.append(
                f"    static const int* {prefix}_cat_dims_{name_s}[{len(ins)}] = {{{arr_dims}}};"
            )
            lines.append(
                f"    static const int {prefix}_cat_ranks_{name_s}[{len(ins)}] = {{{arr_ranks}}};"
            )
            lines.append(
                f"    concat_axis({prefix}_cat_inputs_{name_s}, {prefix}_cat_dims_{name_s}, {prefix}_cat_ranks_{name_s}, {len(ins)}, "
                f"{tensor_expr(out0)}, {shape_expr(out0)}, {rank_expr(out0)}, {axis});"
            )

        else:
            raise CodegenError(f"Internal error: unsupported op dispatch for {op}")

        lines.append("")
        node_idx += 1

    # Close the last layer group.
    if current_layer is not None:
        lines.append(f"#endif /* {_layer_macro_name(prefix, current_layer)} */")

    output_copy_lines: List[str] = []
    for i, oname in enumerate(outputs):
        output_copy_lines.append(
            copy_stmt(tensor_expr(oname), f"outputs[{i}]", tensors[oname], quant)
        )

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    rendered = render_template(
        "model_c.mako",
        prefix=prefix,
        timestamp=timestamp,
        num_inputs=len(inputs),
        num_outputs=len(outputs),
        tensors=tensors,
        inputs=inputs,
        outputs=outputs,
        custom_kernels_header=custom_kernels_header,
        act_ctype=act_ctype,
        shape_defs="\n".join(shape_lines),
        buffer_defs="\n".join(buffer_lines),
        runtime_helpers=render_runtime_helpers(quant),
        include_math=not (quant and quant.enabled),
        ops_body="\n".join(lines),
        output_copies="\n".join(output_copy_lines),
    )
    return rendered, layer_keys


def _ctype_size(ctype: str) -> int:
    mapping = {
        "float": 4,
        "double": 8,
        "int8_t": 1,
        "uint8_t": 1,
        "int16_t": 2,
        "uint16_t": 2,
        "int32_t": 4,
        "uint32_t": 4,
        "int64_t": 8,
        "uint64_t": 8,
    }
    if ctype not in mapping:
        raise CodegenError(f"Unsupported C type in memory breakdown: {ctype}")
    return mapping[ctype]


def _resolve_onnx_path(
    onnx_path: Path,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory[str]]]:
    if onnx_path.suffix.lower() != ".zip":
        return onnx_path, None

    tmp_dir = tempfile.TemporaryDirectory(prefix="onnx_codegen_")
    extract_dir = Path(tmp_dir.name)
    with zipfile.ZipFile(onnx_path, "r") as zf:
        zf.extractall(extract_dir)

    onnx_files = sorted(extract_dir.rglob("*.onnx"))
    if not onnx_files:
        tmp_dir.cleanup()
        raise CodegenError(
            f"Zip archive '{onnx_path}' does not contain any .onnx files"
        )

    return onnx_files[0], tmp_dir


def _render_compare_main(
    prefix: str,
    input_elem_types: Sequence[int],
    output_elem_types: Sequence[int],
) -> str:
    prefix_upper = prefix.upper()
    lines = [
        "#include <stdio.h>",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "",
        f'#include "{prefix}_model.h"',
        "",
        "static int read_exact(const char* path, void* dst, size_t nbytes) {",
        '    FILE* fp = fopen(path, "rb");',
        "    if (fp == NULL) return -1;",
        "    size_t got = fread(dst, 1, nbytes, fp);",
        "    fclose(fp);",
        "    return got == nbytes ? 0 : -2;",
        "}",
        "",
        "static int write_exact(const char* path, const void* src, size_t nbytes) {",
        '    FILE* fp = fopen(path, "wb");',
        "    if (fp == NULL) return -1;",
        "    size_t wrote = fwrite(src, 1, nbytes, fp);",
        "    fclose(fp);",
        "    return wrote == nbytes ? 0 : -2;",
        "}",
        "",
        "int main(void) {",
    ]

    for idx, elem_type in enumerate(input_elem_types):
        ctype = c_type_for_elem_type(elem_type)
        lines.append(
            f"    {ctype} input_{idx}[{prefix_upper}_INPUT_{idx}_SIZE] = {{0}};"
        )
    for idx, elem_type in enumerate(output_elem_types):
        ctype = c_type_for_elem_type(elem_type)
        lines.append(
            f"    {ctype} output_{idx}[{prefix_upper}_OUTPUT_{idx}_SIZE] = {{0}};"
        )

    lines.append("")
    for idx in range(len(input_elem_types)):
        lines.append(
            f'    if (read_exact("input_{idx}.bin", input_{idx}, sizeof(input_{idx})) != 0) return {10 + idx};'
        )

    input_args = ", ".join(f"input_{idx}" for idx in range(len(input_elem_types)))
    output_args = ", ".join(f"output_{idx}" for idx in range(len(output_elem_types)))
    lines.extend(
        [
            f"    const void* inputs[{len(input_elem_types)}] = {{{input_args}}};",
            f"    void* outputs[{len(output_elem_types)}] = {{{output_args}}};",
            f"    int rc = {prefix}_infer(inputs, outputs);",
            "    if (rc != 0) return rc;",
        ]
    )

    for idx in range(len(output_elem_types)):
        lines.append(
            f'    if (write_exact("output_{idx}.bin", output_{idx}, sizeof(output_{idx})) != 0) return {40 + idx};'
        )

    lines.extend(["    return 0;", "}", ""])
    return "\n".join(lines)


def _compiler_command(cc: Optional[str]) -> str:
    if cc:
        return cc
    return os.environ.get("CC", "gcc")


def _generate_compare_inputs(
    tensors: Dict[str, TensorInfo],
    inputs: Sequence[str],
    quant: Optional[QuantConfig],
    random_cases: int,
    seed: int,
) -> List[Tuple[str, List[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    cases: List[Tuple[str, List[np.ndarray]]] = []

    def make_case_arrays(case_idx: Optional[int]) -> List[np.ndarray]:
        arrays: List[np.ndarray] = []
        for name in inputs:
            info = tensors[name]
            elem_type = runtime_elem_type(info.elem_type, quant)
            dtype = numpy_dtype_for_elem_type(elem_type)
            shape = tuple(info.shape)
            if case_idx is None:
                arr = np.zeros(shape, dtype=dtype)
            else:
                if elem_type == TensorProto.UINT8:
                    base = rng.integers(0, 17, size=shape, dtype=np.int32)
                    arr = base.astype(dtype)
                elif elem_type in (
                    TensorProto.INT8,
                    TensorProto.INT16,
                    TensorProto.INT32,
                ):
                    base = rng.integers(-8, 9, size=shape, dtype=np.int32)
                    arr = base.astype(dtype)
                else:
                    base = rng.integers(-8, 9, size=shape, dtype=np.int32)
                    arr = base.astype(np.float32)
            arrays.append(np.ascontiguousarray(arr))
        return arrays

    cases.append(("zeros", make_case_arrays(None)))
    for case_idx in range(random_cases):
        cases.append((f"random_{case_idx}", make_case_arrays(case_idx)))
    return cases


def _prepare_reference_feed(
    input_names: Sequence[str],
    ref_tensors: Dict[str, TensorInfo],
    c_inputs: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    for name, carr in zip(input_names, c_inputs):
        ref_dtype = numpy_dtype_for_elem_type(ref_tensors[name].elem_type)
        feed[name] = np.ascontiguousarray(carr.astype(ref_dtype, copy=False))
    return feed


def compare_generated_c_to_onnx(
    onnx_path: Path,
    out_dir: Path,
    prefix: str,
    *,
    reference_onnx_path: Optional[Path] = None,
    skip_shape_inference: bool,
    quant: Optional[QuantConfig] = None,
    random_cases: int = 2,
    seed: int = 0,
    atol: float = 1e-5,
    cc: Optional[str] = None,
) -> CompareResult:
    """Compare generated C inference against the reference ONNX model.

    The ONNX reference is always evaluated from the float-parameter model. This
    is intentional: some exported models store convolution parameters as
    ``float32`` even though their numeric magnitude is already integer-consistent
    with the generated C path.

    A temporary ``main.c`` is created only for the comparison executable and is
    deleted automatically afterwards.
    """

    resolved_onnx, tmp_dir = _resolve_onnx_path(onnx_path)
    ref_source = reference_onnx_path if reference_onnx_path is not None else onnx_path
    resolved_ref_onnx, ref_tmp_dir = _resolve_onnx_path(ref_source)
    try:
        codegen_model = read_model(
            resolved_onnx, skip_shape_inference=skip_shape_inference
        )
        codegen_model = sanitize_quantized_model(codegen_model, quant)
        tensors, const_arrays, nodes, inputs, outputs = build_graph(codegen_model)
        if quant:
            nodes = _fuse_requant(nodes, const_arrays, tensors)
        nodes = _fuse_zero_pad_into_conv1d(nodes, const_arrays, tensors, outputs)
        if quant:
            _retype_integer_valued_float_constants(tensors, const_arrays, nodes)
        _assign_quantized_runtime_tensor_types(
            tensors, const_arrays, nodes, inputs, quant
        )

        ref_model = read_model(
            resolved_ref_onnx, skip_shape_inference=skip_shape_inference
        )
        ref_model = sanitize_quantized_model(ref_model, None)
        ref_tensors, _, _, ref_inputs, _ = build_graph(ref_model)

        if list(inputs) != list(ref_inputs):
            raise CodegenError(
                "Generated C inputs do not match ONNX reference inputs; "
                "cannot compare safely."
            )

        try:
            from onnx.reference import ReferenceEvaluator
        except ImportError as exc:
            raise CodegenError(
                "ONNX reference evaluator is unavailable in this environment."
            ) from exc

        ref_eval = ReferenceEvaluator(ref_model)

        model_h_path = out_dir / f"{prefix}_model.h"
        model_c_path = out_dir / f"{prefix}_model.c"
        kernels_c_path = out_dir / f"{prefix}_kernels.c"
        if not model_h_path.exists() or not model_c_path.exists():
            raise CodegenError(
                "Generated C files not found for comparison; run code generation first."
            )

        input_elem_types = [
            runtime_elem_type(tensors[name].elem_type, quant) for name in inputs
        ]
        output_elem_types = [
            runtime_elem_type(tensors[name].elem_type, quant) for name in outputs
        ]
        output_dtypes = [numpy_dtype_for_elem_type(t) for t in output_elem_types]
        output_shapes = [tuple(tensors[name].shape) for name in outputs]

        compiler = _compiler_command(cc)
        cases = _generate_compare_inputs(tensors, inputs, quant, random_cases, seed)
        case_results: List[CompareCaseResult] = []

        with tempfile.TemporaryDirectory(
            prefix="onnx_codegen_compare_"
        ) as build_dir_name:
            build_dir = Path(build_dir_name)
            main_c_path = build_dir / "main.c"
            exe_path = build_dir / "compare_model"
            main_c_path.write_text(
                _render_compare_main(prefix, input_elem_types, output_elem_types),
                encoding="utf-8",
            )

            compile_cmd = [
                compiler,
                "-std=c99",
                str(main_c_path),
                str(model_c_path),
            ]
            if kernels_c_path.exists():
                compile_cmd.append(str(kernels_c_path))
            compile_cmd += [
                "-I",
                str(out_dir),
                "-lm",
                "-o",
                str(exe_path),
            ]
            compile_res = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if compile_res.returncode != 0:
                raise CodegenError(
                    "Failed to compile generated C for comparison.\n"
                    + compile_res.stderr
                )

            for case_name, c_inputs in cases:
                case_dir = build_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                for idx, arr in enumerate(c_inputs):
                    arr.tofile(case_dir / f"input_{idx}.bin")

                run_res = subprocess.run(
                    [str(exe_path)],
                    cwd=case_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if run_res.returncode != 0:
                    raise CodegenError(
                        f"Generated C comparison executable failed for case '{case_name}'.\n"
                        + run_res.stderr
                    )

                c_outputs: List[np.ndarray] = []
                for idx, (dtype, shape) in enumerate(zip(output_dtypes, output_shapes)):
                    raw = np.fromfile(case_dir / f"output_{idx}.bin", dtype=dtype)
                    expected = int(np.prod(shape, dtype=np.int64))
                    if raw.size != expected:
                        raise CodegenError(
                            f"Output '{idx}' size mismatch in case '{case_name}': "
                            f"expected {expected} values, got {raw.size}."
                        )
                    c_outputs.append(raw.reshape(shape))

                ref_feed = _prepare_reference_feed(inputs, ref_tensors, c_inputs)
                ref_outputs = ref_eval.run(None, ref_feed)

                case_match = True
                case_max_abs_diff = 0.0
                for c_out, ref_out in zip(c_outputs, ref_outputs):
                    ref_arr = np.asarray(ref_out)
                    if c_out.shape != ref_arr.shape:
                        case_match = False
                        case_max_abs_diff = float("inf")
                        break
                    diff = np.abs(c_out.astype(np.float64) - ref_arr.astype(np.float64))
                    max_abs_diff = float(np.max(diff)) if diff.size else 0.0
                    case_max_abs_diff = max(case_max_abs_diff, max_abs_diff)
                    if not np.allclose(
                        c_out.astype(np.float64),
                        ref_arr.astype(np.float64),
                        rtol=0.0,
                        atol=atol,
                    ):
                        case_match = False

                case_results.append(
                    CompareCaseResult(
                        name=case_name,
                        matches=case_match,
                        max_abs_diff=case_max_abs_diff,
                    )
                )

        return CompareResult(
            matches=all(case.matches for case in case_results),
            cases=case_results,
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if ref_tmp_dir is not None:
            ref_tmp_dir.cleanup()


def generate_library(
    onnx_path: Path,
    out_dir: Path,
    prefix: str,
    skip_shape_inference: bool,
    custom_kernels_header: Optional[str] = None,
    quant: Optional[QuantConfig] = None,
    rogue: int = 0,
) -> GenerationResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_onnx, tmp_dir = _resolve_onnx_path(onnx_path)
    try:
        model = read_model(resolved_onnx, skip_shape_inference=skip_shape_inference)
        model = sanitize_quantized_model(model, quant)
        tensors, const_arrays, nodes, inputs, outputs = build_graph(model)

        # Fuse Mul→Add→Div→Floor→Clip into RequantShift (or identity Clip).
        if quant:
            nodes = _fuse_requant(nodes, const_arrays, tensors)
        nodes = _fuse_zero_pad_into_conv1d(nodes, const_arrays, tensors, outputs)

        if quant:
            _retype_integer_valued_float_constants(tensors, const_arrays, nodes)
        _assign_quantized_runtime_tensor_types(
            tensors, const_arrays, nodes, inputs, quant
        )

        # Identify constants that are inlined at codegen time (never
        # referenced as weight symbols in the generated C).  These still
        # need to stay in const_arrays so render_model_source can read
        # their values, but should be excluded from the weights header.
        inlined_consts: Set[str] = set()
        for n in nodes:
            if n.op_type == "Slice":
                for idx in range(1, min(len(n.inputs), 5)):
                    if n.inputs[idx] and n.inputs[idx] in const_arrays:
                        inlined_consts.add(n.inputs[idx])
            elif n.op_type == "Clip":
                for idx in range(1, min(len(n.inputs), 3)):
                    if n.inputs[idx] and n.inputs[idx] in const_arrays:
                        inlined_consts.add(n.inputs[idx])
        # Also drop any constants not referenced by any remaining node.
        referenced = {inp for n in nodes for inp in n.inputs if inp}
        inlined_consts |= {k for k in const_arrays if k not in referenced}

        # Build weight arrays excluding inlined/dead constants.
        weight_arrays = {
            k: v for k, v in const_arrays.items() if k not in inlined_consts
        }
        _, buf_pool = _compute_buffer_assignments(
            tensors, nodes, inputs, outputs, const_arrays, quant
        )
        model_h = render_model_header(prefix, inputs, outputs, tensors, quant)
        kernels_h = render_kernels_header(prefix, rogue, quant)
        kernels_c = render_kernels_source(prefix, rogue, custom_kernels_header, quant)
        weights_h, int8_weight_names = render_weights_header(
            prefix, weight_arrays, tensors, quant
        )
        model_c, layer_keys = render_model_source(
            prefix,
            tensors,
            const_arrays,
            nodes,
            inputs,
            outputs,
            custom_kernels_header,
            int8_weight_names,
            quant,
            rogue
        )
        layer_cfg_h = render_layer_config_header(prefix, layer_keys)

        model_h_path = out_dir / f"{prefix}_model.h"
        model_c_path = out_dir / f"{prefix}_model.c"
        kernels_h_path = out_dir / f"{prefix}_kernels.h"
        kernels_c_path = out_dir / f"{prefix}_kernels.c"
        weights_h_path = out_dir / f"{prefix}_weights.h"
        layer_cfg_h_path = out_dir / f"{prefix}_layer_cfg.h"

        model_h_path.write_text(model_h, encoding="utf-8")
        model_c_path.write_text(model_c, encoding="utf-8")
        kernels_h_path.write_text(kernels_h, encoding="utf-8")
        kernels_c_path.write_text(kernels_c, encoding="utf-8")
        weights_h_path.write_text(weights_h, encoding="utf-8")
        # Only overwrite layer_cfg.h if it doesn't exist — the user
        # may have hand-edited it to disable specific layers.
        if not layer_cfg_h_path.exists():
            layer_cfg_h_path.write_text(layer_cfg_h, encoding="utf-8")

        weights_bytes = int(sum(arr.nbytes for arr in weight_arrays.values()))
        inlined_const_bytes = int(
            sum(
                const_arrays[name].nbytes
                for name in inlined_consts
                if name in const_arrays
            )
        )
        scratch_bytes = int(
            sum(numel * _ctype_size(ctype) for numel, ctype in buf_pool.values())
        )
        input_bytes = int(
            sum(
                tensors[name].numel
                * elem_size_for_elem_type(
                    runtime_elem_type(tensors[name].elem_type, quant)
                )
                for name in inputs
            )
        )
        output_bytes = int(
            sum(
                tensors[name].numel
                * elem_size_for_elem_type(
                    runtime_elem_type(tensors[name].elem_type, quant)
                )
                for name in outputs
            )
        )

        return GenerationResult(
            model_h_path=model_h_path,
            model_c_path=model_c_path,
            weights_h_path=weights_h_path,
            kernels_h_path=kernels_h_path,
            kernels_c_path=kernels_c_path,
            n_inputs=len(inputs),
            n_outputs=len(outputs),
            n_nodes=len(nodes),
            memory=MemoryBreakdown(
                weights_bytes=weights_bytes,
                scratch_bytes=scratch_bytes,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                inlined_const_bytes=inlined_const_bytes,
                num_scratch_buffers=len(buf_pool),
            ),
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _format_c_array(arr: np.ndarray, ctype: str) -> str:
    """Format a flat numpy array as a comma-separated C initialiser string."""
    flat = arr.reshape(-1)
    if ctype == "float":
        return ", ".join(c_float_literal(float(v)) for v in flat)
    return ", ".join(str(int(v)) for v in flat)


def generate_test_data_header(
    onnx_path: Path,
    out_dir: Path,
    prefix: str,
    skip_shape_inference: bool,
    quant: Optional[QuantConfig] = None,
    random_cases: int = 2,
    seed: int = 0,
) -> Path:
    """Generate a ``<prefix>_test_data.h`` with random inputs and golden outputs.

    The golden outputs are produced by the ONNX reference evaluator so the
    generated header can be used to validate the C inference on target MCU
    hardware.
    """
    resolved_onnx, tmp_dir = _resolve_onnx_path(onnx_path)
    try:
        # Quantized model – used for C-side types and input generation.
        model = read_model(resolved_onnx, skip_shape_inference=skip_shape_inference)
        model = sanitize_quantized_model(model, quant)
        tensors, const_arrays, nodes, inputs, outputs = build_graph(model)
        if quant:
            nodes = _fuse_requant(nodes, const_arrays, tensors)
        nodes = _fuse_zero_pad_into_conv1d(nodes, const_arrays, tensors, outputs)
        if quant:
            _retype_integer_valued_float_constants(tensors, const_arrays, nodes)
        _assign_quantized_runtime_tensor_types(
            tensors, const_arrays, nodes, inputs, quant
        )

        # Reference (float) model – used for golden output computation.
        ref_model = read_model(resolved_onnx, skip_shape_inference=skip_shape_inference)
        ref_model = sanitize_quantized_model(ref_model, None)
        ref_tensors, _, _, _, _ = build_graph(ref_model)

        try:
            from onnx.reference import ReferenceEvaluator
        except ImportError as exc:
            raise CodegenError(
                "ONNX reference evaluator is unavailable; "
                "install onnx >= 1.14 with reference support."
            ) from exc

        ref_eval = ReferenceEvaluator(ref_model)

        cases_data = _generate_compare_inputs(
            tensors, inputs, quant, random_cases, seed
        )

        template_cases = []
        for case_name, c_inputs in cases_data:
            # Convert C-typed inputs to the float types the reference model expects.
            ref_feed = _prepare_reference_feed(inputs, ref_tensors, c_inputs)
            ref_outputs = ref_eval.run(None, ref_feed)

            inp_entries = []
            for i, (name, arr) in enumerate(zip(inputs, c_inputs)):
                elem_type = runtime_elem_type(tensors[name].elem_type, quant)
                ctype = c_type_for_elem_type(elem_type)
                sym = f"{prefix}_test_{case_name}_input_{i}"
                inp_entries.append(
                    {
                        "ctype": ctype,
                        "symbol": sym,
                        "numel": int(arr.size),
                        "values": _format_c_array(arr, ctype),
                    }
                )

            out_entries = []
            for i, (name, ref_out) in enumerate(zip(outputs, ref_outputs)):
                elem_type = runtime_elem_type(tensors[name].elem_type, quant)
                ctype = c_type_for_elem_type(elem_type)
                ref_arr = np.asarray(ref_out)
                # Cast to the C-side type so the golden values match the
                # generated inference output type.
                target_dtype = numpy_dtype_for_elem_type(elem_type)
                golden = ref_arr.astype(target_dtype, copy=False)
                sym = f"{prefix}_test_{case_name}_golden_{i}"
                out_entries.append(
                    {
                        "ctype": ctype,
                        "symbol": sym,
                        "numel": int(golden.size),
                        "values": _format_c_array(golden, ctype),
                    }
                )

            template_cases.append(
                {
                    "name": case_name,
                    "inputs": inp_entries,
                    "outputs": out_entries,
                }
            )

        header = render_template(
            "test_data_h.mako",
            guard=f"{prefix.upper()}_TEST_DATA_H",
            prefix=prefix,
            prefix_upper=prefix.upper(),
            num_cases=len(template_cases),
            num_inputs=len(inputs),
            num_outputs=len(outputs),
            cases=template_cases,
        )
        out_path = out_dir / f"{prefix}_test_data.h"
        out_path.write_text(header, encoding="utf-8")
        return out_path
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
