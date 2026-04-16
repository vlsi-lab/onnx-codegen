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
import re
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
    ``weight_bits`` controls the C type used for constant weight arrays,
    ``act_bits`` controls the C type used for intermediate activation buffers.
    A value of 0 means "keep original" (float).
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
            return "uint8_t"  # activations are unsigned in quantized inference
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
    TensorProto.UINT8,
    TensorProto.INT32,
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


def _build_weight_definitions(
    prefix: str,
    const_arrays: Dict[str, np.ndarray],
    quant: Optional[QuantConfig] = None,
) -> Tuple[List[str], Set[str]]:
    """Build C weight array definitions.

    Returns (defs, int8_weight_names) where int8_weight_names is the set of
    tensor names whose float32 values were compressed and stored as int8_t.
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
                # If the rounded values exceed the target range, fall back to
                # int32_t to preserve precision and emit a warning.
                np_dt = quant.weight_np_dtype
                rounded = np.round(f32)
                lo = int(np.iinfo(np_dt).min)
                hi = int(np.iinfo(np_dt).max)
                vmin, vmax = float(np.min(rounded)), float(np.max(rounded))
                if vmin < lo or vmax > hi:
                    warnings.warn(
                        f"Weight '{name}': value range [{vmin:.4g}, {vmax:.4g}] exceeds "
                        f"int{quant.weight_bits} range [{lo}, {hi}]. "
                        f"Storing as int32_t to preserve precision.",
                        stacklevel=3,
                    )
                    int32_arr = rounded.astype(np.int32)
                    vals = ", ".join(str(int(v)) for v in int32_arr)
                    defs.append(
                        f"static const int32_t {sym}[{flat.size}] = {{{vals}}};"
                    )
                    # Not added to int8_names; caller must handle int32_t pointer type
                else:
                    casted = rounded.astype(np_dt)
                    vals = ", ".join(str(int(v)) for v in casted)
                    defs.append(
                        f"static const {forced_weight_ctype} {sym}[{flat.size}] = {{{vals}}};"
                    )
                    if quant.weight_bits == 8:
                        int8_names.add(name)
            elif _float_array_as_int8(f32):
                # Store integer-valued weights as int8_t (4x smaller)
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
    quant: Optional[QuantConfig] = None,
) -> Tuple[str, Set[str]]:
    """Return (rendered_header, int8_weight_names)."""
    defs, int8_names = _build_weight_definitions(prefix, const_arrays, quant)
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

    # Group by C element type so we never alias across types.
    # When quant overrides activations, float tensors become act_ctype.
    def _resolve_ctype(tname: str) -> str:
        orig = c_type_for_elem_type(tensors[tname].elem_type)
        if quant and quant.act_bits and orig == "float":
            return quant.act_ctype
        return orig

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
    prefix: str, inputs: List[str], outputs: List[str], tensors: Dict[str, TensorInfo]
) -> str:
    io_size_macros: List[str] = []
    io_type_macros: List[str] = []
    for i, n in enumerate(inputs):
        io_size_macros.append(
            f"#define {prefix.upper()}_INPUT_{i}_SIZE {tensors[n].numel}"
        )
        io_size_macros.append(
            f"#define {prefix.upper()}_INPUT_{i}_ELEM_SIZE {elem_size_for_elem_type(tensors[n].elem_type)}"
        )
        io_type_macros.append(
            f"#define {prefix.upper()}_INPUT_{i}_ONNX_TYPE {tensors[n].elem_type}"
        )
    for i, n in enumerate(outputs):
        io_size_macros.append(
            f"#define {prefix.upper()}_OUTPUT_{i}_SIZE {tensors[n].numel}"
        )
        io_size_macros.append(
            f"#define {prefix.upper()}_OUTPUT_{i}_ELEM_SIZE {elem_size_for_elem_type(tensors[n].elem_type)}"
        )
        io_type_macros.append(
            f"#define {prefix.upper()}_OUTPUT_{i}_ONNX_TYPE {tensors[n].elem_type}"
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
        TensorProto.UINT8: 1,
        TensorProto.INT32: 4,
    }
    if elem_type not in mapping:
        raise CodegenError(f"Unsupported runtime tensor type {elem_type}")
    return mapping[elem_type]


def copy_stmt(
    src: str, dst: str, info: TensorInfo, quant: Optional[QuantConfig] = None
) -> str:
    ctype = c_type_for_elem_type(info.elem_type)
    if quant and quant.act_bits and ctype == "float":
        ctype = quant.act_ctype
    if ctype == "float":
        return f"    tensor_copy((const float*)({src}), (float*)({dst}), (size_t){info.numel});"
    nbytes = info.numel * (
        quant.act_elem_size
        if quant and quant.act_bits and info.elem_type in FLOAT_TENSOR_TYPES
        else elem_size_for_elem_type(info.elem_type)
    )
    return f"    tensor_copy_bytes((const void*)({src}), (void*)({dst}), (size_t){nbytes});"


def render_runtime_helpers() -> str:
    helpers_path = (
        Path(__file__).resolve().parent / "templates" / "runtime_helpers.c.inc"
    )
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
) -> str:
    if int8_weight_names is None:
        int8_weight_names = set()

    # Resolve the C type for activation buffers (float unless quant overrides)
    act_ctype = quant.act_ctype if quant and quant.act_bits else "float"
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
            ctype = c_type_for_elem_type(tensors[name].elem_type)
            if quant and quant.act_bits and ctype == "float":
                ctype = act_ctype
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
            if len(ins) >= 3 and ins[1] in const_arrays and ins[2] in const_arrays:
                min_v = float(const_arrays[ins[1]].reshape(-1)[0])
                max_v = float(const_arrays[ins[2]].reshape(-1)[0])
            else:
                min_raw = node.attrs.get("min", -np.inf)
                max_raw = node.attrs.get("max", np.inf)
                min_v = float(cast(Union[int, float], min_raw))
                max_v = float(cast(Union[int, float], max_raw))
            # Use ONNXCG_ACT() cast for integer activation types to avoid
            # float→int truncation warnings and overflow (e.g. 255→uint8).
            if quant and quant.act_bits:
                lo_lit = f"ONNXCG_ACT({int(min_v)})"
                hi_lit = f"ONNXCG_ACT({int(max_v)})"
            else:
                lo_lit = c_float_literal(min_v)
                hi_lit = c_float_literal(max_v)
            lines.append(
                f"    tensor_clip({tensor_expr(ins[0])}, {tensor_expr(out0)}, (size_t){out_info.numel}, {lo_lit}, {hi_lit});"
            )

        elif op == "Add":
            lines.append(
                "    tensor_add_broadcast("
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
                    f"{tensor_expr(ins[0])}, (uint8_t*){tensor_expr(out0)}, "
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
                f"    matmul_2d({tensor_expr(ins[0])}, {tensor_expr(ins[1])}, {tensor_expr(out0)}, {m}, {k}, {n});"
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
                f"    gemm_2d({tensor_expr(ins[0])}, {tensor_expr(ins[1])}, {c_expr}, {tensor_expr(out0)}, "
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
                lines.append(
                    f"    conv1d_ncw_i8w_requant("
                    f"(const uint8_t*){tensor_expr(ins[0])}, {tensor_expr(ins[1])}, "
                    f"(uint8_t*){tensor_expr(rq_out)}, "
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
                # Use int8 weight variant when the weight was compressed to int8_t
                conv_func = (
                    "ONNXCG_CONV1D_I8W_FUNC"
                    if ins[1] in int8_weight_names
                    else "ONNXCG_CONV1D_FUNC"
                )
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
                f"    qlinear_conv1d_u8s8u8((const uint8_t*)({tensor_expr(ins[0])}), (const float*)({tensor_expr(ins[1])}), (const uint8_t*)({tensor_expr(ins[2])}), "
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
            lines.append(
                f"    pad_tensor_constant({tensor_expr(ins[0])}, {tensor_expr(out0)}, {shape_expr(ins[0])}, {shape_expr(out0)}, {rank}, "
                f"{prefix}_pad_begin_{name_s}, {prefix}_pad_end_{name_s}, {c_float_literal(pad_value)});"
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
            lines.append(
                f"    slice_tensor({tensor_expr(ins[0])}, {tensor_expr(out0)}, {shape_expr(ins[0])}, {shape_expr(out0)}, {rank}, "
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
        runtime_helpers=render_runtime_helpers(),
        ops_body="\n".join(lines),
        output_copies="\n".join(output_copy_lines),
    )
    return rendered, layer_keys


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


def generate_library(
    onnx_path: Path,
    out_dir: Path,
    prefix: str,
    skip_shape_inference: bool,
    custom_kernels_header: Optional[str] = None,
    quant: Optional[QuantConfig] = None,
) -> Tuple[Path, Path, Path, int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_onnx, tmp_dir = _resolve_onnx_path(onnx_path)
    try:
        model = read_model(resolved_onnx, skip_shape_inference=skip_shape_inference)
        tensors, const_arrays, nodes, inputs, outputs = build_graph(model)

        # Fuse Mul→Add→Div→Floor→Clip into RequantShift (or identity Clip).
        if quant:
            nodes = _fuse_requant(nodes, const_arrays, tensors)

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

        model_h = render_model_header(prefix, inputs, outputs, tensors)
        weights_h, int8_weight_names = render_weights_header(
            prefix, weight_arrays, quant
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
        )
        layer_cfg_h = render_layer_config_header(prefix, layer_keys)

        model_h_path = out_dir / f"{prefix}_model.h"
        model_c_path = out_dir / f"{prefix}_model.c"
        weights_h_path = out_dir / f"{prefix}_weights.h"
        layer_cfg_h_path = out_dir / f"{prefix}_layer_cfg.h"

        model_h_path.write_text(model_h, encoding="utf-8")
        model_c_path.write_text(model_c, encoding="utf-8")
        weights_h_path.write_text(weights_h, encoding="utf-8")
        # Only overwrite layer_cfg.h if it doesn't exist — the user
        # may have hand-edited it to disable specific layers.
        if not layer_cfg_h_path.exists():
            layer_cfg_h_path.write_text(layer_cfg_h, encoding="utf-8")

        return (
            model_h_path,
            model_c_path,
            weights_h_path,
            len(inputs),
            len(outputs),
            len(nodes),
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
