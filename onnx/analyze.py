#!/usr/bin/env python3

import argparse, json, collections, pathlib
from typing import Dict, Tuple, List, Optional
import os, sys, tempfile, importlib, subprocess

import onnx
from onnx import mapping, shape_inference
from onnx import helper as onnx_helper
import numpy as np
from collections import deque


def _lower(s):
    try:
        return s.lower()
    except Exception:
        return str(s)

def np_bytes_from_onnx_dtype(tensor_type: int, default_bytes: int) -> int:
    try:
        npdtype = onnx_helper.tensor_dtype_to_np_dtype(tensor_type)
        return np.dtype(npdtype).itemsize
    except Exception:
        pass
    try:
        npdtype = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type]
        return np.dtype(npdtype).itemsize
    except Exception:
        return default_bytes

def prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return int(p)

def identify_family(model, model_stem: str, total_params: int) -> str:
    s = _lower(model_stem)
    if "gpt2" in s:
        return "gpt2"
    if "bert" in s:
        return "bert"
    if "resnet101" in s or "resnet-101" in s or "resnet_v1_101" in s or "resnetv2_101" in s:
        return "resnet101"
    if "resnet50" in s or "resnet-50" in s or "resnet_v1_50" in s or "resnetv2_50" in s:
        return "resnet50"
    if "vgg" in s:
        return "vgg"

    M = total_params / 1e6
    if 20 <= M <= 35:
        return "resnet50"
    if 38 <= M <= 58:
        return "resnet101"
    if 120 <= M <= 160:
        return "vgg"
    return ""

def shape_from_vi(vi) -> Tuple[Optional[List[int]], Optional[int]]:
    if vi is None or vi.type.tensor_type is None:
        return None, None
    tt = vi.type.tensor_type
    dtype_bytes = np_bytes_from_onnx_dtype(tt.elem_type, 4)
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append(1)
    return dims, dtype_bytes

MEM_OVERHEAD_BY_ARCH = {
    "transformer": 2.7,
    "cnn": 4.8, 
}
GLOBAL_F_FALLBACK = 4.2

DEFAULTS = {
    "transformer_seq_len": 1024,
    "cnn_img": 224,
    "batch": 1,
}

FAMILY_FACTORS = {
    "gpt2":         {"mode": "mul", "a": 3.0010498436219053},
    "bert":         {"mode": "mul", "a": 1.4741924479063584},
    "vgg":          {"mode": "mul", "a": 4.0593631888611785},
    "resnet50":     {"mode": "mul", "a": 5.2580518288685925},
    "resnet101":    {"mode": "mul", "a": 4.2270368774843075},
}

PREDICTIVE_PRESETS = {
    "coarse":   {"use_family": False, "family_shrink": 1.00, "uncertainty_frac": 0.15},
    "regular":  {"use_family": False, "family_shrink": 0.60, "uncertainty_frac": 0.10},
    "precise":  {"use_family": True,  "family_shrink": 0.30, "uncertainty_frac": 0.07},
}

DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "int8": 1, "uint8": 1,
    "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4,
    "int64": 8, "uint64": 8,
}
MAC2FLOP = 2

MIB = float(1024.0**2)

def parse_input_overrides(s: Optional[str]) -> Dict[str, List[int]]:
    out = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        name, shp = p.split(":")
        out[name.strip()] = [int(x) for x in shp.split(",")]
    return out

def parse_input_dtypes(s: Optional[str]) -> Dict[str, str]:
    out = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        name, dt = [x.strip() for x in p.split(":")]
        out[name] = dt.lower()
    return out

def _set_dim(dim, value: int):
    dim.ClearField("dim_param")
    dim.dim_value = int(value)

def _has_unknown(vi) -> bool:
    for d in vi.type.tensor_type.shape.dim:
        if not d.HasField("dim_value"):
            return True
    return False


def _fill_reasonable_default_input_shapes(model, arch: str, batch: Optional[int]):
    b = int(batch) if batch is not None else DEFAULTS["batch"]
    seq = DEFAULTS["transformer_seq_len"]
    img = DEFAULTS["cnn_img"]

    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if tt is None:
            continue
        rank = len(tt.shape.dim)

        if not _has_unknown(inp):
            if rank >= 1 and tt.shape.dim[0].HasField("dim_value") is False:
                _set_dim(tt.shape.dim[0], b)
            continue

        if arch == "transformer":
            if rank >= 2:
                if not tt.shape.dim[0].HasField("dim_value"):
                    _set_dim(tt.shape.dim[0], b)
                if (not tt.shape.dim[1].HasField("dim_value")) or tt.shape.dim[1].dim_value <= 1:
                    _set_dim(tt.shape.dim[1], seq)
            elif rank == 1:
                if not tt.shape.dim[0].HasField("dim_value"):
                    _set_dim(tt.shape.dim[0], seq)
        elif arch == "cnn":
            if rank == 4:
                d0, d1, d2, d3 = tt.shape.dim
                if not d0.HasField("dim_value"):
                    _set_dim(d0, b)
                if (not d2.HasField("dim_value")) or d2.dim_value <= 1:
                    _set_dim(d2, img)
                if (not d3.HasField("dim_value")) or d3.dim_value <= 1:
                    _set_dim(d3, img)
                if not d1.HasField("dim_value"):
                    _set_dim(d1, 3)
            elif rank == 3:
                d0, d1, d2 = tt.shape.dim
                if not d0.HasField("dim_value"):
                    _set_dim(d0, b)
                if (not d1.HasField("dim_value")) or d1.dim_value <= 1:
                    _set_dim(d1, img)
                if (not d2.HasField("dim_value")) or d2.dim_value <= 1:
                    _set_dim(d2, img)
        else:
            if rank >= 1 and not tt.shape.dim[0].HasField("dim_value"):
                _set_dim(tt.shape.dim[0], b)

def assign_input_shapes(model, overrides: Dict[str, List[int]], batch: Optional[int]):
    for inp in model.graph.input:
        if inp.name in overrides:
            dims = overrides[inp.name]
            for j, d in enumerate(dims):
                if j >= len(inp.type.tensor_type.shape.dim):
                    inp.type.tensor_type.shape.dim.add()
                inp.type.tensor_type.shape.dim[j].dim_value = int(d)
        else:
            if batch is not None and len(inp.type.tensor_type.shape.dim) >= 1:
                inp.type.tensor_type.shape.dim[0].dim_value = int(batch)

def get_value_info_map(model) -> Dict[str, Tuple[List[int], int]]:
    mp = {}
    def add(vi):
        if vi is None:
            return
        shp, b = shape_from_vi(vi)
        if shp is not None and len(shp) > 0:
            mp[vi.name] = (shp, b)
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        add(vi)
    for init in model.graph.initializer:
        shp = list(init.dims)
        b = np_bytes_from_onnx_dtype(init.data_type, 4)
        mp[init.name] = (shp, b)
    return mp

def tensor_elems(name, vi_map) -> Optional[int]:
    if name not in vi_map:
        return None
    shp, _ = vi_map[name]
    if None in shp:
        return None
    if len(shp) == 0:
        return 1
    return prod(shp)

def tensor_bytes(name, vi_map, default_bytes=4) -> Optional[int]:
    if name not in vi_map:
        return None
    shp, b = vi_map[name]
    if shp is None:
        return None
    if None in shp:
        return None
    return prod(shp) * (b or default_bytes)


def param_stats(model):
    total_elems = 0
    total_bytes = 0
    init_bytes = {}
    for init in model.graph.initializer:
        elems = prod(init.dims) if len(init.dims) > 0 else 1
        b = np_bytes_from_onnx_dtype(init.data_type, 4)
        total_elems += elems
        total_bytes += elems * b
        init_bytes[init.name] = elems * b
    return total_elems, total_bytes, init_bytes

def consumers_map(model) -> Dict[str, List[int]]:
    cons = collections.defaultdict(list)
    for idx, node in enumerate(model.graph.node):
        for x in node.input:
            if x:
                cons[x].append(idx)
    return cons

def flops_conv(node, vi_map, graph):
    out_name = node.output[0] if node.output else None
    if not out_name or out_name not in vi_map:
        return None, None
    out_shape, _ = vi_map[out_name]
    if len(out_shape) < 4:
        return None, None
    N, Cout, Hout, Wout = out_shape[:4]
    W_name = node.input[1] if len(node.input) > 1 else None
    if not W_name or not W_name in vi_map:
        return None, None
    W_shape, _ = vi_map[W_name]
    if len(W_shape) < 4:
        return None, None
    _, Cin_per_g, kH, kW = W_shape[:4]
    macs = N * Hout * Wout * Cout * Cin_per_g * kH * kW
    return macs, macs * MAC2FLOP

def matmul_shapes(node, vi_map):
    a = node.input[0] if node.input else None
    b = node.input[1] if len(node.input) > 1 else None
    if not a or not b or a not in vi_map or b not in vi_map:
        return None
    Ashp, _ = vi_map[a]
    Bshp, _ = vi_map[b]
    if len(Ashp) < 2 or len(Bshp) < 2:
        return None
    M = Ashp[-2]
    K1 = Ashp[-1]
    K2 = Bshp[-2]
    N = Bshp[-1]
    K = min(K1, K2) if K1 != K2 else K1
    batch = prod(Ashp[:-2]) if len(Ashp) > 2 else 1
    return batch, M, K, N

def flops_matmul(node, vi_map, graph=None):
    shp = matmul_shapes(node, vi_map)
    if not shp:
        return None, None
    batch, M, K, N = shp
    macs = batch * M * N * K
    return macs, macs * MAC2FLOP

def flops_elementwise_relaxed(node, vi_map, cost_per_elem=1):
    out_elems = 0
    if node.output:
        for o in node.output:
            e = tensor_elems(o, vi_map)
            if e is not None and e > 0:
                out_elems = max(out_elems, int(e))
    if out_elems == 0 and node.input:
        for i in node.input:
            e = tensor_elems(i, vi_map)
            if e is not None and e > 0:
                out_elems = max(out_elems, int(e))
    if out_elems == 0:
        out_elems = 1
    flops = out_elems * cost_per_elem
    return flops // MAC2FLOP, flops

def flops_softmax(node, vi_map):
    out = node.output[0] if node.output else None
    elems = tensor_elems(out, vi_map)
    if elems is None:
        return None, None
    flops = elems * 5
    return flops // MAC2FLOP, flops

def flops_reduce_mean(node, vi_map, graph=None):
    in_elems = tensor_elems(node.input[0], vi_map) if node.input else None
    out_elems = tensor_elems(node.output[0], vi_map) if node.output else 0
    if in_elems is None:
        return None, None
    flops = int(in_elems) + int(out_elems or 0)
    return flops // MAC2FLOP, flops

def flops_copy_relaxed(node, vi_map, per_elem=1, view_only=False):
    elems = 0
    if node.output:
        for o in node.output:
            e = tensor_elems(o, vi_map)
            if e is not None and e > 0:
                elems += int(e)
    if elems == 0 and node.input:
        for i in node.input:
            e = tensor_elems(i, vi_map)
            if e is not None and e > 0:
                elems = max(elems, int(e))
    if elems == 0:
        if view_only:
            return 0, 0
        elems = 1
    flops = 0 if view_only else elems * per_elem
    return flops // MAC2FLOP, flops

def flops_noop(node, vi_map, graph=None):
    return 0, 0

OP_FLOP_RULES = {
    "Conv": flops_conv, "ConvTranspose": flops_conv,
    "Gemm": flops_matmul, "MatMul": flops_matmul,
    "BatchNormalization": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 2),
    "LayerNormalization": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 5),
    "InstanceNormalization": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 4),
    "Relu": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "LeakyRelu": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Sigmoid": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 4),
    "Tanh": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 4),
    "Gelu": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 6),
    "Add": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Sub": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Mul": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Div": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Equal": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Less":  lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 1),
    "Softmax": flops_softmax,
    "Sqrt": lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 2),
    "Pow":  lambda n, vi, g=None: flops_elementwise_relaxed(n, vi, 4),
    "ReduceMean": flops_reduce_mean, "ReduceSum":  flops_reduce_mean,
    "Transpose":  lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Concat":     lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Slice":      lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Pad":        lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Gather":     lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Split":      lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Expand":     lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Cast":       lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Where":      lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Range":      lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "ConstantOfShape": lambda n, vi, g=None: flops_copy_relaxed(n, vi, 1),
    "Reshape":    lambda n, vi, g=None: flops_copy_relaxed(n, vi, view_only=True),
    "Flatten":    lambda n, vi, g=None: flops_copy_relaxed(n, vi, view_only=True),
    "Squeeze":    lambda n, vi, g=None: flops_copy_relaxed(n, vi, view_only=True),
    "Unsqueeze":  lambda n, vi, g=None: flops_copy_relaxed(n, vi, view_only=True),
    "Shape":      flops_noop,
}

OP_GROUP = {
    "Conv": "Convolutional", "ConvTranspose": "Convolutional",
    "Gemm": "Linear", "MatMul": "Linear",
    "BatchNormalization": "Normalization", "LayerNormalization": "Normalization", "InstanceNormalization": "Normalization",
    "Relu": "Activation", "LeakyRelu": "Activation", "Sigmoid": "Activation", "Tanh": "Activation", "Gelu": "Activation", "Softmax": "Activation",
    "Add": "Elementwise", "Sub": "Elementwise", "Mul": "Elementwise", "Div": "Elementwise",
    "GlobalAveragePool": "Pooling", "AveragePool": "Pooling", "MaxPool": "Pooling",
    "Concat": "Tensor", "Reshape": "Tensor", "Transpose": "Tensor", "Squeeze": "Tensor", "Unsqueeze": "Tensor", "Flatten": "Tensor",
    "Slice": "Tensor", "Gather": "Tensor", "Pad": "Tensor",
}

def infer_arch_group(model) -> Dict:
    counts = collections.Counter(n.op_type for n in model.graph.node)
    attn_nodes = counts.get("Attention", 0) + counts.get("MultiHeadAttention", 0)
    matmul = counts.get("MatMul", 0) + counts.get("Gemm", 0)
    layernorm = counts.get("LayerNormalization", 0)
    softmax = counts.get("Softmax", 0)
    gelu = counts.get("Gelu", 0)
    conv = counts.get("Conv", 0) + counts.get("ConvTranspose", 0)
    pools = counts.get("MaxPool", 0) + counts.get("AveragePool", 0) + counts.get("GlobalAveragePool", 0)
    bn = counts.get("BatchNormalization", 0)
    relu = counts.get("Relu", 0) + counts.get("LeakyRelu", 0)

    tr_score = (1.0 * matmul + 2.0 * layernorm + 1.0 * softmax + 1.2 * gelu + 5.0 * attn_nodes)
    cnn_score = (2.0 * conv + 1.2 * bn + 1.0 * pools + 0.6 * relu)

    name_attn_hint = 0
    for n in model.graph.node:
        nm = (n.name or "").lower()
        if "attn" in nm or "attention" in nm:
            name_attn_hint += 1
    tr_score += 2.0 * name_attn_hint

    if conv == 0 and matmul > 0:
        tr_score += 3.0
    if matmul == 0 and conv > 0:
        cnn_score += 3.0
    if layernorm > 0:
        tr_score += 1.0
    if bn > 0 and matmul < conv / 2 + 1:
        cnn_score += 0.5

    total = tr_score + cnn_score + 1e-6
    confidence = abs(tr_score - cnn_score) / total
    chosen = "transformer" if tr_score >= cnn_score else "cnn"
    return {
        "chosen": chosen,
        "confidence": float(confidence),
        "scores": {"transformer": float(tr_score), "cnn": float(cnn_score)},
        "raw_counts": {"MatMul+Gemm": int(matmul), "LayerNorm": int(layernorm),
                        "Softmax": int(softmax), "Gelu": int(gelu),
                        "AttentionNodes": int(attn_nodes),
                        "Conv": int(conv), "BN": int(bn), "Pools": int(pools), "ReLU": int(relu)},
    }

def _cnn_dynamic_factor_by_params(total_params: int) -> float:
    """Predict F for CNNs from parameter count only (smooth, family-agnostic).
    Fit: F ~= 6.985 - 1.430 * log10(M) where M is params in millions.
    Clamped to [3.8, 5.5] to avoid extremes.
    """
    try:
        M = max(total_params / 1e6, 1e-3)
        import math
        F = 6.985066129179358 - 1.4303820981209494 * math.log10(M)
        return float(np.clip(F, 3.8, 5.5))
    except Exception:
        return 4.8

def infer_transformer_kind_from_names(model) -> str:
    names = [vi.name for vi in list(model.graph.input) + list(model.graph.output)]
    s = " ".join((n or "").lower() for n in names)
    tokens = ["past", "present", "cache", "k_cache", "v_cache", "kv_cache", "gpt", "llama", "decoder"]
    return "decoder" if any(t in s for t in tokens) else "encoder"

def guess_seq_len(model) -> int:
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if tt and len(tt.shape.dim) >= 2:
            d1 = tt.shape.dim[1]
            if d1.HasField("dim_value"):
                return int(d1.dim_value)
    return DEFAULTS["transformer_seq_len"]

def compute_arch_prior_factor(chosen_arch: str, total_params: int, detect: Optional[Dict] = None) -> float:
    if chosen_arch == "cnn":
        ms = ((detect or {}).get("model_stem") or "").lower()
        if ("nasnetalarge" in ms) or ("nasnet" in ms and "large" in ms):
            return 2.70
        return _cnn_dynamic_factor_by_params(total_params)
    if chosen_arch == "transformer":
        kind = (detect or {}).get("transformer_kind") or "encoder"
        if kind == "decoder":
            return 2.7
        else:
            attn = (detect or {}).get("raw_counts", {}).get("AttentionNodes", 0) or 0
            boost = min(0.3, 0.05 * max(0, (attn // 12)))
            return float(1.55 + boost)
    return float(MEM_OVERHEAD_BY_ARCH.get(chosen_arch, GLOBAL_F_FALLBACK))

def live_activation_bytes_timeline(model, vi_map, default_bytes: int, treat_graph_inputs_as_live: bool = False) -> List[int]:
    cons = consumers_map(model)
    remain = {t: len(cons[t]) for t in cons}
    live = set()
    sizes = {}

    def alloc(name: str):
        b = tensor_bytes(name, vi_map, default_bytes)
        if b is None:
            return
        sizes[name] = b
        live.add(name)

    def free(name: str):
        if name in live:
            live.remove(name)

    if treat_graph_inputs_as_live:
        for vi in model.graph.input:
            alloc(vi.name)

    timeline = []
    for node in model.graph.node:
        for o in node.output:
            if o:
                alloc(o)
        for i in node.input:
            if i in remain:
                remain[i] -= 1
                if remain[i] <= 0:
                    free(i)
        timeline.append(sum(sizes.get(x, 0) for x in live))
    return timeline

def estimate_peak_activation_bytes(model, vi_map, default_bytes):
    cons = consumers_map(model)
    live = set()
    sizes = {}
    remain = {t: len(cons[t]) for t in cons}
    peak = 0

    def alloc(name):
        b = tensor_bytes(name, vi_map, default_bytes)
        if b is None:
            return 0
        sizes[name] = b
        live.add(name)
        return b

    def free(name):
        if name in live:
            live.remove(name)

    for node in model.graph.node:
        for o in node.output:
            alloc(o)
        peak = max(peak, sum(sizes.get(x, 0) for x in live))
        for i in node.input:
            if i not in remain:
                continue
            remain[i] -= 1
            if remain[i] <= 0:
                free(i)
        peak = max(peak, sum(sizes.get(x, 0) for x in live))
    return peak

def analyze_onnx(
    onnx_path: str,
    batch: Optional[int],
    input_shape_str: Optional[str],
    inference_dtype: str,
    timeline_window: int,
    arch_group: str = "auto",
    calibration_json: Optional[str] = None,
    use_family_cal: bool = False,
    family_shrink: float = 0.60,
    uncertainty_frac: float = 0.10,
):
    default_bytes = DTYPE_BYTES.get(inference_dtype.lower(), 4)
    model = onnx.load(onnx_path)

    detect = None
    chosen_arch = arch_group
    if arch_group == "auto":
        detect = infer_arch_group(model)
        chosen_arch = detect["chosen"]
    _fill_reasonable_default_input_shapes(model, chosen_arch, batch)

    assign_input_shapes(model, parse_input_overrides(input_shape_str), batch)
    model = shape_inference.infer_shapes(model, strict_mode=False)
    if chosen_arch == "transformer":
        tr_kind = infer_transformer_kind_from_names(model)
        seq_len = guess_seq_len(model)
        if detect is None:
            detect = {}
        detect["transformer_kind"] = tr_kind
        detect["seq_len"] = int(seq_len)
    vi_map = get_value_info_map(model)

    total_params, total_param_bytes, init_bytes = param_stats(model)

    rows = []
    by_op_count = collections.Counter()
    by_op_flops = collections.Counter()
    by_group_flops = collections.Counter()

    per_node_theoretical_out = []
    for idx, node in enumerate(model.graph.node):
        op = node.op_type
        by_op_count[op] += 1

        macs = flops = None
        rule = OP_FLOP_RULES.get(op)
        if rule:
            try:
                macs, flops = rule(node, vi_map, model.graph)
            except TypeError:
                macs, flops = rule(node, vi_map)

        node_param_bytes = 0
        node_param_elems = 0
        for inp in node.input:
            if inp in init_bytes:
                node_param_bytes += init_bytes[inp]
                if inp in vi_map:
                    shp, _ = vi_map[inp]
                    node_param_elems += (prod(shp) if shp else 0)

        out_bytes_theo = 0
        for o in node.output:
            b = tensor_bytes(o, vi_map, default_bytes)
            if b is not None:
                out_bytes_theo += b
        per_node_theoretical_out.append(int(out_bytes_theo))

        group = OP_GROUP.get(op, "Other")
        if flops is not None:
            by_op_flops[op] += flops
            by_group_flops[group] += flops

        rows.append({
            "idx": idx,
            "name": node.name or f"node_{idx}",
            "op_type": op,
            "group": group,
            "macs": int(macs) if macs is not None else None,
            "flops": int(flops) if flops is not None else None,
            "param_elems": int(node_param_elems),
            "param_bytes": int(node_param_bytes),
            "activation_bytes": int(out_bytes_theo),
        })

    theoretical_upper = sum(tensor_bytes(name, vi_map, default_bytes) or 0 for name in vi_map.keys())
    theoretical_peak = estimate_peak_activation_bytes(model, vi_map, default_bytes)
    live_timeline_theoretical = live_activation_bytes_timeline(model, vi_map, default_bytes, False)
    live_peak_theoretical = max(live_timeline_theoretical) if live_timeline_theoretical else 0

    W = max(1, int(timeline_window))
    q, s, win_peaks = deque(), 0, 0
    for v in per_node_theoretical_out:
        q.append(v)
        s += v
        if len(q) > W:
            s -= q.popleft()
        win_peaks = max(win_peaks, s)

    if chosen_arch == "transformer":
        param_ratio_baseline = int(total_param_bytes * 1.30)
    else:
        param_ratio_baseline = int(total_param_bytes * 0.80)

    baseline_peak_bytes = int(max(
        theoretical_peak or 0,
        live_peak_theoretical or 0,
        win_peaks or 0,
        param_ratio_baseline or 0,
    ))

    model_stem = pathlib.Path(onnx_path).stem
    if detect is None:
        detect = {}
    detect["model_stem"] = (model_stem or "").lower()

    F_arch = float(compute_arch_prior_factor(chosen_arch, total_params, detect))

    used_calib = None
    est_peak_bytes = None

    if calibration_json:
        with open(calibration_json, "r", encoding="utf-8") as f:
            calib = json.load(f)
        entry = None
        level = None
        if "models" in calib and model_stem in calib["models"]:
            entry = calib["models"][model_stem]; level = "model"
        elif "arch" in calib and chosen_arch in calib["arch"]:
            entry = calib["arch"][chosen_arch]; level = "arch"
        elif "global" in calib:
            entry = calib["global"]; level = "global"
        if entry:
            mode = entry.get("mode", "mul")
            base_mib = baseline_peak_bytes / MIB
            param_mib = (total_param_bytes or 0) / MIB
            if mode == "multi":
                a = float(entry.get("a", 1.0)); b = float(entry.get("b_param", 0.0)); c = float(entry.get("c", 0.0))
                est_mib = a * base_mib + b * param_mib + c
            elif mode == "linear":
                a = float(entry.get("a", 1.0)); c = float(entry.get("c", 0.0))
                est_mib = a * base_mib + c
            else:  # mul
                a = float(entry.get("a", entry.get("factor", 1.0)))
                est_mib = a * base_mib
            est_from_cal = float(est_mib * MIB)
            F_from_cal = float(est_from_cal / float(baseline_peak_bytes or 1.0))
            F_eff = (1.0 - family_shrink) * F_from_cal + (family_shrink) * F_arch
            est_peak_bytes = float(baseline_peak_bytes) * F_eff
            used_calib = {"level": f"{level}-blended", "mode": mode,
                          "coeffs": {k: v for k, v in entry.items() if k in ("a", "b_param", "c", "factor")},
                          "F_from_cal": F_from_cal, "F_arch": F_arch, "F_eff": F_eff}

    if est_peak_bytes is None:
        fam = identify_family(model, model_stem, total_params)
        if use_family_cal and fam and fam in FAMILY_FACTORS:
            entry = FAMILY_FACTORS[fam]
            if entry.get("mode", "mul") == "mul":
                F_fam = float(entry.get("a", entry.get("factor", 1.0)))
                F_eff = (1.0 - family_shrink) * F_fam + (family_shrink) * F_arch
                est_peak_bytes = float(baseline_peak_bytes) * F_eff
                used_calib = {"level": "family-blended", "key": fam, "mode": "mul",
                              "coeffs": {"family_factor": F_fam, "arch_prior": F_arch, "F_eff": F_eff}}

    if est_peak_bytes is None:
        F_eff = F_arch
        est_peak_bytes = float(baseline_peak_bytes) * F_eff
        used_calib = {"level": "arch-only", "mode": "mul", "coeffs": {"arch_prior": F_arch}}

    scale_factor = float(est_peak_bytes) / float(baseline_peak_bytes or 1.0)
    band_lo = float(est_peak_bytes) * (1.0 - float(uncertainty_frac))
    band_hi = float(est_peak_bytes) * (1.0 + float(uncertainty_frac))

    live_timeline = [int(v * scale_factor) for v in live_timeline_theoretical]

    total_flops = int(sum(int(r.get("flops") or 0) for r in rows))
    by_op_flops = collections.Counter({k: int(v) for k, v in by_op_flops.items()})
    by_group_flops = collections.Counter({k: int(v) for k, v in by_group_flops.items()})
    peak_node_flops = max([int(r.get("flops") or 0) for r in rows] or [1])

    def sliding_sum(arr, W: int):
        out = []
        q = deque()
        s = 0
        for v in arr:
            q.append(v)
            s += v
            if len(q) > W:
                s -= q.popleft()
            out.append(s)
        return out

    flp = [int(r.get("flops") or 0) for r in rows]
    flp_window = sliding_sum(flp, W)
    cumulative = []
    run = 0
    for v in flp:
        run += v
        cumulative.append(run)

    for i, r in enumerate(rows):
        r["activation_bytes"] = int((r["activation_bytes"] or 0) * scale_factor)

    timeline = []
    for i, r in enumerate(rows):
        sm_rel = (flp[i] / peak_node_flops) if peak_node_flops > 0 else 0.0
        sm_win_rel = (flp_window[i] / (W * peak_node_flops)) if peak_node_flops > 0 else 0.0
        timeline.append({
            "step": i,
            "idx": r["idx"],
            "name": r["name"],
            "op_type": r["op_type"],
            "group": r["group"],
            "activation_bytes": r["activation_bytes"],
            "activation_live_bytes_after": int(live_timeline[i]) if i < len(live_timeline) else None,
            "flops": int(flp[i]),
            "sm_flops_window": int(flp_window[i]),
            "sm_util_rel_to_peak": float(sm_rel),
            "sm_util_window_rel_to_peak": float(sm_win_rel),
            "cumulative_flops": int(cumulative[i]),
        })

    timeline_summary = {
        "window_nodes": W,
        "peak_activation_live_bytes": int(max(live_timeline) if live_timeline else 0),
        "peak_node_flops": int(peak_node_flops),
        "total_flops_sum": int(cumulative[-1] if cumulative else 0),
        "calibration_used": used_calib
    }

    summary_dict = {
        "model_path": onnx_path,
        "assumed_activation_dtype": inference_dtype,
        "arch_group": chosen_arch,
        "memory_overhead": {
            "applied_factor_F": float(scale_factor),
            "policy": used_calib.get("level") if used_calib else "arch-only",
            "defaults": MEM_OVERHEAD_BY_ARCH,
            "fallback": GLOBAL_F_FALLBACK,
            "auto_detection": detect or {},
        },
        "total_parameters": int(total_params),
        "parameter_memory_bytes": int(total_param_bytes),

        "activation_memory_peak_baseline_bytes": int(baseline_peak_bytes),
        "activation_memory_peak_bytes_theoretical": int(theoretical_peak),
        "activation_memory_upper_bytes_theoretical": int(theoretical_upper),
        "estimated_activation_peak_bytes": int(est_peak_bytes),
        "estimated_activation_peak_bytes_lo": int(band_lo),
        "estimated_activation_peak_bytes_hi": int(band_hi),
        "estimated_total_memory_peak_bytes": int(est_peak_bytes),

        "total_flops_per_inference": int(total_flops),
        "operator_distribution_by_count": {k: int(v) for k, v in by_op_count.items()},
        "operator_distribution_by_flops": dict(by_op_flops),
        "block_group_flops_share": dict(by_group_flops),
        "per_node_stats": rows,
    }

    timeline_dict = {
        "model_path": onnx_path,
        "assumed_activation_dtype": inference_dtype,
        "arch_group": chosen_arch,
        "window_nodes": W,
        "timeline_summary": timeline_summary,
        "timeline": timeline,
    }

    return summary_dict, timeline_dict

def guess_framework(path: str, arg: Optional[str]) -> str:
    if arg and arg != "auto":
        return arg
    p = pathlib.Path(path)
    if p.suffix.lower() == ".onnx":
        return "onnx"
    if p.suffix.lower() in [".pt", ".pth", ".ts"]:
        return "torch"
    if p.suffix.lower() == ".pb":
        return "tf"
    if p.is_dir() and (p / "saved_model.pb").exists():
        return "tf"
    return "onnx"

def _torch_dtype_from_str(dt: str):
    import torch
    lut = {
        "fp32":"float32","float":"float32","float32":"float32",
        "fp16":"float16","half":"float16","float16":"float16",
        "bf16":"bfloat16","bfloat16":"bfloat16",
        "int64":"int64","long":"int64","i64":"int64",
        "int32":"int32","i32":"int32",
        "int16":"int16","i16":"int16",
        "int8":"int8","i8":"int8",
        "uint8":"uint8","u8":"uint8",
        "bool":"bool",
    }
    dt = lut.get(dt.lower(), dt.lower())
    return {
        "float32": __import__("torch").float32,
        "float16": __import__("torch").float16,
        "bfloat16": __import__("torch").bfloat16,
        "int64": __import__("torch").int64,
        "int32": __import__("torch").int32,
        "int16": __import__("torch").int16,
        "int8": __import__("torch").int8,
        "uint8": __import__("torch").uint8,
        "bool": __import__("torch").bool,
    }.get(dt, __import__("torch").float32)

def build_torch_dummy_inputs(input_shapes: Dict[str, List[int]],
                             input_dtypes: Dict[str, str]) -> Tuple[List[str], Tuple]:
    import torch
    names = list(input_shapes.keys())
    tensors = []
    for name in names:
        shape = input_shapes[name]
        if name.lower().endswith(("ids","id")) or "mask" in name.lower():
            default_dt = "int64"
        else:
            default_dt = "float32"
        dt_str = input_dtypes.get(name, default_dt)
        dtype = _torch_dtype_from_str(dt_str)

        if dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            t = torch.zeros(*shape, dtype=dtype)
        else:
            t = torch.randn(*shape, dtype=dtype)
        tensors.append(t)
    return names, tuple(tensors)

def autoguess_torch_inputs(src_path: str, model) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    base = os.path.splitext(os.path.basename(src_path))[0].lower()
    batch = DEFAULTS["batch"]
    seq = DEFAULTS["transformer_seq_len"]

    if "nasnet" in base and "large" in base:
        img = 331
    elif "inception" in base:
        img = 299
    else:
        img = DEFAULTS["cnn_img"]

    arg_names: List[str] = []
    try:
        if hasattr(model, "forward") and hasattr(model.forward, "schema"):
            args = list(model.forward.schema.arguments)[1:]
            arg_names = [a.name for a in args]
    except Exception:
        pass

    if not arg_names:
        if any(t in base for t in ["bert","roberta","gpt","llama","bloom","t5","deberta","mistral"]):
            arg_names = ["input_ids", "attention_mask"]
        else:
            arg_names = ["pixel_values"]

    shapes: Dict[str, List[int]] = {}
    dtypes: Dict[str, str] = {}

    for name in arg_names:
        n = name.lower()
        if ("ids" in n) or ("token" in n) or (n.endswith("id")):
            shapes[name] = [batch, seq]
            dtypes[name] = "int64"
        elif "mask" in n:
            shapes[name] = [batch, seq]
            dtypes[name] = "int64"
        else:
            shapes[name] = [batch, 3, img, img]
            dtypes[name] = "fp32"

    return shapes, dtypes

def export_torch_to_onnx(src_path: str,
                         input_shapes: Dict[str, List[int]],
                         input_dtypes: Dict[str, str],
                         opset: int,
                         loader_spec: Optional[str],
                         ckpt_path: Optional[str],
                         out_dir: str,
                         torch_output_names: Optional[str] = None) -> str:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch가 설치되어 있지 않습니다. `pip install torch` 후 재시도해주세요.") from e

    base = os.path.splitext(os.path.basename(src_path))[0]  # 예: nasnetalarge
    onnx_out = os.path.join(out_dir, f"{base}.onnx")

    model = None
    scripted = False

    if src_path and os.path.isfile(src_path) and src_path.lower().endswith((".pt",".pth",".ts")):
        try:
            model = torch.jit.load(src_path, map_location="cpu")
            scripted = True
        except Exception:
            scripted = False

    if model is None:
        if not loader_spec:
            raise RuntimeError(
                "일반 PyTorch 체크포인트는 모델 생성기가 필요합니다. "
                "--torch-loader \"패키지.모듈:함수명\" 형식으로 제공해주세요."
            )
        mod_name, fn_name = [x.strip() for x in loader_spec.split(":", 1)]
        mod = importlib.import_module(mod_name)
        build_fn = getattr(mod, fn_name)
        model = build_fn()
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                model.load_state_dict(sd["state_dict"], strict=False)
            elif isinstance(sd, dict):
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    pass

    model.eval()

    if not input_shapes:
        guessed_shapes, guessed_dtypes = autoguess_torch_inputs(src_path, model)
        input_shapes = guessed_shapes
        input_dtypes = {**guessed_dtypes, **input_dtypes}

    input_names, dummy_tuple = build_torch_dummy_inputs(input_shapes, input_dtypes)
    dynamic_axes = {n: {i: f"{n}_d{i}" for i in range(len(input_shapes[n]))} for n in input_names}

    output_names = None
    if torch_output_names:
        output_names = [x.strip() for x in torch_output_names.split(",") if x.strip()]

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_tuple,
            onnx_out,
            input_names=input_names,
            output_names=output_names,
            opset_version=int(opset),
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    return onnx_out

def export_tf_to_onnx(src_path: str, opset: int, out_dir: str) -> str:
    onnx_out = os.path.join(out_dir, "model.onnx")
    try:
        import tf2onnx
        from tf2onnx import convert
        import tensorflow as tf
        if os.path.isdir(src_path) and os.path.exists(os.path.join(src_path, "saved_model.pb")):
            model_proto, _ = convert.from_saved_model(src_path, opset=opset)
            with open(onnx_out, "wb") as f:
                f.write(model_proto.SerializeToString())
            return onnx_out
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--opset", str(opset)]
        if os.path.isdir(src_path):
            cmd += ["--saved-model", src_path]
        else:
            cmd += ["--graphdef", src_path]
        cmd += ["--output", onnx_out]
        subprocess.run(cmd, check=True)
        return onnx_out
    except Exception as e:
        raise RuntimeError(
            "TensorFlow 또는 tf2onnx가 설치되어 있어야 합니다. "
            "`pip install tensorflow tf2onnx` 후 재시도해주세요."
        ) from e

def main():
    ap = argparse.ArgumentParser(description="Analyze model (ONNX/Torch/TF) -> summary.json + timeline.json (activation-only, arch-prior, predictive)")
    ap.add_argument("--model", required=True, help="Path to model (.onnx | .pt/.pth/.ts | TF SavedModel dir/.pb)")
    ap.add_argument("--framework", default="auto", choices=["auto", "onnx", "torch", "tf"])

    ap.add_argument("--arch-group", type=str, default="auto", choices=["auto", "transformer", "cnn"], help="Force or auto-detect arch prior")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--input-shape", type=str, default=None, help="name:dim0,dim1;name2:... overrides (also used for Torch export dummy)")
    ap.add_argument("--input-dtypes", type=str, default=None, help="Torch export input dtypes. e.g., input_ids:int64;attention_mask:int64;pixel_values:fp16")
    ap.add_argument("--inference-dtype", type=str, default="fp32")
    ap.add_argument("--timeline-window", type=int, default=64)
    ap.add_argument("--summary-json", type=str, default=None, help="default: json/<stem>_summary.json")
    ap.add_argument("--timeline-json", type=str, default=None, help="default: timeline/<stem>_timeline.json")
    ap.add_argument("--calibration-json", type=str, default=None, help="Optional calibration JSON (model/arch/global). Will be blended toward arch prior.")

    ap.add_argument("--predictive-preset", type=str, default="regular", choices=list(PREDICTIVE_PRESETS.keys()),
                    help="High-level behavior: coarse | regular | precise")
    ap.add_argument("--use-family-cal", action="store_true", help="Allow using FAMILY multipliers (will be blended toward arch prior)")
    ap.add_argument("--family-shrink", type=float, default=0.60, help="Blend factor toward arch prior when using calibration (0=no shrink, 1=arch-only)")
    ap.add_argument("--uncertainty-frac", type=float, default=0.10, help="± band around the central estimate (e.g., 0.10 ⇒ ±10%)")

    ap.add_argument("--opset", type=int, default=17, help="Export opset for ONNX when converting from torch/tf")
    ap.add_argument("--torch-loader", type=str, default=None, help="PyTorch model builder. e.g., mypkg.my_model:build_model")
    ap.add_argument("--torch-ckpt", type=str, default=None, help="PyTorch checkpoint path (state_dict/obj both tried)")
    ap.add_argument("--torch-output-names", type=str, default=None, help="Comma-separated output tensor names for Torch export")

    args = ap.parse_args()

    p = pathlib.Path(args.model)
    name = p.name if p.is_dir() else p.stem
    summary_out = args.summary_json or str(pathlib.Path("json") / f"{name}_summary.json")
    timeline_out = args.timeline_json or str(pathlib.Path("timeline") / f"{name}_timeline.json")
    pathlib.Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(timeline_out).parent.mkdir(parents=True, exist_ok=True)

    preset = PREDICTIVE_PRESETS.get(args.predictive_preset, PREDICTIVE_PRESETS["regular"])
    use_family = args.use_family_cal or preset["use_family"]
    family_shrink = args.family_shrink if args.family_shrink is not None else preset["family_shrink"]
    uncertainty_frac = args.uncertainty_frac if args.uncertainty_frac is not None else preset["uncertainty_frac"]

    fw = guess_framework(args.model, args.framework)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = args.model
        if fw == "torch":
            input_shapes = parse_input_overrides(args.input_shape)
            input_dtypes = parse_input_dtypes(args.input_dtypes)
            onnx_path = export_torch_to_onnx(
                src_path=args.model,
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                opset=args.opset,
                loader_spec=args.torch_loader,
                ckpt_path=args.torch_ckpt,
                out_dir=tmpdir,
                torch_output_names=args.torch_output_names,
            )
        elif fw == "tf":
            onnx_path = export_tf_to_onnx(args.model, args.opset, tmpdir)
        elif fw != "onnx":
            raise NotImplementedError("Unsupported framework")

        summary_dict, timeline_dict = analyze_onnx(
            onnx_path=onnx_path,
            batch=args.batch,
            input_shape_str=args.input_shape,
            inference_dtype=args.inference_dtype,
            timeline_window=args.timeline_window,
            arch_group=args.arch_group,
            calibration_json=args.calibration_json,
            use_family_cal=use_family,
            family_shrink=float(np.clip(family_shrink, 0.0, 1.0)),
            uncertainty_frac=float(np.clip(uncertainty_frac, 0.0, 0.5)),
        )

    with open(summary_out, "w") as f:
        json.dump(summary_dict, f, indent=2)
    with open(timeline_out, "w") as f:
        json.dump(timeline_dict, f, indent=2)

    print(f"Summary JSON  -> {summary_out}")
    print(f"Timeline JSON -> {timeline_out}")

if __name__ == "__main__":
    main()


