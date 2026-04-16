from __future__ import annotations

import argparse
from pathlib import Path

from .core import generate_library, sanitize_symbol, QuantConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pure-C inference code from ONNX"
    )
    parser.add_argument("--onnx", required=True, type=Path, help="Path to ONNX model")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Generated file/function prefix (default: ONNX file stem sanitized)",
    )
    parser.add_argument(
        "--skip-shape-inference",
        action="store_true",
        help="Skip ONNX shape inference pass",
    )
    parser.add_argument(
        "--custom-kernels-header",
        type=str,
        default=None,
        help=(
            "Optional header included by generated model C file to provide custom "
            "kernel declarations/macros (e.g., custom conv1d implementation)."
        ),
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        metavar="NwMa",
        help=(
            "Force weight/activation types to N-bit integers.  "
            "Format: <weight_bits>w<act_bits>a  (e.g. 8w8a, 8w16a, 16w16a).  "
            "Weights are rounded+clamped; a warning is emitted when values "
            "exceed the target range."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path: Path = args.onnx
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file does not exist: {onnx_path}")

    prefix = sanitize_symbol(args.prefix if args.prefix else onnx_path.stem.lower())
    quant = QuantConfig.parse(args.quant) if args.quant else None
    model_h_path, model_c_path, weights_h_path, n_inputs, n_outputs, n_nodes = (
        generate_library(
            onnx_path=onnx_path,
            out_dir=args.out_dir,
            prefix=prefix,
            skip_shape_inference=args.skip_shape_inference,
            custom_kernels_header=args.custom_kernels_header,
            quant=quant,
        )
    )

    print(f"Generated: {model_h_path}")
    print(f"Generated: {model_c_path}")
    print(f"Generated: {weights_h_path}")
    print(f"Inputs: {n_inputs}, Outputs: {n_outputs}, Nodes: {n_nodes}")
    return 0
