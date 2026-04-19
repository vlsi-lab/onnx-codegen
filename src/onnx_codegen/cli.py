from __future__ import annotations

import argparse
from pathlib import Path

from .core import (
    CompareResult,
    GenerationResult,
    QuantConfig,
    compare_generated_c_to_onnx,
    generate_library,
    generate_test_data_header,
    sanitize_symbol,
)


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
    parser.add_argument(
        "--rogue",
        type=int,
        choices=(0, 1),
        default=0,
        metavar="0|1",
        help="Use Rogue kernels: 1 to enable, 0 to disable.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Compile the generated C and compare its outputs against the ONNX "
            "reference model using a temporary main.c harness."
        ),
    )
    parser.add_argument(
        "--compare-onnx",
        type=Path,
        default=None,
        help=(
            "Optional ONNX model used only as comparison reference. "
            "Useful when generating C from an integer-form model but comparing "
            "against the original float-parameter ONNX."
        ),
    )
    parser.add_argument(
        "--compare-random-cases",
        type=int,
        default=2,
        help="Number of random input cases to run in addition to the zero-input case.",
    )
    parser.add_argument(
        "--compare-seed",
        type=int,
        default=0,
        help="Seed used to generate random comparison inputs.",
    )
    parser.add_argument(
        "--cc",
        type=str,
        default=None,
        help="C compiler used by --compare (default: $CC or gcc).",
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help=(
            "Generate a <prefix>_test_data.h header containing random input "
            "vectors and golden (expected) outputs computed via the ONNX "
            "reference evaluator.  Useful for on-target MCU deployment tests."
        ),
    )
    parser.add_argument(
        "--test-data-cases",
        type=int,
        default=2,
        help="Number of random test-data cases (in addition to the zero-input case).",
    )
    parser.add_argument(
        "--test-data-seed",
        type=int,
        default=0,
        help="RNG seed for test-data generation.",
    )
    return parser.parse_args()


def _print_compare_summary(result: CompareResult) -> None:
    status = "PASS" if result.matches else "FAIL"
    print(f"Comparison: {status}")
    for case in result.cases:
        verdict = "ok" if case.matches else "mismatch"
        print(f"  - {case.name}: {verdict} (max_abs_diff={case.max_abs_diff:.6g})")


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n} B ({n / 1024.0:.1f} KiB)"
    return f"{n} B ({n / (1024.0 * 1024.0):.2f} MiB)"


def _print_memory_breakdown(result: GenerationResult) -> None:
    mem = result.memory
    print("Memory breakdown:")
    print(f"  - Weights/header consts: {_format_bytes(mem.weights_bytes)}")
    print(f"  - Inlined consts: {_format_bytes(mem.inlined_const_bytes)}")
    print(f"  - Scratch buffers: {_format_bytes(mem.scratch_bytes)}")
    print(f"  - Inputs: {_format_bytes(mem.input_bytes)}")
    print(f"  - Outputs: {_format_bytes(mem.output_bytes)}")
    print(f"  - Total consts: {_format_bytes(mem.total_const_bytes)}")
    print(f"  - Total runtime: {_format_bytes(mem.total_runtime_bytes)}")
    print(f"  - Total estimated model memory: {_format_bytes(mem.total_bytes)}")


def main() -> int:
    args = parse_args()
    onnx_path: Path = args.onnx
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file does not exist: {onnx_path}")

    prefix = sanitize_symbol(args.prefix if args.prefix else onnx_path.stem.lower())
    quant = QuantConfig.parse(args.quant) if args.quant else None
    result = generate_library(
        onnx_path=onnx_path,
        out_dir=args.out_dir,
        prefix=prefix,
        skip_shape_inference=args.skip_shape_inference,
        custom_kernels_header=args.custom_kernels_header,
        quant=quant,
        rogue=args.rogue
    )

    print(f"Generated: {result.model_h_path}")
    print(f"Generated: {result.model_c_path}")
    if result.kernels_h_path.exists():
        print(f"Generated: {result.kernels_h_path}")
    if result.kernels_c_path.exists():
        print(f"Generated: {result.kernels_c_path}")
    print(f"Generated: {result.weights_h_path}")
    print(
        f"Inputs: {result.n_inputs}, Outputs: {result.n_outputs}, Nodes: {result.n_nodes}"
    )
    _print_memory_breakdown(result)
    if args.test_data:
        td_path = generate_test_data_header(
            onnx_path=onnx_path,
            out_dir=args.out_dir,
            prefix=prefix,
            skip_shape_inference=args.skip_shape_inference,
            quant=quant,
            random_cases=args.test_data_cases,
            seed=args.test_data_seed,
        )
        print(f"Generated: {td_path}")
    if args.compare:
        result = compare_generated_c_to_onnx(
            onnx_path=onnx_path,
            out_dir=args.out_dir,
            prefix=prefix,
            reference_onnx_path=args.compare_onnx,
            skip_shape_inference=args.skip_shape_inference,
            quant=quant,
            random_cases=args.compare_random_cases,
            seed=args.compare_seed,
            cc=args.cc,
        )
        _print_compare_summary(result)
        if not result.matches:
            return 1
    return 0
