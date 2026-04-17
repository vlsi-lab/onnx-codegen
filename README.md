# onnx-codegen

Generate self-contained C inference code from ONNX models, targeting bare-metal MCU deployment.

## Features

- Pure C output with no external runtime dependencies
- Static-shape ONNX graphs (recommended for MCU targets)
- Quantised inference: `8w8a`, `8w16a`, `16w16a` weight/activation bit-widths
  with mixed `act storage + int32 accumulator/logit` runtime types when needed
- Fused Conv1D + requantisation kernels (pulp-nn style)
- Liveness-based scratch buffer allocation to minimise SRAM usage
- Per-layer `#if` guards for selective compilation

## Supported ONNX Ops

Identity, Relu, LeakyRelu, Sigmoid, Tanh, Clip, Add, Mul, Div, Floor,
MatMul, Gemm, Conv (1D NCW, 2D NCHW, groups), MaxPool, AveragePool,
GlobalAveragePool, Flatten, Reshape, Transpose, Squeeze, Unsqueeze,
Softmax, BatchNormalization, Concat, Pad, Slice.

## Installation

```bash
pip install .
```

Or in development mode:

```bash
pip install -e .
```

## Usage

```bash
onnx-codegen --onnx model.onnx --out-dir output/ --prefix my_model --quant 8w8a
```

The CLI prints a memory breakdown after generation, including weights/constants,
scratch buffers, input/output tensors, and total estimated model memory.

To generate the C and immediately compare it against the ONNX reference with a
temporary `main.c` harness:

```bash
onnx-codegen --onnx model.onnx --out-dir output/ --prefix my_model --compare
```

If the generated C comes from an integer-form model but the reference should be
the original float-parameter ONNX, pass it explicitly:

```bash
onnx-codegen --onnx model_int.onnx --compare --compare-onnx model_float.onnx
```

Options:
- `--onnx` — Path to `.onnx` file or `.zip` containing one
- `--out-dir` — Output directory (default: `.`)
- `--prefix` — C symbol prefix (default: sanitised model filename)
- `--quant NwMa` — Force N-bit weights and requantized activation storage
  (e.g. `8w8a`); accumulators/residual sums/logits may still use `int32`
- `--skip-shape-inference` — Skip ONNX shape inference pass
- `--custom-kernels-header` — Extra header to include in generated `.c`
- `--compare` — Compile the generated C and compare it against the ONNX model
- `--compare-onnx` — Alternate ONNX file used only as the comparison reference
- `--compare-random-cases` — Additional random inputs for `--compare` (default: `2`)
- `--compare-seed` — RNG seed for `--compare` (default: `0`)
- `--cc` — C compiler used by `--compare` (default: `$CC` or `gcc`)

## Output Files

| File | Description |
|------|-------------|
| `<prefix>_model.h` | Public API: `<prefix>_init()`, `<prefix>_run()` |
| `<prefix>_model.c` | Inference implementation with all kernels inlined |
| `<prefix>_weights.h` | Static constant weight arrays |
| `<prefix>_layer_cfg.h` | Per-layer enable/disable switches |

## License

Apache-2.0
