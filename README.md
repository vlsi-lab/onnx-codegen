# onnx-codegen

Generate self-contained C inference code from ONNX models, targeting bare-metal MCU deployment.

## Features

- Pure C output with no external runtime dependencies
- Static-shape ONNX graphs (recommended for MCU targets)
- Quantised inference: `8w8a`, `8w16a`, `16w16a` weight/activation bit-widths
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

Options:
- `--onnx` — Path to `.onnx` file or `.zip` containing one
- `--out-dir` — Output directory (default: `.`)
- `--prefix` — C symbol prefix (default: sanitised model filename)
- `--quant NwMa` — Force N-bit weights, M-bit activations (e.g. `8w8a`)
- `--skip-shape-inference` — Skip ONNX shape inference pass
- `--custom-kernels-header` — Extra header to include in generated `.c`

## Output Files

| File | Description |
|------|-------------|
| `<prefix>_model.h` | Public API: `<prefix>_init()`, `<prefix>_run()` |
| `<prefix>_model.c` | Inference implementation with all kernels inlined |
| `<prefix>_weights.h` | Static constant weight arrays |
| `<prefix>_layer_cfg.h` | Per-layer enable/disable switches |

## License

Apache-2.0
