# onnx-codegen

Generate self-contained C inference code from ONNX models.

## Notes

The generated Conv1D kernel code is structurally based on the blocked im2col and matmul approach used in `pulp-nn`, but the generator emits standalone code and does not require `pulp-nn` as a build-time dependency.
