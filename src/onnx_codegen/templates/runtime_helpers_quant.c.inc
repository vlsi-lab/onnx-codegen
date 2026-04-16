/*
 * Integer-only helpers for quantized builds.
 * ONNXCG_ACT_T is the storage type used for requantized activations
 * (for example uint8_t in 8w8a). Accumulators and logits use int32_t.
 * conv1d / linear kernels live in the generated <prefix>_kernels.h/.c files.
 */
#ifndef ONNXCG_ACT_T
#define ONNXCG_ACT_T uint8_t
#endif

#define ONNXCG_ACT(x) ((ONNXCG_ACT_T)(x))

static size_t tensor_numel(const int* dims, int rank) {
    size_t n = 1;
    for (int i = 0; i < rank; ++i) {
        n *= (size_t)dims[i];
    }
    return n;
}

static void tensor_copy_bytes(const void* in, void* out, size_t nbytes) {
    memcpy(out, in, nbytes);
}

static inline int64_t clamp_i64(int64_t x, int64_t lo, int64_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline ONNXCG_ACT_T clamp_i32_to_act(int32_t x) {
    if (sizeof(ONNXCG_ACT_T) == 1) {
        return (ONNXCG_ACT_T)clamp_i32(x, 0, 255);
    }
    if (sizeof(ONNXCG_ACT_T) == 2) {
        return (ONNXCG_ACT_T)clamp_i32(x, -32768, 32767);
    }
    return (ONNXCG_ACT_T)x;
}

static void tensor_clip_act(
    const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n,
    ONNXCG_ACT_T min_v, ONNXCG_ACT_T max_v) {
    for (size_t i = 0; i < n; ++i) {
        ONNXCG_ACT_T x = in[i];
        if (x < min_v) x = min_v;
        if (x > max_v) x = max_v;
        out[i] = x;
    }
}

static void tensor_clip_i32(
    const int32_t* in, int32_t* out, size_t n,
    int32_t min_v, int32_t max_v) {
    for (size_t i = 0; i < n; ++i) {
        int32_t x = in[i];
        if (x < min_v) x = min_v;
        if (x > max_v) x = max_v;
        out[i] = x;
    }
}

static void tensor_clip_i32_to_act(
    const int32_t* in, ONNXCG_ACT_T* out, size_t n,
    int32_t min_v, int32_t max_v) {
    for (size_t i = 0; i < n; ++i) {
        int32_t x = clamp_i32(in[i], min_v, max_v);
        out[i] = clamp_i32_to_act(x);
    }
}

static void tensor_add_broadcast_act_act_i32(
    const ONNXCG_ACT_T* a, const int* a_dims, int a_rank,
    const ONNXCG_ACT_T* b, const int* b_dims, int b_rank,
    int32_t* out, const int* out_dims, int out_rank) {
    size_t out_n = tensor_numel(out_dims, out_rank);
    size_t a_strides[8] = {0};
    size_t b_strides[8] = {0};
    size_t out_strides[8] = {0};

    if (out_rank > 8) return;

    out_strides[out_rank - 1] = 1;
    for (int i = out_rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];
    }

    if (a_rank > 0) {
        a_strides[a_rank - 1] = 1;
        for (int i = a_rank - 2; i >= 0; --i) {
            a_strides[i] = a_strides[i + 1] * (size_t)a_dims[i + 1];
        }
    }
    if (b_rank > 0) {
        b_strides[b_rank - 1] = 1;
        for (int i = b_rank - 2; i >= 0; --i) {
            b_strides[i] = b_strides[i + 1] * (size_t)b_dims[i + 1];
        }
    }

    for (size_t linear = 0; linear < out_n; ++linear) {
        size_t rem = linear;
        size_t a_off = 0;
        size_t b_off = 0;
        for (int od = 0; od < out_rank; ++od) {
            int idx = (int)(rem / out_strides[od]);
            rem %= out_strides[od];

            int ad = od - (out_rank - a_rank);
            if (ad >= 0 && a_rank > 0 && a_dims[ad] != 1) {
                a_off += (size_t)idx * a_strides[ad];
            }
            int bd = od - (out_rank - b_rank);
            if (bd >= 0 && b_rank > 0 && b_dims[bd] != 1) {
                b_off += (size_t)idx * b_strides[bd];
            }
        }
        out[linear] = (int32_t)a[a_off] + (int32_t)b[b_off];
    }
}

static void tensor_add_broadcast_i32(
    const int32_t* a, const int* a_dims, int a_rank,
    const int32_t* b, const int* b_dims, int b_rank,
    int32_t* out, const int* out_dims, int out_rank) {
    size_t out_n = tensor_numel(out_dims, out_rank);
    size_t a_strides[8] = {0};
    size_t b_strides[8] = {0};
    size_t out_strides[8] = {0};

    if (out_rank > 8) return;

    out_strides[out_rank - 1] = 1;
    for (int i = out_rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];
    }

    if (a_rank > 0) {
        a_strides[a_rank - 1] = 1;
        for (int i = a_rank - 2; i >= 0; --i) {
            a_strides[i] = a_strides[i + 1] * (size_t)a_dims[i + 1];
        }
    }
    if (b_rank > 0) {
        b_strides[b_rank - 1] = 1;
        for (int i = b_rank - 2; i >= 0; --i) {
            b_strides[i] = b_strides[i + 1] * (size_t)b_dims[i + 1];
        }
    }

    for (size_t linear = 0; linear < out_n; ++linear) {
        size_t rem = linear;
        size_t a_off = 0;
        size_t b_off = 0;
        for (int od = 0; od < out_rank; ++od) {
            int idx = (int)(rem / out_strides[od]);
            rem %= out_strides[od];

            int ad = od - (out_rank - a_rank);
            if (ad >= 0 && a_rank > 0 && a_dims[ad] != 1) {
                a_off += (size_t)idx * a_strides[ad];
            }
            int bd = od - (out_rank - b_rank);
            if (bd >= 0 && b_rank > 0 && b_dims[bd] != 1) {
                b_off += (size_t)idx * b_strides[bd];
            }
        }
        out[linear] = a[a_off] + b[b_off];
    }
}

static void pad_tensor_constant_act(
    const ONNXCG_ACT_T* in,
    ONNXCG_ACT_T* out,
    const int* in_dims,
    const int* out_dims,
    int rank,
    const int* pad_begin,
    const int* pad_end,
    ONNXCG_ACT_T value) {
    size_t in_strides[8] = {0};
    size_t out_strides[8] = {0};
    if (rank > 8) return;

    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * (size_t)in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];
    }

    size_t out_n = tensor_numel(out_dims, rank);
    for (size_t linear = 0; linear < out_n; ++linear) {
        size_t rem = linear;
        int out_idx[8] = {0};
        for (int d = 0; d < rank; ++d) {
            out_idx[d] = (int)(rem / out_strides[d]);
            rem %= out_strides[d];
        }

        int inside = 1;
        size_t in_off = 0;
        for (int d = 0; d < rank; ++d) {
            int id = out_idx[d] - pad_begin[d];
            if (id < 0 || id >= in_dims[d]) {
                inside = 0;
                break;
            }
            in_off += (size_t)id * in_strides[d];
        }
        out[linear] = inside ? in[in_off] : value;
        (void)pad_end;
    }
}

static void pad_tensor_constant_i32(
    const int32_t* in,
    int32_t* out,
    const int* in_dims,
    const int* out_dims,
    int rank,
    const int* pad_begin,
    const int* pad_end,
    int32_t value) {
    size_t in_strides[8] = {0};
    size_t out_strides[8] = {0};
    if (rank > 8) return;

    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * (size_t)in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];
    }

    size_t out_n = tensor_numel(out_dims, rank);
    for (size_t linear = 0; linear < out_n; ++linear) {
        size_t rem = linear;
        int out_idx[8] = {0};
        for (int d = 0; d < rank; ++d) {
            out_idx[d] = (int)(rem / out_strides[d]);
            rem %= out_strides[d];
        }

        int inside = 1;
        size_t in_off = 0;
        for (int d = 0; d < rank; ++d) {
            int id = out_idx[d] - pad_begin[d];
            if (id < 0 || id >= in_dims[d]) {
                inside = 0;
                break;
            }
            in_off += (size_t)id * in_strides[d];
        }
        out[linear] = inside ? in[in_off] : value;
        (void)pad_end;
    }
}

static void slice_tensor_act(
    const ONNXCG_ACT_T* in,
    ONNXCG_ACT_T* out,
    const int* in_dims,
    const int* out_dims,
    int rank,
    const int* starts,
    const int* steps) {
    size_t in_strides[8] = {0};
    size_t out_strides[8] = {0};
    if (rank > 8) return;

    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * (size_t)in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];
    }

    size_t out_n = tensor_numel(out_dims, rank);
    for (size_t linear = 0; linear < out_n; ++linear) {
        size_t rem = linear;
        int out_idx[8] = {0};
        for (int d = 0; d < rank; ++d) {
            out_idx[d] = (int)(rem / out_strides[d]);
            rem %= out_strides[d];
        }

        size_t in_off = 0;
        for (int d = 0; d < rank; ++d) {
            int id = starts[d] + out_idx[d] * steps[d];
            in_off += (size_t)id * in_strides[d];
        }
        out[linear] = in[in_off];
    }
}

static void slice_tensor_i32(
    const int32_t* in,
    int32_t* out,
    const int* in_dims,
    const int* out_dims,
    int rank,
    const int* starts,
    const int* steps) {
    size_t in_strides[8] = {0};
    size_t out_strides[8] = {0};
    if (rank > 8) return;

    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * (size_t)in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];
    }

    size_t out_n = tensor_numel(out_dims, rank);
    for (size_t linear = 0; linear < out_n; ++linear) {
        size_t rem = linear;
        int out_idx[8] = {0};
        for (int d = 0; d < rank; ++d) {
            out_idx[d] = (int)(rem / out_strides[d]);
            rem %= out_strides[d];
        }

        size_t in_off = 0;
        for (int d = 0; d < rank; ++d) {
            int id = starts[d] + out_idx[d] * steps[d];
            in_off += (size_t)id * in_strides[d];
        }
        out[linear] = in[in_off];
    }
}

static void requant_channel_ncw(
    const int32_t* in, ONNXCG_ACT_T* out,
    const int32_t* kappa, const int32_t* lambda,
    int shift, int lo, int hi,
    int n, int c, int w) {
    for (int bn = 0; bn < n; ++bn) {
        for (int ch = 0; ch < c; ++ch) {
            for (int x = 0; x < w; ++x) {
                size_t idx = ((size_t)bn * c + (size_t)ch) * w + (size_t)x;
                int64_t rq = (int64_t)in[idx] * (int64_t)kappa[ch] + (int64_t)lambda[ch];
                int32_t res = (int32_t)(rq >> shift);
                out[idx] = clamp_i32_to_act(clamp_i32(res, lo, hi));
            }
        }
    }
}
