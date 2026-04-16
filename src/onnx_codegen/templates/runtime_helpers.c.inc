/*
 * Type configuration:
 * - ONNXCG_ACT_T: element type for activations/intermediates (default: float).
 *   Set via --quant flag, e.g. --quant 8w8a makes this int8_t.
 * - conv1d / linear kernels live in the generated <prefix>_kernels.h/.c files.
 */
#ifndef ONNXCG_ACT_T
#define ONNXCG_ACT_T float
#endif

/* Convenience: cast a literal to the activation type. */
#define ONNXCG_ACT(x) ((ONNXCG_ACT_T)(x))

static size_t tensor_numel(const int* dims, int rank) {
    size_t n = 1;
    for (int i = 0; i < rank; ++i) {
        n *= (size_t)dims[i];
    }
    return n;
}

static void tensor_copy(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n) {
    memcpy(out, in, n * sizeof(ONNXCG_ACT_T));
}

static void tensor_copy_bytes(const void* in, void* out, size_t nbytes) {
    memcpy(out, in, nbytes);
}

static inline int clamp_int(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static void quantize_linear_u8(
    const float* in,
    uint8_t* out,
    size_t n,
    const float* scale,
    const int* scale_dims,
    int scale_rank,
    const uint8_t* zero_point,
    const int* zp_dims,
    int zp_rank) {
    (void)scale_dims;
    (void)scale_rank;
    (void)zp_dims;
    (void)zp_rank;
    float s = scale[0];
    uint8_t z = zero_point[0];
    if (s == 0.0f) s = 1.0f;
    for (size_t i = 0; i < n; ++i) {
        int q = (int)lrintf(in[i] / s) + (int)z;
        out[i] = (uint8_t)clamp_int(q, 0, 255);
    }
}

static void quantize_linear_s8(
    const float* in,
    int8_t* out,
    size_t n,
    const float* scale,
    const int* scale_dims,
    int scale_rank,
    const int8_t* zero_point,
    const int* zp_dims,
    int zp_rank) {
    (void)scale_dims;
    (void)scale_rank;
    (void)zp_dims;
    (void)zp_rank;
    float s = scale[0];
    int8_t z = zero_point[0];
    if (s == 0.0f) s = 1.0f;
    for (size_t i = 0; i < n; ++i) {
        int q = (int)lrintf(in[i] / s) + (int)z;
        out[i] = (int8_t)clamp_int(q, -128, 127);
    }
}

static void dequantize_linear_u8(
    const uint8_t* in,
    float* out,
    size_t n,
    const float* scale,
    const int* scale_dims,
    int scale_rank,
    const uint8_t* zero_point,
    const int* zp_dims,
    int zp_rank) {
    (void)scale_dims;
    (void)scale_rank;
    (void)zp_dims;
    (void)zp_rank;
    float s = scale[0];
    uint8_t z = zero_point[0];
    for (size_t i = 0; i < n; ++i) {
        out[i] = ((float)((int)in[i] - (int)z)) * s;
    }
}

static void dequantize_linear_s8(
    const int8_t* in,
    float* out,
    size_t n,
    const float* scale,
    const int* scale_dims,
    int scale_rank,
    const int8_t* zero_point,
    const int* zp_dims,
    int zp_rank) {
    (void)scale_dims;
    (void)scale_rank;
    (void)zp_dims;
    (void)zp_rank;
    float s = scale[0];
    int8_t z = zero_point[0];
    for (size_t i = 0; i < n; ++i) {
        out[i] = ((float)((int)in[i] - (int)z)) * s;
    }
}

static void tensor_relu(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = (in[i] > ONNXCG_ACT(0)) ? in[i] : ONNXCG_ACT(0);
    }
}

static void tensor_leaky_relu(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n, ONNXCG_ACT_T alpha) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = (in[i] >= ONNXCG_ACT(0)) ? in[i] : ONNXCG_ACT(alpha * in[i]);
    }
}

static void tensor_sigmoid(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = ONNXCG_ACT(1.0f / (1.0f + expf(-(float)in[i])));
    }
}

static void tensor_tanh(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = ONNXCG_ACT(tanhf((float)in[i]));
    }
}

static void tensor_clip(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n, ONNXCG_ACT_T min_v, ONNXCG_ACT_T max_v) {
    for (size_t i = 0; i < n; ++i) {
        ONNXCG_ACT_T x = in[i];
        if (x < min_v) x = min_v;
        if (x > max_v) x = max_v;
        out[i] = x;
    }
}

static void tensor_add_broadcast(
    const ONNXCG_ACT_T* a, const int* a_dims, int a_rank,
    const ONNXCG_ACT_T* b, const int* b_dims, int b_rank,
    ONNXCG_ACT_T* out, const int* out_dims, int out_rank) {
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

static void tensor_mul_broadcast(
    const ONNXCG_ACT_T* a, const int* a_dims, int a_rank,
    const ONNXCG_ACT_T* b, const int* b_dims, int b_rank,
    ONNXCG_ACT_T* out, const int* out_dims, int out_rank) {
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
        out[linear] = a[a_off] * b[b_off];
    }
}

static void tensor_div_broadcast(
    const ONNXCG_ACT_T* a, const int* a_dims, int a_rank,
    const ONNXCG_ACT_T* b, const int* b_dims, int b_rank,
    ONNXCG_ACT_T* out, const int* out_dims, int out_rank) {
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
        out[linear] = a[a_off] / b[b_off];
    }
}

static void tensor_floor(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = ONNXCG_ACT(floorf((float)in[i]));
    }
}

static void pad_tensor_constant(
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

static void slice_tensor(
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

/* Standalone per-channel requant for NCW tensors (fallback). */
static void requant_channel_ncw(
    const ONNXCG_ACT_T* in, uint8_t* out,
    const int32_t* kappa, const int32_t* lambda, int shift, int lo, int hi,
    int n, int c, int w) {
    for (int bn = 0; bn < n; ++bn) {
        for (int ch = 0; ch < c; ++ch) {
            for (int ix = 0; ix < w; ++ix) {
                size_t idx = (size_t)bn * c * w + (size_t)ch * w + (size_t)ix;
                int64_t rq = (int64_t)in[idx] * kappa[ch] + lambda[ch];
                int32_t res = (int32_t)(rq >> shift);
                if (res < lo) res = lo;
                if (res > hi) res = hi;
                out[idx] = (uint8_t)res;
            }
        }
    }
}

static void conv2d_nchw(
    const ONNXCG_ACT_T* x, const float* w, const float* b, ONNXCG_ACT_T* y,
    int n, int cin, int hin, int win,
    int cout, int kh, int kw,
    int sh, int sw, int pt, int pl, int pb, int pr,
    int dh, int dw, int groups,
    int hout, int wout) {
    int cin_g = cin / groups;
    int cout_g = cout / groups;

    for (int bn = 0; bn < n; ++bn) {
        for (int oc = 0; oc < cout; ++oc) {
            int g = oc / cout_g;
            for (int oy = 0; oy < hout; ++oy) {
                for (int ox = 0; ox < wout; ++ox) {
                    float acc = b ? b[oc] : 0.0f;
                    for (int ic = 0; ic < cin_g; ++ic) {
                        int ic_abs = g * cin_g + ic;
                        for (int ky = 0; ky < kh; ++ky) {
                            int iy = oy * sh - pt + ky * dh;
                            if (iy < 0 || iy >= hin) continue;
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * sw - pl + kx * dw;
                                if (ix < 0 || ix >= win) continue;
                                size_t x_idx = ((size_t)bn * cin + (size_t)ic_abs) * hin * win + (size_t)iy * win + (size_t)ix;
                                size_t w_idx = ((size_t)oc * cin_g + (size_t)ic) * kh * kw + (size_t)ky * kw + (size_t)kx;
                                acc += (float)x[x_idx] * w[w_idx];
                            }
                        }
                    }
                    size_t y_idx = ((size_t)bn * cout + (size_t)oc) * hout * wout + (size_t)oy * wout + (size_t)ox;
                    y[y_idx] = ONNXCG_ACT(acc);
                }
            }
        }
    }
}

static void maxpool2d_nchw(
    const ONNXCG_ACT_T* x, ONNXCG_ACT_T* y,
    int n, int c, int hin, int win,
    int kh, int kw,
    int sh, int sw,
    int pt, int pl,
    int dh, int dw,
    int hout, int wout) {
    for (int bn = 0; bn < n; ++bn) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oy = 0; oy < hout; ++oy) {
                for (int ox = 0; ox < wout; ++ox) {
                    ONNXCG_ACT_T best = x[((size_t)bn * c + (size_t)ch) * hin * win];
                    int found = 0;
                    for (int ky = 0; ky < kh; ++ky) {
                        int iy = oy * sh - pt + ky * dh;
                        if (iy < 0 || iy >= hin) continue;
                        for (int kx = 0; kx < kw; ++kx) {
                            int ix = ox * sw - pl + kx * dw;
                            if (ix < 0 || ix >= win) continue;
                            size_t x_idx = ((size_t)bn * c + (size_t)ch) * hin * win + (size_t)iy * win + (size_t)ix;
                            if (!found || x[x_idx] > best) { best = x[x_idx]; found = 1; }
                        }
                    }
                    size_t y_idx = ((size_t)bn * c + (size_t)ch) * hout * wout + (size_t)oy * wout + (size_t)ox;
                    y[y_idx] = best;
                }
            }
        }
    }
}

static void avgpool2d_nchw(
    const ONNXCG_ACT_T* x, ONNXCG_ACT_T* y,
    int n, int c, int hin, int win,
    int kh, int kw,
    int sh, int sw,
    int pt, int pl,
    int dh, int dw,
    int hout, int wout,
    int count_include_pad) {
    for (int bn = 0; bn < n; ++bn) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oy = 0; oy < hout; ++oy) {
                for (int ox = 0; ox < wout; ++ox) {
                    float acc = 0.0f;
                    int count = 0;
                    for (int ky = 0; ky < kh; ++ky) {
                        int iy = oy * sh - pt + ky * dh;
                        for (int kx = 0; kx < kw; ++kx) {
                            int ix = ox * sw - pl + kx * dw;
                            int in = (iy >= 0 && iy < hin && ix >= 0 && ix < win);
                            if (in) {
                                size_t x_idx = ((size_t)bn * c + (size_t)ch) * hin * win + (size_t)iy * win + (size_t)ix;
                                acc += (float)x[x_idx];
                                count++;
                            } else if (count_include_pad) {
                                count++;
                            }
                        }
                    }
                    float denom = (count > 0) ? (float)count : 1.0f;
                    size_t y_idx = ((size_t)bn * c + (size_t)ch) * hout * wout + (size_t)oy * wout + (size_t)ox;
                    y[y_idx] = ONNXCG_ACT(acc / denom);
                }
            }
        }
    }
}

static void global_avg_pool_nchw(const ONNXCG_ACT_T* x, ONNXCG_ACT_T* y, int n, int c, int h, int w) {
    int hw = h * w;
    for (int bn = 0; bn < n; ++bn) {
        for (int ch = 0; ch < c; ++ch) {
            float acc = 0.0f;
            for (int i = 0; i < hw; ++i) {
                size_t x_idx = ((size_t)bn * c + (size_t)ch) * hw + (size_t)i;
                acc += (float)x[x_idx];
            }
            size_t y_idx = ((size_t)bn * c + (size_t)ch);
            y[y_idx] = ONNXCG_ACT(acc / (float)hw);
        }
    }
}

static void transpose_tensor(
    const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out,
    const int* in_dims, const int* perm, int rank) {
    size_t in_strides[8] = {0};
    size_t out_dims[8] = {0};
    size_t out_strides[8] = {0};
    if (rank > 8) return;

    in_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * (size_t)in_dims[i + 1];
    }

    for (int i = 0; i < rank; ++i) out_dims[i] = (size_t)in_dims[perm[i]];

    out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    size_t total = 1;
    for (int i = 0; i < rank; ++i) total *= out_dims[i];

    for (size_t out_idx = 0; out_idx < total; ++out_idx) {
        size_t rem = out_idx;
        int out_multi[8] = {0};
        for (int d = 0; d < rank; ++d) {
            out_multi[d] = (int)(rem / out_strides[d]);
            rem %= out_strides[d];
        }

        size_t in_idx = 0;
        for (int d = 0; d < rank; ++d) {
            int in_dim = perm[d];
            in_idx += (size_t)out_multi[d] * in_strides[in_dim];
        }
        out[out_idx] = in[in_idx];
    }
}

static void softmax_axis(const ONNXCG_ACT_T* in, ONNXCG_ACT_T* out, const int* dims, int rank, int axis) {
    size_t outer = 1;
    size_t inner = 1;
    int ax = axis;
    if (ax < 0) ax += rank;
    for (int i = 0; i < ax; ++i) outer *= (size_t)dims[i];
    for (int i = ax + 1; i < rank; ++i) inner *= (size_t)dims[i];
    int axis_dim = dims[ax];

    for (size_t o = 0; o < outer; ++o) {
        for (size_t inr = 0; inr < inner; ++inr) {
            size_t base = o * (size_t)axis_dim * inner + inr;
            float m = -INFINITY;
            for (int a = 0; a < axis_dim; ++a) {
                float v = (float)in[base + (size_t)a * inner];
                if (v > m) m = v;
            }
            float s = 0.0f;
            for (int a = 0; a < axis_dim; ++a) {
                float e = expf((float)in[base + (size_t)a * inner] - m);
                out[base + (size_t)a * inner] = ONNXCG_ACT(e);
                s += e;
            }
            if (s == 0.0f) s = 1.0f;
            for (int a = 0; a < axis_dim; ++a) {
                out[base + (size_t)a * inner] = ONNXCG_ACT((float)out[base + (size_t)a * inner] / s);
            }
        }
    }
}

static void batch_norm_nchw(
    const ONNXCG_ACT_T* x, ONNXCG_ACT_T* y,
    const float* scale, const float* bias,
    const float* mean, const float* var,
    int n, int c, int spatial, float eps) {
    for (int bn = 0; bn < n; ++bn) {
        for (int ch = 0; ch < c; ++ch) {
            float inv = 1.0f / sqrtf(var[ch] + eps);
            float a = scale[ch] * inv;
            float bb = bias[ch] - mean[ch] * a;
            for (int i = 0; i < spatial; ++i) {
                size_t idx = ((size_t)bn * c + (size_t)ch) * (size_t)spatial + (size_t)i;
                y[idx] = ONNXCG_ACT(a * (float)x[idx] + bb);
            }
        }
    }
}

static void concat_axis(
    const ONNXCG_ACT_T** inputs, const int** in_dims, const int* in_ranks, int n_inputs,
    ONNXCG_ACT_T* out, const int* out_dims, int out_rank, int axis) {
    (void)out_dims;
    int ax = axis;
    if (ax < 0) ax += out_rank;

    size_t outer = 1;
    size_t inner = 1;
    for (int i = 0; i < ax; ++i) outer *= (size_t)out_dims[i];
    for (int i = ax + 1; i < out_rank; ++i) inner *= (size_t)out_dims[i];

    size_t axis_offset = 0;
    for (int inp = 0; inp < n_inputs; ++inp) {
        int rank = in_ranks[inp];
        int local_ax = ax - (out_rank - rank);
        int axis_dim = in_dims[inp][local_ax];
        size_t chunk = (size_t)axis_dim * inner;
        for (size_t o = 0; o < outer; ++o) {
            const ONNXCG_ACT_T* src = inputs[inp] + o * chunk;
            ONNXCG_ACT_T* dst = out + o * (size_t)out_dims[ax] * inner + axis_offset * inner;
            memcpy(dst, src, chunk * sizeof(ONNXCG_ACT_T));
        }
        axis_offset += (size_t)axis_dim;
    }
}
