#ifndef ${guard}
#define ${guard}

#include <stddef.h>
#include <stdint.h>

%if rogue == 1:
#include "dma.h"
#include "dma_sdk.h"
#include "rogue_dma_utils.h"
#include "mage_defs.h"
%endif

#ifndef ONNXCG_ACT_T
#define ONNXCG_ACT_T ${act_ctype}
#endif

#ifndef ONNXCG_ACT
#define ONNXCG_ACT(x) ((ONNXCG_ACT_T)(x))
#endif

/*
 * Kernel override hooks:
 * - Define ONNXCG_CONV1D_FUNC / ONNXCG_CONV1D_I8W_FUNC /
 *   ONNXCG_CONV1D_I32X_I8W_FUNC / ONNXCG_CONV1D_I8W_REQUANT_FUNC /
 *   ONNXCG_CONV1D_I32X_I8W_REQUANT_FUNC to route conv1d calls to an accelerator.
 * - Define ONNXCG_MATMUL2D_FUNC / ONNXCG_GEMM2D_FUNC to route linear kernels.
 * - Define ONNXCG_QLINEAR_CONV1D_FUNC to override QLinearConv.
 * - Define ONNXCG_DISABLE_DEFAULT_CONV1D_IMPL,
 *   ONNXCG_DISABLE_DEFAULT_CONV1D_REQUANT_IMPL,
 *   ONNXCG_DISABLE_DEFAULT_LINEAR_IMPL,
 *   ONNXCG_DISABLE_DEFAULT_QLINEAR_CONV1D_IMPL before compiling
 *   ${prefix}_kernels.c to omit the default software kernels.
 */
#ifndef ONNXCG_CONV1D_FUNC
#define ONNXCG_CONV1D_FUNC ${prefix}_kernel_conv1d_ncw
#endif

#ifndef ONNXCG_CONV1D_I8W_FUNC
#define ONNXCG_CONV1D_I8W_FUNC ${prefix}_kernel_conv1d_ncw_i8w
#endif

#ifndef ONNXCG_CONV1D_I32X_I8W_FUNC
#define ONNXCG_CONV1D_I32X_I8W_FUNC ${prefix}_kernel_conv1d_ncw_i32x_i8w
#endif

%if rogue == 1:
#ifndef ONNXCG_CONV1D_REQUANT_ROGUE_FUNC
#define ONNXCG_CONV1D_REQUANT_ROGUE_FUNC ${prefix}_rogue_dconv1d_requant
#endif
%endif

#ifndef ONNXCG_CONV1D_I8W_I32B_FUNC
#define ONNXCG_CONV1D_I8W_I32B_FUNC ${prefix}_kernel_conv1d_ncw_i8w_i32b
#endif

#ifndef ONNXCG_CONV1D_I8W_REQUANT_FUNC
#define ONNXCG_CONV1D_I8W_REQUANT_FUNC ${prefix}_kernel_conv1d_ncw_i8w_requant
#endif

#ifndef ONNXCG_CONV1D_I32X_I8W_REQUANT_FUNC
#define ONNXCG_CONV1D_I32X_I8W_REQUANT_FUNC ${prefix}_kernel_conv1d_ncw_i32x_i8w_requant
#endif

#ifndef ONNXCG_MATMUL2D_FUNC
#define ONNXCG_MATMUL2D_FUNC ${prefix}_kernel_matmul_2d
#endif

#ifndef ONNXCG_GEMM2D_FUNC
#define ONNXCG_GEMM2D_FUNC ${prefix}_kernel_gemm_2d
#endif

#ifndef ONNXCG_QLINEAR_CONV1D_FUNC
#define ONNXCG_QLINEAR_CONV1D_FUNC ${prefix}_kernel_qlinear_conv1d_u8s8u8
#endif

void ${prefix}_kernel_matmul_2d(
    const ONNXCG_ACT_T* a, const ONNXCG_ACT_T* b, ONNXCG_ACT_T* out,
    int m, int k, int n);

void ${prefix}_kernel_gemm_2d(
    const ONNXCG_ACT_T* a, const ONNXCG_ACT_T* b, const ONNXCG_ACT_T* c,
    ONNXCG_ACT_T* out, int m, int k, int n,
    int trans_a, int trans_b, ONNXCG_ACT_T alpha, ONNXCG_ACT_T beta,
    int c_rank, const int* c_dims);

% if quant_enabled:
void ${prefix}_kernel_conv1d_ncw_i8w(
    const ONNXCG_ACT_T* x, const int8_t* w, const int32_t* b, int32_t* y,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);

void ${prefix}_kernel_conv1d_ncw_i32x_i8w(
    const int32_t* x, const int8_t* w, const int32_t* b, int32_t* y,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);

%if rogue == 1:
void ${prefix}_rogue_dconv1d_requant(
    uint32_t ch_in_ptr,
    uint32_t w_ptr, 
    uint32_t ch_out_ptr,
    int32_t * kappa,
    int32_t * lambda,
    int shift,
    uint32_t ch_in,
    uint32_t ch_out,
    uint32_t time_lenght,
    uint32_t kernel_size,
    uint32_t dilation,
    uint32_t src_data_type,
    uint32_t dst_data_type
);
%endif

void ${prefix}_kernel_conv1d_ncw_i8w_requant(
    const ONNXCG_ACT_T* x, const int8_t* w, ONNXCG_ACT_T* y,
    const int32_t* kappa, const int32_t* lambda, int shift,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);

void ${prefix}_kernel_conv1d_ncw_i32x_i8w_requant(
    const int32_t* x, const int8_t* w, ONNXCG_ACT_T* y,
    const int32_t* kappa, const int32_t* lambda, int shift,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);
% else:
void ${prefix}_kernel_qlinear_conv1d_u8s8u8(
    const uint8_t* x,
    const float* x_scale,
    const uint8_t* x_zp,
    const int8_t* w,
    const float* w_scale,
    const int8_t* w_zp,
    const float* y_scale,
    const uint8_t* y_zp,
    const int32_t* b,
    uint8_t* y,
    int n,
    int cin,
    int lin,
    int cout,
    int k,
    int stride,
    int pad_l,
    int pad_r,
    int dilation,
    int groups,
    int lout);

void ${prefix}_kernel_conv1d_ncw(
    const ONNXCG_ACT_T* x, const float* w, const float* b, ONNXCG_ACT_T* y,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);

void ${prefix}_kernel_conv1d_ncw_i8w(
    const ONNXCG_ACT_T* x, const int8_t* w, const float* b, ONNXCG_ACT_T* y,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);

void ${prefix}_kernel_conv1d_ncw_i8w_i32b(
    const ONNXCG_ACT_T* x, const int8_t* w, const int32_t* b, ONNXCG_ACT_T* y,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);

void ${prefix}_kernel_conv1d_ncw_i8w_requant(
    const uint8_t* x, const int8_t* w, uint8_t* y,
    const int32_t* kappa, const int32_t* lambda, int shift,
    int n, int cin, int lin,
    int cout, int k,
    int stride, int pad_l, int pad_r, int dilation, int groups, int lout);
% endif

#endif /* ${guard} */
