/*
 * PULP builtin configuration - generated.
 *
 * Configuration for selecting between compiler builtins (__builtin_pulp_*)
 * and inline assembly implementations.
 *
 * Usage:
 *   Define PULPNN_USE_ASM before including to use assembly versions
 *   instead of compiler builtins.
 */

#ifndef PULP_NN_BUILTINS_CONFIG_H
#define PULP_NN_BUILTINS_CONFIG_H

#ifdef PULPNN_USE_ASM
/* Use assembly implementations */
#include "pulp_builtins_asm.h"

#define PULP_BEXTRACT(x,size,off)     asm_bextract(x,size,off)
#define PULP_BEXTRACTU(x,size,off)    asm_bextractu(x,size,off)
#define PULP_BINSERT(x,v,size,off)    asm_binsert(x,v,size,off)
#define PULP_PACK4(a,b,c,d)           asm_pack4(a,b,c,d)
#define PULP_MAXU4(a,b)               asm_maxu4(a,b)
#define PULP_MINU4(a,b)               asm_minu4(a,b)
#define PULP_AVG4(a,b)                asm_avg4(a,b)
#define PULP_CLIPU_R(x,maxval)        asm_clipu_r(x,maxval)
#define PULP_CLIP_R(x,minval,maxval)  asm_clip_r(x,minval,maxval)
#define PULP_FL1(x)                   asm_fl1(x)
#define PULP_CLB(x)                   asm_clb(x)
#define PULP_ROTR(x,n)                asm_rotr(x,n)
#define PULP_DOTSP4(a,b)              asm_dotsp4(a,b)
#define PULP_SDOTUSP4(a,b)            asm_sdotusp4(a,b)
#define PULP_DOTUP4(a,b)              asm_dotup4(a,b)
#define PULP_MAC(a,b,c)               asm_mac(a,b,c)
#define PULP_SRA_I(a,shift)           asm_sra_i(a,shift)
#define PULP_SRL_I(a,shift)           asm_srl_i(a,shift)
#define PULP_SLL_I(a,shift)           asm_sll_i(a,shift)

#else  /* Use compiler builtins (default) */

#define PULP_BEXTRACT(x,size,off)     __builtin_pulp_bextract(x,size,off)
#define PULP_BEXTRACTU(x,size,off)    __builtin_pulp_bextractu(x,size,off)
#define PULP_BINSERT(x,v,size,off)    __builtin_pulp_binsert(x,v,size,off)
#define PULP_PACK4(a,b,c,d)           __builtin_pulp_pack4(a,b,c,d)
#define PULP_MAXU4(a,b)               __builtin_pulp_maxu4(a,b)
#define PULP_MINU4(a,b)               __builtin_pulp_minu4(a,b)
#define PULP_AVG4(a,b)                __builtin_pulp_avg4(a,b)
#define PULP_CLIPU_R(x,maxval)        __builtin_pulp_clipu_r(x,maxval)
#define PULP_CLIP_R(x,minval,maxval)  __builtin_pulp_clip_r(x,minval,maxval)
#define PULP_FL1(x)                   __builtin_pulp_fl1(x)
#define PULP_CLB(x)                   __builtin_pulp_clb(x)
#define PULP_ROTR(x,n)                __builtin_pulp_rotr(x,n)
#define PULP_DOTSP4(a,b)              __builtin_pulp_dotsp4(a,b)
#define PULP_SDOTUSP4(a,b)            __builtin_pulp_sdotusp4(a,b)
#define PULP_DOTUP4(a,b)              __builtin_pulp_dotup4(a,b)
#define PULP_MAC(a,b,c)               __builtin_pulp_mac(a,b,c)
#define PULP_SRA_I(a,shift)           __builtin_pulp_sra(a,shift)
#define PULP_SRL_I(a,shift)           __builtin_pulp_srl(a,shift)
#define PULP_SLL_I(a,shift)           __builtin_pulp_sll(a,shift)

#endif  /* PULPNN_USE_ASM */

#endif  /* PULP_NN_BUILTINS_CONFIG_H */
