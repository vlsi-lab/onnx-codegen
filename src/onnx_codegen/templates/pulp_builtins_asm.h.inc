/*
 * PULP builtin assembly implementations - generated.
 *
 * Alternative to __builtin_pulp_* functions for cases where
 * compiler builtin support is unavailable.
 *
 * Usage:
 *   Define PULPNN_USE_ASM before including to use assembly versions
 *   instead of compiler builtins.
 */

#ifndef PULP_NN_BUILTINS_ASM_H
#define PULP_NN_BUILTINS_ASM_H

#include <stdint.h>

/* ====================================================================
 *  Bit Extraction & Manipulation
 * ==================================================================== */

/**
 * @brief Signed bit extraction using assembly
 * Extract 'size' bits from 'x' starting at offset 'off'
 */
static inline int32_t asm_bextract(int32_t x, uint32_t size, uint32_t off) {
    int32_t result;
    asm volatile (
        "p.bextract %0, %1, %2(%3)"
        : "=r" (result)
        : "r" (x), "r" (off), "r" (size)
    );
    return result;
}

/**
 * @brief Unsigned bit extraction using assembly
 * Extract 'size' bits from 'x' starting at offset 'off'
 */
static inline uint32_t asm_bextractu(uint32_t x, uint32_t size, uint32_t off) {
    uint32_t result;
    asm volatile (
        "p.bextractu %0, %1, %2(%3)"
        : "=r" (result)
        : "r" (x), "r" (off), "r" (size)
    );
    return result;
}

/**
 * @brief Bit insertion using assembly
 * Insert 'value' of 'size' bits into 'x' at offset 'off'
 */
static inline int32_t asm_binsert(int32_t x, uint32_t value, uint32_t size, uint32_t off) {
    int32_t result;
    asm volatile (
        "p.binsert %0, %1, %2(%3)"
        : "=r" (result)
        : "r" (x), "r" (value), "r" (off), "r" (size)
    );
    return result;
}

/* ====================================================================
 *  Vector Operations
 * ==================================================================== */

/**
 * @brief Pack 4 signed values into a 32-bit word
 */
static inline int32_t asm_pack4(int32_t a, int32_t b, int32_t c, int32_t d) {
    int32_t result;
    int32_t a8 = a & 0xFF;
    int32_t b8 = b & 0xFF;
    int32_t c8 = c & 0xFF;
    int32_t d8 = d & 0xFF;
    result = (d8 << 24) | (c8 << 16) | (b8 << 8) | a8;
    return result;
}

/**
 * @brief Maximum of 4 unsigned 8-bit elements
 */
static inline uint32_t asm_maxu4(uint32_t a, uint32_t b) {
    uint32_t result;
    asm volatile (
        "p.max4 %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/**
 * @brief Minimum of 4 unsigned 8-bit elements
 */
static inline uint32_t asm_minu4(uint32_t a, uint32_t b) {
    uint32_t result;
    asm volatile (
        "p.min4 %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/**
 * @brief Average of 4 unsigned 8-bit elements
 */
static inline uint32_t asm_avg4(uint32_t a, uint32_t b) {
    uint32_t result;
    asm volatile (
        "p.avg4 %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/* ====================================================================
 *  Clipping and Rounding
 * ==================================================================== */

/**
 * @brief Clip unsigned value
 * Result is max(0, min(maxval, value))
 */
static inline uint32_t asm_clipu_r(int32_t value, uint32_t maxval) {
    if (value < 0) return 0;
    else if (value > (int32_t)maxval) return maxval;
    else return (uint32_t)value;
}

/**
 * @brief Clip signed value with rounding
 */
static inline int32_t asm_clip_r(int32_t value, int32_t minval, int32_t maxval) {
    if (value < minval) return minval;
    else if (value > maxval) return maxval;
    else return value;
}

/* ====================================================================
 *  Bit Counting and Search
 * ==================================================================== */

/**
 * @brief Find last set bit (Find First 1 from left)
 * Returns position of the most significant '1' bit (0-31)
 */
static inline int32_t asm_fl1(uint32_t x) {
    int32_t result;
    asm volatile (
        "p.fl1 %0, %1"
        : "=r" (result)
        : "r" (x)
    );
    return result;
}

/**
 * @brief Count leading bits
 */
static inline int32_t asm_clb(int32_t x) {
    int32_t result;
    asm volatile (
        "p.clb %0, %1"
        : "=r" (result)
        : "r" (x)
    );
    return result;
}

/**
 * @brief Rotate right
 */
static inline uint32_t asm_rotr(uint32_t x, uint32_t n) {
    uint32_t result;
    asm volatile (
        "p.rotr %0, %1, %2"
        : "=r" (result)
        : "r" (x), "r" (n)
    );
    return result;
}

/* ====================================================================
 *  Multiplier and MAC Operations
 * ==================================================================== */

/**
 * @brief Signed dot product of 4 signed 8-bit elements
 * Returns sum of (a[i] * b[i]) for i in [0,3]
 */
static inline int32_t asm_dotsp4(int32_t a, int32_t b) {
    int32_t result;
    asm volatile (
        "p.dotsp4 %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/**
 * @brief Signed*Unsigned dot product of 4 elements
 * a is signed 8-bit, b is unsigned 8-bit
 */
static inline int32_t asm_sdotusp4(int32_t a, uint32_t b) {
    int32_t result;
    asm volatile (
        "p.sdotusp4 %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/**
 * @brief Unsigned dot product of 4 unsigned 8-bit elements
 */
static inline uint32_t asm_dotup4(uint32_t a, uint32_t b) {
    uint32_t result;
    asm volatile (
        "p.dotup4 %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/**
 * @brief Multiply and accumulate (signed)
 */
static inline int32_t asm_mac(int32_t a, int32_t b, int32_t c) {
    int32_t result = c;
    asm volatile (
        "p.mac %0, %1, %2"
        : "+r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

/* ====================================================================
 *  Shift Operations
 * ==================================================================== */

/**
 * @brief Arithmetic shift right by immediate
 */
static inline int32_t asm_sra_i(int32_t a, uint32_t shift) {
    int32_t result;
    asm volatile (
        "p.sra %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (shift)
    );
    return result;
}

/**
 * @brief Logical shift right by immediate
 */
static inline uint32_t asm_srl_i(uint32_t a, uint32_t shift) {
    uint32_t result;
    asm volatile (
        "p.srl %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (shift)
    );
    return result;
}

/**
 * @brief Logical shift left by immediate
 */
static inline uint32_t asm_sll_i(uint32_t a, uint32_t shift) {
    uint32_t result;
    asm volatile (
        "p.sll %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (shift)
    );
    return result;
}

#endif /* PULP_NN_BUILTINS_ASM_H */
