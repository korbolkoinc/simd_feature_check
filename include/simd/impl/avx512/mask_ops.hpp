#ifndef LIB_SIMD_IMPL_AVX512_MASK_OPS_HPP_z512k
#define LIB_SIMD_IMPL_AVX512_MASK_OPS_HPP_z512k

#include "simd/arch/detection.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#if SIMD_ARCH_X86 && SIMD_HAS_AVX512F

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct mask_ops<T, N, avx512_tag>
{
    using register_t = typename register_type<T, avx512_tag>::type;
    using mask_t = typename mask_register_type<T, avx512_tag>::type;

    static SIMD_INLINE void cmp_eq(mask_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_cmp_ps_mask(*a, *b, _CMP_EQ_OQ);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_cmp_pd_mask(*a, *b, _CMP_EQ_OQ);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_cmpeq_epi8_mask(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_cmpeq_epi16_mask(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_cmpeq_epi32_mask(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_cmpeq_epi64_mask(*a, *b);
    }

    static SIMD_INLINE void cmp_neq(mask_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_cmp_ps_mask(*a, *b, _CMP_NEQ_UQ);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_cmp_pd_mask(*a, *b, _CMP_NEQ_UQ);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_cmpneq_epi8_mask(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_cmpneq_epi16_mask(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_cmpneq_epi32_mask(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_cmpneq_epi64_mask(*a, *b);
    }

    static SIMD_INLINE void cmp_lt(mask_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_cmp_ps_mask(*a, *b, _CMP_LT_OS);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_cmp_pd_mask(*a, *b, _CMP_LT_OS);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 1)
            *dst = _mm512_cmplt_epi8_mask(*a, *b);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 2)
            *dst = _mm512_cmplt_epi16_mask(*a, *b);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 4)
            *dst = _mm512_cmplt_epi32_mask(*a, *b);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 8)
            *dst = _mm512_cmplt_epi64_mask(*a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_cmplt_epu8_mask(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_cmplt_epu16_mask(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_cmplt_epu32_mask(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_cmplt_epu64_mask(*a, *b);
    }

    static SIMD_INLINE void cmp_le(mask_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_cmp_ps_mask(*a, *b, _CMP_LE_OS);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_cmp_pd_mask(*a, *b, _CMP_LE_OS);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 1)
            *dst = _mm512_cmple_epi8_mask(*a, *b);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 2)
            *dst = _mm512_cmple_epi16_mask(*a, *b);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 4)
            *dst = _mm512_cmple_epi32_mask(*a, *b);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 8)
            *dst = _mm512_cmple_epi64_mask(*a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_cmple_epu8_mask(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_cmple_epu16_mask(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_cmple_epu32_mask(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_cmple_epu64_mask(*a, *b);
    }

    static SIMD_INLINE void cmp_gt(mask_t* dst, const register_t* a, const register_t* b)
    {
        cmp_lt(dst, b, a);
    }

    static SIMD_INLINE void cmp_ge(mask_t* dst, const register_t* a, const register_t* b)
    {
        cmp_le(dst, b, a);
    }

    static SIMD_INLINE void logical_and(mask_t* dst, const mask_t* a, const mask_t* b)
    {
        if constexpr (sizeof(mask_t) == 1)
            *dst = static_cast<mask_t>(*a & *b);
        else if constexpr (sizeof(mask_t) == 2)
            *dst = _mm512_kand(*a, *b);
        else if constexpr (sizeof(mask_t) == 4)
            *dst = static_cast<mask_t>(*a & *b);
        else
            *dst = static_cast<mask_t>(*a & *b);
    }

    static SIMD_INLINE void logical_or(mask_t* dst, const mask_t* a, const mask_t* b)
    {
        if constexpr (sizeof(mask_t) == 1)
            *dst = static_cast<mask_t>(*a | *b);
        else if constexpr (sizeof(mask_t) == 2)
            *dst = _mm512_kor(*a, *b);
        else if constexpr (sizeof(mask_t) == 4)
            *dst = static_cast<mask_t>(*a | *b);
        else
            *dst = static_cast<mask_t>(*a | *b);
    }

    static SIMD_INLINE void logical_xor(mask_t* dst, const mask_t* a, const mask_t* b)
    {
        if constexpr (sizeof(mask_t) == 1)
            *dst = static_cast<mask_t>(*a ^ *b);
        else if constexpr (sizeof(mask_t) == 2)
            *dst = _mm512_kxor(*a, *b);
        else if constexpr (sizeof(mask_t) == 4)
            *dst = static_cast<mask_t>(*a ^ *b);
        else
            *dst = static_cast<mask_t>(*a ^ *b);
    }

    static SIMD_INLINE void logical_not(mask_t* dst, const mask_t* src)
    {
        constexpr size_t elems = 64 / sizeof(T);
        mask_t all = static_cast<mask_t>((elems == 64) ? ~mask_t(0) : (mask_t(1) << elems) - 1);
        if constexpr (sizeof(mask_t) == 2)
            *dst = _mm512_kxor(*src, all);
        else
            *dst = static_cast<mask_t>(*src ^ all);
    }

    static SIMD_INLINE auto to_bitmask(const mask_t* src) -> decltype(*src)
    {
        return *src;
    }

    static SIMD_INLINE bool any(const mask_t* src)
    {
        return *src != 0;
    }

    static SIMD_INLINE bool all(const mask_t* src)
    {
        constexpr size_t elems = 64 / sizeof(T);
        mask_t all_mask =
            static_cast<mask_t>((elems == 64) ? ~mask_t(0) : (mask_t(1) << elems) - 1);
        return (*src & all_mask) == all_mask;
    }

    static SIMD_INLINE int count(const mask_t* src)
    {
        if constexpr (sizeof(mask_t) <= 2)
            return _mm_popcnt_u32(static_cast<unsigned>(*src));
        else if constexpr (sizeof(mask_t) == 4)
            return _mm_popcnt_u32(*src);
        else
            return static_cast<int>(_mm_popcnt_u64(*src));
    }

    static SIMD_INLINE void blend(register_t* dst, const register_t* a, const register_t* b,
                                  const mask_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_mask_blend_ps(*mask, *a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_mask_blend_pd(*mask, *a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_mask_blend_epi8(*mask, *a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_mask_blend_epi16(*mask, *a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_mask_blend_epi32(*mask, *a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_mask_blend_epi64(*mask, *a, *b);
    }
};

} // namespace vector_simd::detail

#endif

#endif // LIB_SIMD_IMPL_AVX512_MASK_OPS_HPP_z512k
