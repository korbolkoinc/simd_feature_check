#ifndef LIB_SIMD_IMPL_AVX512_VECTOR_OPS_HPP_z512v
#define LIB_SIMD_IMPL_AVX512_VECTOR_OPS_HPP_z512v

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
struct vector_ops<T, N, avx512_tag>
{
    using register_t = typename register_type<T, avx512_tag>::type;

    static SIMD_INLINE void add(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_add_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_add_pd(*a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_add_epi8(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_add_epi16(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_add_epi32(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_add_epi64(*a, *b);
    }

    static SIMD_INLINE void sub(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_sub_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_sub_pd(*a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_sub_epi8(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_sub_epi16(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_sub_epi32(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_sub_epi64(*a, *b);
    }

    static SIMD_INLINE void mul(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_mul_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_mul_pd(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_mullo_epi32(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_mullo_epi16(*a, *b);
        else if constexpr (sizeof(T) == 1)
        {
            __m512i even = _mm512_mullo_epi16(*a, *b);
            __m512i odd = _mm512_mullo_epi16(_mm512_srli_epi16(*a, 8), _mm512_srli_epi16(*b, 8));
            odd = _mm512_slli_epi16(odd, 8);
            *dst = _mm512_ternarylogic_epi32(even, odd, _mm512_set1_epi8(static_cast<char>(0xFF)),
                                             0xCA);
        }
        else if constexpr (sizeof(T) == 8)
        {
            __m512i lo_a = _mm512_and_si512(*a, _mm512_set1_epi64(0xFFFFFFFF));
            __m512i lo_b = _mm512_and_si512(*b, _mm512_set1_epi64(0xFFFFFFFF));
            __m512i hi_a = _mm512_srli_epi64(*a, 32);
            __m512i mul_ll = _mm512_mul_epu32(lo_a, lo_b);
            __m512i mul_hl = _mm512_mul_epu32(hi_a, lo_b);
            __m512i mul_lh = _mm512_mul_epu32(lo_a, _mm512_srli_epi64(*b, 32));
            *dst = _mm512_add_epi64(mul_ll,
                                    _mm512_slli_epi64(_mm512_add_epi64(mul_hl, mul_lh), 32));
        }
    }

    static SIMD_INLINE void div(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_div_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_div_pd(*a, *b);
        else
        {
            constexpr size_t elems = 64 / sizeof(T);
            alignas(64) T ta[elems], tb[elems], tc[elems];
            _mm512_store_si512(reinterpret_cast<void*>(ta), *a);
            _mm512_store_si512(reinterpret_cast<void*>(tb), *b);
            for (size_t i = 0; i < elems; ++i) tc[i] = tb[i] ? ta[i] / tb[i] : 0;
            *dst = _mm512_load_si512(reinterpret_cast<const void*>(tc));
        }
    }

    static SIMD_INLINE void bitwise_and(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(*a), _mm512_castps_si512(*b)));
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_castsi512_pd(
                _mm512_and_si512(_mm512_castpd_si512(*a), _mm512_castpd_si512(*b)));
        else
            *dst = _mm512_and_si512(*a, *b);
    }

    static SIMD_INLINE void bitwise_or(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_castsi512_ps(
                _mm512_or_si512(_mm512_castps_si512(*a), _mm512_castps_si512(*b)));
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_castsi512_pd(
                _mm512_or_si512(_mm512_castpd_si512(*a), _mm512_castpd_si512(*b)));
        else
            *dst = _mm512_or_si512(*a, *b);
    }

    static SIMD_INLINE void bitwise_xor(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_castsi512_ps(
                _mm512_xor_si512(_mm512_castps_si512(*a), _mm512_castps_si512(*b)));
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_castsi512_pd(
                _mm512_xor_si512(_mm512_castpd_si512(*a), _mm512_castpd_si512(*b)));
        else
            *dst = _mm512_xor_si512(*a, *b);
    }

    static SIMD_INLINE void bitwise_not(register_t* dst, const register_t* src)
    {
        __m512i ones = _mm512_set1_epi32(-1);
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(*src), ones));
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(*src), ones));
        else
            *dst = _mm512_xor_si512(*src, ones);
    }

    static SIMD_INLINE void min(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_min_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_min_pd(*a, *b);
        else if constexpr (std::is_same_v<T, int8_t>)
            *dst = _mm512_min_epi8(*a, *b);
        else if constexpr (std::is_same_v<T, uint8_t>)
            *dst = _mm512_min_epu8(*a, *b);
        else if constexpr (std::is_same_v<T, int16_t>)
            *dst = _mm512_min_epi16(*a, *b);
        else if constexpr (std::is_same_v<T, uint16_t>)
            *dst = _mm512_min_epu16(*a, *b);
        else if constexpr (std::is_same_v<T, int32_t>)
            *dst = _mm512_min_epi32(*a, *b);
        else if constexpr (std::is_same_v<T, uint32_t>)
            *dst = _mm512_min_epu32(*a, *b);
        else if constexpr (std::is_same_v<T, int64_t>)
            *dst = _mm512_min_epi64(*a, *b);
        else if constexpr (std::is_same_v<T, uint64_t>)
            *dst = _mm512_min_epu64(*a, *b);
    }

    static SIMD_INLINE void max(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_max_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_max_pd(*a, *b);
        else if constexpr (std::is_same_v<T, int8_t>)
            *dst = _mm512_max_epi8(*a, *b);
        else if constexpr (std::is_same_v<T, uint8_t>)
            *dst = _mm512_max_epu8(*a, *b);
        else if constexpr (std::is_same_v<T, int16_t>)
            *dst = _mm512_max_epi16(*a, *b);
        else if constexpr (std::is_same_v<T, uint16_t>)
            *dst = _mm512_max_epu16(*a, *b);
        else if constexpr (std::is_same_v<T, int32_t>)
            *dst = _mm512_max_epi32(*a, *b);
        else if constexpr (std::is_same_v<T, uint32_t>)
            *dst = _mm512_max_epu32(*a, *b);
        else if constexpr (std::is_same_v<T, int64_t>)
            *dst = _mm512_max_epi64(*a, *b);
        else if constexpr (std::is_same_v<T, uint64_t>)
            *dst = _mm512_max_epu64(*a, *b);
    }

    static SIMD_INLINE void clamp(register_t* dst, const register_t* val, const register_t* lo,
                                  const register_t* hi)
    {
        register_t tmp;
        max(&tmp, val, lo);
        min(dst, &tmp, hi);
    }

    static SIMD_INLINE void add_sat(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, int8_t>)
            *dst = _mm512_adds_epi8(*a, *b);
        else if constexpr (std::is_same_v<T, uint8_t>)
            *dst = _mm512_adds_epu8(*a, *b);
        else if constexpr (std::is_same_v<T, int16_t>)
            *dst = _mm512_adds_epi16(*a, *b);
        else if constexpr (std::is_same_v<T, uint16_t>)
            *dst = _mm512_adds_epu16(*a, *b);
        else
            add(dst, a, b);
    }

    static SIMD_INLINE void sub_sat(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, int8_t>)
            *dst = _mm512_subs_epi8(*a, *b);
        else if constexpr (std::is_same_v<T, uint8_t>)
            *dst = _mm512_subs_epu8(*a, *b);
        else if constexpr (std::is_same_v<T, int16_t>)
            *dst = _mm512_subs_epi16(*a, *b);
        else if constexpr (std::is_same_v<T, uint16_t>)
            *dst = _mm512_subs_epu16(*a, *b);
        else
            sub(dst, a, b);
    }

    static SIMD_INLINE void shift_left(register_t* dst, const register_t* src, int count)
    {
        if constexpr (sizeof(T) == 2)
            *dst = _mm512_slli_epi16(*src, count);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_slli_epi32(*src, count);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_slli_epi64(*src, count);
    }

    static SIMD_INLINE void shift_right(register_t* dst, const register_t* src, int count)
    {
        if constexpr (std::is_signed_v<T> && sizeof(T) == 2)
            *dst = _mm512_srai_epi16(*src, count);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 4)
            *dst = _mm512_srai_epi32(*src, count);
        else if constexpr (std::is_signed_v<T> && sizeof(T) == 8)
            *dst = _mm512_srai_epi64(*src, count);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_srli_epi16(*src, count);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_srli_epi32(*src, count);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_srli_epi64(*src, count);
    }

    static SIMD_INLINE T reduce_add(const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            return _mm512_reduce_add_ps(*src);
        else if constexpr (std::is_same_v<T, double>)
            return _mm512_reduce_add_pd(*src);
        else if constexpr (sizeof(T) == 4)
            return static_cast<T>(_mm512_reduce_add_epi32(*src));
        else if constexpr (sizeof(T) == 8)
            return static_cast<T>(_mm512_reduce_add_epi64(*src));
        else
        {
            constexpr size_t elems = 64 / sizeof(T);
            alignas(64) T tmp[elems];
            _mm512_store_si512(reinterpret_cast<void*>(tmp), *src);
            T sum = 0;
            for (size_t i = 0; i < elems; ++i) sum += tmp[i];
            return sum;
        }
    }

    static SIMD_INLINE T reduce_min(const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            return _mm512_reduce_min_ps(*src);
        else if constexpr (std::is_same_v<T, double>)
            return _mm512_reduce_min_pd(*src);
        else if constexpr (std::is_same_v<T, int32_t>)
            return _mm512_reduce_min_epi32(*src);
        else if constexpr (std::is_same_v<T, uint32_t>)
            return static_cast<T>(_mm512_reduce_min_epu32(*src));
        else if constexpr (std::is_same_v<T, int64_t>)
            return _mm512_reduce_min_epi64(*src);
        else
        {
            constexpr size_t elems = 64 / sizeof(T);
            alignas(64) T tmp[elems];
            _mm512_store_si512(reinterpret_cast<void*>(tmp), *src);
            T m = tmp[0];
            for (size_t i = 1; i < elems; ++i) m = tmp[i] < m ? tmp[i] : m;
            return m;
        }
    }

    static SIMD_INLINE T reduce_max(const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            return _mm512_reduce_max_ps(*src);
        else if constexpr (std::is_same_v<T, double>)
            return _mm512_reduce_max_pd(*src);
        else if constexpr (std::is_same_v<T, int32_t>)
            return _mm512_reduce_max_epi32(*src);
        else if constexpr (std::is_same_v<T, uint32_t>)
            return static_cast<T>(_mm512_reduce_max_epu32(*src));
        else if constexpr (std::is_same_v<T, int64_t>)
            return _mm512_reduce_max_epi64(*src);
        else
        {
            constexpr size_t elems = 64 / sizeof(T);
            alignas(64) T tmp[elems];
            _mm512_store_si512(reinterpret_cast<void*>(tmp), *src);
            T m = tmp[0];
            for (size_t i = 1; i < elems; ++i) m = tmp[i] > m ? tmp[i] : m;
            return m;
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE T dot(const register_t* a, const register_t* b)
    {
        register_t prod;
        mul(&prod, a, b);
        return reduce_add(&prod);
    }

    static SIMD_INLINE void interleave_lo(register_t* dst, const register_t* a,
                                          const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_unpacklo_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_unpacklo_pd(*a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_unpacklo_epi8(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_unpacklo_epi16(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_unpacklo_epi32(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_unpacklo_epi64(*a, *b);
    }

    static SIMD_INLINE void interleave_hi(register_t* dst, const register_t* a,
                                          const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_unpackhi_ps(*a, *b);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_unpackhi_pd(*a, *b);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_unpackhi_epi8(*a, *b);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_unpackhi_epi16(*a, *b);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_unpackhi_epi32(*a, *b);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_unpackhi_epi64(*a, *b);
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void ternary_logic(register_t* dst, const register_t* a,
                                          const register_t* b, const register_t* c, int imm)
    {
        *dst = _mm512_ternarylogic_epi32(*a, *b, *c, imm);
    }

    template <typename U = T, std::enable_if_t<(sizeof(U) == 4), int> = 0>
    static SIMD_INLINE void compress(register_t* dst, const register_t* src, __mmask16 mask)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_maskz_compress_ps(mask, *src);
        else
            *dst = _mm512_maskz_compress_epi32(mask, *src);
    }

    template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
    static SIMD_INLINE void compress(register_t* dst, const register_t* src, __mmask8 mask)
    {
        if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_maskz_compress_pd(mask, *src);
        else
            *dst = _mm512_maskz_compress_epi64(mask, *src);
    }

    template <typename U = T, std::enable_if_t<(sizeof(U) == 4), int> = 0>
    static SIMD_INLINE void expand(register_t* dst, const register_t* src, __mmask16 mask)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_maskz_expand_ps(mask, *src);
        else
            *dst = _mm512_maskz_expand_epi32(mask, *src);
    }

    template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
    static SIMD_INLINE void expand(register_t* dst, const register_t* src, __mmask8 mask)
    {
        if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_maskz_expand_pd(mask, *src);
        else
            *dst = _mm512_maskz_expand_epi64(mask, *src);
    }

    static SIMD_INLINE void permute(register_t* dst, const register_t* src, const __m512i* idx)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_permutexvar_ps(*idx, *src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_permutexvar_pd(*idx, *src);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_permutexvar_epi32(*idx, *src);
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_permutexvar_epi64(*idx, *src);
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_permutexvar_epi16(*idx, *src);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_permutexvar_epi8(*idx, *src);
    }
};

} // namespace vector_simd::detail

#endif

#endif // LIB_SIMD_IMPL_AVX512_VECTOR_OPS_HPP_z512v
