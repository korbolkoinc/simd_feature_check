#ifndef LIB_SIMD_IMPL_AVX_MASK_OPS_HPP_avx256m
#define LIB_SIMD_IMPL_AVX_MASK_OPS_HPP_avx256m

#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include <simd/feature_check.hpp>

#if SIMD_ARCH_X86 && SIMD_HAS_AVX

#include <bit>
#include <immintrin.h>
#include <type_traits>

namespace vector_simd::detail
{
template <typename T, size_t N>
struct mask_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::AVX>::compile_time>>
{
    using mask_register_t = typename mask_register_type<T, avx_tag>::type;
    using register_t = typename register_type<T, avx_tag>::type;

    static SIMD_INLINE void set_true(mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *mask = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *mask = _mm256_cmp_pd(_mm256_setzero_pd(), _mm256_setzero_pd(), _CMP_EQ_OQ);
        }
        else
        {
#if SIMD_AVX2
            *mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
#else
            // AVX1 fallback: use two 128-bit operations
            __m128i zero = _mm_setzero_si128();
            __m128i ones = _mm_cmpeq_epi32(zero, zero);
            *mask = _mm256_insertf128_si256(_mm256_castsi128_si256(ones), ones, 1);
#endif
        }
    }

    static SIMD_INLINE void set_false(mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *mask = _mm256_setzero_ps();
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *mask = _mm256_setzero_pd();
        }
        else
        {
            *mask = _mm256_setzero_si256();
        }
    }

    static SIMD_INLINE void load(mask_register_t* dst, const bool* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) int32_t tmp[8] = {0};
            for (size_t i = 0; i < 8; ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm256_load_ps(reinterpret_cast<const float*>(tmp));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) int64_t tmp[4] = {0};
            for (size_t i = 0; i < 4; ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm256_load_pd(reinterpret_cast<const double*>(tmp));
        }
        else
        {
            alignas(32) int tmp[32 / sizeof(T)] = {0};
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
        }
    }

    static SIMD_INLINE void store(const mask_register_t* src, bool* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *src);
            for (size_t i = 0; i < 8; ++i)
            {
                dst[i] = reinterpret_cast<const int32_t*>(tmp)[i] != 0;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            for (size_t i = 0; i < 4; ++i)
            {
                dst[i] = reinterpret_cast<const int64_t*>(tmp)[i] != 0;
            }
        }
        else
        {
            alignas(32) T tmp[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                dst[i] = tmp[i] != 0;
            }
        }
    }

    static SIMD_INLINE bool extract(const mask_register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *src);
            return reinterpret_cast<const int32_t*>(tmp)[index % 8] != 0;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            return reinterpret_cast<const int64_t*>(tmp)[index % 4] != 0;
        }
        else
        {
            alignas(32) T tmp[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % (32 / sizeof(T))] != 0;
        }
    }

    static SIMD_INLINE void logical_and(mask_register_t* dst, const mask_register_t* a,
                                        const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_and_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_and_pd(*a, *b);
        }
        else
        {
#if SIMD_AVX2
            *dst = _mm256_and_si256(*a, *b);
#else
            // AVX1 fallback: split into two 128-bit operations
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_and_si128(a_lo, b_lo);
            __m128i result_hi = _mm_and_si128(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void logical_or(mask_register_t* dst, const mask_register_t* a,
                                       const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_or_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_or_pd(*a, *b);
        }
        else
        {
#if SIMD_AVX2
            *dst = _mm256_or_si256(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_or_si128(a_lo, b_lo);
            __m128i result_hi = _mm_or_si128(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void logical_xor(mask_register_t* dst, const mask_register_t* a,
                                        const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_xor_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_xor_pd(*a, *b);
        }
        else
        {
#if SIMD_AVX2
            *dst = _mm256_xor_si256(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_xor_si128(a_lo, b_lo);
            __m128i result_hi = _mm_xor_si128(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void logical_not(mask_register_t* dst, const mask_register_t* a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m256 ones = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ);
            *dst = _mm256_xor_ps(*a, ones);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m256d ones = _mm256_cmp_pd(_mm256_setzero_pd(), _mm256_setzero_pd(), _CMP_EQ_OQ);
            *dst = _mm256_xor_pd(*a, ones);
        }
        else
        {
#if SIMD_AVX2
            __m256i ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
            *dst = _mm256_xor_si256(*a, ones);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i ones_lo = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
            __m128i ones_hi = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
            __m128i result_lo = _mm_xor_si128(a_lo, ones_lo);
            __m128i result_hi = _mm_xor_si128(a_hi, ones_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void cmp_eq(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_cmp_ps(*a, *b, _CMP_EQ_OQ);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_cmp_pd(*a, *b, _CMP_EQ_OQ);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpeq_epi8(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmpeq_epi8(a_lo, b_lo);
            __m128i result_hi = _mm_cmpeq_epi8(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpeq_epi16(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmpeq_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_cmpeq_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpeq_epi32(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmpeq_epi32(a_lo, b_lo);
            __m128i result_hi = _mm_cmpeq_epi32(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpeq_epi64(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            // SSE2 doesn't have cmpeq_epi64, so we fallback to generic approach
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            alignas(32) int64_t tmp[32 / sizeof(T)] = {0};
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] == b_arr[i] ? -1 : 0;
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
#endif
        }
    }

    static SIMD_INLINE void cmp_neq(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        cmp_eq(dst, a, b);
        logical_not(dst, dst);
    }

    static SIMD_INLINE void cmp_lt(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_cmp_ps(*a, *b, _CMP_LT_OQ);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_cmp_pd(*a, *b, _CMP_LT_OQ);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi8(*b, *a); // b > a == a < b
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmplt_epi8(a_lo, b_lo);
            __m128i result_hi = _mm_cmplt_epi8(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi16(*b, *a); // b > a == a < b
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmplt_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_cmplt_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi32(*b, *a); // b > a == a < b
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmplt_epi32(a_lo, b_lo);
            __m128i result_hi = _mm_cmplt_epi32(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                           std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                           std::is_same_v<T, uint64_t>)
        {
            // Generic fallback for unsigned types and 64-bit types
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            alignas(32) int64_t tmp[32 / sizeof(T)] = {0};
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] < b_arr[i] ? -1 : 0;
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
        }
    }

    static SIMD_INLINE void cmp_le(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_cmp_ps(*a, *b, _CMP_LE_OQ);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_cmp_pd(*a, *b, _CMP_LE_OQ);
        }
        else
        {
            mask_register_t lt, eq;
            cmp_lt(&lt, a, b);
            cmp_eq(&eq, a, b);
            logical_or(dst, &lt, &eq);
        }
    }

    static SIMD_INLINE void cmp_gt(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_cmp_ps(*a, *b, _CMP_GT_OQ);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_cmp_pd(*a, *b, _CMP_GT_OQ);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi8(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmpgt_epi8(a_lo, b_lo);
            __m128i result_hi = _mm_cmpgt_epi8(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi16(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmpgt_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_cmpgt_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi32(*a, *b);
#else
            // AVX1 fallback
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_cmpgt_epi32(a_lo, b_lo);
            __m128i result_hi = _mm_cmpgt_epi32(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                           std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                           std::is_same_v<T, uint64_t>)
        {
            // Generic fallback for unsigned types and 64-bit types
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            alignas(32) int64_t tmp[32 / sizeof(T)] = {0};
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] > b_arr[i] ? -1 : 0;
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
        }
    }

    static SIMD_INLINE void cmp_ge(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_cmp_ps(*a, *b, _CMP_GE_OQ);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_cmp_pd(*a, *b, _CMP_GE_OQ);
        }
        else
        {
            mask_register_t gt, eq;
            cmp_gt(&gt, a, b);
            cmp_eq(&eq, a, b);
            logical_or(dst, &gt, &eq);
        }
    }

    static SIMD_INLINE uint64_t to_bitmask(const mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return _mm256_movemask_ps(*mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return _mm256_movemask_pd(*mask);
        }
        else
        {
#if SIMD_AVX2
            return _mm256_movemask_epi8(*mask);
#else
            // AVX1 fallback: combine two 128-bit movemask operations
            __m128i lo = _mm256_extractf128_si256(*mask, 0);
            __m128i hi = _mm256_extractf128_si256(*mask, 1);
            uint32_t lo_mask = _mm_movemask_epi8(lo);
            uint32_t hi_mask = _mm_movemask_epi8(hi);
            return static_cast<uint64_t>(lo_mask) | (static_cast<uint64_t>(hi_mask) << 16);
#endif
        }
    }

    static SIMD_INLINE bool any(const mask_register_t* mask) { return to_bitmask(mask) != 0; }

    static SIMD_INLINE bool all(const mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return to_bitmask(mask) == 0xFF; // 8 bits for 8 floats
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return to_bitmask(mask) == 0xF; // 4 bits for 4 doubles
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            return to_bitmask(mask) == 0xFFFFFFFFULL; // 32 bits for 32 bytes
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            constexpr uint64_t full_mask = shift == 1   ? 0xFFFFFFFFULL
                                           : shift == 2 ? 0x55555555ULL
                                           : shift == 4 ? 0x11111111ULL
                                           : shift == 8 ? 0x01010101ULL
                                                        : 0;
            return (to_bitmask(mask) & full_mask) == full_mask;
        }
        else
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            constexpr uint64_t full_mask = shift == 1   ? 0xFFFFFFFFULL
                                           : shift == 2 ? 0x55555555ULL
                                           : shift == 4 ? 0x11111111ULL
                                           : shift == 8 ? 0x01010101ULL
                                                        : 0;
            return (to_bitmask(mask) & full_mask) == full_mask;
        }
    }

    static SIMD_INLINE int count(const mask_register_t* mask)
    {
        uint64_t bits = to_bitmask(mask);
        if constexpr (std::is_same_v<T, float>)
        {
            return std::popcount(bits & 0xFF); // 8 bits for 8 floats
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return std::popcount(bits & 0xF); // 4 bits for 4 doubles
        }
        else
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            if constexpr (shift == 1)
            {
                return std::popcount(bits & 0xFFFFFFFFULL); // All 32 bits for bytes
            }
            else
            {
                int count = 0;
                constexpr int elements = 32 / sizeof(T);
                for (int i = 0; i < elements; ++i)
                {
                    if (bits & (1ULL << (i * shift)))
                    {
                        count++;
                    }
                }
                return count;
            }
        }
    }
};
} // namespace vector_simd::detail

#endif

#endif // End of include guard: LIB_SIMD_IMPL_AVX_MASK_OPS_HPP_avx256m