#ifndef SIMD_HPP_al9nn6
#define SIMD_HPP_al9nn6

#include <array>
#include <bit>
#include <cmath>
#include <simd/arch/detection.hpp>
#include <simd/arch/tags.hpp>
#include <simd/common.hpp>
#include <simd/core/concepts.hpp>
#include <simd/core/types.hpp>
#include <simd/feature_check.hpp>
#include <simd/mask/mask.hpp>
#include <simd/operations/forward_decl.hpp>
#include <simd/registers/types.hpp>
#include <simd/vector/base.hpp>
#include <simd/vector/vector.hpp>

namespace vector_simd
{

// Fixed-size vector types
template <size_t N>
using float_v = Vector<float, N>;
template <size_t N>
using double_v = Vector<double, N>;
template <size_t N>
using int8_v = Vector<int8_t, N>;
template <size_t N>
using uint8_v = Vector<uint8_t, N>;
template <size_t N>
using int16_v = Vector<int16_t, N>;
template <size_t N>
using uint16_v = Vector<uint16_t, N>;
template <size_t N>
using int32_v = Vector<int32_t, N>;
template <size_t N>
using uint32_v = Vector<uint32_t, N>;
template <size_t N>
using int64_v = Vector<int64_t, N>;
template <size_t N>
using uint64_v = Vector<uint64_t, N>;

using float_vn = float_v<detail::native_width<float>::value>;
using double_vn = double_v<detail::native_width<double>::value>;
using int8_vn = int8_v<detail::native_width<int8_t>::value>;
using uint8_vn = uint8_v<detail::native_width<uint8_t>::value>;
using int16_vn = int16_v<detail::native_width<int16_t>::value>;
using uint16_vn = uint16_v<detail::native_width<uint16_t>::value>;
using int32_vn = int32_v<detail::native_width<int32_t>::value>;
using uint32_vn = uint32_v<detail::native_width<uint32_t>::value>;
using int64_vn = int64_v<detail::native_width<int64_t>::value>;
using uint64_vn = uint64_v<detail::native_width<uint64_t>::value>;

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2
namespace detail {

#if SIMD_AVX

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
            __m128i zero = _mm_setzero_si128();
            __m128i ones_lo = _mm_cmpeq_epi32(zero, zero);
            __m128i ones_hi = _mm_cmpeq_epi32(zero, zero);
            *mask = _mm256_insertf128_si256(_mm256_castsi128_si256(ones_lo), ones_hi, 1);
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
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_and_si128(a_lo, b_lo);
            __m128i res_hi = _mm_and_si128(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
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
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_xor_si128(a_lo, b_lo);
            __m128i res_hi = _mm_xor_si128(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
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
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);

            __m128i ones_lo = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
            __m128i ones_hi = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());

            __m128i res_lo = _mm_xor_si128(a_lo, ones_lo);
            __m128i res_hi = _mm_xor_si128(a_hi, ones_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
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
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_cmpeq_epi8(a_lo, b_lo);
            __m128i res_hi = _mm_cmpeq_epi8(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpeq_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_cmpeq_epi16(a_lo, b_lo);
            __m128i res_hi = _mm_cmpeq_epi16(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                           std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpeq_epi32(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_cmpeq_epi32(a_lo, b_lo);
            __m128i res_hi = _mm_cmpeq_epi32(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
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
            *dst = _mm256_cmpgt_epi8(*b, *a);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_cmplt_epi8(a_lo, b_lo);
            __m128i res_hi = _mm_cmplt_epi8(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi16(*b, *a);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_cmplt_epi16(a_lo, b_lo);
            __m128i res_hi = _mm_cmplt_epi16(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_cmpgt_epi32(*b, *a);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_cmplt_epi32(a_lo, b_lo);
            __m128i res_hi = _mm_cmplt_epi32(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else
        {
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            alignas(32) T result[32 / sizeof(T)];
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                result[i] = (a_arr[i] < b_arr[i]) ? ~T(0) : T(0);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(result));
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
            cmp_lt(dst, a, b);
            mask_register_t eq_mask;
            cmp_eq(&eq_mask, a, b);
            logical_or(dst, dst, &eq_mask);
        }
    }

    static SIMD_INLINE void cmp_gt(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        cmp_lt(dst, b, a);
    }

    static SIMD_INLINE void cmp_ge(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        cmp_le(dst, b, a);
    }
};

#endif // SIMD_AVX

#if SIMD_ARCH_X86 && SIMD_AVX512

template <typename T, size_t N>
struct vector_ops<T, N,
                  std::enable_if_t<simd::FeatureDetector<simd::Feature::AVX512F>::compile_time>>
{
    using register_t = typename register_type<T, avx512_tag>::type;
    using mask_t = typename mask_register_type<T, avx512_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm512_set1_ps(value);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm512_set1_pd(value);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm512_set1_epi8(static_cast<char>(value));
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm512_set1_epi16(static_cast<short>(value));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm512_set1_epi32(static_cast<int>(value));
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm512_set1_epi64(static_cast<long long>(value));
        }
    }

    // Additional AVX-512 operations would be implemented here...
    // For brevity, not all operations are shown
};

#endif // SIMD_AVX512

template <typename T, size_t N, typename Func>
SIMD_INLINE auto dispatch_simd_function(Func&& sse2_impl, Func&& avx_impl, Func&& avx2_impl,
                                        Func&& avx512_impl, Func&& fallback_impl)
{
    if constexpr (simd::FeatureDetector<simd::Feature::AVX512F>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX512F>())
        {
            return std::forward<Func>(avx512_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::AVX2>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX2>())
        {
            return std::forward<Func>(avx2_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::AVX>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX>())
        {
            return std::forward<Func>(avx_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::SSE2>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::SSE2>())
        {
            return std::forward<Func>(sse2_impl);
        }
    }

    return std::forward<Func>(fallback_impl);
}

} // namespace detail
} // namespace vector_simd

#endif
#endif /* End of include guard: SIMD_HPP_al9nn6 */
