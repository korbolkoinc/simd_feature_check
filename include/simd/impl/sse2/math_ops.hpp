#ifndef LIB_SIMD_IMPL_SSE2_MATH_OPS_HPP_ffv57r
#define LIB_SIMD_IMPL_SSE2_MATH_OPS_HPP_ffv57r

#include "simd/arch/detection.hpp"
#include "simd/impl/sse2/vector_ops.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2

#include <cmath>
#include <emmintrin.h>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct math_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void abs(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const __m128 sign_mask = _mm_set1_ps(-0.0f);
            *dst = _mm_andnot_ps(sign_mask, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            const __m128d sign_mask = _mm_set1_pd(-0.0);
            *dst = _mm_andnot_pd(sign_mask, *src);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSSE3
            *dst = _mm_abs_epi8(*src);
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            for (int i = 0; i < 16; ++i)
            {
                tmp[i] = std::abs(tmp[i]);
            }
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_SSSE3
            *dst = _mm_abs_epi16(*src);
#else
            __m128i sign = _mm_srai_epi16(*src, 15);
            __m128i inv = _mm_xor_si128(*src, sign);
            *dst = _mm_sub_epi16(inv, sign);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSSE3
            *dst = _mm_abs_epi32(*src);
#else
            // Manual abs using SSE2 instructions
            __m128i sign = _mm_srai_epi32(*src, 31);
            __m128i inv = _mm_xor_si128(*src, sign);
            *dst = _mm_sub_epi32(inv, sign);
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::abs(tmp[i]);
            }
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
        else
        {
            *dst = *src;
        }
    }

    static SIMD_INLINE void sqrt(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_sqrt_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_sqrt_pd(*src);
        }
        else
        {
            alignas(16) T src_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(src_arr), *src);

            alignas(16) T result[16 / sizeof(T)];
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                result[i] = static_cast<T>(std::sqrt(static_cast<double>(src_arr[i])));
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(result));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void sin(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::sin(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::sin(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void cos(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::cos(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::cos(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void tan(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::tan(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::tan(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void exp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::exp(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::exp(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void log(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::log(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::log(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    static SIMD_INLINE void fmadd(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
#if SIMD_FMA
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_fmadd_ps(*a, *b, *c);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_fmadd_pd(*a, *b, *c);
        }
        else
        {
            register_t tmp;
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                mul(&tmp, a, b);
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                add(dst, &tmp, c);
        }
#else
        register_t tmp;
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::mul(&tmp,
                                                                                             a, b);
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::add(dst,
                                                                                             &tmp,
                                                                                             c);
#endif
    }

    static SIMD_INLINE void fmsub(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
#if SIMD_FMA
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_fmsub_ps(*a, *b, *c);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_fmsub_pd(*a, *b, *c);
        }
        else
        {
            register_t tmp;
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                mul(&tmp, a, b);
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                sub(dst, &tmp, c);
        }
#else
        // No FMA instructions available, use separate multiply and subtract
        register_t tmp;
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::mul(&tmp,
                                                                                             a, b);
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::sub(dst,
                                                                                             &tmp,
                                                                                             c);
#endif
    }
};

} // namespace vector_simd::detail

#endif

#endif // End of include guard: LIB_SIMD_IMPL_SSE2_MATH_OPS_HPP_ffv57r
