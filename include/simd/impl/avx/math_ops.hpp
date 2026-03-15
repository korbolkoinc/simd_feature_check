#ifndef LIB_SIMD_IMPL_AVX_MATH_OPS_HPP_avx256m
#define LIB_SIMD_IMPL_AVX_MATH_OPS_HPP_avx256m

#include "simd/arch/detection.hpp"
#include "simd/impl/avx/vector_ops.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#if SIMD_ARCH_X86 && SIMD_HAS_AVX

#include <cmath>
#include <immintrin.h>
#include <type_traits>

namespace vector_simd::detail
{

namespace avx_math_detail
{

SIMD_INLINE __m256 exp_ps(__m256 x)
{
    static const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    static const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    static const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    static const __m256 c1     = _mm256_set1_ps(0.693359375f);
    static const __m256 c2     = _mm256_set1_ps(-2.12194440e-4f);
    static const __m256 p0     = _mm256_set1_ps(1.9875691500e-4f);
    static const __m256 p1     = _mm256_set1_ps(1.3981999507e-3f);
    static const __m256 p2     = _mm256_set1_ps(8.3334519073e-3f);
    static const __m256 p3     = _mm256_set1_ps(4.1665795894e-2f);
    static const __m256 p4     = _mm256_set1_ps(1.6666665459e-1f);
    static const __m256 p5     = _mm256_set1_ps(5.0000001201e-1f);
    static const __m256 one    = _mm256_set1_ps(1.0f);
    static const __m256 half   = _mm256_set1_ps(0.5f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 z = _mm256_add_ps(_mm256_mul_ps(x, log2ef), half);
    __m256i n = _mm256_cvttps_epi32(z);
    __m256 fn = _mm256_cvtepi32_ps(n);

    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(fn, c1));
    r = _mm256_sub_ps(r, _mm256_mul_ps(fn, c2));

    __m256 y = p0;
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p1);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p2);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p3);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p4);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p5);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), one);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), one);

    n = _mm256_add_epi32(n, _mm256_set1_epi32(0x7f));
    __m256i pow2n = _mm256_slli_epi32(n, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
}

SIMD_INLINE __m256 log_ps(__m256 x)
{
    static const __m256 min_norm = _mm256_set1_ps(1.17549435e-38f);
    static const __m256 one      = _mm256_set1_ps(1.0f);
    static const __m256 half     = _mm256_set1_ps(0.5f);
    static const __m256 sqrthf   = _mm256_set1_ps(0.707106781186547524f);
    static const __m256 ln2_hi   = _mm256_set1_ps(0.693359375f);
    static const __m256 ln2_lo   = _mm256_set1_ps(-2.12194440e-4f);
    static const __m256 p0       = _mm256_set1_ps(7.0376836292e-2f);
    static const __m256 p1       = _mm256_set1_ps(-1.1514610310e-1f);
    static const __m256 p2       = _mm256_set1_ps(1.1676998740e-1f);
    static const __m256 p3       = _mm256_set1_ps(-1.2420140846e-1f);
    static const __m256 p4       = _mm256_set1_ps(1.4249322787e-1f);
    static const __m256 p5       = _mm256_set1_ps(-1.6668057665e-1f);
    static const __m256 p6       = _mm256_set1_ps(2.0000714765e-1f);
    static const __m256 p7       = _mm256_set1_ps(-2.4999993993e-1f);
    static const __m256 p8       = _mm256_set1_ps(3.3333331174e-1f);

    __m256 invalid = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
    x = _mm256_max_ps(x, min_norm);

    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(~0x7f800000)));
    x = _mm256_or_ps(x, half);

    emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
    __m256 e = _mm256_cvtepi32_ps(emm0);
    e = _mm256_add_ps(e, one);

    __m256 mask = _mm256_cmp_ps(x, sqrthf, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = p0;
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p1);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p2);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p3);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p4);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p5);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p6);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p7);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p8);
    y = _mm256_mul_ps(y, _mm256_mul_ps(x, z));

    y = _mm256_add_ps(y, _mm256_mul_ps(e, ln2_lo));
    y = _mm256_sub_ps(y, _mm256_mul_ps(z, half));
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, _mm256_mul_ps(e, ln2_hi));
    x = _mm256_or_ps(x, invalid);
    return x;
}

SIMD_INLINE void sincos_ps(__m256 x, __m256* s, __m256* c)
{
    static const __m256 dp1       = _mm256_set1_ps(-0.78515625f);
    static const __m256 dp2       = _mm256_set1_ps(-2.4187564849853515625e-4f);
    static const __m256 dp3       = _mm256_set1_ps(-3.77489497744594108e-8f);
    static const __m256 fopi      = _mm256_set1_ps(1.2732395447351628f);
    static const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    static const __m256 one       = _mm256_set1_ps(1.0f);
    static const __m256 half      = _mm256_set1_ps(0.5f);
    static const __m256 sc_p0     = _mm256_set1_ps(-1.9515295891e-4f);
    static const __m256 sc_p1     = _mm256_set1_ps(8.3321608736e-3f);
    static const __m256 sc_p2     = _mm256_set1_ps(-1.6666654611e-1f);
    static const __m256 cc_p0     = _mm256_set1_ps(2.443315711809948e-5f);
    static const __m256 cc_p1     = _mm256_set1_ps(-1.388731625493765e-3f);
    static const __m256 cc_p2     = _mm256_set1_ps(4.166664568298827e-2f);

    __m256 sign_bit_sin = _mm256_and_ps(x, sign_mask);
    x = _mm256_andnot_ps(sign_mask, x);

    __m256 y = _mm256_mul_ps(x, fopi);
    __m256i emm2 = _mm256_cvttps_epi32(y);
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);

    __m256i emm4 = emm2;
    __m256i emm0 = _mm256_and_si256(emm2, _mm256_set1_epi32(4));
    emm0 = _mm256_slli_epi32(emm0, 29);
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(2));
    emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());

    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(emm0);
    __m256 poly_mask = _mm256_castsi256_ps(emm2);

    x = _mm256_add_ps(x, _mm256_add_ps(_mm256_mul_ps(y, dp1),
                      _mm256_add_ps(_mm256_mul_ps(y, dp2), _mm256_mul_ps(y, dp3))));

    emm4 = _mm256_sub_epi32(emm4, _mm256_set1_epi32(2));
    emm4 = _mm256_andnot_si256(emm4, _mm256_set1_epi32(4));
    emm4 = _mm256_slli_epi32(emm4, 29);
    __m256 sign_bit_cos = _mm256_castsi256_ps(emm4);
    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    __m256 z = _mm256_mul_ps(x, x);

    __m256 sy = cc_p0;
    sy = _mm256_add_ps(_mm256_mul_ps(sy, z), cc_p1);
    sy = _mm256_add_ps(_mm256_mul_ps(sy, z), cc_p2);
    sy = _mm256_mul_ps(_mm256_mul_ps(sy, z), z);
    sy = _mm256_sub_ps(sy, _mm256_mul_ps(z, half));
    sy = _mm256_add_ps(sy, one);

    __m256 cy = sc_p0;
    cy = _mm256_add_ps(_mm256_mul_ps(cy, z), sc_p1);
    cy = _mm256_add_ps(_mm256_mul_ps(cy, z), sc_p2);
    cy = _mm256_mul_ps(_mm256_mul_ps(cy, z), z);
    cy = _mm256_add_ps(_mm256_mul_ps(cy, x), x);

    __m256 xmm1 = _mm256_andnot_ps(poly_mask, cy);
    __m256 xmm2 = _mm256_and_ps(poly_mask, sy);
    __m256 xmm3 = _mm256_or_ps(xmm1, xmm2);

    xmm1 = _mm256_andnot_ps(poly_mask, sy);
    xmm2 = _mm256_and_ps(poly_mask, cy);
    *c = _mm256_or_ps(xmm1, xmm2);
    *s = _mm256_xor_ps(xmm3, sign_bit_sin);
    *c = _mm256_xor_ps(*c, sign_bit_cos);
}

SIMD_INLINE __m256 sin_ps(__m256 x)
{
    __m256 s, c;
    sincos_ps(x, &s, &c);
    return s;
}

SIMD_INLINE __m256 cos_ps(__m256 x)
{
    __m256 s, c;
    sincos_ps(x, &s, &c);
    return c;
}

SIMD_INLINE __m256 tan_ps(__m256 x)
{
    __m256 s, c;
    sincos_ps(x, &s, &c);
    return _mm256_div_ps(s, c);
}

} // namespace avx_math_detail

template <typename T, size_t N>
struct math_ops<T, N, avx_tag>
{
    using register_t = typename register_type<T, avx_tag>::type;

    static SIMD_INLINE void abs(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            *dst = _mm256_andnot_ps(sign_mask, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            const __m256d sign_mask = _mm256_set1_pd(-0.0);
            *dst = _mm256_andnot_pd(sign_mask, *src);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_abs_epi8(*src);
#else
            alignas(32) int8_t tmp[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            for (int i = 0; i < 32; ++i) tmp[i] = tmp[i] < 0 ? -tmp[i] : tmp[i];
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_abs_epi16(*src);
#else
            __m128i lo = _mm256_extractf128_si256(*src, 0), hi = _mm256_extractf128_si256(*src, 1);
            __m128i slo = _mm_srai_epi16(lo, 15), shi = _mm_srai_epi16(hi, 15);
            lo = _mm_sub_epi16(_mm_xor_si128(lo, slo), slo);
            hi = _mm_sub_epi16(_mm_xor_si128(hi, shi), shi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_abs_epi32(*src);
#else
            __m128i lo = _mm256_extractf128_si256(*src, 0), hi = _mm256_extractf128_si256(*src, 1);
            __m128i slo = _mm_srai_epi32(lo, 31), shi = _mm_srai_epi32(hi, 31);
            lo = _mm_sub_epi32(_mm_xor_si128(lo, slo), slo);
            hi = _mm_sub_epi32(_mm_xor_si128(hi, shi), shi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
#endif
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
            *dst = _mm256_sqrt_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_sqrt_pd(*src);
        }
        else
        {
            alignas(32) T arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(arr), *src);
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
                arr[i] = static_cast<T>(std::sqrt(static_cast<double>(arr[i])));
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(arr));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void sin(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = avx_math_detail::sin_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            for (int i = 0; i < 4; ++i) tmp[i] = std::sin(tmp[i]);
            *dst = _mm256_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void cos(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = avx_math_detail::cos_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            for (int i = 0; i < 4; ++i) tmp[i] = std::cos(tmp[i]);
            *dst = _mm256_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void tan(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = avx_math_detail::tan_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            for (int i = 0; i < 4; ++i) tmp[i] = std::tan(tmp[i]);
            *dst = _mm256_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void exp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = avx_math_detail::exp_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            for (int i = 0; i < 4; ++i) tmp[i] = std::exp(tmp[i]);
            *dst = _mm256_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void log(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = avx_math_detail::log_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            for (int i = 0; i < 4; ++i) tmp[i] = std::log(tmp[i]);
            *dst = _mm256_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rsqrt(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m256 approx = _mm256_rsqrt_ps(*src);
            __m256 h = _mm256_set1_ps(0.5f);
            __m256 three = _mm256_set1_ps(3.0f);
            *dst = _mm256_mul_ps(_mm256_mul_ps(approx, h),
                                 _mm256_sub_ps(three, _mm256_mul_ps(*src, _mm256_mul_ps(approx, approx))));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(*src));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rcp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m256 approx = _mm256_rcp_ps(*src);
            __m256 two = _mm256_set1_ps(2.0f);
            *dst = _mm256_sub_ps(_mm256_mul_ps(two, approx),
                                 _mm256_mul_ps(_mm256_mul_ps(approx, approx), *src));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_div_pd(_mm256_set1_pd(1.0), *src);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void floor(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_floor_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_floor_pd(*src);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void ceil(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_ceil_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_ceil_pd(*src);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void round(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_round_ps(*src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_round_pd(*src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void trunc(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_round_ps(*src, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_round_pd(*src, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
    }

    static SIMD_INLINE void fmadd(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
#if SIMD_FMA
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm256_fmadd_ps(*a, *b, *c);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm256_fmadd_pd(*a, *b, *c);
        else
        {
            register_t tmp;
            vector_ops<T, N, avx_tag>::mul(&tmp, a, b);
            vector_ops<T, N, avx_tag>::add(dst, &tmp, c);
        }
#else
        register_t tmp;
        vector_ops<T, N, avx_tag>::mul(&tmp, a, b);
        vector_ops<T, N, avx_tag>::add(dst, &tmp, c);
#endif
    }

    static SIMD_INLINE void fmsub(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
#if SIMD_FMA
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm256_fmsub_ps(*a, *b, *c);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm256_fmsub_pd(*a, *b, *c);
        else
        {
            register_t tmp;
            vector_ops<T, N, avx_tag>::mul(&tmp, a, b);
            vector_ops<T, N, avx_tag>::sub(dst, &tmp, c);
        }
#else
        register_t tmp;
        vector_ops<T, N, avx_tag>::mul(&tmp, a, b);
        vector_ops<T, N, avx_tag>::sub(dst, &tmp, c);
#endif
    }
};

template <typename T, size_t N>
struct math_ops<T, N, avx2_tag> : math_ops<T, N, avx_tag>
{
};

} // namespace vector_simd::detail

#endif

#endif // LIB_SIMD_IMPL_AVX_MATH_OPS_HPP_avx256m
