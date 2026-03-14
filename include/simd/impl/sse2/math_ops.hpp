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

namespace sse2_math_detail
{

SIMD_INLINE __m128 exp_ps(__m128 x)
{
    static const __m128 exp_hi = _mm_set1_ps(88.3762626647949f);
    static const __m128 exp_lo = _mm_set1_ps(-88.3762626647949f);
    static const __m128 log2ef = _mm_set1_ps(1.44269504088896341f);
    static const __m128 c1     = _mm_set1_ps(0.693359375f);
    static const __m128 c2     = _mm_set1_ps(-2.12194440e-4f);
    static const __m128 p0     = _mm_set1_ps(1.9875691500e-4f);
    static const __m128 p1     = _mm_set1_ps(1.3981999507e-3f);
    static const __m128 p2     = _mm_set1_ps(8.3334519073e-3f);
    static const __m128 p3     = _mm_set1_ps(4.1665795894e-2f);
    static const __m128 p4     = _mm_set1_ps(1.6666665459e-1f);
    static const __m128 p5     = _mm_set1_ps(5.0000001201e-1f);
    static const __m128 one    = _mm_set1_ps(1.0f);
    static const __m128 half   = _mm_set1_ps(0.5f);

    x = _mm_min_ps(x, exp_hi);
    x = _mm_max_ps(x, exp_lo);

    __m128 z = _mm_add_ps(_mm_mul_ps(x, log2ef), half);
    __m128i n = _mm_cvttps_epi32(z);
    __m128  fn = _mm_cvtepi32_ps(n);

    __m128 r = _mm_sub_ps(x, _mm_mul_ps(fn, c1));
    r = _mm_sub_ps(r, _mm_mul_ps(fn, c2));

    __m128 y = p0;
    y = _mm_add_ps(_mm_mul_ps(y, r), p1);
    y = _mm_add_ps(_mm_mul_ps(y, r), p2);
    y = _mm_add_ps(_mm_mul_ps(y, r), p3);
    y = _mm_add_ps(_mm_mul_ps(y, r), p4);
    y = _mm_add_ps(_mm_mul_ps(y, r), p5);
    y = _mm_add_ps(_mm_mul_ps(y, r), one);
    y = _mm_add_ps(_mm_mul_ps(y, r), one);

    __m128i pow2n = _mm_slli_epi32(_mm_add_epi32(n, _mm_set1_epi32(0x7f)), 23);
    return _mm_mul_ps(y, _mm_castsi128_ps(pow2n));
}

SIMD_INLINE __m128 log_ps(__m128 x)
{
    static const __m128 min_norm = _mm_set1_ps(1.17549435e-38f);
    static const __m128 nan_mask = _mm_set1_ps(-0.0f);
    static const __m128 one      = _mm_set1_ps(1.0f);
    static const __m128 half     = _mm_set1_ps(0.5f);
    static const __m128 sqrthf   = _mm_set1_ps(0.707106781186547524f);
    static const __m128 ln2      = _mm_set1_ps(0.693147180559945f);
    static const __m128 ln2_hi   = _mm_set1_ps(0.693359375f);
    static const __m128 ln2_lo   = _mm_set1_ps(-2.12194440e-4f);
    static const __m128 p0       = _mm_set1_ps(7.0376836292e-2f);
    static const __m128 p1       = _mm_set1_ps(-1.1514610310e-1f);
    static const __m128 p2       = _mm_set1_ps(1.1676998740e-1f);
    static const __m128 p3       = _mm_set1_ps(-1.2420140846e-1f);
    static const __m128 p4       = _mm_set1_ps(1.4249322787e-1f);
    static const __m128 p5       = _mm_set1_ps(-1.6668057665e-1f);
    static const __m128 p6       = _mm_set1_ps(2.0000714765e-1f);
    static const __m128 p7       = _mm_set1_ps(-2.4999993993e-1f);
    static const __m128 p8       = _mm_set1_ps(3.3333331174e-1f);

    __m128 invalid = _mm_cmple_ps(x, _mm_setzero_ps());
    x = _mm_max_ps(x, min_norm);

    __m128i emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
    x = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(~0x7f800000)));
    x = _mm_or_ps(x, half);

    emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
    __m128 e = _mm_cvtepi32_ps(emm0);
    e = _mm_add_ps(e, one);

    __m128 mask = _mm_cmplt_ps(x, sqrthf);
    __m128 tmp  = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);

    __m128 z = _mm_mul_ps(x, x);
    __m128 y = p0;
    y = _mm_add_ps(_mm_mul_ps(y, x), p1);
    y = _mm_add_ps(_mm_mul_ps(y, x), p2);
    y = _mm_add_ps(_mm_mul_ps(y, x), p3);
    y = _mm_add_ps(_mm_mul_ps(y, x), p4);
    y = _mm_add_ps(_mm_mul_ps(y, x), p5);
    y = _mm_add_ps(_mm_mul_ps(y, x), p6);
    y = _mm_add_ps(_mm_mul_ps(y, x), p7);
    y = _mm_add_ps(_mm_mul_ps(y, x), p8);
    y = _mm_mul_ps(y, _mm_mul_ps(x, z));

    __m128 tmp2 = _mm_mul_ps(e, ln2_hi);
    y = _mm_add_ps(y, tmp2);
    tmp2 = _mm_mul_ps(e, ln2_lo);
    y = _mm_sub_ps(y, _mm_mul_ps(z, half));
    x = _mm_add_ps(x, y);
    x = _mm_add_ps(x, _mm_mul_ps(e, ln2));
    x = _mm_or_ps(x, invalid);
    return x;
}

SIMD_INLINE void sincos_ps(__m128 x, __m128* s, __m128* c)
{
    static const __m128 dp1       = _mm_set1_ps(-0.78515625f);
    static const __m128 dp2       = _mm_set1_ps(-2.4187564849853515625e-4f);
    static const __m128 dp3       = _mm_set1_ps(-3.77489497744594108e-8f);
    static const __m128 fopi      = _mm_set1_ps(1.2732395447351628f);
    static const __m128 thr_f     = _mm_set1_ps(8388608.0f);
    static const __m128 sign_mask = _mm_set1_ps(-0.0f);
    static const __m128 one       = _mm_set1_ps(1.0f);
    static const __m128 half      = _mm_set1_ps(0.5f);
    static const __m128 sc_p0     = _mm_set1_ps(-1.9515295891e-4f);
    static const __m128 sc_p1     = _mm_set1_ps(8.3321608736e-3f);
    static const __m128 sc_p2     = _mm_set1_ps(-1.6666654611e-1f);
    static const __m128 cc_p0     = _mm_set1_ps(2.443315711809948e-5f);
    static const __m128 cc_p1     = _mm_set1_ps(-1.388731625493765e-3f);
    static const __m128 cc_p2     = _mm_set1_ps(4.166664568298827e-2f);

    __m128 xmm1, xmm2, xmm3, sign_bit_sin, y;
    __m128i emm0, emm2, emm4;

    sign_bit_sin = _mm_and_ps(x, sign_mask);
    x = _mm_andnot_ps(sign_mask, x);

    y = _mm_mul_ps(x, fopi);
    emm2 = _mm_cvttps_epi32(y);
    emm2 = _mm_add_epi32(emm2, _mm_set1_epi32(1));
    emm2 = _mm_and_si128(emm2, _mm_set1_epi32(~1));
    y = _mm_cvtepi32_ps(emm2);

    emm4 = emm2;
    emm0 = _mm_and_si128(emm2, _mm_set1_epi32(4));
    emm0 = _mm_slli_epi32(emm0, 29);
    emm2 = _mm_and_si128(emm2, _mm_set1_epi32(2));
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

    __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);
    __m128 poly_mask = _mm_castsi128_ps(emm2);

    xmm1 = _mm_mul_ps(y, dp1);
    xmm2 = _mm_mul_ps(y, dp2);
    xmm3 = _mm_mul_ps(y, dp3);
    x = _mm_add_ps(x, _mm_add_ps(xmm1, _mm_add_ps(xmm2, xmm3)));

    emm4 = _mm_sub_epi32(emm4, _mm_set1_epi32(2));
    emm4 = _mm_andnot_si128(emm4, _mm_set1_epi32(4));
    emm4 = _mm_slli_epi32(emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps(emm4);
    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    __m128 z = _mm_mul_ps(x, x);
    __m128 sy = sc_p0;
    sy = _mm_add_ps(_mm_mul_ps(sy, z), sc_p1);
    sy = _mm_add_ps(_mm_mul_ps(sy, z), sc_p2);
    sy = _mm_mul_ps(_mm_mul_ps(sy, z), z);
    sy = _mm_sub_ps(sy, _mm_mul_ps(z, half));
    sy = _mm_add_ps(sy, one);

    __m128 cy = cc_p0;
    cy = _mm_add_ps(_mm_mul_ps(cy, z), cc_p1);
    cy = _mm_add_ps(_mm_mul_ps(cy, z), cc_p2);
    cy = _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(cy, z), z), _mm_mul_ps(z, half));
    cy = _mm_add_ps(cy, one);
    cy = _mm_mul_ps(cy, x);

    xmm1 = _mm_andnot_ps(poly_mask, cy);
    xmm2 = _mm_and_ps(poly_mask, sy);
    xmm3 = _mm_or_ps(xmm1, xmm2);

    xmm1 = _mm_andnot_ps(poly_mask, sy);
    xmm2 = _mm_and_ps(poly_mask, cy);
    *c = _mm_or_ps(xmm1, xmm2);
    *s = _mm_xor_ps(xmm3, sign_bit_sin);
    *c = _mm_xor_ps(*c, sign_bit_cos);
}

SIMD_INLINE __m128 sin_ps(__m128 x)
{
    __m128 s, c;
    sincos_ps(x, &s, &c);
    return s;
}

SIMD_INLINE __m128 cos_ps(__m128 x)
{
    __m128 s, c;
    sincos_ps(x, &s, &c);
    return c;
}

SIMD_INLINE __m128 tan_ps(__m128 x)
{
    __m128 s, c;
    sincos_ps(x, &s, &c);
    return _mm_div_ps(s, c);
}

} // namespace sse2_math_detail

template <typename T, size_t N>
struct math_ops<T, N, sse2_tag>
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
            *dst = sse2_math_detail::sin_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::sin(tmp[0]);
            tmp[1] = std::sin(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void cos(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = sse2_math_detail::cos_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::cos(tmp[0]);
            tmp[1] = std::cos(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void tan(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = sse2_math_detail::tan_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::tan(tmp[0]);
            tmp[1] = std::tan(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void exp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = sse2_math_detail::exp_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::exp(tmp[0]);
            tmp[1] = std::exp(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void log(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = sse2_math_detail::log_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::log(tmp[0]);
            tmp[1] = std::log(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rsqrt(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m128 approx = _mm_rsqrt_ps(*src);
            __m128 half   = _mm_set1_ps(0.5f);
            __m128 three  = _mm_set1_ps(3.0f);
            *dst = _mm_mul_ps(_mm_mul_ps(approx, half),
                              _mm_sub_ps(three, _mm_mul_ps(*src, _mm_mul_ps(approx, approx))));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = 1.0 / std::sqrt(tmp[0]);
            tmp[1] = 1.0 / std::sqrt(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rcp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m128 approx  = _mm_rcp_ps(*src);
            __m128 two     = _mm_set1_ps(2.0f);
            *dst = _mm_sub_ps(_mm_mul_ps(two, approx),
                              _mm_mul_ps(_mm_mul_ps(approx, approx), *src));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = 1.0 / tmp[0];
            tmp[1] = 1.0 / tmp[1];
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void floor(register_t* dst, const register_t* src)
    {
#if SIMD_SSE4_1
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_floor_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_floor_pd(*src);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            __m128i  n = _mm_cvttps_epi32(*src);
            __m128   fn = _mm_cvtepi32_ps(n);
            __m128   neg = _mm_cmplt_ps(fn, *src);
            *dst = _mm_sub_ps(fn, _mm_and_ps(neg, _mm_set1_ps(1.0f)));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::floor(tmp[0]);
            tmp[1] = std::floor(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
#endif
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void ceil(register_t* dst, const register_t* src)
    {
#if SIMD_SSE4_1
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_ceil_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_ceil_pd(*src);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            __m128i  n = _mm_cvttps_epi32(*src);
            __m128   fn = _mm_cvtepi32_ps(n);
            __m128   pos = _mm_cmpgt_ps(fn, *src);
            *dst = _mm_add_ps(fn, _mm_and_ps(pos, _mm_set1_ps(1.0f)));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::ceil(tmp[0]);
            tmp[1] = std::ceil(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
#endif
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void round(register_t* dst, const register_t* src)
    {
#if SIMD_SSE4_1
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_round_ps(*src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_round_pd(*src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            static const __m128 magic = _mm_set1_ps(12582912.0f);
            __m128 round = _mm_sub_ps(_mm_add_ps(*src, magic), magic);
            *dst = round;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::round(tmp[0]);
            tmp[1] = std::round(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
#endif
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void trunc(register_t* dst, const register_t* src)
    {
#if SIMD_SSE4_1
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_round_ps(*src, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_round_pd(*src, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cvtepi32_ps(_mm_cvttps_epi32(*src));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            tmp[0] = std::trunc(tmp[0]);
            tmp[1] = std::trunc(tmp[1]);
            *dst = _mm_load_pd(tmp);
        }
#endif
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
