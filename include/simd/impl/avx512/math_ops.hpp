#ifndef LIB_SIMD_IMPL_AVX512_MATH_OPS_HPP_z512m
#define LIB_SIMD_IMPL_AVX512_MATH_OPS_HPP_z512m

#include "simd/arch/detection.hpp"
#include "simd/impl/avx512/vector_ops.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#if SIMD_ARCH_X86 && SIMD_HAS_AVX512F

#include <cmath>
#include <immintrin.h>
#include <type_traits>

namespace vector_simd::detail
{

namespace avx512_math_detail
{

SIMD_INLINE __m512 exp_ps(__m512 x)
{
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c1     = _mm512_set1_ps(0.693359375f);
    const __m512 c2     = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 p0     = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 p1     = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 p2     = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 p3     = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 p4     = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 p5     = _mm512_set1_ps(5.0000001201e-1f);
    const __m512 one    = _mm512_set1_ps(1.0f);
    const __m512 half   = _mm512_set1_ps(0.5f);

    x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    __m512 z = _mm512_add_ps(_mm512_mul_ps(x, log2ef), half);
    __m512i n = _mm512_cvttps_epi32(z);
    __m512 fn = _mm512_cvtepi32_ps(n);

    __m512 r = _mm512_sub_ps(x, _mm512_mul_ps(fn, c1));
    r = _mm512_sub_ps(r, _mm512_mul_ps(fn, c2));

    __m512 y = _mm512_fmadd_ps(p0, r, p1);
    y = _mm512_fmadd_ps(y, r, p2);
    y = _mm512_fmadd_ps(y, r, p3);
    y = _mm512_fmadd_ps(y, r, p4);
    y = _mm512_fmadd_ps(y, r, p5);
    y = _mm512_fmadd_ps(y, r, one);
    y = _mm512_fmadd_ps(y, r, one);

    n = _mm512_add_epi32(n, _mm512_set1_epi32(0x7f));
    __m512i pow2n = _mm512_slli_epi32(n, 23);
    return _mm512_mul_ps(y, _mm512_castsi512_ps(pow2n));
}

SIMD_INLINE __m512 log_ps(__m512 x)
{
    const __m512 min_norm = _mm512_set1_ps(1.17549435e-38f);
    const __m512 one      = _mm512_set1_ps(1.0f);
    const __m512 half     = _mm512_set1_ps(0.5f);
    const __m512 sqrthf   = _mm512_set1_ps(0.707106781186547524f);
    const __m512 ln2      = _mm512_set1_ps(0.693147180559945f);
    const __m512 ln2_hi   = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo   = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 p0       = _mm512_set1_ps(7.0376836292e-2f);
    const __m512 p1       = _mm512_set1_ps(-1.1514610310e-1f);
    const __m512 p2       = _mm512_set1_ps(1.1676998740e-1f);
    const __m512 p3       = _mm512_set1_ps(-1.2420140846e-1f);
    const __m512 p4       = _mm512_set1_ps(1.4249322787e-1f);
    const __m512 p5       = _mm512_set1_ps(-1.6668057665e-1f);
    const __m512 p6       = _mm512_set1_ps(2.0000714765e-1f);
    const __m512 p7       = _mm512_set1_ps(-2.4999993993e-1f);
    const __m512 p8       = _mm512_set1_ps(3.3333331174e-1f);

    __mmask16 invalid = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS);
    x = _mm512_max_ps(x, min_norm);

    __m512i emm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
    x = _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(~0x7f800000)));
    x = _mm512_castsi512_ps(
        _mm512_or_si512(_mm512_castps_si512(x), _mm512_castps_si512(half)));

    emm0 = _mm512_sub_epi32(emm0, _mm512_set1_epi32(0x7f));
    __m512 e = _mm512_cvtepi32_ps(emm0);
    e = _mm512_add_ps(e, one);

    __mmask16 lt_mask = _mm512_cmp_ps_mask(x, sqrthf, _CMP_LT_OS);
    __m512 tmp = _mm512_maskz_mov_ps(lt_mask, x);
    x = _mm512_sub_ps(x, one);
    e = _mm512_mask_sub_ps(e, lt_mask, e, one);
    x = _mm512_add_ps(x, tmp);

    __m512 z = _mm512_mul_ps(x, x);
    __m512 y = _mm512_fmadd_ps(p0, x, p1);
    y = _mm512_fmadd_ps(y, x, p2);
    y = _mm512_fmadd_ps(y, x, p3);
    y = _mm512_fmadd_ps(y, x, p4);
    y = _mm512_fmadd_ps(y, x, p5);
    y = _mm512_fmadd_ps(y, x, p6);
    y = _mm512_fmadd_ps(y, x, p7);
    y = _mm512_fmadd_ps(y, x, p8);
    y = _mm512_mul_ps(y, _mm512_mul_ps(x, z));

    y = _mm512_fmadd_ps(e, ln2_hi, y);
    y = _mm512_fnmadd_ps(z, half, y);
    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, ln2, x);
    x = _mm512_mask_or_ps(x, invalid, x, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FC00000)));
    return x;
}

SIMD_INLINE void sincos_ps(__m512 x, __m512* s, __m512* c)
{
    const __m512 dp1       = _mm512_set1_ps(-0.78515625f);
    const __m512 dp2       = _mm512_set1_ps(-2.4187564849853515625e-4f);
    const __m512 dp3       = _mm512_set1_ps(-3.77489497744594108e-8f);
    const __m512 fopi      = _mm512_set1_ps(1.2732395447351628f);
    const __m512 one       = _mm512_set1_ps(1.0f);
    const __m512 half      = _mm512_set1_ps(0.5f);
    const __m512 sc_p0     = _mm512_set1_ps(-1.9515295891e-4f);
    const __m512 sc_p1     = _mm512_set1_ps(8.3321608736e-3f);
    const __m512 sc_p2     = _mm512_set1_ps(-1.6666654611e-1f);
    const __m512 cc_p0     = _mm512_set1_ps(2.443315711809948e-5f);
    const __m512 cc_p1     = _mm512_set1_ps(-1.388731625493765e-3f);
    const __m512 cc_p2     = _mm512_set1_ps(4.166664568298827e-2f);

    __m512 sign_bit_sin =
        _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x80000000)));
    x = _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7FFFFFFF)));

    __m512 y = _mm512_mul_ps(x, fopi);
    __m512i emm2 = _mm512_cvttps_epi32(y);
    emm2 = _mm512_add_epi32(emm2, _mm512_set1_epi32(1));
    emm2 = _mm512_and_si512(emm2, _mm512_set1_epi32(~1));
    y = _mm512_cvtepi32_ps(emm2);

    __m512i emm4 = emm2;
    __m512i emm0 = _mm512_and_si512(emm2, _mm512_set1_epi32(4));
    emm0 = _mm512_slli_epi32(emm0, 29);
    emm2 = _mm512_and_si512(emm2, _mm512_set1_epi32(2));
    __mmask16 poly_mask = _mm512_cmpeq_epi32_mask(emm2, _mm512_setzero_si512());

    __m512 swap_sign_bit_sin = _mm512_castsi512_ps(emm0);

    x = _mm512_fmadd_ps(y, dp1, x);
    x = _mm512_fmadd_ps(y, dp2, x);
    x = _mm512_fmadd_ps(y, dp3, x);

    emm4 = _mm512_sub_epi32(emm4, _mm512_set1_epi32(2));
    emm4 = _mm512_andnot_si512(emm4, _mm512_set1_epi32(4));
    emm4 = _mm512_slli_epi32(emm4, 29);
    __m512 sign_bit_cos = _mm512_castsi512_ps(emm4);
    sign_bit_sin = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_castps_si512(sign_bit_sin), _mm512_castps_si512(swap_sign_bit_sin)));

    __m512 z = _mm512_mul_ps(x, x);

    __m512 sy = _mm512_fmadd_ps(sc_p0, z, sc_p1);
    sy = _mm512_fmadd_ps(sy, z, sc_p2);
    sy = _mm512_mul_ps(_mm512_mul_ps(sy, z), z);
    sy = _mm512_fnmadd_ps(z, half, sy);
    sy = _mm512_add_ps(sy, one);

    __m512 cy = _mm512_fmadd_ps(cc_p0, z, cc_p1);
    cy = _mm512_fmadd_ps(cy, z, cc_p2);
    cy = _mm512_fmadd_ps(_mm512_mul_ps(cy, z), z, _mm512_fnmadd_ps(z, half, one));
    cy = _mm512_mul_ps(cy, x);

    *s = _mm512_mask_blend_ps(poly_mask, cy, sy);
    *c = _mm512_mask_blend_ps(poly_mask, sy, cy);
    *s = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_castps_si512(*s), _mm512_castps_si512(sign_bit_sin)));
    *c = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_castps_si512(*c), _mm512_castps_si512(sign_bit_cos)));
}

SIMD_INLINE __m512 sin_ps(__m512 x)
{
    __m512 s, c;
    sincos_ps(x, &s, &c);
    return s;
}

SIMD_INLINE __m512 cos_ps(__m512 x)
{
    __m512 s, c;
    sincos_ps(x, &s, &c);
    return c;
}

SIMD_INLINE __m512 tan_ps(__m512 x)
{
    __m512 s, c;
    sincos_ps(x, &s, &c);
    return _mm512_div_ps(s, c);
}

} // namespace avx512_math_detail

template <typename T, size_t N>
struct math_ops<T, N, avx512_tag>
{
    using register_t = typename register_type<T, avx512_tag>::type;

    static SIMD_INLINE void abs(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_abs_ps(*src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_abs_pd(*src);
        else if constexpr (std::is_same_v<T, int8_t>)
            *dst = _mm512_abs_epi8(*src);
        else if constexpr (std::is_same_v<T, int16_t>)
            *dst = _mm512_abs_epi16(*src);
        else if constexpr (std::is_same_v<T, int32_t>)
            *dst = _mm512_abs_epi32(*src);
        else if constexpr (std::is_same_v<T, int64_t>)
            *dst = _mm512_abs_epi64(*src);
        else
            *dst = *src;
    }

    static SIMD_INLINE void sqrt(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_sqrt_ps(*src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_sqrt_pd(*src);
        else
        {
            constexpr size_t elems = 64 / sizeof(T);
            alignas(64) T arr[elems];
            _mm512_store_si512(reinterpret_cast<void*>(arr), *src);
            for (size_t i = 0; i < elems; ++i)
                arr[i] = static_cast<T>(std::sqrt(static_cast<double>(arr[i])));
            *dst = _mm512_load_si512(reinterpret_cast<const void*>(arr));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void sin(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = avx512_math_detail::sin_ps(*src);
        else
        {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, *src);
            for (int i = 0; i < 8; ++i) tmp[i] = std::sin(tmp[i]);
            *dst = _mm512_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void cos(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = avx512_math_detail::cos_ps(*src);
        else
        {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, *src);
            for (int i = 0; i < 8; ++i) tmp[i] = std::cos(tmp[i]);
            *dst = _mm512_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void tan(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = avx512_math_detail::tan_ps(*src);
        else
        {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, *src);
            for (int i = 0; i < 8; ++i) tmp[i] = std::tan(tmp[i]);
            *dst = _mm512_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void exp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = avx512_math_detail::exp_ps(*src);
        else
        {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, *src);
            for (int i = 0; i < 8; ++i) tmp[i] = std::exp(tmp[i]);
            *dst = _mm512_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void log(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = avx512_math_detail::log_ps(*src);
        else
        {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, *src);
            for (int i = 0; i < 8; ++i) tmp[i] = std::log(tmp[i]);
            *dst = _mm512_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rsqrt(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m512 approx = _mm512_rsqrt14_ps(*src);
            __m512 h = _mm512_set1_ps(0.5f);
            __m512 three = _mm512_set1_ps(3.0f);
            *dst = _mm512_mul_ps(_mm512_mul_ps(approx, h),
                                 _mm512_fnmadd_ps(*src, _mm512_mul_ps(approx, approx),
                                                  three));
        }
        else
        {
            __m512d approx = _mm512_rsqrt14_pd(*src);
            __m512d h = _mm512_set1_pd(0.5);
            __m512d three = _mm512_set1_pd(3.0);
            *dst = _mm512_mul_pd(_mm512_mul_pd(approx, h),
                                 _mm512_fnmadd_pd(*src, _mm512_mul_pd(approx, approx),
                                                  three));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rcp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m512 approx = _mm512_rcp14_ps(*src);
            __m512 two = _mm512_set1_ps(2.0f);
            *dst = _mm512_mul_ps(approx, _mm512_fnmadd_ps(approx, *src, two));
        }
        else
        {
            __m512d approx = _mm512_rcp14_pd(*src);
            __m512d two = _mm512_set1_pd(2.0);
            *dst = _mm512_mul_pd(approx, _mm512_fnmadd_pd(approx, *src, two));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void floor(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_floor_ps(*src);
        else
            *dst = _mm512_floor_pd(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void ceil(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_ceil_ps(*src);
        else
            *dst = _mm512_ceil_pd(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void round(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_roundscale_ps(*src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        else
            *dst = _mm512_roundscale_pd(*src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void trunc(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_roundscale_ps(*src, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        else
            *dst = _mm512_roundscale_pd(*src, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    static SIMD_INLINE void fmadd(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_fmadd_ps(*a, *b, *c);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_fmadd_pd(*a, *b, *c);
        else
        {
            register_t tmp;
            vector_ops<T, N, avx512_tag>::mul(&tmp, a, b);
            vector_ops<T, N, avx512_tag>::add(dst, &tmp, c);
        }
    }

    static SIMD_INLINE void fmsub(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_fmsub_ps(*a, *b, *c);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_fmsub_pd(*a, *b, *c);
        else
        {
            register_t tmp;
            vector_ops<T, N, avx512_tag>::mul(&tmp, a, b);
            vector_ops<T, N, avx512_tag>::sub(dst, &tmp, c);
        }
    }
};

} // namespace vector_simd::detail

#endif

#endif // LIB_SIMD_IMPL_AVX512_MATH_OPS_HPP_z512m
