#ifndef LIB_SIMD_IMPL_SSE2_MASK_OPS_HPP_ymzp0y
#define LIB_SIMD_IMPL_SSE2_MASK_OPS_HPP_ymzp0y

#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include <simd/feature_check.hpp>

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2

#include <bit>
#include <emmintrin.h>
#include <type_traits>

namespace vector_simd::detail
{
template <typename T, size_t N>
struct mask_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using mask_register_t = typename mask_register_type<T, sse2_tag>::type;
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void set_true(mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *mask = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *mask = _mm_cmpeq_pd(_mm_setzero_pd(), _mm_setzero_pd());
        }
        else
        {
            *mask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
        }
    }

    static SIMD_INLINE void set_false(mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *mask = _mm_setzero_ps();
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *mask = _mm_setzero_pd();
        }
        else
        {
            *mask = _mm_setzero_si128();
        }
    }

    static SIMD_INLINE void load(mask_register_t* dst, const bool* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) int32_t tmp[4] = {0};
            for (size_t i = 0; i < 4; ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm_load_ps(reinterpret_cast<const float*>(tmp));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) int64_t tmp[2] = {0};
            for (size_t i = 0; i < 2; ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm_load_pd(reinterpret_cast<const double*>(tmp));
        }
        else
        {
            alignas(16) int tmp[16 / sizeof(T)] = {0};
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
    }

    static SIMD_INLINE void store(const mask_register_t* src, bool* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (size_t i = 0; i < 4; ++i)
            {
                dst[i] = reinterpret_cast<const int32_t*>(tmp)[i] != 0;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (size_t i = 0; i < 2; ++i)
            {
                dst[i] = reinterpret_cast<const int64_t*>(tmp)[i] != 0;
            }
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                dst[i] = tmp[i] != 0;
            }
        }
    }

    static SIMD_INLINE bool extract(const mask_register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            return reinterpret_cast<const int32_t*>(tmp)[index % 4] != 0;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            return reinterpret_cast<const int64_t*>(tmp)[index % 2] != 0;
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % (16 / sizeof(T))] != 0;
        }
    }

    static SIMD_INLINE void logical_and(mask_register_t* dst, const mask_register_t* a,
                                        const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_and_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_and_pd(*a, *b);
        }
        else
        {
            *dst = _mm_and_si128(*a, *b);
        }
    }

    static SIMD_INLINE void logical_or(mask_register_t* dst, const mask_register_t* a,
                                       const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_or_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_or_pd(*a, *b);
        }
        else
        {
            *dst = _mm_or_si128(*a, *b);
        }
    }

    static SIMD_INLINE void logical_xor(mask_register_t* dst, const mask_register_t* a,
                                        const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_xor_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_xor_pd(*a, *b);
        }
        else
        {
            *dst = _mm_xor_si128(*a, *b);
        }
    }

    static SIMD_INLINE void logical_not(mask_register_t* dst, const mask_register_t* a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m128 ones = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
            *dst = _mm_xor_ps(*a, ones);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m128d ones = _mm_cmpeq_pd(_mm_setzero_pd(), _mm_setzero_pd());
            *dst = _mm_xor_pd(*a, ones);
        }
        else
        {
            __m128i ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
            *dst = _mm_xor_si128(*a, ones);
        }
    }

    static SIMD_INLINE void cmp_eq(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmpeq_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmpeq_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_cmpeq_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_cmpeq_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                           std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_cmpeq_epi32(*a, *b);
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
            *dst = _mm_cmplt_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmplt_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            *dst = _mm_cmplt_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_cmplt_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            *dst = _mm_cmplt_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                           std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                           std::is_same_v<T, uint64_t>)
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            alignas(16) int tmp[16 / sizeof(T)] = {0};
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] < b_arr[i] ? -1 : 0;
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
    }

    static SIMD_INLINE void cmp_le(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmple_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmple_pd(*a, *b);
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
            *dst = _mm_cmpgt_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmpgt_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            *dst = _mm_cmpgt_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_cmpgt_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            *dst = _mm_cmpgt_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                           std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                           std::is_same_v<T, uint64_t>)
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            alignas(16) int tmp[16 / sizeof(T)] = {0};
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] > b_arr[i] ? -1 : 0;
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
    }

    static SIMD_INLINE void cmp_ge(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmpge_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmpge_pd(*a, *b);
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
            return _mm_movemask_ps(*mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return _mm_movemask_pd(*mask);
        }
        else
        {
            return _mm_movemask_epi8(*mask);
        }
    }

    static SIMD_INLINE bool any(const mask_register_t* mask) { return to_bitmask(mask) != 0; }

    static SIMD_INLINE bool all(const mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return to_bitmask(mask) == 0xF;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return to_bitmask(mask) == 0x3;
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            return to_bitmask(mask) == 0xFFFF;
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            constexpr uint64_t full_mask = shift == 1   ? 0xFFFF
                                           : shift == 2 ? 0x5555
                                           : shift == 4 ? 0x1111
                                           : shift == 8 ? 0x0101
                                                        : 0;
            return (to_bitmask(mask) & full_mask) == full_mask;
        }
        else
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            constexpr uint64_t full_mask = shift == 1   ? 0xFFFF
                                           : shift == 2 ? 0x5555
                                           : shift == 4 ? 0x1111
                                           : shift == 8 ? 0x0101
                                                        : 0;
            return (to_bitmask(mask) & full_mask) == full_mask;
        }
    }

    static SIMD_INLINE int count(const mask_register_t* mask)
    {
        uint64_t bits = to_bitmask(mask);
        if constexpr (std::is_same_v<T, float>)
        {
            return std::popcount(bits & 0xF);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return std::popcount(bits & 0x3);
        }
        else
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            if constexpr (shift == 1)
            {
                return std::popcount(bits & 0xFFFF);
            }
            else
            {
                int count = 0;
                constexpr int elements = 16 / sizeof(T);
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

#endif // End of include guard: LIB_SIMD_IMPL_SSE2_MASK_OPS_HPP_ymzp0y