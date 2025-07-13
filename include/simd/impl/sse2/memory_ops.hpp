#ifndef LIB_SIMD_IMPL_SSE2_Memory_OPS_HPP_gjaxql
#define LIB_SIMD_IMPL_SSE2_Memory_OPS_HPP_gjaxql

#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include <simd/feature_check.hpp>

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2
#include <algorithm>
#include <emmintrin.h>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct memory_ops<T, N,
                  std::enable_if_t<std::is_same_v<current_isa, sse2_tag> ||
                                   std::is_base_of_v<sse2_tag, current_isa>>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void load(register_t* dst, const T* src) { load_unaligned(dst, src); }

    static SIMD_INLINE void load_aligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_load_ps(src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_load_pd(src);
        }
        else
        {
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
        }
    }

    static SIMD_INLINE void load_unaligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_loadu_ps(src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_loadu_pd(src);
        }
        else
        {
            *dst = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        }
    }

    static SIMD_INLINE void store(const register_t* src, T* dst) { store_unaligned(src, dst); }

    static SIMD_INLINE void store_aligned(const register_t* src, T* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(dst, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(dst, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), *src);
        }
    }

    static SIMD_INLINE void store_unaligned(const register_t* src, T* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_storeu_ps(dst, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_storeu_pd(dst, *src);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), *src);
        }
    }

    static SIMD_INLINE void prefetch(const T* ptr, int hint)
    {
#if SIMD_SSE3
        switch (hint)
        {
            case _MM_HINT_T0:
                _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
                break;
            case _MM_HINT_T1:
                _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1);
                break;
            case _MM_HINT_T2:
                _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T2);
                break;
            case _MM_HINT_NTA:
                _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_NTA);
                break;
            default:
                _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
                break;
        }
#else
        (void)ptr;
        (void)hint;
#endif
    }

    template <typename IndexT>
    static SIMD_INLINE void gather(register_t* dst, const T* base,
                                   const typename register_type<IndexT, sse2_tag>::type* indices)
    {
        alignas(16) IndexT idx_arr[16 / sizeof(IndexT)];
        _mm_store_si128(reinterpret_cast<__m128i*>(idx_arr), *indices);

        alignas(16) T result[16 / sizeof(T)];
        for (size_t i = 0; i < 16 / sizeof(T); ++i)
        {
            if (i < N)
            {
                result[i] = base[idx_arr[i]];
            }
            else
            {
                result[i] = 0;
            }
        }

        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_load_ps(result);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_load_pd(result);
        }
        else
        {
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(result));
        }
    }

    template <typename IndexT>
    static SIMD_INLINE void scatter(const register_t* src, T* base,
                                    const typename register_type<IndexT, sse2_tag>::type* indices)
    {
        alignas(16) T src_arr[16 / sizeof(T)];
        alignas(16) IndexT idx_arr[16 / sizeof(IndexT)];

        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(src_arr, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(src_arr, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(src_arr), *src);
        }

        _mm_store_si128(reinterpret_cast<__m128i*>(idx_arr), *indices);

        for (size_t i = 0; i < std::min(N, 16 / sizeof(T)); ++i)
        {
            base[idx_arr[i]] = src_arr[i];
        }
    }
};
} // namespace vector_simd::detail

#endif // SIMD_ARCH_X86 && SIMD_HAS_SSE2

#endif // End of include guard: LIB_SIMD_IMPL_SSE2_MEMORY_OPS_HPP_gjaxql