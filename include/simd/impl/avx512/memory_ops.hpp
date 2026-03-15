#ifndef LIB_SIMD_IMPL_AVX512_MEMORY_OPS_HPP_z512e
#define LIB_SIMD_IMPL_AVX512_MEMORY_OPS_HPP_z512e

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
struct memory_ops<T, N, avx512_tag>
{
    using register_t = typename register_type<T, avx512_tag>::type;
    using mask_t = typename mask_register_type<T, avx512_tag>::type;
    static constexpr size_t elements = 64 / sizeof(T);

    static SIMD_INLINE void load_aligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_load_ps(src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_load_pd(src);
        else
            *dst = _mm512_load_si512(reinterpret_cast<const void*>(src));
    }

    static SIMD_INLINE void load_unaligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_loadu_ps(src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_loadu_pd(src);
        else
            *dst = _mm512_loadu_si512(reinterpret_cast<const void*>(src));
    }

    static SIMD_INLINE void store_aligned(T* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm512_store_ps(dst, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm512_store_pd(dst, *src);
        else
            _mm512_store_si512(reinterpret_cast<void*>(dst), *src);
    }

    static SIMD_INLINE void store_unaligned(T* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm512_storeu_ps(dst, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm512_storeu_pd(dst, *src);
        else
            _mm512_storeu_si512(reinterpret_cast<void*>(dst), *src);
    }

    static SIMD_INLINE void store_nt(T* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm512_stream_ps(dst, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm512_stream_pd(dst, *src);
        else
            _mm512_stream_si512(reinterpret_cast<void*>(dst), *src);
    }

    static SIMD_INLINE void load_masked(register_t* dst, const T* src, mask_t mask)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_maskz_load_ps(mask, src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_maskz_load_pd(mask, src);
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_maskz_load_epi32(mask, reinterpret_cast<const void*>(src));
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_maskz_load_epi64(mask, reinterpret_cast<const void*>(src));
    }

    static SIMD_INLINE void store_masked(T* dst, const register_t* src, mask_t mask)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm512_mask_store_ps(dst, mask, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm512_mask_store_pd(dst, mask, *src);
        else if constexpr (sizeof(T) == 4)
            _mm512_mask_store_epi32(reinterpret_cast<void*>(dst), mask, *src);
        else if constexpr (sizeof(T) == 8)
            _mm512_mask_store_epi64(reinterpret_cast<void*>(dst), mask, *src);
    }

    static SIMD_INLINE void broadcast(register_t* dst, const T* val)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_set1_ps(*val);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_set1_pd(*val);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm512_set1_epi8(static_cast<char>(*val));
        else if constexpr (sizeof(T) == 2)
            *dst = _mm512_set1_epi16(static_cast<short>(*val));
        else if constexpr (sizeof(T) == 4)
            *dst = _mm512_set1_epi32(static_cast<int>(*val));
        else if constexpr (sizeof(T) == 8)
            *dst = _mm512_set1_epi64(static_cast<long long>(*val));
    }

    static SIMD_INLINE void set_zero(register_t* dst)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm512_setzero_ps();
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm512_setzero_pd();
        else
            *dst = _mm512_setzero_si512();
    }

    static SIMD_INLINE void prefetch(const T* addr)
    {
        _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
    }

    static SIMD_INLINE void prefetch_nt(const T* addr)
    {
        _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA);
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, float>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m512i* vindex)
    {
        *dst = _mm512_i32gather_ps(*vindex, base, sizeof(float));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, double>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m256i* vindex)
    {
        *dst = _mm512_i32gather_pd(*vindex, base, sizeof(double));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, int32_t>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m512i* vindex)
    {
        *dst = _mm512_i32gather_epi32(*vindex, base, sizeof(int32_t));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, int64_t>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m256i* vindex)
    {
        *dst = _mm512_i32gather_epi64(*vindex, reinterpret_cast<const void*>(base),
                                      sizeof(int64_t));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, float>, int> = 0>
    static SIMD_INLINE void scatter(T* base, const register_t* src, const __m512i* vindex)
    {
        _mm512_i32scatter_ps(base, *vindex, *src, sizeof(float));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, double>, int> = 0>
    static SIMD_INLINE void scatter(T* base, const register_t* src, const __m256i* vindex)
    {
        _mm512_i32scatter_pd(base, *vindex, *src, sizeof(double));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, int32_t>, int> = 0>
    static SIMD_INLINE void scatter(T* base, const register_t* src, const __m512i* vindex)
    {
        _mm512_i32scatter_epi32(base, *vindex, *src, sizeof(int32_t));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, int64_t>, int> = 0>
    static SIMD_INLINE void scatter(T* base, const register_t* src, const __m256i* vindex)
    {
        _mm512_i32scatter_epi64(base, *vindex, *src, sizeof(int64_t));
    }
};

} // namespace vector_simd::detail

#endif

#endif // LIB_SIMD_IMPL_AVX512_MEMORY_OPS_HPP_z512e
