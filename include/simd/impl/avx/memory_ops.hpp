#ifndef LIB_SIMD_IMPL_AVX_MEMORY_OPS_HPP_avx256
#define LIB_SIMD_IMPL_AVX_MEMORY_OPS_HPP_avx256

#include "simd/arch/detection.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#if SIMD_ARCH_X86 && SIMD_HAS_AVX

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct memory_ops<T, N, avx_tag>
{
    using register_t = typename register_type<T, avx_tag>::type;
    static constexpr size_t elements = 32 / sizeof(T);

    static SIMD_INLINE void load_aligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm256_load_ps(src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm256_load_pd(src);
        else
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
    }

    static SIMD_INLINE void load_unaligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm256_loadu_ps(src);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm256_loadu_pd(src);
        else
            *dst = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    }

    static SIMD_INLINE void store_aligned(T* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm256_store_ps(dst, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm256_store_pd(dst, *src);
        else
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst), *src);
    }

    static SIMD_INLINE void store_unaligned(T* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm256_storeu_ps(dst, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm256_storeu_pd(dst, *src);
        else
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), *src);
    }

    static SIMD_INLINE void store_nt(T* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
            _mm256_stream_ps(dst, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm256_stream_pd(dst, *src);
        else
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), *src);
    }

    static SIMD_INLINE void broadcast(register_t* dst, const T* val)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm256_broadcast_ss(val);
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm256_broadcast_sd(val);
        else if constexpr (sizeof(T) == 1)
            *dst = _mm256_set1_epi8(static_cast<char>(*val));
        else if constexpr (sizeof(T) == 2)
            *dst = _mm256_set1_epi16(static_cast<short>(*val));
        else if constexpr (sizeof(T) == 4)
            *dst = _mm256_set1_epi32(static_cast<int>(*val));
        else if constexpr (sizeof(T) == 8)
            *dst = _mm256_set1_epi64x(static_cast<long long>(*val));
    }

    static SIMD_INLINE void set_zero(register_t* dst)
    {
        if constexpr (std::is_same_v<T, float>)
            *dst = _mm256_setzero_ps();
        else if constexpr (std::is_same_v<T, double>)
            *dst = _mm256_setzero_pd();
        else
            *dst = _mm256_setzero_si256();
    }

    static SIMD_INLINE void prefetch(const T* addr)
    {
        _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
    }

    static SIMD_INLINE void prefetch_nt(const T* addr)
    {
        _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA);
    }

#if SIMD_AVX2
    template <typename U = T, std::enable_if_t<std::is_same_v<U, float>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m256i* vindex)
    {
        *dst = _mm256_i32gather_ps(base, *vindex, sizeof(float));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, double>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m128i* vindex)
    {
        *dst = _mm256_i32gather_pd(base, *vindex, sizeof(double));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, int32_t>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m256i* vindex)
    {
        *dst = _mm256_i32gather_epi32(base, *vindex, sizeof(int32_t));
    }

    template <typename U = T, std::enable_if_t<std::is_same_v<U, int64_t>, int> = 0>
    static SIMD_INLINE void gather(register_t* dst, const T* base, const __m128i* vindex)
    {
        *dst = _mm256_i32gather_epi64(reinterpret_cast<const long long*>(base), *vindex,
                                      sizeof(int64_t));
    }
#endif

    template <typename U = T, std::enable_if_t<(sizeof(U) >= 4), int> = 0>
    static SIMD_INLINE void scatter(T* base, const register_t* src, const int* indices)
    {
        alignas(32) T tmp[elements];
        if constexpr (std::is_same_v<T, float>)
            _mm256_store_ps(tmp, *src);
        else if constexpr (std::is_same_v<T, double>)
            _mm256_store_pd(tmp, *src);
        else
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
        for (size_t i = 0; i < elements; ++i)
            base[indices[i]] = tmp[i];
    }
};

template <typename T, size_t N>
struct memory_ops<T, N, avx2_tag> : memory_ops<T, N, avx_tag>
{
};

} // namespace vector_simd::detail

#endif

#endif // LIB_SIMD_IMPL_AVX_MEMORY_OPS_HPP_avx256
