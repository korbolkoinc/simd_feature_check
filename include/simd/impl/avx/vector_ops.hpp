#ifndef LIB_SIMD_IMPL_AVX_VECTOR_OPS_HPP_avx256
#define LIB_SIMD_IMPL_AVX_VECTOR_OPS_HPP_avx256

#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include <simd/feature_check.hpp>

#if SIMD_ARCH_X86 && SIMD_HAS_AVX
#include <algorithm>
#include <immintrin.h>
#include <type_traits>

namespace vector_simd
{
namespace detail
{

#if SIMD_AVX2
template <size_t I>
static SIMD_INLINE int8_t extract_epi8_helper_avx(const __m256i& src)
{
    static_assert(I < 32, "Index out of range for epi8 extraction");
    if constexpr (I < 16)
    {
        return static_cast<int8_t>(_mm_extract_epi8(_mm256_extracti128_si256(src, 0), I));
    }
    else
    {
        return static_cast<int8_t>(_mm_extract_epi8(_mm256_extracti128_si256(src, 1), I - 16));
    }
}

template <size_t I>
static SIMD_INLINE uint8_t extract_epu8_helper_avx(const __m256i& src)
{
    static_assert(I < 32, "Index out of range for epu8 extraction");
    if constexpr (I < 16)
    {
        return static_cast<uint8_t>(_mm_extract_epi8(_mm256_extracti128_si256(src, 0), I));
    }
    else
    {
        return static_cast<uint8_t>(_mm_extract_epi8(_mm256_extracti128_si256(src, 1), I - 16));
    }
}

template <size_t I>
static SIMD_INLINE int32_t extract_epi32_helper_avx(const __m256i& src)
{
    static_assert(I < 8, "Index out of range for epi32 extraction");
    if constexpr (I < 4)
    {
        return static_cast<int32_t>(_mm_extract_epi32(_mm256_extracti128_si256(src, 0), I));
    }
    else
    {
        return static_cast<int32_t>(_mm_extract_epi32(_mm256_extracti128_si256(src, 1), I - 4));
    }
}

template <size_t I>
static SIMD_INLINE uint32_t extract_epu32_helper_avx(const __m256i& src)
{
    static_assert(I < 8, "Index out of range for epu32 extraction");
    if constexpr (I < 4)
    {
        return static_cast<uint32_t>(_mm_extract_epi32(_mm256_extracti128_si256(src, 0), I));
    }
    else
    {
        return static_cast<uint32_t>(_mm_extract_epi32(_mm256_extracti128_si256(src, 1), I - 4));
    }
}

template <size_t I>
static SIMD_INLINE int64_t extract_epi64_helper_avx(const __m256i& src)
{
    static_assert(I < 4, "Index out of range for epi64 extraction");
    if constexpr (I < 2)
    {
        return static_cast<int64_t>(_mm_extract_epi64(_mm256_extracti128_si256(src, 0), I));
    }
    else
    {
        return static_cast<int64_t>(_mm_extract_epi64(_mm256_extracti128_si256(src, 1), I - 2));
    }
}

template <size_t I>
static SIMD_INLINE uint64_t extract_epu64_helper_avx(const __m256i& src)
{
    static_assert(I < 4, "Index out of range for epu64 extraction");
    if constexpr (I < 2)
    {
        return static_cast<uint64_t>(_mm_extract_epi64(_mm256_extracti128_si256(src, 0), I));
    }
    else
    {
        return static_cast<uint64_t>(_mm_extract_epi64(_mm256_extracti128_si256(src, 1), I - 2));
    }
}

template <size_t I>
static SIMD_INLINE __m256i insert_epi8_helper_avx(__m256i dst, int value)
{
    static_assert(I < 32, "Index out of range for epi8 insertion");
    if constexpr (I < 16)
    {
        __m128i lo = _mm256_extracti128_si256(dst, 0);
        lo = _mm_insert_epi8(lo, value, I);
        return _mm256_inserti128_si256(dst, lo, 0);
    }
    else
    {
        __m128i hi = _mm256_extracti128_si256(dst, 1);
        hi = _mm_insert_epi8(hi, value, I - 16);
        return _mm256_inserti128_si256(dst, hi, 1);
    }
}

template <size_t I>
static SIMD_INLINE __m256i insert_epi32_helper_avx(__m256i dst, int value)
{
    static_assert(I < 8, "Index out of range for epi32 insertion");
    if constexpr (I < 4)
    {
        __m128i lo = _mm256_extracti128_si256(dst, 0);
        lo = _mm_insert_epi32(lo, value, I);
        return _mm256_inserti128_si256(dst, lo, 0);
    }
    else
    {
        __m128i hi = _mm256_extracti128_si256(dst, 1);
        hi = _mm_insert_epi32(hi, value, I - 4);
        return _mm256_inserti128_si256(dst, hi, 1);
    }
}

template <size_t I>
static SIMD_INLINE __m256i insert_epi64_helper_avx(__m256i dst, long long value)
{
    static_assert(I < 4, "Index out of range for epi64 insertion");
    if constexpr (I < 2)
    {
        __m128i lo = _mm256_extracti128_si256(dst, 0);
        lo = _mm_insert_epi64(lo, value, I);
        return _mm256_inserti128_si256(dst, lo, 0);
    }
    else
    {
        __m128i hi = _mm256_extracti128_si256(dst, 1);
        hi = _mm_insert_epi64(hi, value, I - 2);
        return _mm256_inserti128_si256(dst, hi, 1);
    }
}

template <typename T, size_t MaxIndex>
struct runtime_dispatcher
{
    template <template <size_t> class HelperFunc, typename RetType, typename... Args>
    static SIMD_INLINE RetType extract_dispatch(size_t index, Args&&... args)
    {
        return extract_dispatch_impl<HelperFunc, RetType, MaxIndex>(index,
                                                                    std::forward<Args>(args)...);
    }

    template <template <size_t> class HelperFunc, typename RetType, typename... Args>
    static SIMD_INLINE RetType insert_dispatch(size_t index, Args&&... args)
    {
        return insert_dispatch_impl<HelperFunc, RetType, MaxIndex>(index,
                                                                   std::forward<Args>(args)...);
    }

private:
    template <template <size_t> class HelperFunc, typename RetType, size_t Index, typename... Args>
    static SIMD_INLINE RetType extract_dispatch_impl(size_t runtime_index, Args&&... args)
    {
        if constexpr (Index < MaxIndex)
        {
            if (runtime_index == Index)
                return HelperFunc<Index>::call(std::forward<Args>(args)...);
            else
                return extract_dispatch_impl<HelperFunc, RetType, Index + 1>(
                    runtime_index, std::forward<Args>(args)...);
        }
        else
        {
            return RetType{}; // Default value
        }
    }

    template <template <size_t> class HelperFunc, typename RetType, size_t Index, typename... Args>
    static SIMD_INLINE RetType insert_dispatch_impl(size_t runtime_index, Args&&... args)
    {
        if constexpr (Index < MaxIndex)
        {
            if (runtime_index == Index)
                return HelperFunc<Index>::call(std::forward<Args>(args)...);
            else
                return insert_dispatch_impl<HelperFunc, RetType, Index + 1>(
                    runtime_index, std::forward<Args>(args)...);
        }
        else
        {
            return std::get<0>(std::forward_as_tuple(args...));
        }
    }
};

template <size_t I>
struct extract_epi8_wrapper_avx
{
    static SIMD_INLINE int8_t call(const __m256i& src) { return extract_epi8_helper_avx<I>(src); }
};

template <size_t I>
struct extract_epu8_wrapper_avx
{
    static SIMD_INLINE uint8_t call(const __m256i& src) { return extract_epu8_helper_avx<I>(src); }
};

template <size_t I>
struct extract_epi32_wrapper_avx
{
    static SIMD_INLINE int32_t call(const __m256i& src) { return extract_epi32_helper_avx<I>(src); }
};

template <size_t I>
struct extract_epu32_wrapper_avx
{
    static SIMD_INLINE uint32_t call(const __m256i& src)
    {
        return extract_epu32_helper_avx<I>(src);
    }
};

template <size_t I>
struct extract_epi64_wrapper_avx
{
    static SIMD_INLINE int64_t call(const __m256i& src) { return extract_epi64_helper_avx<I>(src); }
};

template <size_t I>
struct extract_epu64_wrapper_avx
{
    static SIMD_INLINE uint64_t call(const __m256i& src)
    {
        return extract_epu64_helper_avx<I>(src);
    }
};

template <size_t I>
struct insert_epi8_wrapper_avx
{
    static SIMD_INLINE __m256i call(__m256i dst, int value)
    {
        return insert_epi8_helper_avx<I>(dst, value);
    }
};

template <size_t I>
struct insert_epi32_wrapper_avx
{
    static SIMD_INLINE __m256i call(__m256i dst, int value)
    {
        return insert_epi32_helper_avx<I>(dst, value);
    }
};

template <size_t I>
struct insert_epi64_wrapper_avx
{
    static SIMD_INLINE __m256i call(__m256i dst, long long value)
    {
        return insert_epi64_helper_avx<I>(dst, value);
    }
};

static SIMD_INLINE int8_t extract_epi8_runtime_avx(const __m256i& src, size_t index)
{
    return runtime_dispatcher<int8_t, 32>::extract_dispatch<extract_epi8_wrapper_avx, int8_t>(
        index % 32, src);
}

static SIMD_INLINE uint8_t extract_epu8_runtime_avx(const __m256i& src, size_t index)
{
    return runtime_dispatcher<uint8_t, 32>::extract_dispatch<extract_epu8_wrapper_avx, uint8_t>(
        index % 32, src);
}

static SIMD_INLINE int32_t extract_epi32_runtime_avx(const __m256i& src, size_t index)
{
    return runtime_dispatcher<int32_t, 8>::extract_dispatch<extract_epi32_wrapper_avx, int32_t>(
        index % 8, src);
}

static SIMD_INLINE uint32_t extract_epu32_runtime_avx(const __m256i& src, size_t index)
{
    return runtime_dispatcher<uint32_t, 8>::extract_dispatch<extract_epu32_wrapper_avx, uint32_t>(
        index % 8, src);
}

static SIMD_INLINE int64_t extract_epi64_runtime_avx(const __m256i& src, size_t index)
{
    return runtime_dispatcher<int64_t, 4>::extract_dispatch<extract_epi64_wrapper_avx, int64_t>(
        index % 4, src);
}

static SIMD_INLINE uint64_t extract_epu64_runtime_avx(const __m256i& src, size_t index)
{
    return runtime_dispatcher<uint64_t, 4>::extract_dispatch<extract_epu64_wrapper_avx, uint64_t>(
        index % 4, src);
}

static SIMD_INLINE __m256i insert_epi8_runtime_avx(__m256i dst, size_t index, int value)
{
    return runtime_dispatcher<__m256i, 32>::insert_dispatch<insert_epi8_wrapper_avx, __m256i>(
        index % 32, dst, value);
}

static SIMD_INLINE __m256i insert_epi32_runtime_avx(__m256i dst, size_t index, int value)
{
    return runtime_dispatcher<__m256i, 8>::insert_dispatch<insert_epi32_wrapper_avx, __m256i>(
        index % 8, dst, value);
}

static SIMD_INLINE __m256i insert_epi64_runtime_avx(__m256i dst, size_t index, long long value)
{
    return runtime_dispatcher<__m256i, 4>::insert_dispatch<insert_epi64_wrapper_avx, __m256i>(
        index % 4, dst, value);
}
#endif // SIMD_AVX2

template <typename T, size_t N>
struct vector_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::AVX>::compile_time>>
{
    using register_t = typename register_type<T, avx_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_set1_ps(value);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_set1_pd(value);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm256_set1_epi8(static_cast<char>(value));
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm256_set1_epi16(static_cast<short>(value));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm256_set1_epi32(static_cast<int>(value));
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm256_set1_epi64x(static_cast<long long>(value));
        }
    }

    static SIMD_INLINE T extract(const register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *src);
            return tmp[index % 8];
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            return tmp[index % 4];
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_AVX2
            return extract_epi8_runtime_avx(*src, index);
#else
            alignas(32) int8_t tmp[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 32];
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            return extract_epu8_runtime_avx(*src, index);
#else
            alignas(32) uint8_t tmp[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 32];
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            alignas(32) int16_t tmp[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 16];
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            alignas(32) uint16_t tmp[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 16];
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            return extract_epi32_runtime_avx(*src, index);
#else
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 8];
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            return extract_epu32_runtime_avx(*src, index);
#else
            alignas(32) uint32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 8];
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
#if SIMD_AVX2
            return extract_epi64_runtime_avx(*src, index);
#else
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }
        else if constexpr (std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            return extract_epu64_runtime_avx(*src, index);
#else
            alignas(32) uint64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }
    }

    static SIMD_INLINE void insert(register_t* dst, size_t index, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *dst);
            tmp[index % 8] = value;
            *dst = _mm256_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *dst);
            tmp[index % 4] = value;
            *dst = _mm256_load_pd(tmp);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = insert_epi8_runtime_avx(*dst, index, static_cast<int>(value));
#else
            alignas(32) int8_t tmp[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *dst);
            tmp[index % 32] = value;
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            alignas(32) int16_t tmp[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *dst);
            tmp[index % 16] = static_cast<int16_t>(value);
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = insert_epi32_runtime_avx(*dst, index, static_cast<int>(value));
#else
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *dst);
            tmp[index % 8] = static_cast<int32_t>(value);
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            *dst = insert_epi64_runtime_avx(*dst, index, static_cast<long long>(value));
#else
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *dst);
            tmp[index % 4] = static_cast<int64_t>(value);
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
#endif
        }
    }

    static SIMD_INLINE void add(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_add_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_add_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi8(*a, *b);
#else
            // Fallback for AVX1
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_add_epi8(a_lo, b_lo);
            __m128i result_hi = _mm_add_epi8(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_add_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_add_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi32(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_add_epi32(a_lo, b_lo);
            __m128i result_hi = _mm_add_epi32(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi64(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_add_epi64(a_lo, b_lo);
            __m128i result_hi = _mm_add_epi64(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void sub(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_sub_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_sub_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_sub_epi8(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_sub_epi8(a_lo, b_lo);
            __m128i result_hi = _mm_sub_epi8(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_sub_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_sub_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_sub_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_sub_epi32(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_sub_epi32(a_lo, b_lo);
            __m128i result_hi = _mm_sub_epi32(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_sub_epi64(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_sub_epi64(a_lo, b_lo);
            __m128i result_hi = _mm_sub_epi64(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void mul(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_mul_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_mul_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_mullo_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_mullo_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_mullo_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_mullo_epi32(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            // Use SSE 4.1 if available, otherwise fallback
#if SIMD_SSE4_1
            __m128i result_lo = _mm_mullo_epi32(a_lo, b_lo);
            __m128i result_hi = _mm_mullo_epi32(a_hi, b_hi);
#else
            __m128i tmp1_lo = _mm_mul_epu32(a_lo, b_lo);
            __m128i tmp2_lo = _mm_mul_epu32(_mm_srli_si128(a_lo, 4), _mm_srli_si128(b_lo, 4));
            __m128i result_lo =
                _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1_lo, _MM_SHUFFLE(0, 0, 2, 0)),
                                   _mm_shuffle_epi32(tmp2_lo, _MM_SHUFFLE(0, 0, 2, 0)));

            __m128i tmp1_hi = _mm_mul_epu32(a_hi, b_hi);
            __m128i tmp2_hi = _mm_mul_epu32(_mm_srli_si128(a_hi, 4), _mm_srli_si128(b_hi, 4));
            __m128i result_hi =
                _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1_hi, _MM_SHUFFLE(0, 0, 2, 0)),
                                   _mm_shuffle_epi32(tmp2_hi, _MM_SHUFFLE(0, 0, 2, 0)));
#endif
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else
        {
            // Fallback for unsupported types
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                a_arr[i] *= b_arr[i];
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
        }
    }

    static SIMD_INLINE void div(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_div_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_div_pd(*a, *b);
        }
        else
        {
            // Integer division fallback
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                a_arr[i] /= b_arr[i];
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
        }
    }

    static SIMD_INLINE void min(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_min_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_min_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_min_epi8(*a, *b);
#else
            alignas(32) int8_t a_arr[32], b_arr[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 32; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_min_epu8(*a, *b);
#else
            alignas(32) uint8_t a_arr[32], b_arr[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 32; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_min_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_min_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_min_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_min_epu16(*a, *b);
#else
            alignas(32) uint16_t a_arr[16], b_arr[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 16; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_min_epi32(*a, *b);
#else
            alignas(32) int32_t a_arr[8], b_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_min_epu32(*a, *b);
#else
            alignas(32) uint32_t a_arr[8], b_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else
        {
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
        }
    }

    static SIMD_INLINE void max(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_max_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_max_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_max_epi8(*a, *b);
#else
            alignas(32) int8_t a_arr[32], b_arr[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 32; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_max_epu8(*a, *b);
#else
            alignas(32) uint8_t a_arr[32], b_arr[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 32; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_max_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);
            __m128i result_lo = _mm_max_epi16(a_lo, b_lo);
            __m128i result_hi = _mm_max_epi16(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_max_epu16(*a, *b);
#else
            alignas(32) uint16_t a_arr[16], b_arr[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 16; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_max_epi32(*a, *b);
#else
            alignas(32) int32_t a_arr[8], b_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_max_epu32(*a, *b);
#else
            alignas(32) uint32_t a_arr[8], b_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
#endif
        }
        else
        {
            alignas(32) T a_arr[32 / sizeof(T)];
            alignas(32) T b_arr[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a_arr), *a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), *b);

            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_arr));
        }
    }

    static SIMD_INLINE void bitwise_and(register_t* dst, const register_t* a, const register_t* b)
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
            __m128i result_lo = _mm_and_si128(a_lo, b_lo);
            __m128i result_hi = _mm_and_si128(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void bitwise_or(register_t* dst, const register_t* a, const register_t* b)
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

    static SIMD_INLINE void bitwise_xor(register_t* dst, const register_t* a, const register_t* b)
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
            __m128i result_lo = _mm_xor_si128(a_lo, b_lo);
            __m128i result_hi = _mm_xor_si128(a_hi, b_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void bitwise_not(register_t* dst, const register_t* a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const __m256 ones = _mm256_cmp_ps(*a, *a, _CMP_EQ_OQ);
            *dst = _mm256_xor_ps(*a, ones);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            const __m256d ones = _mm256_cmp_pd(*a, *a, _CMP_EQ_OQ);
            *dst = _mm256_xor_pd(*a, ones);
        }
        else
        {
#if SIMD_AVX2
            const __m256i ones = _mm256_cmpeq_epi32(*a, *a);
            *dst = _mm256_xor_si256(*a, ones);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            const __m128i ones_lo = _mm_cmpeq_epi32(a_lo, a_lo);
            const __m128i ones_hi = _mm_cmpeq_epi32(a_hi, a_hi);
            __m128i result_lo = _mm_xor_si128(a_lo, ones_lo);
            __m128i result_hi = _mm_xor_si128(a_hi, ones_hi);
            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi, 1);
#endif
        }
    }

    static SIMD_INLINE void blend(register_t* dst, const register_t* a, const register_t* b,
                                  const typename mask_register_type<T, avx_tag>::type* mask)
    {
#if SIMD_AVX2
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_blendv_ps(*a, *b, *mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_blendv_pd(*a, *b, *mask);
        }
        else
        {
            *dst = _mm256_blendv_epi8(*a, *b, *mask);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_blendv_ps(*a, *b, *mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_blendv_pd(*a, *b, *mask);
        }
        else
        {
            *dst = _mm256_or_ps(_mm256_and_ps(*mask, *b), _mm256_andnot_ps(*mask, *a));
        }
#endif
    }

    static SIMD_INLINE void select(register_t* dst,
                                   const typename mask_register_type<T, avx_tag>::type* mask,
                                   const register_t* a, const register_t* b)
    {
        // !Note: operands reversed because masks are different semantics
        blend(dst, b, a, mask);
    }

    static SIMD_INLINE T horizontal_sum(const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m256 sum = *src;
            __m128 sum_lo = _mm256_extractf128_ps(sum, 0);
            __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
            __m128 sum_128 = _mm_add_ps(sum_lo, sum_hi);
            sum_128 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
            sum_128 = _mm_add_ss(sum_128, _mm_shuffle_ps(sum_128, sum_128, 1));
            return _mm_cvtss_f32(sum_128);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m256d sum = *src;
            __m128d sum_lo = _mm256_extractf128_pd(sum, 0);
            __m128d sum_hi = _mm256_extractf128_pd(sum, 1);
            __m128d sum_128 = _mm_add_pd(sum_lo, sum_hi);
            sum_128 = _mm_add_sd(sum_128, _mm_unpackhi_pd(sum_128, sum_128));
            return _mm_cvtsd_f64(sum_128);
        }
        else
        {
            alignas(32) T tmp[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            T sum = 0;
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                sum += tmp[i];
            }
            return sum;
        }
    }

    static SIMD_INLINE T horizontal_min(const register_t* src)
    {
        alignas(32) T tmp[32 / sizeof(T)];
        if constexpr (std::is_same_v<T, float>)
        {
            _mm256_store_ps(tmp, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm256_store_pd(tmp, *src);
        }
        else
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
        }

        T min_val = tmp[0];
        for (size_t i = 1; i < 32 / sizeof(T); ++i)
        {
            min_val = std::min(min_val, tmp[i]);
        }
        return min_val;
    }

    static SIMD_INLINE T horizontal_max(const register_t* src)
    {
        alignas(32) T tmp[32 / sizeof(T)];
        if constexpr (std::is_same_v<T, float>)
        {
            _mm256_store_ps(tmp, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm256_store_pd(tmp, *src);
        }
        else
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
        }

        T max_val = tmp[0];
        for (size_t i = 1; i < 32 / sizeof(T); ++i)
        {
            max_val = std::max(max_val, tmp[i]);
        }
        return max_val;
    }

    static SIMD_INLINE void shuffle(register_t* dst, const register_t* src, const int* indices)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *src);

            alignas(32) float result[8];
            for (size_t i = 0; i < 8; ++i)
            {
                result[i] = tmp[indices[i] % 8];
            }

            *dst = _mm256_load_ps(result);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);

            alignas(32) double result[4];
            for (size_t i = 0; i < 4; ++i)
            {
                result[i] = tmp[indices[i] % 4];
            }

            *dst = _mm256_load_pd(result);
        }
        else
        {
            alignas(32) T tmp[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);

            alignas(32) T result[32 / sizeof(T)];
            for (size_t i = 0; i < 32 / sizeof(T); ++i)
            {
                result[i] = tmp[indices[i] % (32 / sizeof(T))];
            }

            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(result));
        }
    }

    template <typename U>
    static SIMD_INLINE void convert(typename register_type<U, avx_tag>::type* dst,
                                    const register_t* src)
    {
        if constexpr (std::is_same_v<T, float> && std::is_same_v<U, int32_t>)
        {
            *dst = _mm256_cvtps_epi32(*src);
        }
        else if constexpr (std::is_same_v<T, int32_t> && std::is_same_v<U, float>)
        {
            *dst = _mm256_cvtepi32_ps(*src);
        }
        else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, double>)
        {
            __m256 s = *src;
            __m128 s_lo = _mm256_extractf128_ps(s, 0);
            __m128 s_hi = _mm256_extractf128_ps(s, 1);
            dst[0] = _mm256_cvtps_pd(s_lo);
            dst[1] = _mm256_cvtps_pd(s_hi);
        }
        else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, float>)
        {
            __m128 lo = _mm256_cvtpd_ps(src[0]);
            __m128 hi = _mm256_cvtpd_ps(src[1]);
            *dst = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
        }
        else
        {
            // Generic conversion for other types
            alignas(32) T src_arr[32 / sizeof(T)];
            if constexpr (std::is_same_v<T, float>)
            {
                _mm256_store_ps(src_arr, *src);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                _mm256_store_pd(src_arr, *src);
            }
            else
            {
                _mm256_store_si256(reinterpret_cast<__m256i*>(src_arr), *src);
            }

            alignas(32) U dst_arr[32 / sizeof(U)];
            for (size_t i = 0; i < std::min(32 / sizeof(T), 32 / sizeof(U)); ++i)
            {
                dst_arr[i] = static_cast<U>(src_arr[i]);
            }

            if constexpr (std::is_same_v<U, float>)
            {
                *dst = _mm256_load_ps(dst_arr);
            }
            else if constexpr (std::is_same_v<U, double>)
            {
                *dst = _mm256_load_pd(dst_arr);
            }
            else
            {
                *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst_arr));
            }
        }
    }
};

} // namespace detail

} // namespace vector_simd

#endif // SIMD_ARCH_X86 && SIMD_HAS_AVX

#endif // End of include guard: LIB_SIMD_IMPL_AVX_VECTOR_OPS_HPP_avx256