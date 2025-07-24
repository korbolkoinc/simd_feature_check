#ifndef LIB_SIMD_IMPL_SSE2_VECTOR_OPS_HPP_9b0cmo
#define LIB_SIMD_IMPL_SSE2_VECTOR_OPS_HPP_9b0cmo

#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include <simd/feature_check.hpp>

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2
#include <algorithm>
#include <emmintrin.h>
#include <type_traits>

namespace vector_simd
{
namespace detail
{

#if SIMD_SSE4_1
template <size_t I>
static SIMD_INLINE int8_t extract_epi8_helper(const __m128i& src)
{
    static_assert(I < 16, "Index out of range for epi8 extraction");
    return static_cast<int8_t>(_mm_extract_epi8(src, I));
}

template <size_t I>
static SIMD_INLINE uint8_t extract_epu8_helper(const __m128i& src)
{
    static_assert(I < 16, "Index out of range for epu8 extraction");
    return static_cast<uint8_t>(_mm_extract_epi8(src, I));
}

template <size_t I>
static SIMD_INLINE int32_t extract_epi32_helper(const __m128i& src)
{
    static_assert(I < 4, "Index out of range for epi32 extraction");
    return static_cast<int32_t>(_mm_extract_epi32(src, I));
}

template <size_t I>
static SIMD_INLINE uint32_t extract_epu32_helper(const __m128i& src)
{
    static_assert(I < 4, "Index out of range for epu32 extraction");
    return static_cast<uint32_t>(_mm_extract_epi32(src, I));
}

template <size_t I>
static SIMD_INLINE int64_t extract_epi64_helper(const __m128i& src)
{
    static_assert(I < 2, "Index out of range for epi64 extraction");
    return static_cast<int64_t>(_mm_extract_epi64(src, I));
}

template <size_t I>
static SIMD_INLINE uint64_t extract_epu64_helper(const __m128i& src)
{
    static_assert(I < 2, "Index out of range for epu64 extraction");
    return static_cast<uint64_t>(_mm_extract_epi64(src, I));
}

template <size_t I>
static SIMD_INLINE __m128i insert_epi8_helper(__m128i dst, int value)
{
    static_assert(I < 16, "Index out of range for epi8 insertion");
    return _mm_insert_epi8(dst, value, I);
}

template <size_t I>
static SIMD_INLINE __m128i insert_epi32_helper(__m128i dst, int value)
{
    static_assert(I < 4, "Index out of range for epi32 insertion");
    return _mm_insert_epi32(dst, value, I);
}

template <size_t I>
static SIMD_INLINE __m128i insert_epi64_helper(__m128i dst, long long value)
{
    static_assert(I < 2, "Index out of range for epi64 insertion");
    return _mm_insert_epi64(dst, value, I);
}

template <typename T, size_t MaxIndex>
struct runtime_dispatcher_sse
{
    template <template <size_t> class HelperFunc, typename RetType, typename... Args>
    static SIMD_INLINE RetType extract_dispatch(size_t index, Args&&... args)
    {
        return extract_dispatch_impl<HelperFunc, RetType, 0>(index, std::forward<Args>(args)...);
    }

    template <template <size_t> class HelperFunc, typename RetType, typename... Args>
    static SIMD_INLINE RetType insert_dispatch(size_t index, Args&&... args)
    {
        return insert_dispatch_impl<HelperFunc, RetType, 0>(index, std::forward<Args>(args)...);
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
struct extract_epi8_wrapper_sse
{
    static SIMD_INLINE int8_t call(const __m128i& src) { return extract_epi8_helper<I>(src); }
};

template <size_t I>
struct extract_epu8_wrapper_sse
{
    static SIMD_INLINE uint8_t call(const __m128i& src) { return extract_epu8_helper<I>(src); }
};

template <size_t I>
struct extract_epi32_wrapper_sse
{
    static SIMD_INLINE int32_t call(const __m128i& src) { return extract_epi32_helper<I>(src); }
};

template <size_t I>
struct extract_epu32_wrapper_sse
{
    static SIMD_INLINE uint32_t call(const __m128i& src) { return extract_epu32_helper<I>(src); }
};

template <size_t I>
struct extract_epi64_wrapper_sse
{
    static SIMD_INLINE int64_t call(const __m128i& src) { return extract_epi64_helper<I>(src); }
};

template <size_t I>
struct extract_epu64_wrapper_sse
{
    static SIMD_INLINE uint64_t call(const __m128i& src) { return extract_epu64_helper<I>(src); }
};

template <size_t I>
struct insert_epi8_wrapper_sse
{
    static SIMD_INLINE __m128i call(__m128i dst, int value)
    {
        return insert_epi8_helper<I>(dst, value);
    }
};

template <size_t I>
struct insert_epi32_wrapper_sse
{
    static SIMD_INLINE __m128i call(__m128i dst, int value)
    {
        return insert_epi32_helper<I>(dst, value);
    }
};

template <size_t I>
struct insert_epi64_wrapper_sse
{
    static SIMD_INLINE __m128i call(__m128i dst, long long value)
    {
        return insert_epi64_helper<I>(dst, value);
    }
};

static SIMD_INLINE int8_t extract_epi8_runtime(const __m128i& src, size_t index)
{
    return runtime_dispatcher_sse<int8_t, 16>::extract_dispatch<extract_epi8_wrapper_sse, int8_t>(
        index % 16, src);
}

static SIMD_INLINE uint8_t extract_epu8_runtime(const __m128i& src, size_t index)
{
    return runtime_dispatcher_sse<uint8_t, 16>::extract_dispatch<extract_epu8_wrapper_sse, uint8_t>(
        index % 16, src);
}

static SIMD_INLINE int32_t extract_epi32_runtime(const __m128i& src, size_t index)
{
    return runtime_dispatcher_sse<int32_t, 4>::extract_dispatch<extract_epi32_wrapper_sse, int32_t>(
        index % 4, src);
}

static SIMD_INLINE uint32_t extract_epu32_runtime(const __m128i& src, size_t index)
{
    return runtime_dispatcher_sse<uint32_t, 4>::extract_dispatch<extract_epu32_wrapper_sse,
                                                                 uint32_t>(index % 4, src);
}

static SIMD_INLINE int64_t extract_epi64_runtime(const __m128i& src, size_t index)
{
    return runtime_dispatcher_sse<int64_t, 2>::extract_dispatch<extract_epi64_wrapper_sse, int64_t>(
        index % 2, src);
}

static SIMD_INLINE uint64_t extract_epu64_runtime(const __m128i& src, size_t index)
{
    return runtime_dispatcher_sse<uint64_t, 2>::extract_dispatch<extract_epu64_wrapper_sse,
                                                                 uint64_t>(index % 2, src);
}

static SIMD_INLINE __m128i insert_epi8_runtime(__m128i dst, size_t index, int value)
{
    return runtime_dispatcher_sse<__m128i, 16>::insert_dispatch<insert_epi8_wrapper_sse, __m128i>(
        index % 16, dst, value);
}

static SIMD_INLINE __m128i insert_epi32_runtime(__m128i dst, size_t index, int value)
{
    return runtime_dispatcher_sse<__m128i, 4>::insert_dispatch<insert_epi32_wrapper_sse, __m128i>(
        index % 4, dst, value);
}

static SIMD_INLINE __m128i insert_epi64_runtime(__m128i dst, size_t index, long long value)
{
    return runtime_dispatcher_sse<__m128i, 2>::insert_dispatch<insert_epi64_wrapper_sse, __m128i>(
        index % 2, dst, value);
}
#endif // SIMD_SSE4_1

template <typename T, size_t N>
struct vector_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_set1_ps(value);
        }

        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_set1_pd(value);
        }

        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_set1_epi8(static_cast<char>(value));
        }

        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_set1_epi16(static_cast<short>(value));
        }

        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_set1_epi32(static_cast<int>(value));
        }

        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_set1_epi64x(static_cast<long long>(value));
        }
    }

    static SIMD_INLINE T extract(const register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            return tmp[index % 4];
        }

        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            return tmp[index % 2];
        }

        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            return extract_epi8_runtime(*src, index);
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 16];
#endif
        }

        else if constexpr (std::is_same_v<T, uint8_t>)
        {
#if SIMD_SSE4_1
            return extract_epu8_runtime(*src, index);
#else
            alignas(16) uint8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 16];
#endif
        }

        else if constexpr (std::is_same_v<T, int16_t>)
        {
            alignas(16) int16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 8];
        }

        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            alignas(16) uint16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 8];
        }

        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            return extract_epi32_runtime(*src, index);
#else
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }

        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            return extract_epu32_runtime(*src, index);
#else
            alignas(16) uint32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }

        else if constexpr (std::is_same_v<T, int64_t>)
        {
#if SIMD_SSE4_1
            return extract_epi64_runtime(*src, index);
#else
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 2];
#endif
        }

        else if constexpr (std::is_same_v<T, uint64_t>)
        {
#if SIMD_SSE4_1
            return extract_epu64_runtime(*src, index);
#else
            alignas(16) uint64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 2];
#endif
        }
    }

    static SIMD_INLINE void insert(register_t* dst, size_t index, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *dst);
            tmp[index % 4] = value;
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *dst);
            tmp[index % 2] = value;
            *dst = _mm_load_pd(tmp);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_SSE4_1
            *dst = insert_epi8_runtime(*dst, index, static_cast<int>(value));
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 16] = value;
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            alignas(16) int16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 8] = static_cast<int16_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = insert_epi32_runtime(*dst, index, static_cast<int>(value));
#else
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 4] = static_cast<int32_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_SSE4_1
            *dst = insert_epi64_runtime(*dst, index, static_cast<long long>(value));
#else
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 2] = static_cast<int64_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
    }

    static SIMD_INLINE void add(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_add_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_add_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_add_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_add_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_add_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_add_epi64(*a, *b);
        }
    }

    static SIMD_INLINE void sub(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_sub_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_sub_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_sub_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_sub_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_sub_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_sub_epi64(*a, *b);
        }
    }

    static SIMD_INLINE void mul(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_mul_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_mul_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_mullo_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_mullo_epi32(*a, *b);
#else
            __m128i tmp1 = _mm_mul_epu32(*a, *b);
            __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(*a, 4), _mm_srli_si128(*b, 4));
            *dst = _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
                                      _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
#endif
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] *= b_arr[i];
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void div(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_div_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_div_pd(*a, *b);
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] /= b_arr[i];
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void min(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_min_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_min_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epi8(*a, *b);
#else
            alignas(16) int8_t a_arr[16], b_arr[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 16; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_min_epu8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_min_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epu16(*a, *b);
#else
            alignas(16) uint16_t a_arr[8], b_arr[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epi32(*a, *b);
#else
            alignas(16) int32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epu32(*a, *b);
#else
            alignas(16) uint32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void max(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_max_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_max_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epi8(*a, *b);
#else
            alignas(16) int8_t a_arr[16], b_arr[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 16; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_max_epu8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_max_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epu16(*a, *b);
#else
            alignas(16) uint16_t a_arr[8], b_arr[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epi32(*a, *b);
#else
            alignas(16) int32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epu32(*a, *b);
#else
            alignas(16) uint32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void bitwise_and(register_t* dst, const register_t* a, const register_t* b)
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

    static SIMD_INLINE void bitwise_or(register_t* dst, const register_t* a, const register_t* b)
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

    static SIMD_INLINE void bitwise_xor(register_t* dst, const register_t* a, const register_t* b)
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

    static SIMD_INLINE void bitwise_not(register_t* dst, const register_t* a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const __m128 ones = _mm_cmpeq_ps(*a, *a);
            *dst = _mm_xor_ps(*a, ones);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            const __m128d ones = _mm_cmpeq_pd(*a, *a);
            *dst = _mm_xor_pd(*a, ones);
        }
        else
        {
            const __m128i ones = _mm_cmpeq_epi32(*a, *a);
            *dst = _mm_xor_si128(*a, ones);
        }
    }

    static SIMD_INLINE void blend(register_t* dst, const register_t* a, const register_t* b,
                                  const typename mask_register_type<T, sse2_tag>::type* mask)
    {
#if SIMD_SSE4_1
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_blendv_ps(*a, *b, *mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_blendv_pd(*a, *b, *mask);
        }
        else
        {
            *dst = _mm_blendv_epi8(*a, *b, *mask);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_or_ps(_mm_and_ps(*mask, *b), _mm_andnot_ps(*mask, *a));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_or_pd(_mm_and_pd(*mask, *b), _mm_andnot_pd(*mask, *a));
        }
        else
        {
            *dst = _mm_or_si128(_mm_and_si128(*mask, *b), _mm_andnot_si128(*mask, *a));
        }
#endif
    }

    static SIMD_INLINE void select(register_t* dst,
                                   const typename mask_register_type<T, sse2_tag>::type* mask,
                                   const register_t* a, const register_t* b)
    {
        // !Note: operands reversed because masks are different semantics
        blend(dst, b, a, mask);
    }

    static SIMD_INLINE T horizontal_sum(const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m128 sum = *src;
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m128d sum = *src;
            sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
            return _mm_cvtsd_f64(sum);
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            T sum = 0;
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                sum += tmp[i];
            }
            return sum;
        }
    }

    static SIMD_INLINE T horizontal_min(const register_t* src)
    {
        alignas(16) T tmp[16 / sizeof(T)];
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(tmp, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(tmp, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
        }

        T min_val = tmp[0];
        for (size_t i = 1; i < 16 / sizeof(T); ++i)
        {
            min_val = std::min(min_val, tmp[i]);
        }
        return min_val;
    }

    static SIMD_INLINE T horizontal_max(const register_t* src)
    {
        alignas(16) T tmp[16 / sizeof(T)];
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(tmp, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(tmp, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
        }

        T max_val = tmp[0];
        for (size_t i = 1; i < 16 / sizeof(T); ++i)
        {
            max_val = std::max(max_val, tmp[i]);
        }
        return max_val;
    }

    static SIMD_INLINE void shuffle(register_t* dst, const register_t* src, const int* indices)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);

            alignas(16) float result[4];
            for (size_t i = 0; i < 4; ++i)
            {
                result[i] = tmp[indices[i] % 4];
            }

            *dst = _mm_load_ps(result);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);

            alignas(16) double result[2];
            for (size_t i = 0; i < 2; ++i)
            {
                result[i] = tmp[indices[i] % 2];
            }

            *dst = _mm_load_pd(result);
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);

            alignas(16) T result[16 / sizeof(T)];
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                result[i] = tmp[indices[i] % (16 / sizeof(T))];
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(result));
        }
    }

    template <typename U>
    static SIMD_INLINE void convert(typename register_type<U, sse2_tag>::type* dst,
                                    const register_t* src)
    {
        if constexpr (std::is_same_v<T, float> && std::is_same_v<U, int32_t>)
        {
            *dst = _mm_cvtps_epi32(*src);
        }
        else if constexpr (std::is_same_v<T, int32_t> && std::is_same_v<U, float>)
        {
            *dst = _mm_cvtepi32_ps(*src);
        }
        else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, double>)
        {
            __m128 s = *src;
            __m128d lo = _mm_cvtps_pd(s);
            __m128d hi = _mm_cvtps_pd(_mm_movehl_ps(s, s));

            dst[0] = lo;
            dst[1] = hi;
        }
        else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, float>)
        {
            __m128 lo = _mm_cvtpd_ps(src[0]);
            __m128 hi = _mm_cvtpd_ps(src[1]);
            *dst = _mm_movelh_ps(lo, hi);
        }
        else
        {
            alignas(16) T src_arr[16 / sizeof(T)];
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

            alignas(16) U dst_arr[16 / sizeof(U)];
            for (size_t i = 0; i < std::min(16 / sizeof(T), 16 / sizeof(U)); ++i)
            {
                dst_arr[i] = static_cast<U>(src_arr[i]);
            }

            if constexpr (std::is_same_v<U, float>)
            {
                *dst = _mm_load_ps(dst_arr);
            }
            else if constexpr (std::is_same_v<U, double>)
            {
                *dst = _mm_load_pd(dst_arr);
            }
            else
            {
                *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(dst_arr));
            }
        }
    }
};

} // namespace detail

} // namespace vector_simd

#endif // SIMD_ARCH_X86 && SIMD_HAS_SSE2

#endif // End of include guard: LIB_SIMD_IMPL_SSE2_VECTOR_OPS_HPP_9b0cmo