#ifndef SIMD_HPP_al9nn6
#define SIMD_HPP_al9nn6

#include <simd/common.hpp>
#include <simd/feature_check.hpp>

namespace vector_simd
{

constexpr int kVersionMajor = 0;
constexpr int kVersionMinor = 1;
constexpr int kVersionPatch = 0;

constexpr size_t kSSEAlignment = 16;
constexpr size_t kAVXAlignment = 32;
constexpr size_t kAVX512Alignment = 64;

constexpr size_t kDefaultAlignment =
    simd::compile_time::has<simd::Feature::AVX512F>() ? kAVX512Alignment
    : simd::compile_time::has<simd::Feature::AVX>()   ? kAVXAlignment
                                                      : kSSEAlignment;

template <typename T>
concept SimdArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;

template <typename T>
concept SimdFloat = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept SimdInteger = SimdArithmetic<T> && !SimdFloat<T>;

template <typename T>
concept SimdSignedInteger =
    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

template <typename T>
concept SimdUnsignedInteger =
    std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

// Forward declarations
template <SimdArithmetic T, size_t N>
class Vector;

template <SimdArithmetic T, size_t N>
class Mask;

namespace detail
{

struct generic_tag
{
};

struct sse2_tag : generic_tag
{
};

struct sse3_tag : sse2_tag
{
};

struct ssse3_tag : sse3_tag
{
};

struct sse4_1_tag : ssse3_tag
{
};

struct sse4_2_tag : sse4_1_tag
{
};

struct avx_tag : sse4_2_tag
{
};

struct avx2_tag : avx_tag
{
};

struct avx512_tag : avx2_tag
{
};

struct neon_tag : generic_tag
{
};

struct wasm_simd_tag : generic_tag
{
};

struct best_available_tag
{
    using type = std::conditional_t<
        simd::compile_time::has<simd::Feature::AVX512F>(), avx512_tag,
        std::conditional_t<
            simd::compile_time::has<simd::Feature::AVX2>(), avx2_tag,
            std::conditional_t<
                simd::compile_time::has<simd::Feature::AVX>(), avx_tag,
                std::conditional_t<
                    simd::compile_time::has<simd::Feature::SSE42>(), sse4_2_tag,
                    std::conditional_t<
                        simd::compile_time::has<simd::Feature::SSE41>(), sse4_1_tag,
                        std::conditional_t<
                            simd::compile_time::has<simd::Feature::SSSE3>(), ssse3_tag,
                            std::conditional_t<
                                simd::compile_time::has<simd::Feature::SSE3>(), sse3_tag,
                                std::conditional_t<simd::compile_time::has<simd::Feature::SSE2>(),
                                                   sse2_tag, generic_tag
                                >
                            >
                        >
                    >
                >
            >
        >
    >;
};

using current_isa = typename best_available_tag::type;

template <typename T, typename ISA>
struct simd_width;

template <>
struct simd_width<float, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<double, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<int8_t, sse2_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint8_t, sse2_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int16_t, sse2_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint16_t, sse2_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int32_t, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint32_t, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int64_t, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<uint64_t, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<float, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<double, avx_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int8_t, avx_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<uint8_t, avx_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<int16_t, avx_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint16_t, avx_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int32_t, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint32_t, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int64_t, avx_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint64_t, avx_tag>
{
    static constexpr size_t value = 4;
};

template <typename T>
struct simd_width<T, avx2_tag> : simd_width<T, avx_tag>
{
};

template <>
struct simd_width<float, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<double, avx512_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int8_t, avx512_tag>
{
    static constexpr size_t value = 64;
};

template <>
struct simd_width<uint8_t, avx512_tag>
{
    static constexpr size_t value = 64;
};

template <>
struct simd_width<int16_t, avx512_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<uint16_t, avx512_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<int32_t, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint32_t, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int64_t, avx512_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint64_t, avx512_tag>
{
    static constexpr size_t value = 8;
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template <>
struct simd_width<float, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int8_t, neon_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint8_t, neon_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int16_t, neon_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint16_t, neon_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int32_t, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint32_t, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int64_t, neon_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<uint64_t, neon_tag>
{
    static constexpr size_t value = 2;
};

#endif

// Generic fallback for unsupported architectures

template <typename T>
struct simd_width<T, generic_tag>
{
    static constexpr size_t value = 1;
};

template <typename T>
struct native_width
{
    static constexpr size_t value = simd_width<T, current_isa>::value;
};

template <typename T, typename ISA>
struct register_type;

template <>
struct register_type<float, sse2_tag>
{
    using type = __m128;
};

template <>
struct register_type<double, sse2_tag>
{
    using type = __m128d;
};

template <>
struct register_type<int8_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint8_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int16_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint16_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int32_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint32_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int64_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint64_t, sse2_tag>
{
    using type = __m128i;
};

// AVX register type mappings

template <>
struct register_type<float, avx_tag>
{
    using type = __m256;
};

template <>
struct register_type<double, avx_tag>
{
    using type = __m256d;
};

template <>
struct register_type<int8_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint8_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int16_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint16_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int32_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint32_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int64_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint64_t, avx_tag>
{
    using type = __m256i;
};

// Same register types for AVX2

template <typename T>
struct register_type<T, avx2_tag> : register_type<T, avx_tag>
{
};

#ifdef __AVX512F__

template <>
struct register_type<float, avx512_tag>
{
    using type = __m512;
};

template <>
struct register_type<double, avx512_tag>
{
    using type = __m512d;
};

template <>
struct register_type<int8_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint8_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int16_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint16_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int32_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint32_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int64_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint64_t, avx512_tag>
{
    using type = __m512i;
};

#endif

// NEON register type mappings (if needed)

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template <>
struct register_type<float, neon_tag>
{
    using type = float32x4_t;
};

template <>
struct register_type<int8_t, neon_tag>
{
    using type = int8x16_t;
};

template <>
struct register_type<uint8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct register_type<int16_t, neon_tag>
{
    using type = int16x8_t;
};

template <>
struct register_type<uint16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct register_type<int32_t, neon_tag>
{
    using type = int32x4_t;
};

template <>
struct register_type<uint32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct register_type<int64_t, neon_tag>
{
    using type = int64x2_t;
};

template <>
struct register_type<uint64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct register_type<double, neon_tag>
{
#ifdef __aarch64__
    using type = float64x2_t;
#else
    using type = double;
#endif
};
#endif

} // namespace detail

} // namespace vector_simd

#endif /* End of include guard: SIMD_HPP_al9nn6 */
