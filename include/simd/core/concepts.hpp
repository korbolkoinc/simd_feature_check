#ifndef LIB_SIMD_CORE_TYPES_HPP_czk5dm
#define LIB_SIMD_CORE_TYPES_HPP_czk5dm

#include <cstdint>
#include <type_traits>

namespace vector_simd
{
template <typename T>
concept SimdArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int8_t> ||
    std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, uint64_t>;

template <typename T>
concept SimdFloat = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept SimdInteger = SimdArithmetic<T> && !SimdFloat<T>;

template <typename T>
concept SimdSignedInteger = std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                            std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

template <typename T>
concept SimdUnsignedInteger = std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

} // namespace vector_simd
#endif // End of include guard: LIB_SIMD_CORE_TYPES_HPP_czk5dm