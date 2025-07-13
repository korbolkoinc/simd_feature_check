#ifndef SIMD_HPP_al9nn6
#define SIMD_HPP_al9nn6

#include <array>
#include <bit>
#include <cmath>
#include <simd/arch/detection.hpp>
#include <simd/arch/tags.hpp>
#include <simd/common.hpp>
#include <simd/core/concepts.hpp>
#include <simd/core/types.hpp>
#include <simd/feature_check.hpp>
#include <simd/mask/mask.hpp>
#include <simd/operations/forward_decl.hpp>
#include <simd/registers/types.hpp>
#include <simd/vector/base.hpp>
#include <simd/vector/vector.hpp>

namespace vector_simd
{

// Fixed-size vector types
template <size_t N>
using float_v = Vector<float, N>;
template <size_t N>
using double_v = Vector<double, N>;
template <size_t N>
using int8_v = Vector<int8_t, N>;
template <size_t N>
using uint8_v = Vector<uint8_t, N>;
template <size_t N>
using int16_v = Vector<int16_t, N>;
template <size_t N>
using uint16_v = Vector<uint16_t, N>;
template <size_t N>
using int32_v = Vector<int32_t, N>;
template <size_t N>
using uint32_v = Vector<uint32_t, N>;
template <size_t N>
using int64_v = Vector<int64_t, N>;
template <size_t N>
using uint64_v = Vector<uint64_t, N>;

using float_vn = float_v<detail::native_width<float>::value>;
using double_vn = double_v<detail::native_width<double>::value>;
using int8_vn = int8_v<detail::native_width<int8_t>::value>;
using uint8_vn = uint8_v<detail::native_width<uint8_t>::value>;
using int16_vn = int16_v<detail::native_width<int16_t>::value>;
using uint16_vn = uint16_v<detail::native_width<uint16_t>::value>;
using int32_vn = int32_v<detail::native_width<int32_t>::value>;
using uint32_vn = uint32_v<detail::native_width<uint32_t>::value>;
using int64_vn = int64_v<detail::native_width<int64_t>::value>;
using uint64_vn = uint64_v<detail::native_width<uint64_t>::value>;

namespace detail
{

template <typename T, size_t N, typename Func>
SIMD_INLINE auto dispatch_simd_function(Func&& sse2_impl, Func&& avx_impl, Func&& avx2_impl,
                                        Func&& avx512_impl, Func&& fallback_impl)
{
    if constexpr (simd::FeatureDetector<simd::Feature::AVX512F>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX512F>())
        {
            return std::forward<Func>(avx512_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::AVX2>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX2>())
        {
            return std::forward<Func>(avx2_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::AVX>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX>())
        {
            return std::forward<Func>(avx_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::SSE2>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::SSE2>())
        {
            return std::forward<Func>(sse2_impl);
        }
    }

    return std::forward<Func>(fallback_impl);
}

} // namespace detail
} // namespace vector_simd

#endif /* End of include guard: SIMD_HPP_al9nn6 */
