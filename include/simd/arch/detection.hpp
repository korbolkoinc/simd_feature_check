// Todo: remove and merge with simd/arch/types.hpp

#ifndef LIB_SIMD_ARCH_DETECTION_HPP_bg9y40
#define LIB_SIMD_ARCH_DETECTION_HPP_bg9y40
#include "simd/arch/tags.hpp"
#include "simd/core/types.hpp"
#include <simd/feature_check.hpp>

namespace vector_simd
{
constexpr size_t kDefaultAlignment =
    simd::compile_time::has<simd::Feature::AVX512F>() ? kAVX512Alignment
    : simd::compile_time::has<simd::Feature::AVX>()   ? kAVXAlignment
                                                      : kSSEAlignment;
}

#endif // End of include guard: LIB_SIMD_ARCH_DETECTION_HPP_bg9y40