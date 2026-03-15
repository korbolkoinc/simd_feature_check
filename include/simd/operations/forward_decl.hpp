#ifndef LIB_SIMD_OPERATIONS_FORWARD_HPP_g67p2r
#define LIB_SIMD_OPERATIONS_FORWARD_HPP_g67p2r

#include "simd/arch/tags.hpp"

namespace vector_simd::detail
{

template <typename T, size_t N, typename ISA = current_isa>
struct vector_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct mask_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct memory_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct math_ops;

} // namespace vector_simd::detail

#endif // End of include guard: LIB_SIMD_OPERATIONS_FORWARD_HPP_g67p2r
