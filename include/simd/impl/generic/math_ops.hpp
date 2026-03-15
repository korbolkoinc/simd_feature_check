#ifndef LIB_SIMD_IMPL_GENERIC_MATH_OPS_HPP_genm
#define LIB_SIMD_IMPL_GENERIC_MATH_OPS_HPP_genm

#include "simd/arch/tags.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct math_ops<T, N, generic_tag>
{
    using register_t = T;

    static SIMD_INLINE void abs(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_signed_v<T> || std::is_floating_point_v<T>)
            *dst = (*src < T(0)) ? -(*src) : *src;
        else
            *dst = *src;
    }

    static SIMD_INLINE void sqrt(register_t* dst, const register_t* src)
    {
        *dst = static_cast<T>(std::sqrt(static_cast<double>(*src)));
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void sin(register_t* dst, const register_t* src)
    {
        *dst = std::sin(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void cos(register_t* dst, const register_t* src)
    {
        *dst = std::cos(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void tan(register_t* dst, const register_t* src)
    {
        *dst = std::tan(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void exp(register_t* dst, const register_t* src)
    {
        *dst = std::exp(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void log(register_t* dst, const register_t* src)
    {
        *dst = std::log(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rsqrt(register_t* dst, const register_t* src)
    {
        *dst = T(1) / std::sqrt(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void rcp(register_t* dst, const register_t* src)
    {
        *dst = T(1) / *src;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void floor(register_t* dst, const register_t* src)
    {
        *dst = std::floor(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void ceil(register_t* dst, const register_t* src)
    {
        *dst = std::ceil(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void round(register_t* dst, const register_t* src)
    {
        *dst = std::round(*src);
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void trunc(register_t* dst, const register_t* src)
    {
        *dst = std::trunc(*src);
    }

    static SIMD_INLINE void fmadd(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
        *dst = (*a) * (*b) + (*c);
    }

    static SIMD_INLINE void fmsub(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
        *dst = (*a) * (*b) - (*c);
    }
};

} // namespace vector_simd::detail

#endif // LIB_SIMD_IMPL_GENERIC_MATH_OPS_HPP_genm
