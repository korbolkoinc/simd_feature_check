#ifndef LIB_SIMD_IMPL_GENERIC_VECTOR_OPS_HPP_genv
#define LIB_SIMD_IMPL_GENERIC_VECTOR_OPS_HPP_genv

#include "simd/arch/tags.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct vector_ops<T, N, generic_tag>
{
    using register_t = T;

    static SIMD_INLINE void add(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a + *b;
    }

    static SIMD_INLINE void sub(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a - *b;
    }

    static SIMD_INLINE void mul(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a * *b;
    }

    static SIMD_INLINE void div(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a / *b;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void bitwise_and(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a & *b;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void bitwise_or(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a | *b;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void bitwise_xor(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = *a ^ *b;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void bitwise_not(register_t* dst, const register_t* src)
    {
        *dst = ~*src;
    }

    static SIMD_INLINE void min(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a < *b) ? *a : *b;
    }

    static SIMD_INLINE void max(register_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a > *b) ? *a : *b;
    }

    static SIMD_INLINE void clamp(register_t* dst, const register_t* val, const register_t* lo,
                                  const register_t* hi)
    {
        register_t tmp;
        max(&tmp, val, lo);
        min(dst, &tmp, hi);
    }

    static SIMD_INLINE T reduce_add(const register_t* src) { return *src; }
    static SIMD_INLINE T reduce_min(const register_t* src) { return *src; }
    static SIMD_INLINE T reduce_max(const register_t* src) { return *src; }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void shift_left(register_t* dst, const register_t* src, int count)
    {
        *dst = static_cast<T>(*src << count);
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    static SIMD_INLINE void shift_right(register_t* dst, const register_t* src, int count)
    {
        *dst = static_cast<T>(*src >> count);
    }
};

} // namespace vector_simd::detail

#endif // LIB_SIMD_IMPL_GENERIC_VECTOR_OPS_HPP_genv
