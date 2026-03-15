#ifndef LIB_SIMD_IMPL_GENERIC_MASK_OPS_HPP_genk
#define LIB_SIMD_IMPL_GENERIC_MASK_OPS_HPP_genk

#include "simd/arch/tags.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#include <cstddef>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct mask_ops<T, N, generic_tag>
{
    using register_t = T;
    using mask_t = T;

    static SIMD_INLINE void cmp_eq(mask_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a == *b) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void cmp_neq(mask_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a != *b) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void cmp_lt(mask_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a < *b) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void cmp_le(mask_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a <= *b) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void cmp_gt(mask_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a > *b) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void cmp_ge(mask_t* dst, const register_t* a, const register_t* b)
    {
        *dst = (*a >= *b) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void logical_and(mask_t* dst, const mask_t* a, const mask_t* b)
    {
        if constexpr (std::is_integral_v<T>)
            *dst = *a & *b;
        else
            *dst = (*a != T(0) && *b != T(0)) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void logical_or(mask_t* dst, const mask_t* a, const mask_t* b)
    {
        if constexpr (std::is_integral_v<T>)
            *dst = *a | *b;
        else
            *dst = (*a != T(0) || *b != T(0)) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE void logical_not(mask_t* dst, const mask_t* src)
    {
        if constexpr (std::is_integral_v<T>)
            *dst = ~*src;
        else
            *dst = (*src == T(0)) ? static_cast<T>(~T(0)) : T(0);
    }

    static SIMD_INLINE bool any(const mask_t* src) { return *src != T(0); }

    static SIMD_INLINE bool all(const mask_t* src) { return *src != T(0); }

    static SIMD_INLINE int count(const mask_t* src) { return *src != T(0) ? 1 : 0; }
};

} // namespace vector_simd::detail

#endif // LIB_SIMD_IMPL_GENERIC_MASK_OPS_HPP_genk
