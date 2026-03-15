#ifndef LIB_SIMD_IMPL_GENERIC_MEMORY_OPS_HPP_gene
#define LIB_SIMD_IMPL_GENERIC_MEMORY_OPS_HPP_gene

#include "simd/arch/tags.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace vector_simd::detail
{

template <typename T, size_t N>
struct memory_ops<T, N, generic_tag>
{
    using register_t = T;

    static SIMD_INLINE void load_aligned(register_t* dst, const T* src) { *dst = *src; }

    static SIMD_INLINE void load_unaligned(register_t* dst, const T* src) { *dst = *src; }

    static SIMD_INLINE void store_aligned(T* dst, const register_t* src) { *dst = *src; }

    static SIMD_INLINE void store_unaligned(T* dst, const register_t* src) { *dst = *src; }

    static SIMD_INLINE void store_nt(T* dst, const register_t* src) { *dst = *src; }

    static SIMD_INLINE void broadcast(register_t* dst, const T* val) { *dst = *val; }

    static SIMD_INLINE void set_zero(register_t* dst) { *dst = T(0); }

    static SIMD_INLINE void prefetch(const T*) {}

    static SIMD_INLINE void prefetch_nt(const T*) {}
};

} // namespace vector_simd::detail

#endif // LIB_SIMD_IMPL_GENERIC_MEMORY_OPS_HPP_gene
