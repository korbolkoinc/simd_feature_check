#ifndef SIMD_SUPPORT_HPP_d8nx78
#define SIMD_SUPPORT_HPP_d8nx78

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) ||                 \
    defined(__i386__) || defined(_M_IX86)
#define SIMD_ARCH_X86 1
#else
#define SIMD_ARCH_X86 0
#endif

#if defined(_MSC_VER)
#define SIMD_COMPILER_MSVC 1
#define SIMD_COMPILER_GCC 0
#define SIMD_COMPILER_CLANG 0
#include <intrin.h>
#elif defined(__clang__)
#define SIMD_COMPILER_MSVC 0
#define SIMD_COMPILER_GCC 0
#define SIMD_COMPILER_CLANG 1
#include <cpuid.h>
#include <x86intrin.h>
#elif defined(__GNUC__)
#define SIMD_COMPILER_MSVC 0
#define SIMD_COMPILER_GCC 1
#define SIMD_COMPILER_CLANG 0
#include <cpuid.h>
#include <x86intrin.h>
#else
#define SIMD_COMPILER_MSVC 0
#define SIMD_COMPILER_GCC 0
#define SIMD_COMPILER_CLANG 0
#endif

#if SIMD_COMPILER_MSVC
#define SIMD_ALWAYS_INLINE __forceinline
#define SIMD_NEVER_INLINE __declspec(noinline)
#elif SIMD_COMPILER_GCC || SIMD_COMPILER_CLANG
#define SIMD_ALWAYS_INLINE inline __attribute__((always_inline))
#define SIMD_NEVER_INLINE __attribute__((noinline))
#else
#define SIMD_ALWAYS_INLINE inline
#define SIMD_NEVER_INLINE
#endif

#define SIMD_SUPPORT_VERSION_MAJOR 0
#define SIMD_SUPPORT_VERSION_MINOR 1
#define SIMD_SUPPORT_VERSION_PATCH 0

namespace simd
{

inline int get_simd_support() { return 5; }

} // namespace simd

#endif /* End of include guard: SIMD_SUPPORT_HPP_d8nx78 */
