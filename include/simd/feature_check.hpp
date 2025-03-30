#ifndef SIMD_FEATURE_CHECK_d8nx78
#define SIMD_FEATURE_CHECK_d8nx78

#include <cstdint>

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

enum class Feature : uint32_t;
namespace detail
{
class CPUInfo final
{
private:
    enum FunctionID : uint32_t
    {
        VENDOR_INFO = 0x00000000,
        FEATURE_FLAGS = 0x00000001,
        EXTENDED_FEATURES = 0x00000007,
        EXTENDED_STATE = 0x0000000D,
        EXTENDED_FUNCTION_INFO = 0x80000001,
        PROCESSOR_BRAND_STRING_1 = 0x80000002,
        PROCESSOR_BRAND_STRING_2 = 0x80000003,
        PROCESSOR_BRAND_STRING_3 = 0x80000004,
        EXTENDED_FEATURES_FLAGS = 0x80000007,
        AMX_FEATURES = 0x00000019
    };

    struct FeatureRegisters
    {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    };

    struct CpuidResult
    {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    };

#if SIMD_ARCH_X86
    static SIMD_ALWAYS_INLINE CpuidResult cpuid(uint32_t leaf,
                                                uint32_t subleaf = 0) noexcept
    {
        CpuidResult result;

#if SIMD_COMPILER_MSVC
        int regs[4];
        __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
        result.eax = static_cast<uint32_t>(regs[0]);
        result.ebx = static_cast<uint32_t>(regs[1]);
        result.ecx = static_cast<uint32_t>(regs[2]);
        result.edx = static_cast<uint32_t>(regs[3]);
#elif SIMD_COMPILER_GCC || SIMD_COMPILER_CLANG
#if defined(__i386__) && defined(__PIC__)
        __asm__ __volatile__("xchg %%ebx, %1;"
                             "cpuid;"
                             "xchg %%ebx, %1;"
                             : "=a"(result.eax), "=r"(result.ebx),
                               "=c"(result.ecx), "=d"(result.edx)
                             : "0"(leaf), "2"(subleaf));
#else
        __cpuid_count(static_cast<int>(leaf), static_cast<int>(subleaf),
                      result.eax, result.ebx, result.ecx, result.edx);
#endif
#else
        (void)leaf;
        (void)subleaf;
        result.eax = result.ebx = result.ecx = result.edx = 0;
#endif

        return result;
    }

    static SIMD_ALWAYS_INLINE uint64_t xgetbv(uint32_t xcr) noexcept
    {
#if SIMD_COMPILER_MSVC
        return _xgetbv(xcr);
#elif (SIMD_COMPILER_GCC || SIMD_COMPILER_CLANG) && defined(__XSAVE__)
        uint32_t eax, edx;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
        return (static_cast<uint64_t>(edx) << 32) | eax;
#else
        (void)xcr;
        return 0;
#endif
    }
#else
    static SIMD_ALWAYS_INLINE CpuidResult cpuid(uint32_t leaf,
                                                uint32_t subleaf = 0) noexcept
    {
        (void)leaf;
        (void)subleaf;
        return {0, 0, 0, 0};
    }

    static SIMD_ALWAYS_INLINE uint64_t xgetbv(uint32_t xcr) noexcept
    {
        (void)xcr;
        return 0;
    }
#endif

    static SIMD_ALWAYS_INLINE bool has_bit(uint32_t value,
                                           uint32_t bit) noexcept
    {
        return (value & bit) != 0;
    }

public:
    // CPUInfo class public methods will be added here
};
} // namespace detail

enum class Feature : uint32_t
{
    NONE = 0,

    // Legacy features
    MMX = 1,
    SSE = 2,
    SSE2 = 3,
    SSE3 = 4,
    SSSE3 = 5,
    SSE41 = 6,
    SSE42 = 7,

    // AVX features
    AVX = 8,
    AVX2 = 9,
    FMA = 10,

    // Special instructions
    POPCNT = 11,
    LZCNT = 12,
    BMI1 = 13,
    BMI2 = 14,
    F16C = 15,
    MOVBE = 16,

    // AVX-512 Foundation
    AVX512F = 17,
    AVX512CD = 18,

    // AVX-512 extensions
    AVX512DQ = 19,
    AVX512BW = 20,
    AVX512VL = 21,
    AVX512IFMA = 22,
    AVX512VBMI = 23,
    AVX512VBMI2 = 24,
    AVX512VNNI = 25,
    AVX512BITALG = 26,
    AVX512VPOPCNTDQ = 27,
    AVX512VP2INTERSECT = 28,
    AVX512BF16 = 29,
    AVX512FP16 = 30,

    // Intel AMX
    AMX_TILE = 31,
    AMX_INT8 = 32,
    AMX_BF16 = 33,

    // Cryptographic extensions
    AES = 34,
    VAES = 35,
    PCLMULQDQ = 36,
    VPCLMULQDQ = 37,
    SHA = 38,

    // Misc extensions
    RDRND = 39,
    RDSEED = 40,
    ADX = 41,

    // Prefetch instructions
    PREFETCHW = 42,
    PREFETCHWT1 = 43,

    // AVX-512 additional extensions
    AVX512_4VNNIW = 44,
    AVX512_4FMAPS = 45,
    GFNI = 46,

    // Misc
    RDPID = 47,
    SGX = 48,
    CET_IBT = 49,
    CET_SS = 50,

    MAX_FEATURE = CET_SS + 1
};

inline int get_simd_support() { return 5; }

} // namespace simd

#endif /* End of include guard: SIMD_FEATURE_CHECK_d8nx78 */
