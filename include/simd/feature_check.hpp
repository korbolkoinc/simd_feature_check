#ifndef SIMD_FEATURE_CHECK_d8nx78
#define SIMD_FEATURE_CHECK_d8nx78

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

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

    struct FeatureBits
    {
        struct ECX1
        {
            static constexpr uint32_t SSE3 = 1u << 0;
            static constexpr uint32_t PCLMULQDQ = 1u << 1;
            static constexpr uint32_t DTES64 = 1u << 2;
            static constexpr uint32_t MONITOR = 1u << 3;
            static constexpr uint32_t DS_CPL = 1u << 4;
            static constexpr uint32_t VMX = 1u << 5;
            static constexpr uint32_t SMX = 1u << 6;
            static constexpr uint32_t EIST = 1u << 7;
            static constexpr uint32_t TM2 = 1u << 8;
            static constexpr uint32_t SSSE3 = 1u << 9;
            static constexpr uint32_t CNXT_ID = 1u << 10;
            static constexpr uint32_t SDBG = 1u << 11;
            static constexpr uint32_t FMA = 1u << 12;
            static constexpr uint32_t CX16 = 1u << 13;
            static constexpr uint32_t XTPR = 1u << 14;
            static constexpr uint32_t PDCM = 1u << 15;
            static constexpr uint32_t PCID = 1u << 17;
            static constexpr uint32_t DCA = 1u << 18;
            static constexpr uint32_t SSE41 = 1u << 19;
            static constexpr uint32_t SSE42 = 1u << 20;
            static constexpr uint32_t X2APIC = 1u << 21;
            static constexpr uint32_t MOVBE = 1u << 22;
            static constexpr uint32_t POPCNT = 1u << 23;
            static constexpr uint32_t TSC_DEADLINE = 1u << 24;
            static constexpr uint32_t AES = 1u << 25;
            static constexpr uint32_t XSAVE = 1u << 26;
            static constexpr uint32_t OSXSAVE = 1u << 27;
            static constexpr uint32_t AVX = 1u << 28;
            static constexpr uint32_t F16C = 1u << 29;
            static constexpr uint32_t RDRND = 1u << 30;
            static constexpr uint32_t HYPERVISOR = 1u << 31;
        };

        struct EDX1
        {
            static constexpr uint32_t FPU = 1u << 0;
            static constexpr uint32_t VME = 1u << 1;
            static constexpr uint32_t DE = 1u << 2;
            static constexpr uint32_t PSE = 1u << 3;
            static constexpr uint32_t TSC = 1u << 4;
            static constexpr uint32_t MSR = 1u << 5;
            static constexpr uint32_t PAE = 1u << 6;
            static constexpr uint32_t MCE = 1u << 7;
            static constexpr uint32_t CX8 = 1u << 8;
            static constexpr uint32_t APIC = 1u << 9;
            static constexpr uint32_t SEP = 1u << 11;
            static constexpr uint32_t MTRR = 1u << 12;
            static constexpr uint32_t PGE = 1u << 13;
            static constexpr uint32_t MCA = 1u << 14;
            static constexpr uint32_t CMOV = 1u << 15;
            static constexpr uint32_t PAT = 1u << 16;
            static constexpr uint32_t PSE36 = 1u << 17;
            static constexpr uint32_t PSN = 1u << 18;
            static constexpr uint32_t CLFSH = 1u << 19;
            static constexpr uint32_t DS = 1u << 21;
            static constexpr uint32_t ACPI = 1u << 22;
            static constexpr uint32_t MMX = 1u << 23;
            static constexpr uint32_t FXSR = 1u << 24;
            static constexpr uint32_t SSE = 1u << 25;
            static constexpr uint32_t SSE2 = 1u << 26;
            static constexpr uint32_t SS = 1u << 27;
            static constexpr uint32_t HTT = 1u << 28;
            static constexpr uint32_t TM = 1u << 29;
            static constexpr uint32_t PBE = 1u << 31;
        };

        struct EBX7
        {
            static constexpr uint32_t FSGSBASE = 1u << 0;
            static constexpr uint32_t TSC_ADJUST = 1u << 1;
            static constexpr uint32_t SGX = 1u << 2;
            static constexpr uint32_t BMI1 = 1u << 3;
            static constexpr uint32_t HLE = 1u << 4;
            static constexpr uint32_t AVX2 = 1u << 5;
            static constexpr uint32_t FDP_EXCPTN = 1u << 6;
            static constexpr uint32_t SMEP = 1u << 7;
            static constexpr uint32_t BMI2 = 1u << 8;
            static constexpr uint32_t ERMS = 1u << 9;
            static constexpr uint32_t INVPCID = 1u << 10;
            static constexpr uint32_t RTM = 1u << 11;
            static constexpr uint32_t PQM = 1u << 12;
            static constexpr uint32_t FPU_CSDS = 1u << 13;
            static constexpr uint32_t MPX = 1u << 14;
            static constexpr uint32_t PQE = 1u << 15;
            static constexpr uint32_t AVX512F = 1u << 16;
            static constexpr uint32_t AVX512DQ = 1u << 17;
            static constexpr uint32_t RDSEED = 1u << 18;
            static constexpr uint32_t ADX = 1u << 19;
            static constexpr uint32_t SMAP = 1u << 20;
            static constexpr uint32_t AVX512IFMA = 1u << 21;
            static constexpr uint32_t PCOMMIT = 1u << 22;
            static constexpr uint32_t CLFLUSHOPT = 1u << 23;
            static constexpr uint32_t CLWB = 1u << 24;
            static constexpr uint32_t INTEL_PT = 1u << 25;
            static constexpr uint32_t AVX512PF = 1u << 26;
            static constexpr uint32_t AVX512ER = 1u << 27;
            static constexpr uint32_t AVX512CD = 1u << 28;
            static constexpr uint32_t SHA = 1u << 29;
            static constexpr uint32_t AVX512BW = 1u << 30;
            static constexpr uint32_t AVX512VL = 1u << 31;
        };

        struct ECX7
        {
            static constexpr uint32_t PREFETCHWT1 = 1u << 0;
            static constexpr uint32_t AVX512VBMI = 1u << 1;
            static constexpr uint32_t UMIP = 1u << 2;
            static constexpr uint32_t PKU = 1u << 3;
            static constexpr uint32_t OSPKE = 1u << 4;
            static constexpr uint32_t WAITPKG = 1u << 5;
            static constexpr uint32_t AVX512VBMI2 = 1u << 6;
            static constexpr uint32_t CET_SS = 1u << 7;
            static constexpr uint32_t GFNI = 1u << 8;
            static constexpr uint32_t VAES = 1u << 9;
            static constexpr uint32_t VPCLMULQDQ = 1u << 10;
            static constexpr uint32_t AVX512VNNI = 1u << 11;
            static constexpr uint32_t AVX512BITALG = 1u << 12;
            static constexpr uint32_t TME_EN = 1u << 13;
            static constexpr uint32_t AVX512VPOPCNTDQ = 1u << 14;
            static constexpr uint32_t LA57 = 1u << 16;
            static constexpr uint32_t RDPID = 1u << 22;
            static constexpr uint32_t KL = 1u << 23;
            static constexpr uint32_t BUS_LOCK_DETECT = 1u << 24;
            static constexpr uint32_t CLDEMOTE = 1u << 25;
            static constexpr uint32_t MOVDIRI = 1u << 27;
            static constexpr uint32_t MOVDIR64B = 1u << 28;
            static constexpr uint32_t ENQCMD = 1u << 29;
            static constexpr uint32_t SGX_LC = 1u << 30;
            static constexpr uint32_t PKS = 1u << 31;
        };

        struct EDX7
        {
            static constexpr uint32_t SGX_KEYS = 1u << 0;
            static constexpr uint32_t AVX512_4VNNIW = 1u << 2;
            static constexpr uint32_t AVX512_4FMAPS = 1u << 3;
            static constexpr uint32_t FAST_SHORT_REP_MOV = 1u << 4;
            static constexpr uint32_t UINTR = 1u << 5;
            static constexpr uint32_t AVX512_VP2INTERSECT = 1u << 8;
            static constexpr uint32_t SRBDS_CTRL = 1u << 9;
            static constexpr uint32_t MD_CLEAR = 1u << 10;
            static constexpr uint32_t RTM_ALWAYS_ABORT = 1u << 11;
            static constexpr uint32_t TSX_FORCE_ABORT = 1u << 13;
            static constexpr uint32_t SERIALIZE = 1u << 14;
            static constexpr uint32_t HYBRID = 1u << 15;
            static constexpr uint32_t TSXLDTRK = 1u << 16;
            static constexpr uint32_t PCONFIG = 1u << 18;
            static constexpr uint32_t CET_IBT = 1u << 20;
            static constexpr uint32_t AMX_BF16 = 1u << 22;
            static constexpr uint32_t AVX512_FP16 = 1u << 23;
            static constexpr uint32_t AMX_TILE = 1u << 24;
            static constexpr uint32_t AMX_INT8 = 1u << 25;
            static constexpr uint32_t IBRS_IBPB = 1u << 26;
            static constexpr uint32_t STIBP = 1u << 27;
            static constexpr uint32_t L1D_FLUSH = 1u << 28;
            static constexpr uint32_t ARCH_CAPABILITIES = 1u << 29;
            static constexpr uint32_t CORE_CAPABILITIES = 1u << 30;
            static constexpr uint32_t SSBD = 1u << 31;
        };

        struct EAX7_1
        {
            static constexpr uint32_t AVX512_BF16 = 1u << 5;
        };

        struct ECX81
        {
            static constexpr uint32_t LAHF = 1u << 0;
            static constexpr uint32_t LZCNT = 1u << 5;
            static constexpr uint32_t ABM = 1u << 5;
            static constexpr uint32_t PREFETCHW = 1u << 8;
        };
    };

    class CpuidData
    {
    private:
        bool initialized;
        FeatureRegisters regs1;
        FeatureRegisters regs7_0;
        FeatureRegisters regs7_1;
        FeatureRegisters regs81;
        uint64_t xcr0;

    public:
        constexpr CpuidData() noexcept
            : initialized(false), regs1{0, 0, 0, 0}, regs7_0{0, 0, 0, 0},
              regs7_1{0, 0, 0, 0}, regs81{0, 0, 0, 0}, xcr0(0)
        {
        }

        void initialize() noexcept
        {
            if (!initialized)
            {
                CpuidResult res1 = cpuid(FunctionID::FEATURE_FLAGS);
                regs1.eax = res1.eax;
                regs1.ebx = res1.ebx;
                regs1.ecx = res1.ecx;
                regs1.edx = res1.edx;

                CpuidResult res7_0 = cpuid(FunctionID::EXTENDED_FEATURES, 0);
                regs7_0.eax = res7_0.eax;
                regs7_0.ebx = res7_0.ebx;
                regs7_0.ecx = res7_0.ecx;
                regs7_0.edx = res7_0.edx;

                if (res7_0.eax >= 1)
                {
                    CpuidResult res7_1 =
                        cpuid(FunctionID::EXTENDED_FEATURES, 1);
                    regs7_1.eax = res7_1.eax;
                    regs7_1.ebx = res7_1.ebx;
                    regs7_1.ecx = res7_1.ecx;
                    regs7_1.edx = res7_1.edx;
                }

                CpuidResult res81 = cpuid(FunctionID::EXTENDED_FUNCTION_INFO);
                regs81.eax = res81.eax;
                regs81.ebx = res81.ebx;
                regs81.ecx = res81.ecx;
                regs81.edx = res81.edx;

                if (has_bit(regs1.ecx, FeatureBits::ECX1::XSAVE) &&
                    has_bit(regs1.ecx, FeatureBits::ECX1::OSXSAVE))
                {
                    xcr0 = xgetbv(0);
                }

                initialized = true;
            }
        }

        constexpr const FeatureRegisters& get_regs1() const noexcept
        {
            return regs1;
        }

        constexpr const FeatureRegisters& get_regs7_0() const noexcept
        {
            return regs7_0;
        }

        constexpr const FeatureRegisters& get_regs7_1() const noexcept
        {
            return regs7_1;
        }

        constexpr const FeatureRegisters& get_regs81() const noexcept
        {
            return regs81;
        }

        constexpr uint64_t get_xcr0() const noexcept { return xcr0; }
    };

    static CpuidData& get_cpuid_data() noexcept
    {
        static CpuidData data;
        data.initialize();
        return data;
    }

    static constexpr uint64_t XCR0_SSE_STATE = 0x2;
    static constexpr uint64_t XCR0_AVX_STATE = 0x4;
    static constexpr uint64_t XCR0_OPMASK_STATE = 0x20;
    static constexpr uint64_t XCR0_ZMM_HI256_STATE = 0x40;
    static constexpr uint64_t XCR0_HI16_ZMM_STATE = 0x80;

    static constexpr uint64_t XCR0_AVX512_STATE =
        XCR0_OPMASK_STATE | XCR0_ZMM_HI256_STATE | XCR0_HI16_ZMM_STATE;
    static constexpr uint64_t XCR0_AVX_AVX512_STATE =
        XCR0_AVX_STATE | XCR0_AVX512_STATE;

public:
    static SIMD_ALWAYS_INLINE bool has_mmx() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.edx, FeatureBits::EDX1::MMX);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sse() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.edx, FeatureBits::EDX1::SSE);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sse2() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.edx, FeatureBits::EDX1::SSE2);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sse3() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::SSE3);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_ssse3() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::SSSE3);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sse41() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::SSE41);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sse42() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::SSE42);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        bool avx_supported = has_bit(regs.ecx, FeatureBits::ECX1::AVX);
        bool osxsave_supported = has_bit(regs.ecx, FeatureBits::ECX1::OSXSAVE);
        uint64_t xcr0 = get_cpuid_data().get_xcr0();
        bool avx_enabled = (xcr0 & (XCR0_SSE_STATE | XCR0_AVX_STATE)) ==
                           (XCR0_SSE_STATE | XCR0_AVX_STATE);
        return avx_supported && osxsave_supported && avx_enabled;
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx2() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx() && has_bit(regs.ebx, FeatureBits::EBX7::AVX2);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_fma() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_avx() && has_bit(regs.ecx, FeatureBits::ECX1::FMA);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_f16c() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_avx() && has_bit(regs.ecx, FeatureBits::ECX1::F16C);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512f() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        bool avx512f_supported = has_bit(regs.ebx, FeatureBits::EBX7::AVX512F);
        bool osxsave_supported = has_bit(get_cpuid_data().get_regs1().ecx,
                                         FeatureBits::ECX1::OSXSAVE);
        uint64_t xcr0 = get_cpuid_data().get_xcr0();
        bool avx512_enabled =
            (xcr0 & XCR0_AVX_AVX512_STATE) == XCR0_AVX_AVX512_STATE;
        return has_avx() && avx512f_supported && osxsave_supported &&
               avx512_enabled;
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512cd() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() && has_bit(regs.ebx, FeatureBits::EBX7::AVX512CD);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512dq() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() && has_bit(regs.ebx, FeatureBits::EBX7::AVX512DQ);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512bw() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() && has_bit(regs.ebx, FeatureBits::EBX7::AVX512BW);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512vl() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() && has_bit(regs.ebx, FeatureBits::EBX7::AVX512VL);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512ifma() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.ebx, FeatureBits::EBX7::AVX512IFMA);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512vbmi() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.ecx, FeatureBits::ECX7::AVX512VBMI);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512vbmi2() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.ecx, FeatureBits::ECX7::AVX512VBMI2);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512vnni() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.ecx, FeatureBits::ECX7::AVX512VNNI);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512bitalg() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.ecx, FeatureBits::ECX7::AVX512BITALG);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512vpopcntdq() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.ecx, FeatureBits::ECX7::AVX512VPOPCNTDQ);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512fp16() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.edx, FeatureBits::EDX7::AVX512_FP16);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512bf16() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_1();
        return has_avx512f() &&
               has_bit(regs.eax, FeatureBits::EAX7_1::AVX512_BF16);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512vp2intersect() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.edx, FeatureBits::EDX7::AVX512_VP2INTERSECT);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512_4vnniw() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.edx, FeatureBits::EDX7::AVX512_4VNNIW);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_avx512_4fmaps() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_avx512f() &&
               has_bit(regs.edx, FeatureBits::EDX7::AVX512_4FMAPS);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_amx_tile() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.edx, FeatureBits::EDX7::AMX_TILE);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_amx_int8() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_amx_tile() && has_bit(regs.edx, FeatureBits::EDX7::AMX_INT8);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_amx_bf16() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_amx_tile() && has_bit(regs.edx, FeatureBits::EDX7::AMX_BF16);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_popcnt() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::POPCNT);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_lzcnt() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs81();
        return has_bit(regs.ecx, FeatureBits::ECX81::LZCNT);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_bmi1() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ebx, FeatureBits::EBX7::BMI1);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_bmi2() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ebx, FeatureBits::EBX7::BMI2);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_movbe() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::MOVBE);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_prefetchw() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs81();
        return has_bit(regs.ecx, FeatureBits::ECX81::PREFETCHW);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_prefetchwt1() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ecx, FeatureBits::ECX7::PREFETCHWT1);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_aes() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::AES);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_vaes() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ecx, FeatureBits::ECX7::VAES);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_pclmulqdq() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::PCLMULQDQ);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_vpclmulqdq() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ecx, FeatureBits::ECX7::VPCLMULQDQ);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sha() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ebx, FeatureBits::EBX7::SHA);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_rdrnd() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs1();
        return has_bit(regs.ecx, FeatureBits::ECX1::RDRND);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_rdseed() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ebx, FeatureBits::EBX7::RDSEED);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_adx() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ebx, FeatureBits::EBX7::ADX);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_sgx() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ebx, FeatureBits::EBX7::SGX);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_rdpid() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ecx, FeatureBits::ECX7::RDPID);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_cet_ibt() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.edx, FeatureBits::EDX7::CET_IBT);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_cet_ss() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ecx, FeatureBits::ECX7::CET_SS);
#else
        return false;
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_gfni() noexcept
    {
#if SIMD_ARCH_X86
        const auto& regs = get_cpuid_data().get_regs7_0();
        return has_bit(regs.ecx, FeatureBits::ECX7::GFNI);
#else
        return false;
#endif
    }

    static std::string get_vendor_string() noexcept
    {
#if SIMD_ARCH_X86
        char vendor[13] = {0};
        CpuidResult res = cpuid(FunctionID::VENDOR_INFO);

        std::memcpy(vendor, &res.ebx, 4);
        std::memcpy(vendor + 4, &res.edx, 4);
        std::memcpy(vendor + 8, &res.ecx, 4);

        return std::string(vendor);
#else
        return "Unknown";
#endif
    }

    static std::optional<std::array<int, 3>> get_processor_model() noexcept
    {
#if SIMD_ARCH_X86
        CpuidResult res = cpuid(FunctionID::FEATURE_FLAGS);

        int family_id = (res.eax >> 8) & 0xF;
        int model_id = (res.eax >> 4) & 0xF;
        int stepping_id = res.eax & 0xF;

        int extended_family = 0;
        int extended_model = 0;

        if (family_id == 0xF)
        {
            extended_family = ((res.eax >> 20) & 0xFF);
            family_id += extended_family;
        }

        if (family_id == 0xF || family_id == 0x6)
        {
            extended_model = ((res.eax >> 16) & 0xF);
            model_id |= (extended_model << 4);
        }

        return std::array<int, 3>{family_id, model_id, stepping_id};
#else
        return std::nullopt;
#endif
    }

    static std::string get_processor_brand_string() noexcept
    {
#if SIMD_ARCH_X86
        char brand[49] = {0};
        CpuidResult res1 = cpuid(FunctionID::PROCESSOR_BRAND_STRING_1);
        CpuidResult res2 = cpuid(FunctionID::PROCESSOR_BRAND_STRING_2);
        CpuidResult res3 = cpuid(FunctionID::PROCESSOR_BRAND_STRING_3);

        std::memcpy(brand, &res1.eax, 4);
        std::memcpy(brand + 4, &res1.ebx, 4);
        std::memcpy(brand + 8, &res1.ecx, 4);
        std::memcpy(brand + 12, &res1.edx, 4);

        std::memcpy(brand + 16, &res2.eax, 4);
        std::memcpy(brand + 20, &res2.ebx, 4);
        std::memcpy(brand + 24, &res2.ecx, 4);
        std::memcpy(brand + 28, &res2.edx, 4);

        std::memcpy(brand + 32, &res3.eax, 4);
        std::memcpy(brand + 36, &res3.ebx, 4);
        std::memcpy(brand + 40, &res3.ecx, 4);
        std::memcpy(brand + 44, &res3.edx, 4);

        return std::string(brand);
#else
        return "Unknown";
#endif
    }

    static SIMD_ALWAYS_INLINE bool has_feature(Feature feature) noexcept
    {
        switch (feature)
        {
            case Feature::NONE:
                return true;
            case Feature::MMX:
                return has_mmx();
            case Feature::SSE:
                return has_sse();
            case Feature::SSE2:
                return has_sse2();
            case Feature::SSE3:
                return has_sse3();
            case Feature::SSSE3:
                return has_ssse3();
            case Feature::SSE41:
                return has_sse41();
            case Feature::SSE42:
                return has_sse42();
            case Feature::AVX:
                return has_avx();
            case Feature::AVX2:
                return has_avx2();
            case Feature::FMA:
                return has_fma();
            case Feature::POPCNT:
                return has_popcnt();
            case Feature::LZCNT:
                return has_lzcnt();
            case Feature::BMI1:
                return has_bmi1();
            case Feature::BMI2:
                return has_bmi2();
            case Feature::F16C:
                return has_f16c();
            case Feature::MOVBE:
                return has_movbe();
            case Feature::AVX512F:
                return has_avx512f();
            case Feature::AVX512CD:
                return has_avx512cd();
            case Feature::AVX512DQ:
                return has_avx512dq();
            case Feature::AVX512BW:
                return has_avx512bw();
            case Feature::AVX512VL:
                return has_avx512vl();
            case Feature::AVX512IFMA:
                return has_avx512ifma();
            case Feature::AVX512VBMI:
                return has_avx512vbmi();
            case Feature::AVX512VBMI2:
                return has_avx512vbmi2();
            case Feature::AVX512VNNI:
                return has_avx512vnni();
            case Feature::AVX512BITALG:
                return has_avx512bitalg();
            case Feature::AVX512VPOPCNTDQ:
                return has_avx512vpopcntdq();
            case Feature::AVX512VP2INTERSECT:
                return has_avx512vp2intersect();
            case Feature::AVX512BF16:
                return has_avx512bf16();
            case Feature::AVX512FP16:
                return has_avx512fp16();
            case Feature::AMX_TILE:
                return has_amx_tile();
            case Feature::AMX_INT8:
                return has_amx_int8();
            case Feature::AMX_BF16:
                return has_amx_bf16();
            case Feature::AES:
                return has_aes();
            case Feature::VAES:
                return has_vaes();
            case Feature::PCLMULQDQ:
                return has_pclmulqdq();
            case Feature::VPCLMULQDQ:
                return has_vpclmulqdq();
            case Feature::SHA:
                return has_sha();
            case Feature::RDRND:
                return has_rdrnd();
            case Feature::RDSEED:
                return has_rdseed();
            case Feature::ADX:
                return has_adx();
            case Feature::PREFETCHW:
                return has_prefetchw();
            case Feature::PREFETCHWT1:
                return has_prefetchwt1();
            case Feature::AVX512_4VNNIW:
                return has_avx512_4vnniw();
            case Feature::AVX512_4FMAPS:
                return has_avx512_4fmaps();
            case Feature::GFNI:
                return has_gfni();
            case Feature::RDPID:
                return has_rdpid();
            case Feature::SGX:
                return has_sgx();
            case Feature::CET_IBT:
                return has_cet_ibt();
            case Feature::CET_SS:
                return has_cet_ss();
            default:
                return false;
        }
    }
};

class CpuidVersionInfo final
{
private:
    uint32_t stepping_id_;
    uint32_t model_id_;
    uint32_t family_id_;
    uint32_t processor_type_;
    uint32_t extended_model_id_;
    uint32_t extended_family_id_;

public:
    explicit CpuidVersionInfo(uint32_t raw_eax) noexcept
        : stepping_id_((raw_eax) & 0xF), model_id_((raw_eax >> 4) & 0xF),
          family_id_((raw_eax >> 8) & 0xF),
          processor_type_((raw_eax >> 12) & 0x3),
          extended_model_id_((raw_eax >> 16) & 0xF),
          extended_family_id_((raw_eax >> 20) & 0xFF)
    {
    }

    SIMD_ALWAYS_INLINE uint32_t stepping_id() const noexcept
    {
        return stepping_id_;
    }

    SIMD_ALWAYS_INLINE uint32_t model_id() const noexcept
    {
        if (family_id_ == 0x6 || family_id_ == 0xF)
        {
            return (extended_model_id_ << 4) | model_id_;
        }
        return model_id_;
    }

    SIMD_ALWAYS_INLINE uint32_t family_id() const noexcept
    {
        if (family_id_ == 0xF)
        {
            return family_id_ + extended_family_id_;
        }
        return family_id_;
    }

    SIMD_ALWAYS_INLINE uint32_t processor_type() const noexcept
    {
        return processor_type_;
    }
};
} // namespace detail

// Template utilities for compile-time feature detection
#if defined(__MMX__)
#define SIMD_HAS_MMX 1
#else
#define SIMD_HAS_MMX 0
#endif

#if defined(__SSE__)
#define SIMD_HAS_SSE 1
#else
#define SIMD_HAS_SSE 0
#endif

#if defined(__SSE2__)
#define SIMD_HAS_SSE2 1
#else
#define SIMD_HAS_SSE2 0
#endif

#if defined(__SSE3__)
#define SIMD_HAS_SSE3 1
#else
#define SIMD_HAS_SSE3 0
#endif

#if defined(__SSSE3__)
#define SIMD_HAS_SSSE3 1
#else
#define SIMD_HAS_SSSE3 0
#endif

#if defined(__SSE4_1__)
#define SIMD_HAS_SSE41 1
#else
#define SIMD_HAS_SSE41 0
#endif

#if defined(__SSE4_2__)
#define SIMD_HAS_SSE42 1
#else
#define SIMD_HAS_SSE42 0
#endif

#if defined(__AVX__)
#define SIMD_HAS_AVX 1
#else
#define SIMD_HAS_AVX 0
#endif

#if defined(__AVX2__)
#define SIMD_HAS_AVX2 1
#else
#define SIMD_HAS_AVX2 0
#endif

#if defined(__FMA__)
#define SIMD_HAS_FMA 1
#else
#define SIMD_HAS_FMA 0
#endif

#if defined(__F16C__)
#define SIMD_HAS_F16C 1
#else
#define SIMD_HAS_F16C 0
#endif

#if defined(__POPCNT__)
#define SIMD_HAS_POPCNT 1
#else
#define SIMD_HAS_POPCNT 0
#endif

#if defined(__LZCNT__)
#define SIMD_HAS_LZCNT 1
#else
#define SIMD_HAS_LZCNT 0
#endif

#if defined(__BMI__)
#define SIMD_HAS_BMI1 1
#else
#define SIMD_HAS_BMI1 0
#endif

#if defined(__BMI2__)
#define SIMD_HAS_BMI2 1
#else
#define SIMD_HAS_BMI2 0
#endif

#if defined(__MOVBE__)
#define SIMD_HAS_MOVBE 1
#else
#define SIMD_HAS_MOVBE 0
#endif

#if defined(__AVX512F__)
#define SIMD_HAS_AVX512F 1
#else
#define SIMD_HAS_AVX512F 0
#endif

#if defined(__AVX512CD__)
#define SIMD_HAS_AVX512CD 1
#else
#define SIMD_HAS_AVX512CD 0
#endif

#if defined(__AVX512DQ__)
#define SIMD_HAS_AVX512DQ 1
#else
#define SIMD_HAS_AVX512DQ 0
#endif

#if defined(__AVX512BW__)
#define SIMD_HAS_AVX512BW 1
#else
#define SIMD_HAS_AVX512BW 0
#endif

#if defined(__AVX512VL__)
#define SIMD_HAS_AVX512VL 1
#else
#define SIMD_HAS_AVX512VL 0
#endif

#if defined(__AVX512IFMA__)
#define SIMD_HAS_AVX512IFMA 1
#else
#define SIMD_HAS_AVX512IFMA 0
#endif

#if defined(__AVX512VBMI__)
#define SIMD_HAS_AVX512VBMI 1
#else
#define SIMD_HAS_AVX512VBMI 0
#endif

#if defined(__AVX512VBMI2__)
#define SIMD_HAS_AVX512VBMI2 1
#else
#define SIMD_HAS_AVX512VBMI2 0
#endif

#if defined(__AVX512VNNI__)
#define SIMD_HAS_AVX512VNNI 1
#else
#define SIMD_HAS_AVX512VNNI 0
#endif

#if defined(__AVX512BITALG__)
#define SIMD_HAS_AVX512BITALG 1
#else
#define SIMD_HAS_AVX512BITALG 0
#endif

#if defined(__AVX512VPOPCNTDQ__)
#define SIMD_HAS_AVX512VPOPCNTDQ 1
#else
#define SIMD_HAS_AVX512VPOPCNTDQ 0
#endif

#if defined(__AVX512VP2INTERSECT__)
#define SIMD_HAS_AVX512VP2INTERSECT 1
#else
#define SIMD_HAS_AVX512VP2INTERSECT 0
#endif

#if defined(__AVX512BF16__)
#define SIMD_HAS_AVX512BF16 1
#else
#define SIMD_HAS_AVX512BF16 0
#endif

#if defined(__AVX512FP16__)
#define SIMD_HAS_AVX512FP16 1
#else
#define SIMD_HAS_AVX512FP16 0
#endif

#if defined(__AMX_TILE__)
#define SIMD_HAS_AMX_TILE 1
#else
#define SIMD_HAS_AMX_TILE 0
#endif

#if defined(__AMX_INT8__)
#define SIMD_HAS_AMX_INT8 1
#else
#define SIMD_HAS_AMX_INT8 0
#endif

#if defined(__AMX_BF16__)
#define SIMD_HAS_AMX_BF16 1
#else
#define SIMD_HAS_AMX_BF16 0
#endif

#if defined(__AES__)
#define SIMD_HAS_AES 1
#else
#define SIMD_HAS_AES 0
#endif

#if defined(__VAES__)
#define SIMD_HAS_VAES 1
#else
#define SIMD_HAS_VAES 0
#endif

#if defined(__PCLMUL__)
#define SIMD_HAS_PCLMULQDQ 1
#else
#define SIMD_HAS_PCLMULQDQ 0
#endif

#if defined(__VPCLMULQDQ__)
#define SIMD_HAS_VPCLMULQDQ 1
#else
#define SIMD_HAS_VPCLMULQDQ 0
#endif

#if defined(__SHA__)
#define SIMD_HAS_SHA 1
#else
#define SIMD_HAS_SHA 0
#endif

#if defined(__RDRND__)
#define SIMD_HAS_RDRND 1
#else
#define SIMD_HAS_RDRND 0
#endif

#if defined(__RDSEED__)
#define SIMD_HAS_RDSEED 1
#else
#define SIMD_HAS_RDSEED 0
#endif

#if defined(__ADX__)
#define SIMD_HAS_ADX 1
#else
#define SIMD_HAS_ADX 0
#endif

#if defined(__PREFETCHWT1__)
#define SIMD_HAS_PREFETCHWT1 1
#else
#define SIMD_HAS_PREFETCHWT1 0
#endif

#if defined(__AVX512_4VNNIW__)
#define SIMD_HAS_AVX512_4VNNIW 1
#else
#define SIMD_HAS_AVX512_4VNNIW 0
#endif

#if defined(__AVX512_4FMAPS__)
#define SIMD_HAS_AVX512_4FMAPS 1
#else
#define SIMD_HAS_AVX512_4FMAPS 0
#endif

#if defined(__GFNI__)
#define SIMD_HAS_GFNI 1
#else
#define SIMD_HAS_GFNI 0
#endif

#if defined(__RDPID__)
#define SIMD_HAS_RDPID 1
#else
#define SIMD_HAS_RDPID 0
#endif

#if defined(__SGX__)
#define SIMD_HAS_SGX 1
#else
#define SIMD_HAS_SGX 0
#endif

#if defined(__CET_IBT__)
#define SIMD_HAS_CET_IBT 1
#else
#define SIMD_HAS_CET_IBT 0
#endif

#if defined(__CET_SS__)
#define SIMD_HAS_CET_SS 1
#else
#define SIMD_HAS_CET_SS 0
#endif

namespace compile_time
{

constexpr bool mmx = SIMD_HAS_MMX != 0;
constexpr bool sse = SIMD_HAS_SSE != 0;
constexpr bool sse2 = SIMD_HAS_SSE2 != 0;
constexpr bool sse3 = SIMD_HAS_SSE3 != 0;
constexpr bool ssse3 = SIMD_HAS_SSSE3 != 0;
constexpr bool sse41 = SIMD_HAS_SSE41 != 0;
constexpr bool sse42 = SIMD_HAS_SSE42 != 0;
constexpr bool avx = SIMD_HAS_AVX != 0;
constexpr bool avx2 = SIMD_HAS_AVX2 != 0;
constexpr bool fma = SIMD_HAS_FMA != 0;
constexpr bool f16c = SIMD_HAS_F16C != 0;
constexpr bool popcnt = SIMD_HAS_POPCNT != 0;
constexpr bool lzcnt = SIMD_HAS_LZCNT != 0;
constexpr bool bmi1 = SIMD_HAS_BMI1 != 0;
constexpr bool bmi2 = SIMD_HAS_BMI2 != 0;
constexpr bool movbe = SIMD_HAS_MOVBE != 0;
constexpr bool avx512f = SIMD_HAS_AVX512F != 0;
constexpr bool avx512cd = SIMD_HAS_AVX512CD != 0;
constexpr bool avx512dq = SIMD_HAS_AVX512DQ != 0;
constexpr bool avx512bw = SIMD_HAS_AVX512BW != 0;
constexpr bool avx512vl = SIMD_HAS_AVX512VL != 0;
constexpr bool avx512ifma = SIMD_HAS_AVX512IFMA != 0;
constexpr bool avx512vbmi = SIMD_HAS_AVX512VBMI != 0;
constexpr bool avx512vbmi2 = SIMD_HAS_AVX512VBMI2 != 0;
constexpr bool avx512vnni = SIMD_HAS_AVX512VNNI != 0;
constexpr bool avx512bitalg = SIMD_HAS_AVX512BITALG != 0;
constexpr bool avx512vpopcntdq = SIMD_HAS_AVX512VPOPCNTDQ != 0;
constexpr bool avx512vp2intersect = SIMD_HAS_AVX512VP2INTERSECT != 0;
constexpr bool avx512bf16 = SIMD_HAS_AVX512BF16 != 0;
constexpr bool avx512fp16 = SIMD_HAS_AVX512FP16 != 0;
constexpr bool amx_tile = SIMD_HAS_AMX_TILE != 0;
constexpr bool amx_int8 = SIMD_HAS_AMX_INT8 != 0;
constexpr bool amx_bf16 = SIMD_HAS_AMX_BF16 != 0;
constexpr bool aes = SIMD_HAS_AES != 0;
constexpr bool vaes = SIMD_HAS_VAES != 0;
constexpr bool pclmulqdq = SIMD_HAS_PCLMULQDQ != 0;
constexpr bool vpclmulqdq = SIMD_HAS_VPCLMULQDQ != 0;
constexpr bool sha = SIMD_HAS_SHA != 0;
constexpr bool rdrnd = SIMD_HAS_RDRND != 0;
constexpr bool rdseed = SIMD_HAS_RDSEED != 0;
constexpr bool adx = SIMD_HAS_ADX != 0;
constexpr bool prefetchwt1 = SIMD_HAS_PREFETCHWT1 != 0;
constexpr bool avx512_4vnniw = SIMD_HAS_AVX512_4VNNIW != 0;
constexpr bool avx512_4fmaps = SIMD_HAS_AVX512_4FMAPS != 0;
constexpr bool gfni = SIMD_HAS_GFNI != 0;
constexpr bool rdpid = SIMD_HAS_RDPID != 0;
constexpr bool sgx = SIMD_HAS_SGX != 0;
constexpr bool cet_ibt = SIMD_HAS_CET_IBT != 0;
constexpr bool cet_ss = SIMD_HAS_CET_SS != 0;

template <Feature F>
static constexpr bool has() noexcept
{
    if constexpr (F == Feature::MMX)
        return mmx;
    else if constexpr (F == Feature::SSE)
        return sse;
    else if constexpr (F == Feature::SSE2)
        return sse2;
    else if constexpr (F == Feature::SSE3)
        return sse3;
    else if constexpr (F == Feature::SSSE3)
        return ssse3;
    else if constexpr (F == Feature::SSE41)
        return sse41;
    else if constexpr (F == Feature::SSE42)
        return sse42;
    else if constexpr (F == Feature::AVX)
        return avx;
    else if constexpr (F == Feature::AVX2)
        return avx2;
    else if constexpr (F == Feature::FMA)
        return fma;
    else if constexpr (F == Feature::F16C)
        return f16c;
    else if constexpr (F == Feature::POPCNT)
        return popcnt;
    else if constexpr (F == Feature::LZCNT)
        return lzcnt;
    else if constexpr (F == Feature::BMI1)
        return bmi1;
    else if constexpr (F == Feature::BMI2)
        return bmi2;
    else if constexpr (F == Feature::MOVBE)
        return movbe;
    else if constexpr (F == Feature::AVX512F)
        return avx512f;
    else if constexpr (F == Feature::AVX512CD)
        return avx512cd;
    else if constexpr (F == Feature::AVX512DQ)
        return avx512dq;
    else if constexpr (F == Feature::AVX512BW)
        return avx512bw;
    else if constexpr (F == Feature::AVX512VL)
        return avx512vl;
    else if constexpr (F == Feature::AVX512IFMA)
        return avx512ifma;
    else if constexpr (F == Feature::AVX512VBMI)
        return avx512vbmi;
    else if constexpr (F == Feature::AVX512VBMI2)
        return avx512vbmi2;
    else if constexpr (F == Feature::AVX512VNNI)
        return avx512vnni;
    else if constexpr (F == Feature::AVX512BITALG)
        return avx512bitalg;
    else if constexpr (F == Feature::AVX512VPOPCNTDQ)
        return avx512vpopcntdq;
    else if constexpr (F == Feature::AVX512VP2INTERSECT)
        return avx512vp2intersect;
    else if constexpr (F == Feature::AVX512BF16)
        return avx512bf16;
    else if constexpr (F == Feature::AVX512FP16)
        return avx512fp16;
    else if constexpr (F == Feature::AMX_TILE)
        return amx_tile;
    else if constexpr (F == Feature::AMX_INT8)
        return amx_int8;
    else if constexpr (F == Feature::AMX_BF16)
        return amx_bf16;
    else if constexpr (F == Feature::AES)
        return aes;
    else if constexpr (F == Feature::VAES)
        return vaes;
    else if constexpr (F == Feature::PCLMULQDQ)
        return pclmulqdq;
    else if constexpr (F == Feature::VPCLMULQDQ)
        return vpclmulqdq;
    else if constexpr (F == Feature::SHA)
        return sha;
    else if constexpr (F == Feature::RDRND)
        return rdrnd;
    else if constexpr (F == Feature::RDSEED)
        return rdseed;
    else if constexpr (F == Feature::ADX)
        return adx;
    else if constexpr (F == Feature::PREFETCHW)
        return false;
    else if constexpr (F == Feature::PREFETCHWT1)
        return prefetchwt1;
    else if constexpr (F == Feature::AVX512_4VNNIW)
        return avx512_4vnniw;
    else if constexpr (F == Feature::AVX512_4FMAPS)
        return avx512_4fmaps;
    else if constexpr (F == Feature::GFNI)
        return gfni;
    else if constexpr (F == Feature::RDPID)
        return rdpid;
    else if constexpr (F == Feature::SGX)
        return sgx;
    else if constexpr (F == Feature::CET_IBT)
        return cet_ibt;
    else if constexpr (F == Feature::CET_SS)
        return cet_ss;
    else if constexpr (F == Feature::NONE)
        return true;
    else
        return false;
}

} // namespace compile_time

inline int get_simd_support() { return 5; }

} // namespace simd

#endif /* End of include guard: SIMD_FEATURE_CHECK_d8nx78 */
