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
