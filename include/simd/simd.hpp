#ifndef SIMD_HPP_al9nn6
#define SIMD_HPP_al9nn6

#include <array>
#include <simd/common.hpp>
#include <simd/feature_check.hpp>

namespace vector_simd
{

constexpr int kVersionMajor = 0;
constexpr int kVersionMinor = 1;
constexpr int kVersionPatch = 0;

constexpr size_t kSSEAlignment = 16;
constexpr size_t kAVXAlignment = 32;
constexpr size_t kAVX512Alignment = 64;

constexpr size_t kDefaultAlignment =
    simd::compile_time::has<simd::Feature::AVX512F>() ? kAVX512Alignment
    : simd::compile_time::has<simd::Feature::AVX>()   ? kAVXAlignment
                                                      : kSSEAlignment;

template <typename T>
concept SimdArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int8_t> ||
    std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, uint64_t>;

template <typename T>
concept SimdFloat = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept SimdInteger = SimdArithmetic<T> && !SimdFloat<T>;

template <typename T>
concept SimdSignedInteger = std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                            std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

template <typename T>
concept SimdUnsignedInteger = std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

// Forward declarations
template <SimdArithmetic T, size_t N>
class Vector;

template <SimdArithmetic T, size_t N>
class Mask;

namespace detail
{

struct generic_tag
{
};

struct sse2_tag : generic_tag
{
};

struct sse3_tag : sse2_tag
{
};

struct ssse3_tag : sse3_tag
{
};

struct sse4_1_tag : ssse3_tag
{
};

struct sse4_2_tag : sse4_1_tag
{
};

struct avx_tag : sse4_2_tag
{
};

struct avx2_tag : avx_tag
{
};

struct avx512_tag : avx2_tag
{
};

struct neon_tag : generic_tag
{
};

struct wasm_simd_tag : generic_tag
{
};

struct best_available_tag
{
    using type = std::conditional_t<
        simd::compile_time::has<simd::Feature::AVX512F>(), avx512_tag,
        std::conditional_t<
            simd::compile_time::has<simd::Feature::AVX2>(), avx2_tag,
            std::conditional_t<
                simd::compile_time::has<simd::Feature::AVX>(), avx_tag,
                std::conditional_t<
                    simd::compile_time::has<simd::Feature::SSE42>(), sse4_2_tag,
                    std::conditional_t<
                        simd::compile_time::has<simd::Feature::SSE41>(), sse4_1_tag,
                        std::conditional_t<
                            simd::compile_time::has<simd::Feature::SSSE3>(), ssse3_tag,
                            std::conditional_t<
                                simd::compile_time::has<simd::Feature::SSE3>(), sse3_tag,
                                std::conditional_t<simd::compile_time::has<simd::Feature::SSE2>(),
                                                   sse2_tag, generic_tag
                                >
                            >
                        >
                    >
                >
            >
        >
    >;
};

using current_isa = typename best_available_tag::type;

template <typename T, typename ISA>
struct simd_width;

template <>
struct simd_width<float, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<double, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<int8_t, sse2_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint8_t, sse2_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int16_t, sse2_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint16_t, sse2_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int32_t, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint32_t, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int64_t, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<uint64_t, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<float, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<double, avx_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int8_t, avx_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<uint8_t, avx_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<int16_t, avx_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint16_t, avx_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int32_t, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint32_t, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int64_t, avx_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint64_t, avx_tag>
{
    static constexpr size_t value = 4;
};

template <typename T>
struct simd_width<T, avx2_tag> : simd_width<T, avx_tag>
{
};

template <>
struct simd_width<float, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<double, avx512_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int8_t, avx512_tag>
{
    static constexpr size_t value = 64;
};

template <>
struct simd_width<uint8_t, avx512_tag>
{
    static constexpr size_t value = 64;
};

template <>
struct simd_width<int16_t, avx512_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<uint16_t, avx512_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<int32_t, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint32_t, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int64_t, avx512_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint64_t, avx512_tag>
{
    static constexpr size_t value = 8;
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template <>
struct simd_width<float, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int8_t, neon_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint8_t, neon_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int16_t, neon_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint16_t, neon_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int32_t, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint32_t, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int64_t, neon_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<uint64_t, neon_tag>
{
    static constexpr size_t value = 2;
};

#endif

// Generic fallback for unsupported architectures

template <typename T>
struct simd_width<T, generic_tag>
{
    static constexpr size_t value = 1;
};

template <typename T>
struct native_width
{
    static constexpr size_t value = simd_width<T, current_isa>::value;
};

template <typename T, typename ISA>
struct register_type;

template <>
struct register_type<float, sse2_tag>
{
    using type = __m128;
};

template <>
struct register_type<double, sse2_tag>
{
    using type = __m128d;
};

template <>
struct register_type<int8_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint8_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int16_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint16_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int32_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint32_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int64_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint64_t, sse2_tag>
{
    using type = __m128i;
};

// AVX register type mappings

template <>
struct register_type<float, avx_tag>
{
    using type = __m256;
};

template <>
struct register_type<double, avx_tag>
{
    using type = __m256d;
};

template <>
struct register_type<int8_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint8_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int16_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint16_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int32_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint32_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int64_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint64_t, avx_tag>
{
    using type = __m256i;
};

// Same register types for AVX2

template <typename T>
struct register_type<T, avx2_tag> : register_type<T, avx_tag>
{
};

#ifdef __AVX512F__

template <>
struct register_type<float, avx512_tag>
{
    using type = __m512;
};

template <>
struct register_type<double, avx512_tag>
{
    using type = __m512d;
};

template <>
struct register_type<int8_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint8_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int16_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint16_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int32_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint32_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int64_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint64_t, avx512_tag>
{
    using type = __m512i;
};

#endif

// NEON register type mappings (if needed)

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

template <>
struct register_type<float, neon_tag>
{
    using type = float32x4_t;
};

template <>
struct register_type<int8_t, neon_tag>
{
    using type = int8x16_t;
};

template <>
struct register_type<uint8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct register_type<int16_t, neon_tag>
{
    using type = int16x8_t;
};

template <>
struct register_type<uint16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct register_type<int32_t, neon_tag>
{
    using type = int32x4_t;
};

template <>
struct register_type<uint32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct register_type<int64_t, neon_tag>
{
    using type = int64x2_t;
};

template <>
struct register_type<uint64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct register_type<double, neon_tag>
{
#ifdef __aarch64__
    using type = float64x2_t;
#else
    using type = double;
#endif
};

#endif

template <typename T>
struct register_type<T, generic_tag>
{
    using type = T;
};

template <typename T, typename ISA>
struct mask_register_type;

template <typename T>
struct mask_register_type<T, sse2_tag>
{
    using type = typename register_type<T, sse2_tag>::type;
};

template <typename T>
struct mask_register_type<T, avx_tag>
{
    using type = typename register_type<T, avx_tag>::type;
};

template <typename T>
struct mask_register_type<T, avx2_tag> : mask_register_type<T, avx_tag>
{
};

#ifdef __AVX512F__

template <>
struct mask_register_type<float, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct mask_register_type<int8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct mask_register_type<uint8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct mask_register_type<int16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct mask_register_type<uint16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct mask_register_type<int32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct mask_register_type<uint32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct mask_register_type<int64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct mask_register_type<uint64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct mask_register_type<double, neon_tag>
{
#ifdef __aarch64__
    using type = uint64x2_t;
#else
    using type = bool;
#endif
};

#endif

template <typename T>
struct mask_register_type<T, generic_tag>
{
    using type = bool;
};

// forward declarations for vector and mask operations

template <typename T, size_t N, typename ISA = current_isa>
struct vector_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct mask_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct memory_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct math_ops;

template <typename Derived, typename T, size_t N>
class vector_base
{
protected:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

public:
    using value_type = T;
    using size_type = size_t;
    static constexpr size_t size_value = N;

    SIMD_INLINE T operator[](size_t i) const
    {
        assert(i < N && "Index out of bounds");
        return derived().extract(i);
    }

    class reference
    {
    private:
        Derived& vec;
        size_t idx;

    public:
        reference(Derived& v, size_t i) : vec(v), idx(i) {}

        operator T() const { return vec.extract(idx); }

        reference& operator=(T value)
        {
            vec.insert(idx, value);
            return *this;
        }

        reference& operator+=(T value)
        {
            vec.insert(idx, vec.extract(idx) + value);
            return *this;
        }

        reference& operator-=(T value)
        {
            vec.insert(idx, vec.extract(idx) - value);
            return *this;
        }

        reference& operator*=(T value)
        {
            vec.insert(idx, vec.extract(idx) * value);
            return *this;
        }

        reference& operator/=(T value)
        {
            vec.insert(idx, vec.extract(idx) / value);
            return *this;
        }
    };

    SIMD_INLINE reference operator[](size_t i)
    {
        assert(i < N && "Index out of bounds");
        return reference(derived(), i);
    }

    auto _data() { return derived().data(); }
    auto _data() const { return derived().data(); }
};

} // namespace detail

template <SimdArithmetic T, size_t N>
class alignas(kDefaultAlignment) Vector : public detail::vector_base<Vector<T, N>, T, N>
{
private:
    using ops = detail::vector_ops<T, N>;
    using m_ops = detail::mask_ops<T, N>;
    using mem_ops = detail::memory_ops<T, N>;
    using math = detail::math_ops<T, N>;

    static constexpr size_t native_width = detail::native_width<T>::value;
    static constexpr size_t storage_size = (N + native_width - 1) / native_width * native_width;

    using register_t = typename detail::register_type<T, detail::current_isa>::type;
    static constexpr size_t num_registers =
        (storage_size + detail::simd_width<T, detail::current_isa>::value - 1) /
        detail::simd_width<T, detail::current_isa>::value;

    alignas(kDefaultAlignment) std::array<register_t, num_registers> registers;

public:
    using value_type = T;
    using mask_type = Mask<T, N>;
    using size_type = size_t;
    static constexpr size_t size_value = N;

    Vector() = default;

    explicit Vector(T value) { ops::set1(registers.data(), value); }

    explicit Vector(const T* ptr) { mem_ops::load(registers.data(), ptr); }

    Vector(const Vector&) = default;
    Vector(Vector&&) = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) = default;

    Vector(std::initializer_list<T> values)
    {
        std::array<T, storage_size> tmp{};
        size_t count = std::min(values.size(), storage_size);
        std::copy_n(values.begin(), count, tmp.begin());
        mem_ops::load(registers.data(), tmp.data());
    }

    register_t* data() { return registers.data(); }
    const register_t* data() const { return registers.data(); }

    SIMD_INLINE T extract(size_t i) const
    {
        assert(i < N && "Index out of bounds");
        return ops::extract(registers.data(), i);
    }

    SIMD_INLINE void insert(size_t i, T value)
    {
        assert(i < N && "Index out of bounds");
        ops::insert(registers.data(), i, value);
    }

    static Vector load(const T* ptr)
    {
        Vector result;
        mem_ops::load(result.data(), ptr);
        return result;
    }

    static Vector load_aligned(const T* ptr)
    {
        Vector result;
        mem_ops::load_aligned(result.data(), ptr);
        return result;
    }

    static Vector load_unaligned(const T* ptr)
    {
        Vector result;
        mem_ops::load_unaligned(result.data(), ptr);
        return result;
    }

    void store(T* ptr) const { mem_ops::store(registers.data(), ptr); }

    void store_aligned(T* ptr) const { mem_ops::store_aligned(registers.data(), ptr); }

    void store_unaligned(T* ptr) const { mem_ops::store_unaligned(registers.data(), ptr); }

    std::array<T, N> to_array() const
    {
        std::array<T, N> result;
        store(result.data());
        return result;
    }

    template <typename IndexT>
    static Vector gather(const T* base, const Vector<IndexT, N>& indices)
    {
        Vector result;
        mem_ops::gather(result.data(), base, indices.data());
        return result;
    }

    template <typename IndexT>
    void scatter(T* base, const Vector<IndexT, N>& indices) const
    {
        mem_ops::scatter(registers.data(), base, indices.data());
    }

    Vector operator+(const Vector& rhs) const
    {
        Vector result;
        ops::add(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator-(const Vector& rhs) const
    {
        Vector result;
        ops::sub(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator*(const Vector& rhs) const
    {
        Vector result;
        ops::mul(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator/(const Vector& rhs) const
    {
        Vector result;
        ops::div(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector& operator+=(const Vector& rhs)
    {
        ops::add(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator-=(const Vector& rhs)
    {
        ops::sub(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator*=(const Vector& rhs)
    {
        ops::mul(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator/=(const Vector& rhs)
    {
        ops::div(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector operator&(const Vector& rhs) const
    {
        Vector result;
        ops::bitwise_and(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator|(const Vector& rhs) const
    {
        Vector result;
        ops::bitwise_or(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator^(const Vector& rhs) const
    {
        Vector result;
        ops::bitwise_xor(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator~() const
    {
        Vector result;
        ops::bitwise_not(result.data(), registers.data());
        return result;
    }

    Vector& operator&=(const Vector& rhs)
    {
        ops::bitwise_and(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator|=(const Vector& rhs)
    {
        ops::bitwise_or(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator^=(const Vector& rhs)
    {
        ops::bitwise_xor(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    mask_type operator==(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_eq(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator!=(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_neq(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator<(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_lt(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator<=(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_le(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator>(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_gt(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator>=(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_ge(result._data(), registers.data(), rhs.data());
        return result;
    }

    Vector abs() const
    {
        Vector result;
        math::abs(result.data(), registers.data());
        return result;
    }

    Vector sqrt() const
    {
        Vector result;
        math::sqrt(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector sin() const
    {
        Vector result;
        math::sin(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector cos() const
    {
        Vector result;
        math::cos(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector tan() const
    {
        Vector result;
        math::tan(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector exp() const
    {
        Vector result;
        math::exp(result.data(), registers.data());
        return result;
    }

    Vector log() const
    {
        Vector result;
        math::log(result.data(), registers.data());
        return result;
    }

    Vector min(const Vector& rhs) const
    {
        Vector result;
        ops::min(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector max(const Vector& rhs) const
    {
        Vector result;
        ops::max(result.data(), registers.data(), rhs.data());
        return result;
    }

    T hsum() const { return ops::horizontal_sum(registers.data()); }
    
    T hmin() const { return ops::horizontal_min(registers.data()); }

    T hmax() const { return ops::horizontal_max(registers.data()); }

    Vector shuffle(const std::array<int, N>& indices) const
    {
        Vector result;
        ops::shuffle(result.data(), registers.data(), indices.data());
        return result;
    }

    Vector blend(const Vector& rhs, const mask_type& mask) const
    {
        Vector result;
        ops::blend(result.data(), registers.data(), rhs.data(), mask._data());
        return result;
    }

    static Vector select(const mask_type& mask, const Vector& a, const Vector& b)
    {
        Vector result;
        ops::select(result.data(), mask._data(), a.data(), b.data());
        return result;
    }

    Vector fmadd(const Vector& a, const Vector& b) const
    {
        Vector result;
        math::fmadd(result.data(), registers.data(), a.data(), b.data());
        return result;
    }

    Vector fmsub(const Vector& a, const Vector& b) const
    {
        Vector result;
        math::fmsub(result.data(), registers.data(), a.data(), b.data());
        return result;
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<T, U>, int> = 0>
    Vector<U, N> convert() const
    {
        Vector<U, N> result;
        ops::convert(result.data(), registers.data());
        return result;
    }

    static void prefetch(const T* ptr, int hint = 0) { mem_ops::prefetch(ptr, hint); }
};

template <SimdArithmetic T, size_t N>
class alignas(kDefaultAlignment) Mask
{
private:
    using m_ops = detail::mask_ops<T, N>;
    friend class Vector<T, N>;

    using mask_register_t = typename detail::mask_register_type<T, detail::current_isa>::type;
    static constexpr size_t native_width = detail::native_width<T>::value;
    static constexpr size_t storage_size = (N + native_width - 1) / native_width * native_width;
    static constexpr size_t num_registers =
        (storage_size + detail::simd_width<T, detail::current_isa>::value - 1) /
        detail::simd_width<T, detail::current_isa>::value;

    alignas(kDefaultAlignment) std::array<mask_register_t, num_registers> registers;

public:
    using value_type = bool;
    using size_type = size_t;
    static constexpr size_t size_value = N;

    Mask() { m_ops::set_false(registers.data()); }

    explicit Mask(bool value)
    {
        if (value)
        {
            m_ops::set_true(registers.data());
        }
        else
        {
            m_ops::set_false(registers.data());
        }
    }

    explicit Mask(const bool* ptr) { m_ops::load(registers.data(), ptr); }

    Mask(const Mask&) = default;
    Mask(Mask&&) = default;
    Mask& operator=(const Mask&) = default;
    Mask& operator=(Mask&&) = default;

    mask_register_t* _data() { return registers.data(); }
    const mask_register_t* _data() const { return registers.data(); }

    SIMD_INLINE bool operator[](size_t i) const
    {
        assert(i < N && "Index out of bounds");
        return m_ops::extract(registers.data(), i);
    }

    Mask operator&(const Mask& rhs) const
    {
        Mask result;
        m_ops::logical_and(result._data(), registers.data(), rhs._data());
        return result;
    }

    Mask operator|(const Mask& rhs) const
    {
        Mask result;
        m_ops::logical_or(result._data(), registers.data(), rhs._data());
        return result;
    }

    Mask operator^(const Mask& rhs) const
    {
        Mask result;
        m_ops::logical_xor(result._data(), registers.data(), rhs._data());
        return result;
    }

    Mask operator~() const
    {
        Mask result;
        m_ops::logical_not(result._data(), registers.data());
        return result;
    }

    Mask& operator&=(const Mask& rhs)
    {
        m_ops::logical_and(registers.data(), registers.data(), rhs._data());
        return *this;
    }

    Mask& operator|=(const Mask& rhs)
    {
        m_ops::logical_or(registers.data(), registers.data(), rhs._data());
        return *this;
    }

    Mask& operator^=(const Mask& rhs)
    {
        m_ops::logical_xor(registers.data(), registers.data(), rhs._data());
        return *this;
    }

    uint64_t to_bitmask() const { return m_ops::to_bitmask(registers.data()); }

    bool any() const { return m_ops::any(registers.data()); }

    bool all() const { return m_ops::all(registers.data()); }

    bool none() const { return !any(); }

    int count() const { return m_ops::count(registers.data()); }

    void store(bool* ptr) const { m_ops::store(registers.data(), ptr); }

    std::array<bool, N> to_array() const
    {
        std::array<bool, N> result;
        store(result.data());
        return result;
    }
};

// Type aliases for common vector types
// Fixed-size vector types
template <size_t N>
using float_v = Vector<float, N>;
template <size_t N>
using double_v = Vector<double, N>;
template <size_t N>
using int8_v = Vector<int8_t, N>;
template <size_t N>
using uint8_v = Vector<uint8_t, N>;
template <size_t N>
using int16_v = Vector<int16_t, N>;
template <size_t N>
using uint16_v = Vector<uint16_t, N>;
template <size_t N>
using int32_v = Vector<int32_t, N>;
template <size_t N>
using uint32_v = Vector<uint32_t, N>;
template <size_t N>
using int64_v = Vector<int64_t, N>;
template <size_t N>
using uint64_v = Vector<uint64_t, N>;

// Native-width vector types (width depends on best available instruction set)
using float_vn = float_v<detail::native_width<float>::value>;
using double_vn = double_v<detail::native_width<double>::value>;
using int8_vn = int8_v<detail::native_width<int8_t>::value>;
using uint8_vn = uint8_v<detail::native_width<uint8_t>::value>;
using int16_vn = int16_v<detail::native_width<int16_t>::value>;
using uint16_vn = uint16_v<detail::native_width<uint16_t>::value>;
using int32_vn = int32_v<detail::native_width<int32_t>::value>;
using uint32_vn = uint32_v<detail::native_width<uint32_t>::value>;
using int64_vn = int64_v<detail::native_width<int64_t>::value>;
using uint64_vn = uint64_v<detail::native_width<uint64_t>::value>;

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2

namespace detail
{

template <typename T, size_t N>
struct vector_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_set1_ps(value);
        }

        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_set1_pd(value);
        }

        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_set1_epi8(static_cast<char>(value));
        }

        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_set1_epi16(static_cast<short>(value));
        }

        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_set1_epi32(static_cast<int>(value));
        }

        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_set1_epi64x(static_cast<long long>(value));
        }
    }

    static SIMD_INLINE T extract(const register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            return tmp[index % 4];
        }

        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            return tmp[index % 2];
        }

        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            return static_cast<int8_t>(_mm_extract_epi8(*src, index % 16));
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 16];
#endif
        }

        else if constexpr (std::is_same_v<T, uint8_t>)
        {
#if SIMD_SSE4_1
            return static_cast<uint8_t>(_mm_extract_epi8(*src, index % 16));
#else
            alignas(16) uint8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 16];
#endif
        }

        else if constexpr (std::is_same_v<T, int16_t>)
        {
            alignas(16) int16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 8];
        }

        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            alignas(16) uint16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 8];
        }

        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            return static_cast<int32_t>(_mm_extract_epi32(*src, index % 4));
#else
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }

        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            return static_cast<uint32_t>(_mm_extract_epi32(*src, index % 4));
#else
            alignas(16) uint32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }

        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_SSE4_1
            return static_cast<T>(_mm_extract_epi64(*src, index % 2));
#else
            alignas(16) T tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 2];
#endif
        }
    }

    static SIMD_INLINE void insert(register_t* dst, size_t index, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *dst);
            tmp[index % 4] = value;
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *dst);
            tmp[index % 2] = value;
            *dst = _mm_load_pd(tmp);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_insert_epi8(*dst, static_cast<int>(value), index % 16);
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 16] = value;
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            alignas(16) int16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 8] = static_cast<int16_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_insert_epi32(*dst, static_cast<int>(value), index % 4);
#else
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 4] = static_cast<int32_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_insert_epi64(*dst, static_cast<long long>(value), index % 2);
#else
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 2] = static_cast<int64_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
    }
};

}

#endif

} // namespace vector_simd

#endif /* End of include guard: SIMD_HPP_al9nn6 */
