#ifndef LIB_SIMD_VECTOR_VECTOR_HPP_qvwewn
#define LIB_SIMD_VECTOR_VECTOR_HPP_qvwewn

#include "simd/core/concepts.hpp"
#include "simd/core/types.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include "simd/vector/base.hpp"
#include <algorithm>
#include <array>
#include <initializer_list>
#include <simd/feature_check.hpp>

namespace vector_simd
{

template <SimdArithmetic T, size_t N>
class Mask;

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

    static Vector zeros()
    {
        Vector result;
        for (size_t i = 0; i < num_registers; ++i) mem_ops::set_zero(&result.registers[i]);
        return result;
    }

    static Vector ones() { return Vector(T(1)); }

    static Vector iota(T start = T(0))
    {
        std::array<T, storage_size> tmp{};
        for (size_t i = 0; i < N; ++i) tmp[i] = static_cast<T>(start + static_cast<T>(i));
        return Vector::load(tmp.data());
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector rsqrt() const
    {
        Vector result;
        math::rsqrt(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector rcp() const
    {
        Vector result;
        math::rcp(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector floor() const
    {
        Vector result;
        math::floor(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector ceil() const
    {
        Vector result;
        math::ceil(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector round() const
    {
        Vector result;
        math::round(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector trunc() const
    {
        Vector result;
        math::trunc(result.data(), registers.data());
        return result;
    }

    Vector clamp(const Vector& lo, const Vector& hi) const
    {
        Vector result;
        ops::clamp(result.data(), registers.data(), lo.data(), hi.data());
        return result;
    }

    Vector add_sat(const Vector& rhs) const
    {
        Vector result;
        ops::add_sat(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector sub_sat(const Vector& rhs) const
    {
        Vector result;
        ops::sub_sat(result.data(), registers.data(), rhs.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    Vector shift_left(int count) const
    {
        Vector result;
        ops::shift_left(result.data(), registers.data(), count);
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>, int> = 0>
    Vector shift_right(int count) const
    {
        Vector result;
        ops::shift_right(result.data(), registers.data(), count);
        return result;
    }

    T reduce_add() const { return ops::reduce_add(registers.data()); }

    T reduce_min() const { return ops::reduce_min(registers.data()); }

    T reduce_max() const { return ops::reduce_max(registers.data()); }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    T dot(const Vector& rhs) const
    {
        return ops::dot(registers.data(), rhs.data());
    }

    void store_nt(T* ptr) const { mem_ops::store_nt(ptr, registers.data()); }

    Vector interleave_lo(const Vector& rhs) const
    {
        Vector result;
        ops::interleave_lo(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector interleave_hi(const Vector& rhs) const
    {
        Vector result;
        ops::interleave_hi(result.data(), registers.data(), rhs.data());
        return result;
    }

    const T* begin() const
    {
        static thread_local std::array<T, storage_size> buf;
        store(buf.data());
        return buf.data();
    }

    const T* end() const { return begin() + N; }
};
} // namespace vector_simd

#endif // End of include guard: LIB_SIMD_VECTOR_VECTOR_HPP_qvwewn