#ifndef LIB_SIMD_MASK_MASK_HPP_ggewhb
#define LIB_SIMD_MASK_MASK_HPP_ggewhb

#include "simd/arch/detection.hpp"
#include "simd/core/concepts.hpp"
#include "simd/core/types.hpp"
#include "simd/operations/forward_decl.hpp"
#include "simd/registers/types.hpp"
#include <array>
#include <cassert>

namespace vector_simd
{

template <SimdArithmetic T, size_t N>
class Vector;

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

} // namespace vector_simd

#endif // End of include guard: LIB_SIMD_MASK_MASK_HPP_ggewhb