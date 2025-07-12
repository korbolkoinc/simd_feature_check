#ifndef LIB_SIMD_VECTOR_BASE_HPP_fiziy4
#define LIB_SIMD_VECTOR_BASE_HPP_fiziy4

#include <simd/feature_check.hpp>
#include <cassert>

namespace vector_simd::detail
{

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

} // namespace vector_simd::detail

#endif // End of include guard: LIB_SIMD_VECTOR_BASE_HPP_fiziy4