#include "simd/simd.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>

using namespace vector_simd;

static constexpr size_t N = detail::native_width<float>::value;
static constexpr size_t Nd = detail::native_width<double>::value;
static constexpr size_t Ni32 = detail::native_width<int32_t>::value;

class VectorArithmeticTest : public ::testing::Test
{
};

TEST_F(VectorArithmeticTest, AddFloat)
{
    Vector<float, N> a(2.0f), b(3.0f);
    auto c = a + b;
    auto arr = c.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 5.0f);
}

TEST_F(VectorArithmeticTest, SubFloat)
{
    Vector<float, N> a(7.0f), b(3.0f);
    auto c = a - b;
    auto arr = c.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 4.0f);
}

TEST_F(VectorArithmeticTest, MulFloat)
{
    Vector<float, N> a(3.0f), b(4.0f);
    auto c = a * b;
    auto arr = c.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 12.0f);
}

TEST_F(VectorArithmeticTest, DivFloat)
{
    Vector<float, N> a(12.0f), b(4.0f);
    auto c = a / b;
    auto arr = c.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 3.0f);
}

TEST_F(VectorArithmeticTest, AddDouble)
{
    Vector<double, Nd> a(2.5), b(3.5);
    auto c = a + b;
    auto arr = c.to_array();
    for (size_t i = 0; i < Nd; ++i) EXPECT_DOUBLE_EQ(arr[i], 6.0);
}

TEST_F(VectorArithmeticTest, AddInt32)
{
    Vector<int32_t, Ni32> a(10), b(20);
    auto c = a + b;
    auto arr = c.to_array();
    for (size_t i = 0; i < Ni32; ++i) EXPECT_EQ(arr[i], 30);
}

TEST_F(VectorArithmeticTest, CompoundAssign)
{
    Vector<float, N> a(2.0f), b(3.0f);
    a += b;
    auto arr = a.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 5.0f);
}

class VectorMemoryTest : public ::testing::Test
{
};

TEST_F(VectorMemoryTest, LoadStore)
{
    alignas(64) std::array<float, N> input;
    std::iota(input.begin(), input.end(), 1.0f);
    auto v = Vector<float, N>::load_aligned(input.data());
    alignas(64) std::array<float, N> output;
    v.store_aligned(output.data());
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(input[i], output[i]);
}

TEST_F(VectorMemoryTest, LoadUnaligned)
{
    std::array<float, N + 1> buf;
    std::iota(buf.begin(), buf.end(), 0.0f);
    auto v = Vector<float, N>::load_unaligned(buf.data() + 1);
    auto arr = v.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], static_cast<float>(i + 1));
}

TEST_F(VectorMemoryTest, Broadcast)
{
    Vector<float, N> v(42.0f);
    auto arr = v.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 42.0f);
}

TEST_F(VectorMemoryTest, ToArray)
{
    Vector<int32_t, Ni32> v(7);
    auto arr = v.to_array();
    for (size_t i = 0; i < Ni32; ++i) EXPECT_EQ(arr[i], 7);
}

TEST_F(VectorMemoryTest, InitializerList)
{
    if constexpr (N >= 4)
    {
        Vector<float, 4> v = {1.0f, 2.0f, 3.0f, 4.0f};
        auto arr = v.to_array();
        EXPECT_FLOAT_EQ(arr[0], 1.0f);
        EXPECT_FLOAT_EQ(arr[1], 2.0f);
        EXPECT_FLOAT_EQ(arr[2], 3.0f);
        EXPECT_FLOAT_EQ(arr[3], 4.0f);
    }
}

class VectorFactoryTest : public ::testing::Test
{
};

TEST_F(VectorFactoryTest, Zeros)
{
    auto v = Vector<float, N>::zeros();
    auto arr = v.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 0.0f);
}

TEST_F(VectorFactoryTest, Ones)
{
    auto v = Vector<float, N>::ones();
    auto arr = v.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 1.0f);
}

TEST_F(VectorFactoryTest, Iota)
{
    auto v = Vector<float, N>::iota(0.0f);
    auto arr = v.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], static_cast<float>(i));
}

TEST_F(VectorFactoryTest, IotaWithOffset)
{
    auto v = Vector<float, N>::iota(10.0f);
    auto arr = v.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 10.0f + static_cast<float>(i));
}

class VectorMathTest : public ::testing::Test
{
};

TEST_F(VectorMathTest, Abs)
{
    Vector<float, N> v(-5.0f);
    auto r = v.abs();
    auto arr = r.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 5.0f);
}

TEST_F(VectorMathTest, Sqrt)
{
    Vector<float, N> v(16.0f);
    auto r = v.sqrt();
    auto arr = r.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 4.0f);
}

TEST_F(VectorMathTest, MinMax)
{
    Vector<float, N> a(3.0f), b(5.0f);
    auto mn = a.min(b);
    auto mx = a.max(b);
    auto mna = mn.to_array();
    auto mxa = mx.to_array();
    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(mna[i], 3.0f);
        EXPECT_FLOAT_EQ(mxa[i], 5.0f);
    }
}

TEST_F(VectorMathTest, SinCos)
{
    alignas(64) float input[N];
    for (size_t i = 0; i < N; ++i) input[i] = 0.5f * static_cast<float>(i);
    auto v = Vector<float, N>::load(input);
    auto s = v.sin();
    auto c = v.cos();
    auto sa = s.to_array();
    auto ca = c.to_array();
    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(sa[i], std::sin(input[i]), 1e-4f);
        EXPECT_NEAR(ca[i], std::cos(input[i]), 1e-4f);
    }
}

TEST_F(VectorMathTest, ExpLog)
{
    alignas(64) float input[N];
    for (size_t i = 0; i < N; ++i) input[i] = 0.5f + 0.1f * static_cast<float>(i);
    auto v = Vector<float, N>::load(input);
    auto e = v.exp();
    auto l = v.log();
    auto ea = e.to_array();
    auto la = l.to_array();
    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(ea[i], std::exp(input[i]), std::abs(std::exp(input[i])) * 1e-5f);
        EXPECT_NEAR(la[i], std::log(input[i]), 1e-5f);
    }
}

TEST_F(VectorMathTest, Rsqrt)
{
    Vector<float, N> v(4.0f);
    auto r = v.rsqrt();
    auto arr = r.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_NEAR(arr[i], 0.5f, 1e-5f);
}

TEST_F(VectorMathTest, Rcp)
{
    Vector<float, N> v(4.0f);
    auto r = v.rcp();
    auto arr = r.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_NEAR(arr[i], 0.25f, 1e-5f);
}

TEST_F(VectorMathTest, FloorCeilRound)
{
    Vector<float, N> v(3.7f);
    auto fl = v.floor();
    auto ce = v.ceil();
    auto rn = v.round();
    auto fla = fl.to_array();
    auto cea = ce.to_array();
    auto rna = rn.to_array();
    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(fla[i], 3.0f);
        EXPECT_FLOAT_EQ(cea[i], 4.0f);
        EXPECT_FLOAT_EQ(rna[i], 4.0f);
    }
}

TEST_F(VectorMathTest, Fmadd)
{
    Vector<float, N> a(2.0f), b(3.0f), c(4.0f);
    auto r = a.fmadd(b, c);
    auto arr = r.to_array();
    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(arr[i], 10.0f);
}

class VectorBitwiseTest : public ::testing::Test
{
};

TEST_F(VectorBitwiseTest, AndOrXor)
{
    Vector<int32_t, Ni32> a(0xFF00), b(0x0FF0);
    auto and_r = a & b;
    auto or_r = a | b;
    auto xor_r = a ^ b;
    auto anda = and_r.to_array();
    auto ora = or_r.to_array();
    auto xora = xor_r.to_array();
    for (size_t i = 0; i < Ni32; ++i)
    {
        EXPECT_EQ(anda[i], 0xFF00 & 0x0FF0);
        EXPECT_EQ(ora[i], 0xFF00 | 0x0FF0);
        EXPECT_EQ(xora[i], 0xFF00 ^ 0x0FF0);
    }
}

class VectorComparisonTest : public ::testing::Test
{
};

TEST_F(VectorComparisonTest, EqualNotEqual)
{
    Vector<float, N> a(1.0f), b(1.0f), c(2.0f);
    auto eq = a == b;
    auto ne = a != c;
    EXPECT_TRUE(eq.all());
    EXPECT_TRUE(ne.all());
}

TEST_F(VectorComparisonTest, LessGreater)
{
    Vector<float, N> a(1.0f), b(2.0f);
    auto lt = a < b;
    auto gt = b > a;
    auto le = a <= b;
    auto ge = b >= a;
    EXPECT_TRUE(lt.all());
    EXPECT_TRUE(gt.all());
    EXPECT_TRUE(le.all());
    EXPECT_TRUE(ge.all());
}

class MaskTest : public ::testing::Test
{
};

TEST_F(MaskTest, AllNoneAny)
{
    Mask<float, N> all_true(true);
    Mask<float, N> all_false(false);
    EXPECT_TRUE(all_true.all());
    EXPECT_TRUE(all_true.any());
    EXPECT_FALSE(all_true.none());
    EXPECT_FALSE(all_false.all());
    EXPECT_FALSE(all_false.any());
    EXPECT_TRUE(all_false.none());
}

TEST_F(MaskTest, LogicalOps)
{
    Mask<float, N> a(true), b(false);
    auto and_r = a & b;
    auto or_r = a | b;
    auto not_r = ~a;
    EXPECT_TRUE(and_r.none());
    EXPECT_TRUE(or_r.all());
    EXPECT_TRUE(not_r.none());
}

TEST_F(MaskTest, Count)
{
    Mask<float, N> all(true);
    EXPECT_EQ(all.count(), static_cast<int>(N));
    Mask<float, N> none(false);
    EXPECT_EQ(none.count(), 0);
}
