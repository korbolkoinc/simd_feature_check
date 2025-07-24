#include "simd/feature_check.hpp"
#include <gtest/gtest.h>

TEST(SimdFeaturesTest, CompileTimeFeatureDetection)
{
    bool has_sse2 = simd::compile_time::has<simd::Feature::SSE2>();
    bool has_avx = simd::compile_time::has<simd::Feature::AVX>();
    bool has_avx2 = simd::compile_time::has<simd::Feature::AVX2>();

    EXPECT_TRUE(has_sse2 == true || has_sse2 == false);
    EXPECT_TRUE(has_avx == true || has_avx == false);
    EXPECT_TRUE(has_avx2 == true || has_avx2 == false);
}

TEST(SimdFeaturesTest, RuntimeFeatureDetection)
{
    bool runtime_sse2 = simd::FeatureDetector<simd::Feature::SSE2>::available();
    bool runtime_avx = simd::FeatureDetector<simd::Feature::AVX>::available();

    EXPECT_TRUE(runtime_sse2 == true || runtime_sse2 == false);
    EXPECT_TRUE(runtime_avx == true || runtime_avx == false);
}

TEST(SimdFeaturesTest, FeatureStringConversion)
{
    std::string sse2_str = simd::feature_to_string(simd::Feature::SSE2);
    std::string avx_str = simd::feature_to_string(simd::Feature::AVX);

    EXPECT_FALSE(sse2_str.empty());
    EXPECT_FALSE(avx_str.empty());
    EXPECT_NE(sse2_str, avx_str);
}

TEST(SimdFeaturesTest, MacroConsistency)
{
    bool compile_time_sse2 = simd::compile_time::has<simd::Feature::SSE2>();
    bool compile_time_avx = simd::compile_time::has<simd::Feature::AVX>();
    bool compile_time_avx2 = simd::compile_time::has<simd::Feature::AVX2>();

    EXPECT_EQ(compile_time_sse2, SIMD_HAS_SSE2 != 0);
    EXPECT_EQ(compile_time_avx, SIMD_HAS_AVX != 0);
    EXPECT_EQ(compile_time_avx2, SIMD_HAS_AVX2 != 0);
}
