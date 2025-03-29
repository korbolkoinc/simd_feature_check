#include <gtest/gtest.h>
#include "simd/feature_check.hpp"

TEST(SimdFeaturesTest, GetSimdSupport) {
    int simd_support = simd::get_simd_support();
    EXPECT_EQ(5, simd_support);
}
