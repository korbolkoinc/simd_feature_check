#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    std::cout << "===== Runtime Detection =====" << std::endl;

    std::cout << "SSE runtime check: "
              << (simd::runtime::has<simd::Feature::SSE>() ? "Yes" : "No")
              << std::endl;
    std::cout << "AVX runtime check: "
              << (simd::runtime::has<simd::Feature::AVX>() ? "Yes" : "No")
              << std::endl;
    std::cout << "AVX2 runtime check: "
              << (simd::runtime::has<simd::Feature::AVX2>() ? "Yes" : "No")
              << std::endl;

    std::cout << "Maximum runtime SIMD feature: "
              << simd::feature_to_string(simd::runtime::highest_feature())
              << std::endl;

    std::cout << std::endl;
}