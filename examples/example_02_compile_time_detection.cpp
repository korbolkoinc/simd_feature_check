#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    std::cout << "===== Compile-Time Detection =====" << std::endl;

    std::cout << "SSE available at compile time: "
              << (simd::compile_time::sse ? "Yes" : "No") << std::endl;
    std::cout << "AVX available at compile time: "
              << (simd::compile_time::avx ? "Yes" : "No") << std::endl;
    std::cout << "AVX2 available at compile time: "
              << (simd::compile_time::avx2 ? "Yes" : "No") << std::endl;

    std::cout << "SSE2 available (template): "
              << (simd::compile_time::has<simd::Feature::SSE2>() ? "Yes" : "No")
              << std::endl;
    std::cout << "AVX512F available (template): "
              << (simd::compile_time::has<simd::Feature::AVX512F>() ? "Yes"
                                                                    : "No")
              << std::endl;

    std::cout << "Maximum compile-time SIMD feature: "
              << simd::feature_to_string(simd::compile_time::max_feature)
              << std::endl;

    std::cout << std::endl;
}