#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    std::cout << "===== Basic Feature Detection =====" << std::endl;

    std::cout << "SSE support: "
              << (simd::has_feature(simd::Feature::SSE) ? "Yes" : "No")
              << std::endl;
    std::cout << "AVX support: "
              << (simd::has_feature(simd::Feature::AVX) ? "Yes" : "No")
              << std::endl;
    std::cout << "AVX2 support: "
              << (simd::has_feature(simd::Feature::AVX2) ? "Yes" : "No")
              << std::endl;
    std::cout << "AVX-512F support: "
              << (simd::has_feature(simd::Feature::AVX512F) ? "Yes" : "No")
              << std::endl;

    std::cout << "CPU Vendor: " << simd::get_cpu_vendor() << std::endl;

    auto model = simd::get_cpu_model();
    if (model)
    {
        std::cout << "CPU Model: Family " << model->at(0) << ", Model "
                  << model->at(1) << ", Stepping " << model->at(2) << std::endl;
    }

    simd::Feature highest = simd::highest_feature();
    std::cout << "Highest supported SIMD feature: "
              << simd::feature_to_string(highest) << std::endl;

    std::cout << std::endl;
}
