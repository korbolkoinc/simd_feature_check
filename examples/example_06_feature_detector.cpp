// Example 6: Using the FeatureDetector
#include <simd/feature_check.hpp>
#include <iostream>

int main()
{
    std::cout << "===== FeatureDetector Example =====" << std::endl;

    using AVXDetector = simd::FeatureDetector<simd::Feature::AVX>;
    using AVX2Detector = simd::FeatureDetector<simd::Feature::AVX2>;
    using AVX512Detector = simd::FeatureDetector<simd::Feature::AVX512F>;

    std::cout << "AVX Feature:" << std::endl;
    std::cout << "  Name: " << AVXDetector::name() << std::endl;
    std::cout << "  Compile-time support: "
              << (AVXDetector::compile_time ? "Yes" : "No") << std::endl;
    std::cout << "  Runtime support: "
              << (AVXDetector::available() ? "Yes" : "No") << std::endl;

    std::cout << "AVX2 Feature:" << std::endl;
    std::cout << "  Name: " << AVX2Detector::name() << std::endl;
    std::cout << "  Compile-time support: "
              << (AVX2Detector::compile_time ? "Yes" : "No") << std::endl;
    std::cout << "  Runtime support: "
              << (AVX2Detector::available() ? "Yes" : "No") << std::endl;

    std::cout << "AVX-512F Feature:" << std::endl;
    std::cout << "  Name: " << AVX512Detector::name() << std::endl;
    std::cout << "  Compile-time support: "
              << (AVX512Detector::compile_time ? "Yes" : "No") << std::endl;
    std::cout << "  Runtime support: "
              << (AVX512Detector::available() ? "Yes" : "No") << std::endl;

    std::cout << std::endl;
}
