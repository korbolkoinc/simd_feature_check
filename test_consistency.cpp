#include "simd/feature_check.hpp"
#include <iostream>

int main() {
    std::cout << "===== SIMD Feature Consistency Check =====" << std::endl;
    
    std::cout << "Compile-time feature detection (simd::compile_time::has):" << std::endl;
    std::cout << "  SSE2: " << (simd::compile_time::has<simd::Feature::SSE2>() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX:  " << (simd::compile_time::has<simd::Feature::AVX>() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX2: " << (simd::compile_time::has<simd::Feature::AVX2>() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX512F: " << (simd::compile_time::has<simd::Feature::AVX512F>() ? "Yes" : "No") << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "Macro consistency check (SIMD_HAS_*):" << std::endl;
    std::cout << "  SIMD_HAS_SSE2: " << (SIMD_HAS_SSE2 ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD_HAS_AVX:  " << (SIMD_HAS_AVX ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD_HAS_AVX2: " << (SIMD_HAS_AVX2 ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD_HAS_AVX512F: " << (SIMD_HAS_AVX512F ? "Yes" : "No") << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "Convenience macro check (SIMD_*):" << std::endl;
    std::cout << "  SIMD_SSE2: " << (SIMD_SSE2 ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD_AVX:  " << (SIMD_AVX ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD_AVX2: " << (SIMD_AVX2 ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD_AVX512: " << (SIMD_AVX512 ? "Yes" : "No") << std::endl;
    
    std::cout << std::endl;
    
    bool consistent = true;
    
    if (simd::compile_time::has<simd::Feature::SSE2>() != (SIMD_HAS_SSE2 != 0)) {
        std::cout << "ERROR: SSE2 detection inconsistent!" << std::endl;
        consistent = false;
    }
    
    if (simd::compile_time::has<simd::Feature::AVX>() != (SIMD_HAS_AVX != 0)) {
        std::cout << "ERROR: AVX detection inconsistent!" << std::endl;
        consistent = false;
    }
    
    if (simd::compile_time::has<simd::Feature::AVX2>() != (SIMD_HAS_AVX2 != 0)) {
        std::cout << "ERROR: AVX2 detection inconsistent!" << std::endl;
        consistent = false;
    }
    
    if (SIMD_HAS_AVX != SIMD_AVX) {
        std::cout << "ERROR: SIMD_HAS_AVX != SIMD_AVX!" << std::endl;
        consistent = false;
    }
    
    if (SIMD_HAS_AVX2 != SIMD_AVX2) {
        std::cout << "ERROR: SIMD_HAS_AVX2 != SIMD_AVX2!" << std::endl;
        consistent = false;
    }
    
    if (consistent) {
        std::cout << "✓ All feature detection APIs are consistent!" << std::endl;
    } else {
        std::cout << "✗ Inconsistencies found!" << std::endl;
        return 1;
    }
    
    return 0;
}
