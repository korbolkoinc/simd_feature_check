// src/main.cpp
#include "simd/feature_check.hpp"
#include <cstdio>

int main()
{
    // ...
    // Example usage of SIMD feature detection
    if (simd::has_feature(simd::Feature::AVX512F))
    {
        std::printf("AVX512F is supported.\n");
    }
    else
    {
        std::printf("AVX512F is not supported.\n");
    }

    return 0;
}
