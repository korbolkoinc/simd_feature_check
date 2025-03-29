// src/main.cpp
#include "simd/feature_check.hpp"
#include <cstdio>

int main()
{
    int simd_support = simd::get_simd_support();
    (void)printf("SIMD support level: %d\n", simd_support);

    return 0;
}
