#include <iostream>
#include "simd/feature_check.hpp"

int main()
{
    int simd_support = simd::get_simd_support();
    std::cout << "SIMD support level: " << simd_support << std::endl;
   
    return 0;
}
