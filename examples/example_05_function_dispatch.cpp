#include "simd/feature_check.hpp"
#include <iostream>

float* add_vectors_scalar(const float* a, const float* b, float* result,
                          size_t size)
{
    std::cout << "Using scalar implementation" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = a[i] + b[i];
    }
    return result;
}

#if SIMD_HAS_AVX
float* add_vectors_avx(const float* a, const float* b, float* result,
                       size_t size)
{
    std::cout << "Using AVX implementation" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = a[i] + b[i];
    }
    return result;
}
#endif

#if SIMD_HAS_AVX512F
float* add_vectors_avx512(const float* a, const float* b, float* result,
                          size_t size)
{
    std::cout << "Using AVX-512 implementation" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = a[i] + b[i];
    }
    return result;
}
#endif

int main()
{
    std::cout << "===== Function Dispatch =====" << std::endl;

    using AddFunc = float* (*)(const float*, const float*, float*, size_t);

    AddFunc best_impl;

    if (simd::has_feature(simd::Feature::AVX512F))
    {
#if SIMD_HAS_AVX512F
        best_impl = add_vectors_avx512;
#else
        best_impl = add_vectors_scalar;
#endif
    }
    else if (simd::has_feature(simd::Feature::AVX))
    {
#if SIMD_HAS_AVX
        best_impl = add_vectors_avx;
#else
        best_impl = add_vectors_scalar;
#endif
    }
    else
    {
        best_impl = add_vectors_scalar;
    }

    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];

    best_impl(a, b, result, 4);

    std::cout << "Result: [" << result[0] << ", " << result[1] << ", "
              << result[2] << ", " << result[3] << "]" << std::endl;

    std::cout << std::endl;
}
