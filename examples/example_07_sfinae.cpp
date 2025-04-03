#include <algorithm>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <simd/feature_check.hpp>
#include <type_traits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename T>
typename std::enable_if<
    simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        std::is_same<T, float>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with AVX-512 (float)" << std::endl;

    const size_t step = 16;
    const size_t aligned_count = count - (count % step);

    __m512 scale = _mm512_set1_ps(0.1f);
    __m512 add_const = _mm512_set1_ps(1.0f);

    for (size_t i = 0; i < aligned_count; i += step)
    {
        __m512 values = _mm512_loadu_ps(&data[i]);
        __m512 factors = _mm512_loadu_ps(&factor[i]);

        __m512 scaled = _mm512_mul_ps(values, scale);

        __m512 multiplied = _mm512_mul_ps(scaled, factors);

        __m512 result = _mm512_add_ps(multiplied, add_const);

        _mm512_storeu_ps(&data[i], result);
    }

    for (size_t i = aligned_count; i < count; i++)
    {
        data[i] = data[i] * 0.1f * factor[i] + 1.0f;
    }
}

template <typename T>
typename std::enable_if<
    simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        std::is_same<T, double>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with AVX-512 (double)" << std::endl;

    const size_t step = 8;
    const size_t aligned_count = count - (count % step);

    __m512d scale = _mm512_set1_pd(0.1);
    __m512d add_const = _mm512_set1_pd(1.0);

    for (size_t i = 0; i < aligned_count; i += step)
    {
        __m512d values = _mm512_loadu_pd(&data[i]);
        __m512d factors = _mm512_loadu_pd(&factor[i]);

        __m512d scaled = _mm512_mul_pd(values, scale);
        __m512d multiplied = _mm512_mul_pd(scaled, factors);
        __m512d result = _mm512_add_pd(multiplied, add_const);

        _mm512_storeu_pd(&data[i], result);
    }

    for (size_t i = aligned_count; i < count; i++)
    {
        data[i] = data[i] * 0.1 * factor[i] + 1.0;
    }
}

template <typename T>
typename std::enable_if<
    simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        std::is_same<T, int>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with AVX-512 (int)" << std::endl;

    const size_t step = 16;
    const size_t aligned_count = count - (count % step);

    __m512i add_const = _mm512_set1_epi32(1);

    for (size_t i = 0; i < aligned_count; i += step)
    {
        __m512i values = _mm512_loadu_si512(&data[i]);
        __m512i factors = _mm512_loadu_si512(&factor[i]);

        __m512i scaled = _mm512_srai_epi32(values, 2);

        __m512i multiplied = _mm512_mullo_epi32(scaled, factors);

        __m512i result = _mm512_add_epi32(multiplied, add_const);

        _mm512_storeu_si512(&data[i], result);
    }

    for (size_t i = aligned_count; i < count; i++)
    {
        data[i] = (data[i] / 4) * factor[i] + 1;
    }
}

template <typename T>
typename std::enable_if<
    !simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        simd::enable_if_supported<simd::Feature::AVX2, T>::value &&
        std::is_same<T, float>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with AVX2 (float)" << std::endl;

    const size_t step = 8;
    const size_t aligned_count = count - (count % step);

    __m256 scale = _mm256_set1_ps(0.1f);
    __m256 add_const = _mm256_set1_ps(1.0f);

    for (size_t i = 0; i < aligned_count; i += step)
    {
        __m256 values = _mm256_loadu_ps(&data[i]);
        __m256 factors = _mm256_loadu_ps(&factor[i]);

        __m256 scaled = _mm256_mul_ps(values, scale);
        __m256 multiplied = _mm256_mul_ps(scaled, factors);
        __m256 result = _mm256_add_ps(multiplied, add_const);

        _mm256_storeu_ps(&data[i], result);
    }

    for (size_t i = aligned_count; i < count; i++)
    {
        data[i] = data[i] * 0.1f * factor[i] + 1.0f;
    }
}

template <typename T>
typename std::enable_if<
    !simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        simd::enable_if_supported<simd::Feature::AVX2, T>::value &&
        std::is_same<T, double>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with AVX2 (double)" << std::endl;

    const size_t step = 4;
    const size_t aligned_count = count - (count % step);

    __m256d scale = _mm256_set1_pd(0.1);
    __m256d add_const = _mm256_set1_pd(1.0);

    for (size_t i = 0; i < aligned_count; i += step)
    {
        __m256d values = _mm256_loadu_pd(&data[i]);
        __m256d factors = _mm256_loadu_pd(&factor[i]);

        __m256d scaled = _mm256_mul_pd(values, scale);
        __m256d multiplied = _mm256_mul_pd(scaled, factors);
        __m256d result = _mm256_add_pd(multiplied, add_const);

        _mm256_storeu_pd(&data[i], result);
    }

    for (size_t i = aligned_count; i < count; i++)
    {
        data[i] = data[i] * 0.1 * factor[i] + 1.0;
    }
}

template <typename T>
typename std::enable_if<
    !simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        simd::enable_if_supported<simd::Feature::AVX2, T>::value &&
        std::is_same<T, int>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with AVX2 (int)" << std::endl;

    const size_t step = 8;
    const size_t aligned_count = count - (count % step);

    __m256i add_const = _mm256_set1_epi32(1);

    for (size_t i = 0; i < aligned_count; i += step)
    {
        __m256i values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data[i]));
        __m256i factors =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&factor[i]));

        __m256i scaled = _mm256_srai_epi32(values, 2);

        __m256i multiplied = _mm256_mullo_epi32(scaled, factors);

        __m256i result = _mm256_add_epi32(multiplied, add_const);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&data[i]), result);
    }

    for (size_t i = aligned_count; i < count; i++)
    {
        data[i] = (data[i] / 4) * factor[i] + 1;
    }
}

template <typename T>
typename std::enable_if<
    !simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        !simd::enable_if_supported<simd::Feature::AVX2, T>::value,
    void>::type
process_data_avx512(T* data, const T* factor, size_t count)
{
    std::cout << "Processing with scalar operations" << std::endl;

    for (size_t i = 0; i < count; i++)
    {
        if (std::is_floating_point<T>::value)
        {
            data[i] = data[i] * T(0.1) * factor[i] + T(1.0);
        }
        else
        {
            data[i] = (data[i] / 4) * factor[i] + 1;
        }
    }
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generate_test_data(size_t count)
{
    std::vector<T> data(count);
    std::vector<T> factor(count);

    std::random_device rd;
    std::mt19937 gen(rd());

    if (std::is_floating_point<T>::value)
    {
        std::uniform_real_distribution<float> dist(0.1f, 10.0f);
        for (size_t i = 0; i < count; i++)
        {
            data[i] = static_cast<T>(dist(gen));
            factor[i] = static_cast<T>(dist(gen));
        }
    }
    else
    {
        std::uniform_int_distribution<int> dist(1, 100);
        for (size_t i = 0; i < count; i++)
        {
            data[i] = static_cast<T>(dist(gen));
            factor[i] = static_cast<T>(dist(gen) % 10 + 1);
        }
    }

    return {data, factor};
}

template <typename T>
void benchmark_processing(size_t data_size, int iterations)
{
    std::cout << "Benchmarking with type " << typeid(T).name()
              << ", data size: " << data_size << ", iterations: " << iterations
              << std::endl;

    auto [data, factor] = generate_test_data<T>(data_size);

    std::vector<T> data_copy(data.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        std::copy(data.begin(), data.end(), data_copy.begin());
        process_data_avx512(data_copy.data(), factor.data(), data_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Total time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per iteration: " << elapsed.count() / iterations
              << " ms" << std::endl;
    std::cout << "First 5 results: ";
    for (size_t i = 0; i < std::min(size_t(5), data_size); i++)
    {
        std::cout << data_copy[i] << " ";
    }
    std::cout << std::endl << std::endl;

    bool has_inf = false;
    if (std::is_floating_point<T>::value)
    {
        for (size_t i = 0; i < data_size; i++)
        {
            if (std::isinf(data_copy[i]) || std::isnan(data_copy[i]))
            {
                has_inf = true;
                break;
            }
        }
    }

    if (has_inf)
    {
        std::cout << "UYARI: Sonuçlarda inf veya NaN değerler var!"
                  << std::endl;
    }
}

template <typename T>
typename std::enable_if<
    simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        std::is_same<T, float>::value,
    void>::type
simd_fft_butterfly(T* real, T* imag, size_t n)
{
    std::cout << "Performing FFT butterfly with AVX-512" << std::endl;

    const size_t step = 16;

    for (size_t i = 0; i < n; i += step)
    {
        __m512 real_vec = _mm512_loadu_ps(&real[i]);
        __m512 imag_vec = _mm512_loadu_ps(&imag[i]);

        __m512 real_new = _mm512_sub_ps(real_vec, imag_vec);
        __m512 imag_new = _mm512_add_ps(real_vec, imag_vec);

        _mm512_storeu_ps(&real[i], real_new);
        _mm512_storeu_ps(&imag[i], imag_new);
    }
}

template <typename T>
typename std::enable_if<
    simd::enable_if_supported<simd::Feature::AVX512F, T>::value &&
        std::is_same<T, float>::value,
    void>::type
simd_matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n)
{
    std::cout << "Matrix multiply with AVX-512" << std::endl;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j += 16)
        {
            __m512 sum = _mm512_setzero_ps();

            for (int l = 0; l < k; l++)
            {
                __m512 a_val = _mm512_set1_ps(A[i * k + l]);
                __m512 b_vals = _mm512_loadu_ps(&B[l * n + j]);
                sum = _mm512_fmadd_ps(a_val, b_vals, sum);
            }

            _mm512_storeu_ps(&C[i * n + j], sum);
        }
    }
}

int main()
{
    std::cout << "SIMD Performance Benchmark with Real Intrinsics (Fixed)"
              << std::endl;
    std::cout << "-----------------------------------------------------"
              << std::endl;

    benchmark_processing<float>(10000000, 10);
    benchmark_processing<double>(10000000, 10);
    benchmark_processing<int>(10000000, 10);

    std::cout << "Testing different data sizes with float:" << std::endl;
    benchmark_processing<float>(100000, 100);
    benchmark_processing<float>(1000000, 50);
    benchmark_processing<float>(10000000, 20);

    if (simd::has_feature(simd::Feature::AVX512F))
    {
        const size_t fft_size = 1024;
        std::vector<float> real(fft_size), imag(fft_size);

        for (size_t i = 0; i < fft_size; i++)
        {
            real[i] =
                static_cast<float>(std::sin(2 * M_PI * static_cast<double>(i) /
                                            static_cast<double>(fft_size)));
            imag[i] = 0.0f;
        }

        simd_fft_butterfly(real.data(), imag.data(), fft_size);

        std::cout << "FFT butterfly example completed." << std::endl;
    }

    if (simd::has_feature(simd::Feature::AVX512F))
    {
        const int m = 128, k = 128, n = 128;
        std::vector<float> A(m * k, 1.0f);
        std::vector<float> B(k * n, 0.5f);
        std::vector<float> C(m * n, 0.0f);

        simd_matrix_multiply(A.data(), B.data(), C.data(), m, k, n);

        std::cout << "Matrix multiply example completed." << std::endl;
        std::cout << "First few results: ";
        for (size_t i = 0; i < 5; i++)
        {
            std::cout << C[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
