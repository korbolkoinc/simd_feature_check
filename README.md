# simd_feature_check

A modern C++20 library for CPU feature detection and SIMD vector operations. This library provides compile-time and runtime detection of SIMD instruction sets (SSE, AVX, AVX-512, AMX) along with a high-level vector abstraction layer for writing portable SIMD code.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Build Options](#build-options)
- [Library Architecture](#library-architecture)
- [Usage Examples](#usage-examples)
  - [Basic Feature Detection](#basic-feature-detection)
  - [Compile-Time Detection](#compile-time-detection)
  - [Runtime Detection](#runtime-detection)
  - [FeatureDetector Class](#featuredetector-class)
  - [Function Dispatch](#function-dispatch)
  - [Vector Operations](#vector-operations)
  - [Mask Operations](#mask-operations)
  - [Mathematical Functions](#mathematical-functions)
  - [Memory Operations](#memory-operations)
  - [Type Conversions](#type-conversions)
- [Supported Features](#supported-features)
- [Integration](#integration)
- [Testing](#testing)
- [License](#license)

## Overview

simd_feature_check solves two fundamental challenges in SIMD programming:

1. **Feature Detection**: Determining which SIMD instruction sets are available on the target CPU, both at compile time and runtime
2. **Vector Abstraction**: Writing portable SIMD code that automatically adapts to the best available instruction set

The library uses CPUID instructions on x86/x86_64 to detect processor capabilities and provides a clean API for querying supported features. The vector abstraction layer automatically selects the optimal implementation based on detected capabilities.

## Features

- Comprehensive detection of 50+ SIMD features including SSE, AVX, AVX-512, and AMX
- Compile-time detection via template metaprogramming
- Runtime detection using CPUID instructions
- Feature-to-string conversion for logging and debugging
- Highest feature detection for capability reporting
- Vector abstraction supporting multiple data types and sizes
- Automatic dispatch to optimal SIMD implementations
- Cross-platform support with architecture detection
- Modern C++20 with concepts for type safety
- Zero-overhead abstractions

## Requirements

- CMake 3.16 or newer
- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Git

### Compiler Support

| Compiler | Minimum Version | Notes |
|----------|----------------|-------|
| GCC | 10.0 | Full support |
| Clang | 10.0 | Full support |
| MSVC | 2019 (16.8) | Full support |

## Installation

Clone the repository:

```bash
git clone https://github.com/hun756/CPP-Starter-Template.git simd_feature_check
cd simd_feature_check
```

Create a build directory and configure:

```bash
mkdir build && cd build
cmake ..
```

Build the library:

```bash
cmake --build .
```

Run tests:

```bash
ctest
```

## Build Options

The following CMake options are available:

| Option | Default | Description |
|--------|---------|-------------|
| BUILD_SHARED_LIBS | OFF | Build shared libraries instead of static |
| BUILD_EXAMPLES | ON | Build example programs |
| BUILD_TESTS | ON | Build and enable tests |
| BUILD_BENCHMARKS | OFF | Build benchmarking programs |
| ENABLE_COVERAGE | OFF | Enable code coverage reporting |
| ENABLE_SANITIZERS | OFF | Enable sanitizers in debug builds |
| ENABLE_PCH | OFF | Enable precompiled headers |
| ENABLE_LTO | OFF | Enable Link Time Optimization |

Example configuration with sanitizers enabled:

```bash
cmake .. -DENABLE_SANITIZERS=ON -DBUILD_EXAMPLES=ON
```

## Library Architecture

The library is organized into several namespaces and components:

### Core Namespace (simd)

The `simd` namespace contains feature detection functionality:

- `simd::Feature` - Enum class listing all detectable features
- `simd::has_feature()` - Runtime feature check
- `simd::compile_time::` - Compile-time feature detection
- `simd::runtime::` - Runtime feature detection
- `simd::FeatureDetector<T>` - Template class for feature introspection

### Vector SIMD Namespace (vector_simd)

The `vector_simd` namespace provides vector abstractions:

- `Vector<T, N>` - Fixed-size SIMD vector class
- `Mask<T, N>` - Mask type for predicate operations
- Type aliases like `float_v<4>`, `int32_v<8>`, etc.
- Native-width types like `float_vn` for optimal vector width

### Implementation Details

The library uses a layered architecture:

1. `simd/common.hpp` - Architecture and compiler detection macros
2. `simd/feature_check.hpp` - Core feature detection implementation
3. `simd/registers/types.hpp` - Register type mappings for each ISA
4. `simd/vector/vector.hpp` - High-level vector abstraction
5. `simd/impl/` - Architecture-specific implementations

## Usage Examples

### Basic Feature Detection

The simplest way to check for SIMD support:

```cpp
#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    // Check for specific features
    if (simd::has_feature(simd::Feature::SSE)) {
        std::cout << "SSE is supported" << std::endl;
    }
    
    if (simd::has_feature(simd::Feature::AVX)) {
        std::cout << "AVX is supported" << std::endl;
    }
    
    if (simd::has_feature(simd::Feature::AVX2)) {
        std::cout << "AVX2 is supported" << std::endl;
    }
    
    if (simd::has_feature(simd::Feature::AVX512F)) {
        std::cout << "AVX-512 Foundation is supported" << std::endl;
    }
    
    // Get CPU vendor string
    std::cout << "CPU Vendor: " << simd::get_cpu_vendor() << std::endl;
    
    // Get the highest supported SIMD feature
    simd::Feature highest = simd::highest_feature();
    std::cout << "Highest SIMD feature: " 
              << simd::feature_to_string(highest) << std::endl;
    
    return 0;
}
```

### Compile-Time Detection

When you need to know features at compile time for conditional compilation:

```cpp
#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    // Check compile-time availability using boolean constants
    std::cout << "SSE compile-time: " 
              << (simd::compile_time::sse ? "Yes" : "No") << std::endl;
    
    std::cout << "AVX compile-time: " 
              << (simd::compile_time::avx ? "Yes" : "No") << std::endl;
    
    std::cout << "AVX2 compile-time: " 
              << (simd::compile_time::avx2 ? "Yes" : "No") << std::endl;
    
    // Template-based compile-time checks
    std::cout << "SSE2 available: " 
              << (simd::compile_time::has<simd::Feature::SSE2>() ? "Yes" : "No")
              << std::endl;
    
    std::cout << "AVX-512F available: " 
              << (simd::compile_time::has<simd::Feature::AVX512F>() ? "Yes" : "No")
              << std::endl;
    
    // Get maximum compile-time feature
    std::cout << "Max compile-time feature: " 
              << simd::feature_to_string(simd::compile_time::max_feature)
              << std::endl;
    
    return 0;
}
```

Using compile-time detection for conditional compilation:

```cpp
#include "simd/feature_check.hpp"

void process_data(float* data, size_t size)
{
    if constexpr (simd::compile_time::has<simd::Feature::AVX512F>()) {
        // This code is only compiled if AVX-512F is available
        // Compiler will use AVX-512 intrinsics here
        process_avx512(data, size);
    }
    else if constexpr (simd::compile_time::has<simd::Feature::AVX2>()) {
        // Fallback to AVX2
        process_avx2(data, size);
    }
    else {
        // Scalar fallback
        process_scalar(data, size);
    }
}
```

### Runtime Detection

For runtime checks that adapt to the executing CPU:

```cpp
#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    // Runtime checks using template syntax
    std::cout << "SSE runtime: " 
              << (simd::runtime::has<simd::Feature::SSE>() ? "Yes" : "No")
              << std::endl;
    
    std::cout << "AVX runtime: " 
              << (simd::runtime::has<simd::Feature::AVX>() ? "Yes" : "No")
              << std::endl;
    
    std::cout << "AVX2 runtime: " 
              << (simd::runtime::has<simd::Feature::AVX2>() ? "Yes" : "No")
              << std::endl;
    
    // Get highest runtime feature
    std::cout << "Highest runtime feature: " 
              << simd::feature_to_string(simd::runtime::highest_feature())
              << std::endl;
    
    return 0;
}
```

### FeatureDetector Class

The FeatureDetector template provides detailed introspection for any feature:

```cpp
#include "simd/feature_check.hpp"
#include <iostream>

int main()
{
    // Create detectors for specific features
    using AVXDetector = simd::FeatureDetector<simd::Feature::AVX>;
    using AVX2Detector = simd::FeatureDetector<simd::Feature::AVX2>;
    using AVX512Detector = simd::FeatureDetector<simd::Feature::AVX512F>;
    
    // AVX feature information
    std::cout << "AVX Feature:" << std::endl;
    std::cout << "  Name: " << AVXDetector::name() << std::endl;
    std::cout << "  Compile-time support: " 
              << (AVXDetector::compile_time ? "Yes" : "No") << std::endl;
    std::cout << "  Runtime support: " 
              << (AVXDetector::available() ? "Yes" : "No") << std::endl;
    
    // AVX2 feature information
    std::cout << "AVX2 Feature:" << std::endl;
    std::cout << "  Name: " << AVX2Detector::name() << std::endl;
    std::cout << "  Compile-time support: " 
              << (AVX2Detector::compile_time ? "Yes" : "No") << std::endl;
    std::cout << "  Runtime support: " 
              << (AVX2Detector::available() ? "Yes" : "No") << std::endl;
    
    // AVX-512F feature information
    std::cout << "AVX-512F Feature:" << std::endl;
    std::cout << "  Name: " << AVX512Detector::name() << std::endl;
    std::cout << "  Compile-time support: " 
              << (AVX512Detector::compile_time ? "Yes" : "No") << std::endl;
    std::cout << "  Runtime support: " 
              << (AVX512Detector::available() ? "Yes" : "No") << std::endl;
    
    return 0;
}
```

### Function Dispatch

Manual dispatch based on detected features:

```cpp
#include "simd/feature_check.hpp"
#include <iostream>

// Scalar implementation (always available)
float* add_vectors_scalar(const float* a, const float* b, float* result,
                          size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// AVX implementation (conditionally compiled)
#if SIMD_HAS_AVX
float* add_vectors_avx(const float* a, const float* b, float* result,
                       size_t size)
{
    // AVX-optimized implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}
#endif

// AVX-512 implementation (conditionally compiled)
#if SIMD_HAS_AVX512F
float* add_vectors_avx512(const float* a, const float* b, float* result,
                          size_t size)
{
    // AVX-512 optimized implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}
#endif

int main()
{
    using AddFunc = float* (*)(const float*, const float*, float*, size_t);
    
    AddFunc best_impl;
    
    // Select best implementation at runtime
    if (simd::has_feature(simd::Feature::AVX512F)) {
    #if SIMD_HAS_AVX512F
        best_impl = add_vectors_avx512;
    #else
        best_impl = add_vectors_scalar;
    #endif
    }
    else if (simd::has_feature(simd::Feature::AVX)) {
    #if SIMD_HAS_AVX
        best_impl = add_vectors_avx;
    #else
        best_impl = add_vectors_scalar;
    #endif
    }
    else {
        best_impl = add_vectors_scalar;
    }
    
    // Use the selected implementation
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];
    
    best_impl(a, b, result, 4);
    
    std::cout << "Result: [" << result[0] << ", " << result[1] 
              << ", " << result[2] << ", " << result[3] << "]" << std::endl;
    
    return 0;
}
```

### Vector Operations

The library provides a high-level vector abstraction that automatically uses the best available SIMD instructions:

```cpp
#include "simd/simd.hpp"
#include <iostream>

int main()
{
    using namespace vector_simd;
    
    // Create vectors with fixed size
    float_v<4> a{1.0f, 2.0f, 3.0f, 4.0f};
    float_v<4> b{5.0f, 6.0f, 7.0f, 8.0f};
    
    // Basic arithmetic operations
    float_v<4> sum = a + b;           // Element-wise addition
    float_v<4> diff = a - b;          // Element-wise subtraction
    float_v<4> prod = a * b;          // Element-wise multiplication
    float_v<4> quot = b / a;          // Element-wise division
    
    // Compound assignment operators
    float_v<4> c{0.0f, 0.0f, 0.0f, 0.0f};
    c += a;
    c *= b;
    
    // Extract and insert elements
    float val = sum.extract(0);       // Get first element
    sum.insert(0, 10.0f);             // Set first element
    
    // Store results to memory
    float result[4];
    sum.store(result);                // Unaligned store
    sum.store_aligned(result);        // Aligned store (faster)
    
    // Load from memory
    float_v<4> loaded = float_v<4>::load(result);
    float_v<4> aligned = float_v<4>::load_aligned(result);
    
    // Convert to std::array
    std::array<float, 4> arr = sum.to_array();
    
    return 0;
}
```

### Mask Operations

Masks enable conditional operations and predication:

```cpp
#include "simd/simd.hpp"
#include <iostream>

int main()
{
    using namespace vector_simd;
    
    float_v<4> a{1.0f, 5.0f, 3.0f, 8.0f};
    float_v<4> b{4.0f, 2.0f, 6.0f, 7.0f};
    
    // Comparison operations return masks
    auto mask_eq = (a == b);          // Equal comparison
    auto mask_ne = (a != b);          // Not equal
    auto mask_lt = (a < b);           // Less than
    auto mask_le = (a <= b);          // Less than or equal
    auto mask_gt = (a > b);           // Greater than
    auto mask_ge = (a >= b);          // Greater than or equal
    
    // Select based on mask
    float_v<4> min_vals = float_v<4>::select(mask_lt, a, b);  // Element-wise min
    float_v<4> max_vals = float_v<4>::select(mask_gt, a, b);  // Element-wise max
    
    // Blend vectors based on mask
    float_v<4> blended = a.blend(b, mask_gt);  // Take from b where a > b
    
    return 0;
}
```

### Mathematical Functions

Comprehensive math operations with automatic SIMD optimization:

```cpp
#include "simd/simd.hpp"
#include <iostream>
#include <cmath>

int main()
{
    using namespace vector_simd;
    
    float_v<4> a{1.0f, 4.0f, 9.0f, 16.0f};
    float_v<4> b{-2.5f, 3.7f, -1.2f, 0.0f};
    
    // Basic math functions
    float_v<4> abs_val = a.abs();         // Absolute value
    float_v<4> sqrt_val = a.sqrt();       // Square root
    
    // Trigonometric functions
    float_v<4> angles{0.0f, 1.5708f, 3.14159f, 4.71239f};
    float_v<4> sin_val = angles.sin();    // Sine
    float_v<4> cos_val = angles.cos();    // Cosine
    float_v<4> tan_val = angles.tan();    // Tangent
    
    // Exponential and logarithmic
    float_v<4> exp_val = a.exp();         // e^x
    float_v<4> log_val = a.log();         // Natural log
    
    // Rounding functions
    float_v<4> round_vals{1.2f, 3.7f, -2.3f, -4.8f};
    float_v<4> floor_val = round_vals.floor();   // Floor
    float_v<4> ceil_val = round_vals.ceil();     // Ceiling
    float_v<4> round_val = round_vals.round();   // Round
    float_v<4> trunc_val = round_vals.trunc();   // Truncate
    
    // Reciprocal and reciprocal square root (approximate, fast)
    float_v<4> rcp_val = a.rcp();         // 1/x
    float_v<4> rsqrt_val = a.rsqrt();     // 1/sqrt(x)
    
    // Fused multiply-add: a * b + c
    float_v<4> c{1.0f, 1.0f, 1.0f, 1.0f};
    float_v<4> fmadd_result = a.fmadd(b, c);
    
    // Fused multiply-subtract: a * b - c
    float_v<4> fmsub_result = a.fmadd(b, c);
    
    // Min and max
    float_v<4> min_val = a.min(b);
    float_v<4> max_val = a.max(b);
    
    // Clamp to range
    float_v<4> lo{0.0f, 0.0f, 0.0f, 0.0f};
    float_v<4> hi{10.0f, 10.0f, 10.0f, 10.0f};
    float_v<4> clamped = b.clamp(lo, hi);
    
    return 0;
}
```

### Memory Operations

Efficient memory access patterns:

```cpp
#include "simd/simd.hpp"
#include <iostream>

int main()
{
    using namespace vector_simd;
    
    alignas(64) float data[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    // Aligned load (fastest, requires aligned address)
    float_v<4> v1 = float_v<4>::load_aligned(data);
    
    // Unaligned load (works with any address)
    float_v<4> v2 = float_v<4>::load_unaligned(data + 1);
    
    // Aligned store
    alignas(64) float result[4];
    v1.store_aligned(result);
    
    // Unaligned store
    float unaligned_result[4];
    v2.store_unaligned(unaligned_result);
    
    // Non-temporal store (bypasses cache, useful for write-once data)
    alignas(64) float nt_buffer[4];
    v1.store_nt(nt_buffer);
    
    // Gather: load non-contiguous elements
    int32_v<4> indices{0, 2, 4, 6};
    float_v<4> gathered = float_v<4>::gather(data, indices);
    
    // Scatter: store to non-contiguous locations
    float output[16] = {0};
    gathered.scatter(output, indices);
    
    // Prefetch data into cache
    float_v<4>::prefetch(data + 8);
    
    return 0;
}
```

### Type Conversions

Convert between different vector types:

```cpp
#include "simd/simd.hpp"
#include <iostream>

int main()
{
    using namespace vector_simd;
    
    // Integer vectors
    int32_v<4> ints{1, 2, 3, 4};
    
    // Convert to float
    float_v<4> floats = ints.convert<float>();
    
    // Convert to double
    double_v<4> doubles = ints.convert<double>();
    
    // Different integer widths
    int16_v<8> shorts{1, 2, 3, 4, 5, 6, 7, 8};
    int32_v<8> expanded = shorts.convert<int32_t>();
    
    // Saturation arithmetic (prevents overflow)
    uint8_v<16> a{200, 200, 200, 200, 200, 200, 200, 200,
                  200, 200, 200, 200, 200, 200, 200, 200};
    uint8_v<16> b{100, 100, 100, 100, 100, 100, 100, 100,
                  100, 100, 100, 100, 100, 100, 100, 100};
    
    uint8_v<16> sat_add = a.add_sat(b);  // Saturates at 255
    uint8_v<16> sat_sub = b.sub_sat(a);  // Saturates at 0
    
    return 0;
}
```

### Horizontal Operations

Reduce vector elements to scalar values:

```cpp
#include "simd/simd.hpp"
#include <iostream>

int main()
{
    using namespace vector_simd;
    
    float_v<4> a{1.0f, 2.0f, 3.0f, 4.0f};
    
    // Horizontal sum: 1 + 2 + 3 + 4 = 10
    float sum = a.hsum();
    
    // Horizontal min: min(1, 2, 3, 4) = 1
    float min_val = a.hmin();
    
    // Horizontal max: max(1, 2, 3, 4) = 4
    float max_val = a.hmax();
    
    // Dot product
    float_v<4> b{1.0f, 1.0f, 1.0f, 1.0f};
    float dot = a.dot(b);
    
    // Reduce operations (alternative names)
    float reduce_sum = a.reduce_add();
    float reduce_min = a.reduce_min();
    float reduce_max = a.reduce_max();
    
    return 0;
}
```

### Native Width Vectors

Use vectors sized for the best available instruction set:

```cpp
#include "simd/simd.hpp"
#include <iostream>

int main()
{
    using namespace vector_simd;
    
    // These types automatically use the optimal width:
    // - SSE2: 4 floats (128-bit)
    // - AVX: 8 floats (256-bit)
    // - AVX-512: 16 floats (512-bit)
    
    float_vn a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float_vn b = a * 2.0f;
    
    // Process arrays in chunks of native width
    void process_array(float* data, size_t size)
    {
        size_t i = 0;
        
        // Process in native-width chunks
        for (; i + float_vn::size_value <= size; i += float_vn::size_value) {
            float_vn v = float_vn::load_aligned(data + i);
            v = v * 2.0f;
            v.store_aligned(data + i);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            data[i] *= 2.0f;
        }
    }
    
    return 0;
}
```

## Supported Features

The library detects the following SIMD features:

### Legacy SIMD
- MMX
- SSE, SSE2, SSE3, SSSE3
- SSE4.1, SSE4.2

### AVX Family
- AVX
- AVX2
- FMA (Fused Multiply-Add)
- F16C (Half-precision conversion)

### AVX-512 Foundation and Extensions
- AVX-512F (Foundation)
- AVX-512CD (Conflict Detection)
- AVX-512DQ (Doubleword and Quadword)
- AVX-512BW (Byte and Word)
- AVX-512VL (Vector Length)
- AVX-512IFMA (Integer FMA)
- AVX-512VBMI, VBMI2 (Vector Byte Manipulation)
- AVX-512VNNI (Neural Network)
- AVX-512BITALG (Bit Algorithms)
- AVX-512VPOPCNTDQ (Vector Population Count)
- AVX-512VP2INTERSECT
- AVX-512BF16 (BFloat16)
- AVX-512FP16 (Float16)
- AVX-512_4VNNIW, AVX-512_4FMAPS

### Intel AMX (Advanced Matrix Extensions)
- AMX_TILE
- AMX_INT8
- AMX_BF16

### Cryptographic Extensions
- AES, VAES
- PCLMULQDQ, VPCLMULQDQ
- SHA

### Bit Manipulation
- POPCNT (Population Count)
- LZCNT (Leading Zero Count)
- BMI1, BMI2 (Bit Manipulation Instructions)

### Other Extensions
- MOVBE (Move Byte Swap)
- RDRND, RDSEED (Random Number Generation)
- ADX (Multi-Precision Add-Carry)
- PREFETCHW, PREFETCHWT1
- GFNI (Galois Field)
- RDPID (Read Processor ID)
- SGX (Software Guard Extensions)
- CET_IBT, CET_SS (Control-flow Enforcement Technology)

## Integration

### Using as a CMake Subdirectory

Add this to your CMakeLists.txt:

```cmake
add_subdirectory(path/to/simd_feature_check)
target_link_libraries(your_target PRIVATE simd_feature_check::simd_feature_check)
```

### Using as an Installed Package

After installing the library:

```cmake
find_package(simd_feature_check REQUIRED)
target_link_libraries(your_target PRIVATE simd_feature_check::simd_feature_check)
```

### Using Macros in Your Code

The library defines convenience macros for conditional compilation:

```cpp
#include "simd/common.hpp"

#if SIMD_HAS_AVX2
    // AVX2-specific code
#endif

#if SIMD_HAS_AVX512F
    // AVX-512-specific code
#endif

// Alternative syntax (equivalent)
#if SIMD_AVX2
    // AVX2-specific code
#endif
```

## Testing

Run the test suite:

```bash
cd build
ctest --output-on-failure
```

Build and run tests with verbose output:

```bash
ctest -V
```

Run specific test categories:

```bash
ctest -R simd_features    # Run feature detection tests
ctest -R vector_ops       # Run vector operation tests
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
