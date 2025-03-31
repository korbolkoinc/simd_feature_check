#include <benchmark/benchmark.h>
#include "simd/feature_check.hpp"


static void BM_GetSimdSupport(benchmark::State& state) {
    for (auto _ : state) {
        int simd_support = simd::get_simd_support();
        benchmark::DoNotOptimize(simd_support);
    }
}
BENCHMARK(BM_GetSimdSupport)->Iterations(1000000);

BENCHMARK_MAIN();   