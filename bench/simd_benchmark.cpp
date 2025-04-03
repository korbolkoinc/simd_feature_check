#include <benchmark/benchmark.h>
#include "simd/feature_check.hpp"


static void BM_GetSimdSupport(benchmark::State& state) {
    for (auto _ : state) {
        // ...
    }
}
BENCHMARK(BM_GetSimdSupport)->Iterations(1000000);

BENCHMARK_MAIN();   