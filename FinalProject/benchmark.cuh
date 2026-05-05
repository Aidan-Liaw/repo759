#ifndef GAUSSIAN_BLUR_TEST_BENCHMARK_CUH
#define GAUSSIAN_BLUR_TEST_BENCHMARK_CUH

#include "main.cuh"

BenchmarkConfig parse_benchmark_config(int argc, char *argv[]);
void print_usage(const char *program_name);

void initialise_test(const BenchmarkConfig &config, BenchmarkState &state);
BenchmarkResult run_test(const BenchmarkConfig &config, BenchmarkState &state);
void deinitialise_test(BenchmarkState &state);

void print_basic_result(const BenchmarkResult &result);
void print_comprehensive_csv(const BenchmarkConfig &config,
                             const BenchmarkState &state,
                             const BenchmarkResult &result);

#endif  // GAUSSIAN_BLUR_TEST_BENCHMARK_CUH
