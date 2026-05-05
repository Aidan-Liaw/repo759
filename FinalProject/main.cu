#include "benchmark.cuh"
#include "main.cuh"

#include <exception>
#include <iostream>

int main(int argc, char *argv[]) {
    BenchmarkState state{};

    try {
        const BenchmarkConfig config = parse_benchmark_config(argc, argv);
        initialise_test(config, state);
        const BenchmarkResult result = run_test(config, state);

#if !TEST_RUN_BUSY_WAIT_ONLY
#if TEST_COMPREHENSIVE_CSV
        print_comprehensive_csv(config, state, result);
#else
        print_basic_result(result);
#endif
#endif

        deinitialise_test(state);
        return 0;
    } catch (const std::exception &e) {
        try {
            deinitialise_test(state);
        } catch (const std::exception &cleanup_error) {
            std::cerr << "Cleanup failed after earlier error: " << cleanup_error.what() << std::endl;
        }

        std::cerr << e.what() << std::endl;
        return 1;
    }
}
