#ifndef GAUSSIAN_BLUR_TEST_MAIN_CUH
#define GAUSSIAN_BLUR_TEST_MAIN_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "convolution.cuh"

// -----------------------------------------------------------------------------
// Basic benchmark switches
// -----------------------------------------------------------------------------
#ifndef TEST_PERF
#define TEST_PERF 1
#endif

#ifndef TEST_COMPREHENSIVE_CSV
#define TEST_COMPREHENSIVE_CSV 0
#endif

#ifndef TEST_CSV_HEADER
#define TEST_CSV_HEADER 1
#endif

#ifndef TEST_RECORD_EACH_ITERATION
#define TEST_RECORD_EACH_ITERATION 0
#endif

#ifndef TEST_WRITE_OUTPUT_IMAGE
#define TEST_WRITE_OUTPUT_IMAGE 0
#endif

#ifndef TEST_TEST_ITERATIONS
#define TEST_TEST_ITERATIONS 100
#endif

#ifndef TEST_WARMUP_ITERATIONS
#define TEST_WARMUP_ITERATIONS 10
#endif

#if TEST_COMPREHENSIVE_CSV && (TEST_PERF != 1)
#error "TEST_COMPREHENSIVE_CSV requires TEST_PERF=1 so per-iteration elapsed_ms values can be recorded."
#endif

// -----------------------------------------------------------------------------
// Image/input switches
// -----------------------------------------------------------------------------
#ifndef TEST_USE_RGB
#define TEST_USE_RGB 1
#endif

#ifndef TEST_USE_PINNED_HOST_MEMORY
#define TEST_USE_PINNED_HOST_MEMORY 1
#endif

#ifndef TEST_USE_PITCHED_LINEAR_MEMORY
#define TEST_USE_PITCHED_LINEAR_MEMORY 1
#endif

// -----------------------------------------------------------------------------
// Gaussian blur kernel path switches
// -----------------------------------------------------------------------------
#ifndef TEST_USE_FUSED_GAUSSIAN
#define TEST_USE_FUSED_GAUSSIAN 0
#endif

// -----------------------------------------------------------------------------
// Host-side launch/path switches
// -----------------------------------------------------------------------------
#ifndef TEST_USE_CUDA_STREAMS
#define TEST_USE_CUDA_STREAMS 1
#endif

#ifndef TEST_USE_CUDA_GRAPHS
#define TEST_USE_CUDA_GRAPHS 0
#endif

#ifndef TEST_USE_GREEN_CONTEXTS
#define TEST_USE_GREEN_CONTEXTS 0
#endif

// 0 = auto, 1 = CUDA Runtime API Green Contexts, 2 = CUDA Driver API Green Contexts.
#ifndef TEST_GREEN_CONTEXT_API
#define TEST_GREEN_CONTEXT_API 0
#endif

#ifndef TEST_GREEN_CONTEXT_SM_COUNT
#define TEST_GREEN_CONTEXT_SM_COUNT 8
#endif

// -----------------------------------------------------------------------------
// Interference/background-load switches
// -----------------------------------------------------------------------------
#ifndef TEST_USE_BUSY_WAIT_COMPETITOR
#define TEST_USE_BUSY_WAIT_COMPETITOR 0
#endif

#ifndef TEST_RUN_BUSY_WAIT_ONLY
#define TEST_RUN_BUSY_WAIT_ONLY 0
#endif

#ifndef TEST_BUSY_WAIT_THREADS
#define TEST_BUSY_WAIT_THREADS 256
#endif

// 0 means auto: blocks = SM count * TEST_BUSY_WAIT_BLOCKS_PER_SM.
#ifndef TEST_BUSY_WAIT_BLOCKS
#define TEST_BUSY_WAIT_BLOCKS 0
#endif

#ifndef TEST_BUSY_WAIT_BLOCKS_PER_SM
#define TEST_BUSY_WAIT_BLOCKS_PER_SM 4
#endif

#ifndef TEST_BUSY_WAIT_MILLISECONDS
#define TEST_BUSY_WAIT_MILLISECONDS 1000
#endif

// These do not enable MIG/MPS. They only label output for experiment logs.
#ifndef TEST_ASSUME_MIG
#define TEST_ASSUME_MIG 0
#endif

#ifndef TEST_ASSUME_MPS
#define TEST_ASSUME_MPS 0
#endif

// -----------------------------------------------------------------------------
// Green Context API selection
// -----------------------------------------------------------------------------
#if TEST_USE_GREEN_CONTEXTS
#if TEST_GREEN_CONTEXT_API == 1
#define TEST_USE_RUNTIME_GREEN_CONTEXTS 1
#define TEST_USE_DRIVER_GREEN_CONTEXTS 0
#elif TEST_GREEN_CONTEXT_API == 2
#define TEST_USE_RUNTIME_GREEN_CONTEXTS 0
#define TEST_USE_DRIVER_GREEN_CONTEXTS 1
#else
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13010)
#define TEST_USE_RUNTIME_GREEN_CONTEXTS 1
#define TEST_USE_DRIVER_GREEN_CONTEXTS 0
#else
#define TEST_USE_RUNTIME_GREEN_CONTEXTS 0
#define TEST_USE_DRIVER_GREEN_CONTEXTS 1
#endif
#endif
#else
#define TEST_USE_RUNTIME_GREEN_CONTEXTS 0
#define TEST_USE_DRIVER_GREEN_CONTEXTS 0
#endif

#if TEST_USE_RUNTIME_GREEN_CONTEXTS && (!defined(CUDART_VERSION) || (CUDART_VERSION < 13010))
#error "TEST_GREEN_CONTEXT_API=1 requires CUDA Runtime API headers from CUDA 13.1 or newer. Use TEST_GREEN_CONTEXT_API=2 for the Driver API fallback."
#endif

#if TEST_USE_DRIVER_GREEN_CONTEXTS && (!defined(CUDA_VERSION) || (CUDA_VERSION < 12040))
#error "TEST_GREEN_CONTEXT_API=2 requires CUDA Driver API headers with Green Context support."
#endif

// -----------------------------------------------------------------------------
// Error handling macros
// -----------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t cuda_status = (call);                                                                \
        if (cuda_status != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +                 \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(cuda_status)); \
        }                                                                                                \
    } while (0)

#define CHECK_CU(call)                                                                                   \
    do {                                                                                                 \
        CUresult cu_status = (call);                                                                     \
        if (cu_status != CUDA_SUCCESS) {                                                                 \
            const char *cu_error_name = nullptr;                                                         \
            const char *cu_error_string = nullptr;                                                       \
            cuGetErrorName(cu_status, &cu_error_name);                                                   \
            cuGetErrorString(cu_status, &cu_error_string);                                               \
            throw std::runtime_error(std::string("CUDA Driver error at ") + __FILE__ + ":" +           \
                                     std::to_string(__LINE__) + ": " +                                  \
                                     (cu_error_name ? cu_error_name : "unknown") + " - " +              \
                                     (cu_error_string ? cu_error_string : "no error string"));           \
        }                                                                                                \
    } while (0)

#define CHECK_NVJPEG(call)                                                                               \
    do {                                                                                                 \
        nvjpegStatus_t nvjpeg_status = (call);                                                           \
        if (nvjpeg_status != NVJPEG_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error(std::string("nvJPEG error at ") + __FILE__ + ":" +                \
                                     std::to_string(__LINE__) + ": status " +                           \
                                     std::to_string(static_cast<int>(nvjpeg_status)));                    \
        }                                                                                                \
    } while (0)

struct GreenContextState {
    bool enabled = false;
#if TEST_USE_RUNTIME_GREEN_CONTEXTS
    cudaExecutionContext_t critical_ctx = nullptr;
    cudaExecutionContext_t main_ctx = nullptr;
    cudaDevResourceDesc_t critical_desc = nullptr;
    cudaDevResourceDesc_t main_desc = nullptr;
    std::vector<cudaDevResource> groups;
    std::vector<cudaDevResource> main_resources;
#endif
#if TEST_USE_DRIVER_GREEN_CONTEXTS
    CUdevice cu_device = 0;
    CUgreenCtx critical_gc = nullptr;
    CUgreenCtx main_gc = nullptr;
    CUdevResourceDesc critical_desc = nullptr;
    CUdevResourceDesc main_desc = nullptr;
    CUstream critical_cu_stream = nullptr;
    CUstream main_cu_stream = nullptr;
    std::vector<CUdevResource> groups;
    std::vector<CUdevResource> main_resources;
#endif
};

struct BenchmarkStreams {
    cudaStream_t stream = nullptr;
    cudaStream_t busy_wait_stream = nullptr;
    GreenContextState green_context_state{};
};

struct NvjpegState {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t jpeg_handle = nullptr;
};

struct BenchmarkConfig {
    int R = 0;
    int mask_length = 1;
    int threads_per_block = 256;
    int test_iterations = TEST_TEST_ITERATIONS;
    float sigma = 1.0f;
    int gpu_device_index = 0;
    std::string image_path{};
    std::string output_image_path{};
};

struct BenchmarkState {
    BenchmarkStreams benchmark_streams{};
    NvjpegState nvjpeg_state{};

    std::vector<unsigned char> jpeg_buffer{};
    std::vector<float> mask{};

    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling{};
    int widths[NVJPEG_MAX_COMPONENT]{};
    int heights[NVJPEG_MAX_COMPONENT]{};

    int image_width = 0;
    int image_height = 0;
    int channels = 0;
    nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_UNCHANGED;

    unsigned char *gImage_u8 = nullptr;
    float *gImage = nullptr;
    float *gMask = nullptr;
    float *gTemp = nullptr;
    float *gOutput = nullptr;
    unsigned long long *gBusyWaitSink = nullptr;

    std::size_t image_u8_pitch = 0;
    std::size_t image_pitch = 0;
    std::size_t temp_pitch = 0;
    std::size_t output_pitch = 0;

    int busy_wait_blocks = 0;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    float *output = nullptr;
    std::size_t output_length = 0;

    bool nvjpeg_initialised = false;
    bool streams_initialised = false;
    bool output_allocated = false;
};

struct BenchmarkResult {
    float ms = 0.0f;
    float output_sentinel = 0.0f;
    std::vector<float> measurements{};
};

#endif  // GAUSSIAN_BLUR_TEST_MAIN_CUH
