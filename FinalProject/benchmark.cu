#include "benchmark.cuh"
#include "file_handling.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

__global__ void u8_to_float_planar_kernel(const unsigned char *__restrict__ gImage_u8,
                                          float *__restrict__ gImage,
                                          int image_width,
                                          int image_height,
                                          int channels,
                                          std::size_t image_u8_pitch,
                                          std::size_t image_pitch) {
    const unsigned long long thread_idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long plane_pixels = static_cast<unsigned long long>(image_width) * static_cast<unsigned long long>(image_height);
    const unsigned long long pixel_count = plane_pixels * static_cast<unsigned long long>(channels);

    if (thread_idx >= pixel_count) {
        return;
    }

    const int channel_idx = static_cast<int>(thread_idx / plane_pixels);
    const unsigned long long plane_idx = thread_idx - static_cast<unsigned long long>(channel_idx) * plane_pixels;
    const int row_idx = static_cast<int>(plane_idx / static_cast<unsigned long long>(image_width));
    const int column_idx = static_cast<int>(plane_idx - static_cast<unsigned long long>(row_idx) * static_cast<unsigned long long>(image_width));

    const unsigned char *src_channel = gImage_u8 + static_cast<std::size_t>(channel_idx) * image_u8_pitch * image_height;
    const unsigned char *src_row = src_channel + static_cast<std::size_t>(row_idx) * image_u8_pitch;

    float *dst_channel = reinterpret_cast<float *>(reinterpret_cast<char *>(gImage) + static_cast<std::size_t>(channel_idx) * image_pitch * image_height);
    float *dst_row = reinterpret_cast<float *>(reinterpret_cast<char *>(dst_channel) + static_cast<std::size_t>(row_idx) * image_pitch);

    dst_row[column_idx] = static_cast<float>(src_row[column_idx]) * (1.0f / 255.0f);
}

__global__ void busy_wait_kernel(unsigned long long wait_cycles, unsigned long long *busy_wait_sink) {
    const unsigned long long start = clock64();
    unsigned long long now = start;

    do {
        now = clock64();
    } while ((now - start) < wait_cycles);

    if (threadIdx.x == 0 && busy_wait_sink != nullptr) {
        busy_wait_sink[blockIdx.x] = now;
    }
}

cudaError_t launch_u8_to_float_planar(const unsigned char *gImage_u8,
                                      float *gImage,
                                      int image_width,
                                      int image_height,
                                      int channels,
                                      std::size_t image_u8_pitch,
                                      std::size_t image_pitch,
                                      unsigned int threads_per_block,
                                      cudaStream_t stream) {
    const unsigned long long pixel_count = static_cast<unsigned long long>(image_width) *
                                           static_cast<unsigned long long>(image_height) *
                                           static_cast<unsigned long long>(channels);
    const unsigned int number_of_blocks = static_cast<unsigned int>((pixel_count + threads_per_block - 1) / threads_per_block);
    u8_to_float_planar_kernel<<<number_of_blocks, threads_per_block, 0, stream>>>(gImage_u8,
                                                                                  gImage,
                                                                                  image_width,
                                                                                  image_height,
                                                                                  channels,
                                                                                  image_u8_pitch,
                                                                                  image_pitch);
    return cudaGetLastError();
}

cudaError_t run_gaussian_blur(const float *gImage,
                              const float *gMask,
                              float *gTemp,
                              float *gOutput,
                              int image_width,
                              int image_height,
                              int channels,
                              int R,
                              std::size_t image_pitch,
                              std::size_t temp_pitch,
                              std::size_t output_pitch,
                              unsigned int threads_per_block,
                              cudaStream_t stream) {
#if TEST_USE_FUSED_GAUSSIAN
    (void) gTemp;
    (void) temp_pitch;
    return gaussian_blur_fused_channels(gImage,
                                        gMask,
                                        gOutput,
                                        image_width,
                                        image_height,
                                        channels,
                                        R,
                                        static_cast<long long>(image_pitch),
                                        static_cast<long long>(output_pitch),
                                        threads_per_block,
                                        stream,
                                        cudaMemcpyDeviceToDevice);
#else
    return gaussian_blur_separable_channels(gImage,
                                            gMask,
                                            gTemp,
                                            gOutput,
                                            image_width,
                                            image_height,
                                            channels,
                                            R,
                                            static_cast<long long>(image_pitch),
                                            static_cast<long long>(temp_pitch),
                                            static_cast<long long>(output_pitch),
                                            threads_per_block,
                                            stream,
                                            cudaMemcpyDeviceToDevice);
#endif
}

int automatic_busy_wait_blocks(int gpu_device_index) {
    int multiProcessorCount = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, gpu_device_index));
    return std::max(1, multiProcessorCount * TEST_BUSY_WAIT_BLOCKS_PER_SM);
}

unsigned long long busy_wait_cycles(int gpu_device_index) {
    int clockRatekHz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clockRatekHz, cudaDevAttrClockRate, gpu_device_index));
    return static_cast<unsigned long long>(clockRatekHz) * static_cast<unsigned long long>(TEST_BUSY_WAIT_MILLISECONDS);
}

cudaError_t launch_busy_wait_competitor(unsigned long long *gBusyWaitSink,
                                        int gpu_device_index,
                                        cudaStream_t busy_wait_stream) {
    const int busy_wait_blocks = (TEST_BUSY_WAIT_BLOCKS > 0) ? TEST_BUSY_WAIT_BLOCKS : automatic_busy_wait_blocks(gpu_device_index);
    const unsigned long long wait_cycles = busy_wait_cycles(gpu_device_index);
    busy_wait_kernel<<<busy_wait_blocks, TEST_BUSY_WAIT_THREADS, 0, busy_wait_stream>>>(wait_cycles, gBusyWaitSink);
    return cudaGetLastError();
}

#if TEST_USE_RUNTIME_GREEN_CONTEXTS
void append_runtime_main_resources(std::vector<cudaDevResource> &main_resources,
                                   const std::vector<cudaDevResource> &groups,
                                   const cudaDevResource &remainder_res) {
    for (std::size_t idx = 1; idx < groups.size(); ++idx) {
        main_resources.push_back(groups[idx]);
    }

    if (remainder_res.type == cudaDevResourceTypeSm && remainder_res.sm.smCount > 0) {
        main_resources.push_back(remainder_res);
    }

    if (main_resources.empty()) {
        throw std::runtime_error("Green Context split left no SM resources for the busy-wait/background stream");
    }
}
#endif

#if TEST_USE_DRIVER_GREEN_CONTEXTS
void append_driver_main_resources(std::vector<CUdevResource> &main_resources,
                                  const std::vector<CUdevResource> &groups,
                                  const CUdevResource &remainder_res) {
    for (std::size_t idx = 1; idx < groups.size(); ++idx) {
        main_resources.push_back(groups[idx]);
    }

    if (remainder_res.type == CU_DEV_RESOURCE_TYPE_SM && remainder_res.sm.smCount > 0) {
        main_resources.push_back(remainder_res);
    }

    if (main_resources.empty()) {
        throw std::runtime_error("Green Context split left no SM resources for the busy-wait/background stream");
    }
}
#endif

BenchmarkStreams create_benchmark_streams(int gpu_device_index) {
    BenchmarkStreams benchmark_streams{};

#if TEST_USE_RUNTIME_GREEN_CONTEXTS
    cudaDevResource initial_GPU_SM_resources{};
    CHECK_CUDA(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));

    unsigned int nbGroups = 0;
    CHECK_CUDA(cudaDevSmResourceSplitByCount(nullptr,
                                             &nbGroups,
                                             &initial_GPU_SM_resources,
                                             nullptr,
                                             0,
                                             TEST_GREEN_CONTEXT_SM_COUNT));

    if (nbGroups == 0) {
        throw std::runtime_error("Runtime Green Context SM split produced zero groups");
    }

    benchmark_streams.green_context_state.groups.resize(nbGroups);
    cudaDevResource remainder_res{};
    CHECK_CUDA(cudaDevSmResourceSplitByCount(benchmark_streams.green_context_state.groups.data(),
                                             &nbGroups,
                                             &initial_GPU_SM_resources,
                                             &remainder_res,
                                             0,
                                             TEST_GREEN_CONTEXT_SM_COUNT));
    benchmark_streams.green_context_state.groups.resize(nbGroups);

    append_runtime_main_resources(benchmark_streams.green_context_state.main_resources,
                                  benchmark_streams.green_context_state.groups,
                                  remainder_res);

    CHECK_CUDA(cudaDevResourceGenerateDesc(&benchmark_streams.green_context_state.critical_desc,
                                           benchmark_streams.green_context_state.groups.data(),
                                           1));
    CHECK_CUDA(cudaDevResourceGenerateDesc(&benchmark_streams.green_context_state.main_desc,
                                           benchmark_streams.green_context_state.main_resources.data(),
                                           static_cast<unsigned int>(benchmark_streams.green_context_state.main_resources.size())));

    CHECK_CUDA(cudaGreenCtxCreate(&benchmark_streams.green_context_state.critical_ctx,
                                  benchmark_streams.green_context_state.critical_desc,
                                  gpu_device_index,
                                  0));
    CHECK_CUDA(cudaGreenCtxCreate(&benchmark_streams.green_context_state.main_ctx,
                                  benchmark_streams.green_context_state.main_desc,
                                  gpu_device_index,
                                  0));

    CHECK_CUDA(cudaExecutionCtxStreamCreate(&benchmark_streams.stream,
                                            benchmark_streams.green_context_state.critical_ctx,
                                            cudaStreamNonBlocking,
                                            0));
    CHECK_CUDA(cudaExecutionCtxStreamCreate(&benchmark_streams.busy_wait_stream,
                                            benchmark_streams.green_context_state.main_ctx,
                                            cudaStreamNonBlocking,
                                            0));

    benchmark_streams.green_context_state.enabled = true;
#elif TEST_USE_DRIVER_GREEN_CONTEXTS
    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&benchmark_streams.green_context_state.cu_device, gpu_device_index));

    CUdevResource initial_GPU_SM_resources{};
    CHECK_CU(cuDeviceGetDevResource(benchmark_streams.green_context_state.cu_device,
                                    &initial_GPU_SM_resources,
                                    CU_DEV_RESOURCE_TYPE_SM));

    unsigned int nbGroups = 0;
    CHECK_CU(cuDevSmResourceSplitByCount(nullptr,
                                         &nbGroups,
                                         &initial_GPU_SM_resources,
                                         nullptr,
                                         0,
                                         TEST_GREEN_CONTEXT_SM_COUNT));

    if (nbGroups == 0) {
        throw std::runtime_error("Driver Green Context SM split produced zero groups");
    }

    benchmark_streams.green_context_state.groups.resize(nbGroups);
    CUdevResource remainder_res{};
    CHECK_CU(cuDevSmResourceSplitByCount(benchmark_streams.green_context_state.groups.data(),
                                         &nbGroups,
                                         &initial_GPU_SM_resources,
                                         &remainder_res,
                                         0,
                                         TEST_GREEN_CONTEXT_SM_COUNT));
    benchmark_streams.green_context_state.groups.resize(nbGroups);

    append_driver_main_resources(benchmark_streams.green_context_state.main_resources,
                                 benchmark_streams.green_context_state.groups,
                                 remainder_res);

    CHECK_CU(cuDevResourceGenerateDesc(&benchmark_streams.green_context_state.critical_desc,
                                       benchmark_streams.green_context_state.groups.data(),
                                       1));
    CHECK_CU(cuDevResourceGenerateDesc(&benchmark_streams.green_context_state.main_desc,
                                       benchmark_streams.green_context_state.main_resources.data(),
                                       static_cast<unsigned int>(benchmark_streams.green_context_state.main_resources.size())));

    CHECK_CU(cuGreenCtxCreate(&benchmark_streams.green_context_state.critical_gc,
                              benchmark_streams.green_context_state.critical_desc,
                              benchmark_streams.green_context_state.cu_device,
                              CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuGreenCtxCreate(&benchmark_streams.green_context_state.main_gc,
                              benchmark_streams.green_context_state.main_desc,
                              benchmark_streams.green_context_state.cu_device,
                              CU_GREEN_CTX_DEFAULT_STREAM));

    CHECK_CU(cuGreenCtxStreamCreate(&benchmark_streams.green_context_state.critical_cu_stream,
                                    benchmark_streams.green_context_state.critical_gc,
                                    CU_STREAM_NON_BLOCKING,
                                    0));
    CHECK_CU(cuGreenCtxStreamCreate(&benchmark_streams.green_context_state.main_cu_stream,
                                    benchmark_streams.green_context_state.main_gc,
                                    CU_STREAM_NON_BLOCKING,
                                    0));

    benchmark_streams.stream = reinterpret_cast<cudaStream_t>(benchmark_streams.green_context_state.critical_cu_stream);
    benchmark_streams.busy_wait_stream = reinterpret_cast<cudaStream_t>(benchmark_streams.green_context_state.main_cu_stream);
    benchmark_streams.green_context_state.enabled = true;
#else
#if TEST_USE_CUDA_STREAMS || TEST_USE_CUDA_GRAPHS || TEST_USE_BUSY_WAIT_COMPETITOR || TEST_RUN_BUSY_WAIT_ONLY
    CHECK_CUDA(cudaStreamCreateWithFlags(&benchmark_streams.stream, cudaStreamNonBlocking));
#else
    benchmark_streams.stream = 0;
#endif

#if TEST_USE_BUSY_WAIT_COMPETITOR || TEST_RUN_BUSY_WAIT_ONLY
    CHECK_CUDA(cudaStreamCreateWithFlags(&benchmark_streams.busy_wait_stream, cudaStreamNonBlocking));
#else
    benchmark_streams.busy_wait_stream = benchmark_streams.stream;
#endif
#endif

    return benchmark_streams;
}

void destroy_benchmark_streams(BenchmarkStreams &benchmark_streams) {
#if TEST_USE_RUNTIME_GREEN_CONTEXTS
    if (benchmark_streams.stream != nullptr) {
        CHECK_CUDA(cudaStreamDestroy(benchmark_streams.stream));
        benchmark_streams.stream = nullptr;
    }
    if (benchmark_streams.busy_wait_stream != nullptr) {
        CHECK_CUDA(cudaStreamDestroy(benchmark_streams.busy_wait_stream));
        benchmark_streams.busy_wait_stream = nullptr;
    }
    if (benchmark_streams.green_context_state.critical_ctx != nullptr) {
        CHECK_CUDA(cudaExecutionCtxDestroy(benchmark_streams.green_context_state.critical_ctx));
        benchmark_streams.green_context_state.critical_ctx = nullptr;
    }
    if (benchmark_streams.green_context_state.main_ctx != nullptr) {
        CHECK_CUDA(cudaExecutionCtxDestroy(benchmark_streams.green_context_state.main_ctx));
        benchmark_streams.green_context_state.main_ctx = nullptr;
    }
#elif TEST_USE_DRIVER_GREEN_CONTEXTS
    if (benchmark_streams.green_context_state.critical_cu_stream != nullptr) {
        CHECK_CU(cuStreamDestroy(benchmark_streams.green_context_state.critical_cu_stream));
        benchmark_streams.green_context_state.critical_cu_stream = nullptr;
    }
    if (benchmark_streams.green_context_state.main_cu_stream != nullptr) {
        CHECK_CU(cuStreamDestroy(benchmark_streams.green_context_state.main_cu_stream));
        benchmark_streams.green_context_state.main_cu_stream = nullptr;
    }
    if (benchmark_streams.green_context_state.critical_gc != nullptr) {
        CHECK_CU(cuGreenCtxDestroy(benchmark_streams.green_context_state.critical_gc));
        benchmark_streams.green_context_state.critical_gc = nullptr;
    }
    if (benchmark_streams.green_context_state.main_gc != nullptr) {
        CHECK_CU(cuGreenCtxDestroy(benchmark_streams.green_context_state.main_gc));
        benchmark_streams.green_context_state.main_gc = nullptr;
    }
#else
    if (benchmark_streams.busy_wait_stream != nullptr && benchmark_streams.busy_wait_stream != benchmark_streams.stream) {
        CHECK_CUDA(cudaStreamDestroy(benchmark_streams.busy_wait_stream));
        benchmark_streams.busy_wait_stream = nullptr;
    }
#if TEST_USE_CUDA_STREAMS || TEST_USE_CUDA_GRAPHS || TEST_USE_BUSY_WAIT_COMPETITOR || TEST_RUN_BUSY_WAIT_ONLY
    if (benchmark_streams.stream != nullptr) {
        CHECK_CUDA(cudaStreamDestroy(benchmark_streams.stream));
        benchmark_streams.stream = nullptr;
    }
#endif
#endif
}

void allocate_float_pitched(float **ptr, std::size_t *pitch, int image_width, int image_height, int channels) {
#if TEST_USE_PITCHED_LINEAR_MEMORY
    CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void **>(ptr),
                               pitch,
                               static_cast<std::size_t>(image_width) * sizeof(float),
                               static_cast<std::size_t>(image_height) * channels));
#else
    *pitch = static_cast<std::size_t>(image_width) * sizeof(float);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(ptr), (*pitch) * static_cast<std::size_t>(image_height) * channels));
#endif
}

void allocate_u8_pitched(unsigned char **ptr, std::size_t *pitch, int image_width, int image_height, int channels) {
#if TEST_USE_PITCHED_LINEAR_MEMORY
    CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void **>(ptr),
                               pitch,
                               static_cast<std::size_t>(image_width),
                               static_cast<std::size_t>(image_height) * channels));
#else
    *pitch = static_cast<std::size_t>(image_width);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(ptr), (*pitch) * static_cast<std::size_t>(image_height) * channels));
#endif
}

void copy_output_to_host(float *output,
                         const float *gOutput,
                         int image_width,
                         int image_height,
                         int channels,
                         std::size_t output_pitch,
                         cudaStream_t stream) {
    for (int channel_idx = 0; channel_idx < channels; ++channel_idx) {
        const char *src_channel = reinterpret_cast<const char *>(gOutput) + static_cast<std::size_t>(channel_idx) * output_pitch * image_height;
        float *dst_channel = output + static_cast<std::size_t>(channel_idx) * image_width * image_height;
        CHECK_CUDA(cudaMemcpy2DAsync(dst_channel,
                                     static_cast<std::size_t>(image_width) * sizeof(float),
                                     src_channel,
                                     output_pitch,
                                     static_cast<std::size_t>(image_width) * sizeof(float),
                                     image_height,
                                     cudaMemcpyDeviceToHost,
                                     stream));
    }
}

cudaError_t instantiate_graph(cudaGraphExec_t *graph_exec, cudaGraph_t graph) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    return cudaGraphInstantiate(graph_exec, graph, 0);
#else
    return cudaGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
#endif
}

void destroy_nvjpeg_state(NvjpegState &nvjpeg_state) {
    if (nvjpeg_state.jpeg_handle != nullptr) {
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state.jpeg_handle));
        nvjpeg_state.jpeg_handle = nullptr;
    }
    if (nvjpeg_state.handle != nullptr) {
        CHECK_NVJPEG(nvjpegDestroy(nvjpeg_state.handle));
        nvjpeg_state.handle = nullptr;
    }
}

void print_config(const BenchmarkConfig &config, const BenchmarkState &state) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    int multiProcessorCount = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device));

    std::cerr << "image_width=" << state.image_width
              << " image_height=" << state.image_height
              << " channels=" << state.channels
              << " R=" << config.R
              << " mask_length=" << config.mask_length
              << " threads_per_block=" << config.threads_per_block
              << " test_iterations=" << config.test_iterations
              << " sigma=" << config.sigma
              << " multiProcessorCount=" << multiProcessorCount
              << " TEST_PERF=" << TEST_PERF
              << " TEST_COMPREHENSIVE_CSV=" << TEST_COMPREHENSIVE_CSV
              << " TEST_WRITE_OUTPUT_IMAGE=" << TEST_WRITE_OUTPUT_IMAGE
              << " TEST_USE_RGB=" << TEST_USE_RGB
              << " TEST_USE_PINNED_HOST_MEMORY=" << TEST_USE_PINNED_HOST_MEMORY
              << " TEST_USE_PITCHED_LINEAR_MEMORY=" << TEST_USE_PITCHED_LINEAR_MEMORY
              << " TEST_USE_CUDA_STREAMS=" << TEST_USE_CUDA_STREAMS
              << " TEST_USE_CUDA_GRAPHS=" << TEST_USE_CUDA_GRAPHS
              << " TEST_USE_GREEN_CONTEXTS=" << TEST_USE_GREEN_CONTEXTS
              << " TEST_GREEN_CONTEXT_API=" << TEST_GREEN_CONTEXT_API
              << " TEST_USE_RUNTIME_GREEN_CONTEXTS=" << TEST_USE_RUNTIME_GREEN_CONTEXTS
              << " TEST_USE_DRIVER_GREEN_CONTEXTS=" << TEST_USE_DRIVER_GREEN_CONTEXTS
              << " TEST_GREEN_CONTEXT_SM_COUNT=" << TEST_GREEN_CONTEXT_SM_COUNT
              << " TEST_USE_FUSED_GAUSSIAN=" << TEST_USE_FUSED_GAUSSIAN
              << " TEST_USE_BUSY_WAIT_COMPETITOR=" << TEST_USE_BUSY_WAIT_COMPETITOR
              << " TEST_RUN_BUSY_WAIT_ONLY=" << TEST_RUN_BUSY_WAIT_ONLY
              << " TEST_BUSY_WAIT_MILLISECONDS=" << TEST_BUSY_WAIT_MILLISECONDS
              << " TEST_ASSUME_MIG=" << TEST_ASSUME_MIG
              << " TEST_ASSUME_MPS=" << TEST_ASSUME_MPS
              << " CONVOLUTION_USE_SHARED_MEMORY=" << CONVOLUTION_USE_SHARED_MEMORY
              << " CONVOLUTION_USE_CONSTANT_MASK=" << CONVOLUTION_USE_CONSTANT_MASK
              << " CONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED=" << CONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED
              << std::endl;
}

float mean_measurement(const std::vector<float> &measurements) {
    if (measurements.empty()) {
        return 0.0f;
    }
    const float sum = std::accumulate(measurements.begin(), measurements.end(), 0.0f);
    return sum / static_cast<float>(measurements.size());
}

}  // namespace

void print_usage(const char *program_name) {
    std::cerr << "Usage: " << program_name
              << " <image.jpg> <R> <threads_per_block> [test_iterations] [sigma] [gpu_device_index] [output_image_path]"
              << std::endl;
    std::cerr << "Busy-wait-only mode: " << program_name << " [gpu_device_index]" << std::endl;
}

BenchmarkConfig parse_benchmark_config(int argc, char *argv[]) {
    BenchmarkConfig config{};

    if (argc < 4 && !TEST_RUN_BUSY_WAIT_ONLY) {
        print_usage(argv[0]);
        throw std::runtime_error("Missing required command-line arguments");
    }

    if (!TEST_RUN_BUSY_WAIT_ONLY) {
        config.image_path = argv[1];
        config.R = std::stoi(argv[2]);
        config.mask_length = 2 * config.R + 1;
        config.threads_per_block = std::stoi(argv[3]);
        config.test_iterations = (argc > 4) ? std::stoi(argv[4]) : TEST_TEST_ITERATIONS;
        config.sigma = (argc > 5) ? std::stof(argv[5]) : std::max(1.0f, static_cast<float>(config.R) * 0.5f);
        config.gpu_device_index = (argc > 6) ? std::stoi(argv[6]) : 0;
#if TEST_WRITE_OUTPUT_IMAGE
        config.output_image_path = (argc > 7) ? argv[7] : default_output_image_path(config.image_path, TEST_USE_RGB ? 3 : 1);
#endif
    } else {
        config.gpu_device_index = (argc > 1) ? std::stoi(argv[1]) : 0;
    }

    if (config.R < 0 || config.threads_per_block <= 0 || config.test_iterations <= 0) {
        throw std::runtime_error("R, threads_per_block, and test_iterations must be positive/non-negative as appropriate");
    }

    return config;
}

void initialise_test(const BenchmarkConfig &config, BenchmarkState &state) {
    CHECK_CUDA(cudaSetDevice(config.gpu_device_index));
    state.benchmark_streams = create_benchmark_streams(config.gpu_device_index);
    state.streams_initialised = true;

#if TEST_USE_BUSY_WAIT_COMPETITOR || TEST_RUN_BUSY_WAIT_ONLY
    state.busy_wait_blocks = (TEST_BUSY_WAIT_BLOCKS > 0) ? TEST_BUSY_WAIT_BLOCKS : automatic_busy_wait_blocks(config.gpu_device_index);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&state.gBusyWaitSink), static_cast<std::size_t>(state.busy_wait_blocks) * sizeof(unsigned long long)));
#endif

#if TEST_RUN_BUSY_WAIT_ONLY
    return;
#else
    CHECK_NVJPEG(nvjpegCreateSimple(&state.nvjpeg_state.handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(state.nvjpeg_state.handle, &state.nvjpeg_state.jpeg_handle));
    state.nvjpeg_initialised = true;

    state.jpeg_buffer = read_file(config.image_path);
    CHECK_NVJPEG(nvjpegGetImageInfo(state.nvjpeg_state.handle,
                                    state.jpeg_buffer.data(),
                                    state.jpeg_buffer.size(),
                                    &state.nComponents,
                                    &state.subsampling,
                                    state.widths,
                                    state.heights));

    state.image_width = state.widths[0];
    state.image_height = state.heights[0];
    state.channels = TEST_USE_RGB ? 3 : 1;
    state.output_format = TEST_USE_RGB ? NVJPEG_OUTPUT_RGB : NVJPEG_OUTPUT_Y;

    allocate_u8_pitched(&state.gImage_u8,
                        &state.image_u8_pitch,
                        state.image_width,
                        state.image_height,
                        state.channels);

    nvjpegImage_t decoded_image{};
    for (int channel_idx = 0; channel_idx < state.channels; ++channel_idx) {
        decoded_image.channel[channel_idx] = state.gImage_u8 + static_cast<std::size_t>(channel_idx) * state.image_u8_pitch * state.image_height;
        decoded_image.pitch[channel_idx] = state.image_u8_pitch;
    }

    CHECK_NVJPEG(nvjpegDecode(state.nvjpeg_state.handle,
                              state.nvjpeg_state.jpeg_handle,
                              state.jpeg_buffer.data(),
                              state.jpeg_buffer.size(),
                              state.output_format,
                              &decoded_image,
                              state.benchmark_streams.stream));

    allocate_float_pitched(&state.gImage,
                           &state.image_pitch,
                           state.image_width,
                           state.image_height,
                           state.channels);
    allocate_float_pitched(&state.gTemp,
                           &state.temp_pitch,
                           state.image_width,
                           state.image_height,
                           state.channels);
    allocate_float_pitched(&state.gOutput,
                           &state.output_pitch,
                           state.image_width,
                           state.image_height,
                           state.channels);

    state.mask = make_gaussian_mask(config.R, config.sigma);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&state.gMask), static_cast<std::size_t>(config.mask_length) * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(state.gMask,
                               state.mask.data(),
                               static_cast<std::size_t>(config.mask_length) * sizeof(float),
                               cudaMemcpyHostToDevice,
                               state.benchmark_streams.stream));

    CHECK_CUDA(launch_u8_to_float_planar(state.gImage_u8,
                                         state.gImage,
                                         state.image_width,
                                         state.image_height,
                                         state.channels,
                                         state.image_u8_pitch,
                                         state.image_pitch,
                                         static_cast<unsigned int>(config.threads_per_block),
                                         state.benchmark_streams.stream));

#if CONVOLUTION_USE_CONSTANT_MASK
    if (config.R <= CONVOLUTION_MAX_RADIUS) {
        CHECK_CUDA(convolution_upload_mask(state.gMask, config.R, cudaMemcpyDeviceToDevice, state.benchmark_streams.stream));
    }
#endif

    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.stream));

    print_config(config, state);

    for (int idx = 0; idx < TEST_WARMUP_ITERATIONS; ++idx) {
        CHECK_CUDA(run_gaussian_blur(state.gImage,
                                     state.gMask,
                                     state.gTemp,
                                     state.gOutput,
                                     state.image_width,
                                     state.image_height,
                                     state.channels,
                                     config.R,
                                     state.image_pitch,
                                     state.temp_pitch,
                                     state.output_pitch,
                                     static_cast<unsigned int>(config.threads_per_block),
                                     state.benchmark_streams.stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.stream));

#if TEST_USE_CUDA_GRAPHS
    CHECK_CUDA(cudaStreamBeginCapture(state.benchmark_streams.stream, cudaStreamCaptureModeGlobal));
    CHECK_CUDA(run_gaussian_blur(state.gImage,
                                 state.gMask,
                                 state.gTemp,
                                 state.gOutput,
                                 state.image_width,
                                 state.image_height,
                                 state.channels,
                                 config.R,
                                 state.image_pitch,
                                 state.temp_pitch,
                                 state.output_pitch,
                                 static_cast<unsigned int>(config.threads_per_block),
                                 state.benchmark_streams.stream));
    CHECK_CUDA(cudaStreamEndCapture(state.benchmark_streams.stream, &state.graph));
    CHECK_CUDA(instantiate_graph(&state.graph_exec, state.graph));
    CHECK_CUDA(cudaGraphUpload(state.graph_exec, state.benchmark_streams.stream));
    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.stream));
#endif
#endif
}

BenchmarkResult run_test(const BenchmarkConfig &config, BenchmarkState &state) {
    BenchmarkResult result{};

#if TEST_RUN_BUSY_WAIT_ONLY
    CHECK_CUDA(launch_busy_wait_competitor(state.gBusyWaitSink,
                                           config.gpu_device_index,
                                           state.benchmark_streams.busy_wait_stream));
    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.busy_wait_stream));
    return result;
#else
#if TEST_USE_BUSY_WAIT_COMPETITOR
    CHECK_CUDA(launch_busy_wait_competitor(state.gBusyWaitSink,
                                           config.gpu_device_index,
                                           state.benchmark_streams.busy_wait_stream));
#endif

#if TEST_PERF == 1
#if TEST_RECORD_EACH_ITERATION || TEST_COMPREHENSIVE_CSV
    result.measurements.resize(static_cast<std::size_t>(config.test_iterations));
    for (int idx = 0; idx < config.test_iterations; ++idx) {
#if TEST_USE_BUSY_WAIT_COMPETITOR
        CHECK_CUDA(launch_busy_wait_competitor(state.gBusyWaitSink,
                                               config.gpu_device_index,
                                               state.benchmark_streams.busy_wait_stream));
#endif
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, state.benchmark_streams.stream));
#if TEST_USE_CUDA_GRAPHS
        CHECK_CUDA(cudaGraphLaunch(state.graph_exec, state.benchmark_streams.stream));
#else
        CHECK_CUDA(run_gaussian_blur(state.gImage,
                                     state.gMask,
                                     state.gTemp,
                                     state.gOutput,
                                     state.image_width,
                                     state.image_height,
                                     state.channels,
                                     config.R,
                                     state.image_pitch,
                                     state.temp_pitch,
                                     state.output_pitch,
                                     static_cast<unsigned int>(config.threads_per_block),
                                     state.benchmark_streams.stream));
#endif
        CHECK_CUDA(cudaEventRecord(stop, state.benchmark_streams.stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&result.measurements[static_cast<std::size_t>(idx)], start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    result.ms = mean_measurement(result.measurements);
    if (!result.measurements.empty()) {
        const auto minmax = std::minmax_element(result.measurements.begin(), result.measurements.end());
        std::cerr << "min_ms=" << *minmax.first << " max_ms=" << *minmax.second << " mean_ms=" << result.ms << std::endl;
    }
#else
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, state.benchmark_streams.stream));

    for (int idx = 0; idx < config.test_iterations; ++idx) {
#if TEST_USE_CUDA_GRAPHS
        CHECK_CUDA(cudaGraphLaunch(state.graph_exec, state.benchmark_streams.stream));
#else
        CHECK_CUDA(run_gaussian_blur(state.gImage,
                                     state.gMask,
                                     state.gTemp,
                                     state.gOutput,
                                     state.image_width,
                                     state.image_height,
                                     state.channels,
                                     config.R,
                                     state.image_pitch,
                                     state.temp_pitch,
                                     state.output_pitch,
                                     static_cast<unsigned int>(config.threads_per_block),
                                     state.benchmark_streams.stream));
#endif
    }

    CHECK_CUDA(cudaEventRecord(stop, state.benchmark_streams.stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&result.ms, start, stop));
    result.ms /= static_cast<float>(config.test_iterations);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
#endif
#else
    for (int idx = 0; idx < config.test_iterations; ++idx) {
#if TEST_USE_CUDA_GRAPHS
        CHECK_CUDA(cudaGraphLaunch(state.graph_exec, state.benchmark_streams.stream));
#else
        CHECK_CUDA(run_gaussian_blur(state.gImage,
                                     state.gMask,
                                     state.gTemp,
                                     state.gOutput,
                                     state.image_width,
                                     state.image_height,
                                     state.channels,
                                     config.R,
                                     state.image_pitch,
                                     state.temp_pitch,
                                     state.output_pitch,
                                     static_cast<unsigned int>(config.threads_per_block),
                                     state.benchmark_streams.stream));
#endif
    }
    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.stream));
#endif

#if TEST_USE_BUSY_WAIT_COMPETITOR
    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.busy_wait_stream));
#endif

    state.output_length = static_cast<std::size_t>(state.image_width) * state.image_height * state.channels;
#if TEST_USE_PINNED_HOST_MEMORY
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&state.output), state.output_length * sizeof(float)));
#else
    state.output = new float[state.output_length];
#endif
    state.output_allocated = true;

    copy_output_to_host(state.output,
                        state.gOutput,
                        state.image_width,
                        state.image_height,
                        state.channels,
                        state.output_pitch,
                        state.benchmark_streams.stream);
    CHECK_CUDA(cudaStreamSynchronize(state.benchmark_streams.stream));

    result.output_sentinel = state.output[state.output_length - 1];

#if TEST_WRITE_OUTPUT_IMAGE
    write_output_image(config.output_image_path,
                       state.output,
                       state.image_width,
                       state.image_height,
                       state.channels);
    std::cerr << "output_image_path=" << config.output_image_path << std::endl;
#endif

    return result;
#endif
}

void deinitialise_test(BenchmarkState &state) {
#if TEST_USE_CUDA_GRAPHS
    if (state.graph_exec != nullptr) {
        CHECK_CUDA(cudaGraphExecDestroy(state.graph_exec));
        state.graph_exec = nullptr;
    }
    if (state.graph != nullptr) {
        CHECK_CUDA(cudaGraphDestroy(state.graph));
        state.graph = nullptr;
    }
#endif

    if (state.output_allocated && state.output != nullptr) {
#if TEST_USE_PINNED_HOST_MEMORY
        CHECK_CUDA(cudaFreeHost(state.output));
#else
        delete[] state.output;
#endif
        state.output = nullptr;
        state.output_allocated = false;
    }

    if (state.gBusyWaitSink != nullptr) {
        CHECK_CUDA(cudaFree(state.gBusyWaitSink));
        state.gBusyWaitSink = nullptr;
    }
    if (state.gImage_u8 != nullptr) {
        CHECK_CUDA(cudaFree(state.gImage_u8));
        state.gImage_u8 = nullptr;
    }
    if (state.gImage != nullptr) {
        CHECK_CUDA(cudaFree(state.gImage));
        state.gImage = nullptr;
    }
    if (state.gMask != nullptr) {
        CHECK_CUDA(cudaFree(state.gMask));
        state.gMask = nullptr;
    }
    if (state.gTemp != nullptr) {
        CHECK_CUDA(cudaFree(state.gTemp));
        state.gTemp = nullptr;
    }
    if (state.gOutput != nullptr) {
        CHECK_CUDA(cudaFree(state.gOutput));
        state.gOutput = nullptr;
    }

    if (state.nvjpeg_initialised) {
        destroy_nvjpeg_state(state.nvjpeg_state);
        state.nvjpeg_initialised = false;
    }

    if (state.streams_initialised) {
        destroy_benchmark_streams(state.benchmark_streams);
        state.streams_initialised = false;
    }
}

void print_basic_result(const BenchmarkResult &result) {
#if TEST_PERF == 1
    std::cout << std::setprecision(9) << result.ms << std::endl;
    std::cout << std::setprecision(9) << result.output_sentinel << std::endl;
#else
    (void) result;
#endif
}

void print_comprehensive_csv(const BenchmarkConfig &config,
                             const BenchmarkState &state,
                             const BenchmarkResult &result) {
#if TEST_COMPREHENSIVE_CSV
#if TEST_CSV_HEADER
    std::cout << "iteration_idx,elapsed_ms,image_width,image_height,channels,R,sigma,threads_per_block,"
                 "fused,cuda_graphs,green_contexts,green_context_api,green_context_sm_count,"
                 "busy_wait_enabled,busy_wait_blocks,busy_wait_threads,busy_wait_milliseconds,"
                 "mps_assumed,mig_assumed,shared_memory,constant_mask,pitched_memory,pinned_host_memory,"
                 "cuda_streams,warmup_iterations,test_iterations,output_sentinel,output_image_path"
              << std::endl;
#endif

    for (std::size_t idx = 0; idx < result.measurements.size(); ++idx) {
        std::cout << idx << ','
                  << std::setprecision(9) << result.measurements[idx] << ','
                  << state.image_width << ','
                  << state.image_height << ','
                  << state.channels << ','
                  << config.R << ','
                  << std::setprecision(9) << config.sigma << ','
                  << config.threads_per_block << ','
                  << TEST_USE_FUSED_GAUSSIAN << ','
                  << TEST_USE_CUDA_GRAPHS << ','
                  << TEST_USE_GREEN_CONTEXTS << ','
                  << TEST_GREEN_CONTEXT_API << ','
                  << TEST_GREEN_CONTEXT_SM_COUNT << ','
                  << TEST_USE_BUSY_WAIT_COMPETITOR << ','
                  << state.busy_wait_blocks << ','
                  << TEST_BUSY_WAIT_THREADS << ','
                  << TEST_BUSY_WAIT_MILLISECONDS << ','
                  << TEST_ASSUME_MPS << ','
                  << TEST_ASSUME_MIG << ','
                  << CONVOLUTION_USE_SHARED_MEMORY << ','
                  << CONVOLUTION_USE_CONSTANT_MASK << ','
                  << TEST_USE_PITCHED_LINEAR_MEMORY << ','
                  << TEST_USE_PINNED_HOST_MEMORY << ','
                  << TEST_USE_CUDA_STREAMS << ','
                  << TEST_WARMUP_ITERATIONS << ','
                  << config.test_iterations << ','
                  << std::setprecision(9) << result.output_sentinel << ','
                  << config.output_image_path
                  << std::endl;
    }
#else
    (void) config;
    (void) state;
    (void) result;
#endif
}
