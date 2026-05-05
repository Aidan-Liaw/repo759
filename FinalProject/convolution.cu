//
// Created by aidan on 19/04/2026.
// Fixed/extended for separable 2D Gaussian convolution and CUDA latency experiments.
//

#include "convolution.cuh"

#include <climits>
#include <cstddef>
#include <cstdint>

__constant__ float convolution_constant_mask[2 * CONVOLUTION_MAX_RADIUS + 1];

namespace {

static constexpr int tile_width = CONVOLUTION_TILE_WIDTH;
static constexpr bool compile_use_shared_memory = (CONVOLUTION_USE_SHARED_MEMORY != 0);
static constexpr bool compile_use_constant_mask = (CONVOLUTION_USE_CONSTANT_MASK != 0);
static constexpr bool assume_mask_already_uploaded = (CONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED != 0);

__device__ __forceinline__ long long clamp_index(long long idx, long long upper_exclusive) {
    idx = idx < 0 ? 0 : idx;
    idx = idx >= upper_exclusive ? (upper_exclusive - 1) : idx;
    return idx;
}

__device__ __forceinline__ const float *channel_base_ptr(const float *image,
                                                         long long pitch,
                                                         long long height,
                                                         long long channel_idx) {
    return reinterpret_cast<const float *>(reinterpret_cast<const char *>(image) + channel_idx * pitch * height);
}

__device__ __forceinline__ float *channel_base_ptr(float *image,
                                                   long long pitch,
                                                   long long height,
                                                   long long channel_idx) {
    return reinterpret_cast<float *>(reinterpret_cast<char *>(image) + channel_idx * pitch * height);
}

__device__ __forceinline__ const float *image_row_ptr(const float *image,
                                                      long long pitch,
                                                      long long height,
                                                      long long channel_idx,
                                                      long long row_idx) {
    return reinterpret_cast<const float *>(reinterpret_cast<const char *>(channel_base_ptr(image, pitch, height, channel_idx)) + row_idx * pitch);
}

__device__ __forceinline__ float *output_row_ptr(float *output,
                                                 long long output_pitch,
                                                 long long height,
                                                 long long channel_idx,
                                                 long long row_idx) {
    return reinterpret_cast<float *>(reinterpret_cast<char *>(channel_base_ptr(output, output_pitch, height, channel_idx)) + row_idx * output_pitch);
}

// Square-image compatibility helpers.
__device__ __forceinline__ const float *image_row_ptr(const float *image, long long pitch, long long row_idx) {
    return reinterpret_cast<const float *>(reinterpret_cast<const char *>(image) + row_idx * pitch);
}

__device__ __forceinline__ float *output_row_ptr(float *output, long long output_pitch, long long row_idx) {
    return reinterpret_cast<float *>(reinterpret_cast<char *>(output) + row_idx * output_pitch);
}

template <bool use_constant_mask>
__device__ __forceinline__ float mask_value(const float *mask, long long idx) {
    if constexpr (use_constant_mask) {
        return convolution_constant_mask[idx];
    } else {
        return mask[idx];
    }
}

template <bool use_constant_mask>
__global__ void horizontal_convolve_global_channels_kernel(const float *__restrict__ image,
                                                           float *__restrict__ output,
                                                           long long width,
                                                           long long height,
                                                           long long channels,
                                                           long long pitch,
                                                           long long output_pitch,
                                                           const float *__restrict__ mask,
                                                           long long R) {
    const unsigned long long thread_idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long plane_pixels = static_cast<unsigned long long>(width) * static_cast<unsigned long long>(height);
    const unsigned long long pixel_count = plane_pixels * static_cast<unsigned long long>(channels);

    if (thread_idx >= pixel_count) {
        return;
    }

    const long long channel_idx = static_cast<long long>(thread_idx / plane_pixels);
    const unsigned long long plane_idx = thread_idx - static_cast<unsigned long long>(channel_idx) * plane_pixels;
    const long long row_idx = static_cast<long long>(plane_idx / static_cast<unsigned long long>(width));
    const long long column_idx = static_cast<long long>(plane_idx - static_cast<unsigned long long>(row_idx) * static_cast<unsigned long long>(width));

    float sum = 0.0f;
#pragma unroll 1
    for (long long j = -R; j <= R; ++j) {
        const long long image_column_idx = clamp_index(column_idx + j, width);
        sum += image_row_ptr(image, pitch, height, channel_idx, row_idx)[image_column_idx] * mask_value<use_constant_mask>(mask, R - j);
    }

    output_row_ptr(output, output_pitch, height, channel_idx, row_idx)[column_idx] = sum;
}

template <bool use_constant_mask>
__global__ void vertical_convolve_global_channels_kernel(const float *__restrict__ image,
                                                         float *__restrict__ output,
                                                         long long width,
                                                         long long height,
                                                         long long channels,
                                                         long long pitch,
                                                         long long output_pitch,
                                                         const float *__restrict__ mask,
                                                         long long R) {
    const unsigned long long thread_idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long plane_pixels = static_cast<unsigned long long>(width) * static_cast<unsigned long long>(height);
    const unsigned long long pixel_count = plane_pixels * static_cast<unsigned long long>(channels);

    if (thread_idx >= pixel_count) {
        return;
    }

    const long long channel_idx = static_cast<long long>(thread_idx / plane_pixels);
    const unsigned long long plane_idx = thread_idx - static_cast<unsigned long long>(channel_idx) * plane_pixels;
    const long long row_idx = static_cast<long long>(plane_idx / static_cast<unsigned long long>(width));
    const long long column_idx = static_cast<long long>(plane_idx - static_cast<unsigned long long>(row_idx) * static_cast<unsigned long long>(width));

    float sum = 0.0f;
#pragma unroll 1
    for (long long j = -R; j <= R; ++j) {
        const long long image_row_idx = clamp_index(row_idx + j, height);
        sum += image_row_ptr(image, pitch, height, channel_idx, image_row_idx)[column_idx] * mask_value<use_constant_mask>(mask, R - j);
    }

    output_row_ptr(output, output_pitch, height, channel_idx, row_idx)[column_idx] = sum;
}

template <bool use_constant_mask>
__global__ void horizontal_convolve_shared_channels_kernel(const float *__restrict__ image,
                                                           float *__restrict__ output,
                                                           long long width,
                                                           long long height,
                                                           long long channels,
                                                           long long pitch,
                                                           long long output_pitch,
                                                           const float *__restrict__ mask,
                                                           long long R) {
    const int tile_height = blockDim.x / tile_width;
    if (tile_height <= 0) {
        return;
    }

    const int local_idx = threadIdx.x;
    const int local_x = local_idx % tile_width;
    const int local_y = local_idx / tile_width;

    const long long channel_idx = static_cast<long long>(blockIdx.z);
    if (channel_idx >= channels) {
        return;
    }

    const long long block_column_idx = static_cast<long long>(blockIdx.x) * tile_width;
    const long long block_row_idx = static_cast<long long>(blockIdx.y) * tile_height;
    const long long column_idx = block_column_idx + local_x;
    const long long row_idx = block_row_idx + local_y;

    extern __shared__ float shared_memory[];
    const int shared_row_width = tile_width + static_cast<int>(2 * R);
    float *sImage = shared_memory;
    float *sMask = sImage + static_cast<std::size_t>(shared_row_width) * tile_height;

    const long long image_elems = static_cast<long long>(shared_row_width) * tile_height;
    for (long long idx = local_idx; idx < image_elems; idx += blockDim.x) {
        const long long shared_y = idx / shared_row_width;
        const long long shared_x = idx - shared_y * shared_row_width;
        const long long image_row_idx = clamp_index(block_row_idx + shared_y, height);
        const long long image_column_idx = clamp_index(block_column_idx + shared_x - R, width);
        sImage[idx] = image_row_ptr(image, pitch, height, channel_idx, image_row_idx)[image_column_idx];
    }

    if constexpr (!use_constant_mask) {
        for (long long idx = local_idx; idx < 2 * R + 1; idx += blockDim.x) {
            sMask[idx] = mask[2 * R - idx];
        }
    }

    __syncthreads();

    if (local_y >= tile_height || row_idx >= height || column_idx >= width) {
        return;
    }

    float sum = 0.0f;
    const long long shared_output_base = static_cast<long long>(local_y) * shared_row_width + local_x + R;
#pragma unroll 1
    for (long long j = -R; j <= R; ++j) {
        const float coefficient = use_constant_mask ? mask_value<true>(mask, R - j) : sMask[j + R];
        sum += sImage[shared_output_base + j] * coefficient;
    }

    output_row_ptr(output, output_pitch, height, channel_idx, row_idx)[column_idx] = sum;
}

template <bool use_constant_mask>
__global__ void vertical_convolve_shared_channels_kernel(const float *__restrict__ image,
                                                         float *__restrict__ output,
                                                         long long width,
                                                         long long height,
                                                         long long channels,
                                                         long long pitch,
                                                         long long output_pitch,
                                                         const float *__restrict__ mask,
                                                         long long R) {
    const int tile_height = blockDim.x / tile_width;
    if (tile_height <= 0) {
        return;
    }

    const int local_idx = threadIdx.x;
    const int local_x = local_idx % tile_width;
    const int local_y = local_idx / tile_width;

    const long long channel_idx = static_cast<long long>(blockIdx.z);
    if (channel_idx >= channels) {
        return;
    }

    const long long block_column_idx = static_cast<long long>(blockIdx.x) * tile_width;
    const long long block_row_idx = static_cast<long long>(blockIdx.y) * tile_height;
    const long long column_idx = block_column_idx + local_x;
    const long long row_idx = block_row_idx + local_y;

    extern __shared__ float shared_memory[];
    const int shared_row_width = tile_width;
    const int shared_tile_height = tile_height + static_cast<int>(2 * R);
    float *sImage = shared_memory;
    float *sMask = sImage + static_cast<std::size_t>(shared_row_width) * shared_tile_height;

    const long long image_elems = static_cast<long long>(shared_row_width) * shared_tile_height;
    for (long long idx = local_idx; idx < image_elems; idx += blockDim.x) {
        const long long shared_y = idx / shared_row_width;
        const long long shared_x = idx - shared_y * shared_row_width;
        const long long image_row_idx = clamp_index(block_row_idx + shared_y - R, height);
        const long long image_column_idx = clamp_index(block_column_idx + shared_x, width);
        sImage[idx] = image_row_ptr(image, pitch, height, channel_idx, image_row_idx)[image_column_idx];
    }

    if constexpr (!use_constant_mask) {
        for (long long idx = local_idx; idx < 2 * R + 1; idx += blockDim.x) {
            sMask[idx] = mask[2 * R - idx];
        }
    }

    __syncthreads();

    if (local_y >= tile_height || row_idx >= height || column_idx >= width) {
        return;
    }

    float sum = 0.0f;
    const long long shared_output_base = static_cast<long long>(local_y + R) * shared_row_width + local_x;
#pragma unroll 1
    for (long long j = -R; j <= R; ++j) {
        const float coefficient = use_constant_mask ? mask_value<true>(mask, R - j) : sMask[j + R];
        sum += sImage[shared_output_base + j * shared_row_width] * coefficient;
    }

    output_row_ptr(output, output_pitch, height, channel_idx, row_idx)[column_idx] = sum;
}

template <bool use_constant_mask>
__global__ void fused_gaussian_convolve_global_channels_kernel(const float *__restrict__ image,
                                                               float *__restrict__ output,
                                                               long long width,
                                                               long long height,
                                                               long long channels,
                                                               long long pitch,
                                                               long long output_pitch,
                                                               const float *__restrict__ mask,
                                                               long long R) {
    const unsigned long long thread_idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long plane_pixels = static_cast<unsigned long long>(width) * static_cast<unsigned long long>(height);
    const unsigned long long pixel_count = plane_pixels * static_cast<unsigned long long>(channels);

    if (thread_idx >= pixel_count) {
        return;
    }

    const long long channel_idx = static_cast<long long>(thread_idx / plane_pixels);
    const unsigned long long plane_idx = thread_idx - static_cast<unsigned long long>(channel_idx) * plane_pixels;
    const long long row_idx = static_cast<long long>(plane_idx / static_cast<unsigned long long>(width));
    const long long column_idx = static_cast<long long>(plane_idx - static_cast<unsigned long long>(row_idx) * static_cast<unsigned long long>(width));

    float sum = 0.0f;
#pragma unroll 1
    for (long long y = -R; y <= R; ++y) {
        const long long image_row_idx = clamp_index(row_idx + y, height);
        const float coefficient_y = mask_value<use_constant_mask>(mask, R - y);
#pragma unroll 1
        for (long long x = -R; x <= R; ++x) {
            const long long image_column_idx = clamp_index(column_idx + x, width);
            const float coefficient_x = mask_value<use_constant_mask>(mask, R - x);
            sum += image_row_ptr(image, pitch, height, channel_idx, image_row_idx)[image_column_idx] * coefficient_x * coefficient_y;
        }
    }

    output_row_ptr(output, output_pitch, height, channel_idx, row_idx)[column_idx] = sum;
}

cudaError_t validate_launch_args(long long width,
                                 long long height,
                                 long long channels,
                                 long long R,
                                 long long pitch,
                                 long long output_pitch,
                                 unsigned int threads_per_block) {
    if (width <= 0 || height <= 0 || channels <= 0 || R < 0 || pitch <= 0 || output_pitch <= 0 || threads_per_block == 0) {
        return cudaErrorInvalidValue;
    }

    if (pitch < width * static_cast<long long>(sizeof(float)) || output_pitch < width * static_cast<long long>(sizeof(float))) {
        return cudaErrorInvalidPitchValue;
    }

    if (compile_use_shared_memory && threads_per_block < static_cast<unsigned int>(tile_width)) {
        return cudaErrorInvalidValue;
    }

    if (compile_use_shared_memory && (threads_per_block / static_cast<unsigned int>(tile_width)) == 0) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

cudaError_t maybe_upload_mask(const float *mask,
                              long long R,
                              cudaMemcpyKind mask_copy_kind,
                              cudaStream_t stream,
                              bool *use_constant_mask) {
    *use_constant_mask = compile_use_constant_mask && (R <= CONVOLUTION_MAX_RADIUS);

    if (!*use_constant_mask || assume_mask_already_uploaded) {
        return cudaSuccess;
    }

    return convolution_upload_mask(mask, R, mask_copy_kind, stream);
}

dim3 shared_grid(long long width, long long height, long long channels, unsigned int threads_per_block) {
    const unsigned int tile_height = threads_per_block / tile_width;
    const unsigned int grid_x = static_cast<unsigned int>((width + tile_width - 1) / tile_width);
    const unsigned int grid_y = static_cast<unsigned int>((height + tile_height - 1) / tile_height);
    const unsigned int grid_z = static_cast<unsigned int>(channels);
    return dim3(grid_x, grid_y, grid_z);
}

unsigned int global_grid_x(long long width, long long height, long long channels, unsigned int threads_per_block) {
    const unsigned long long pixel_count = static_cast<unsigned long long>(width) *
                                           static_cast<unsigned long long>(height) *
                                           static_cast<unsigned long long>(channels);
    return static_cast<unsigned int>((pixel_count + threads_per_block - 1) / threads_per_block);
}

std::size_t horizontal_shared_memory_size(long long R, unsigned int threads_per_block, bool use_constant_mask) {
    const unsigned int tile_height = threads_per_block / tile_width;
    const std::size_t image_elems = static_cast<std::size_t>(tile_width + 2 * R) * tile_height;
    const std::size_t mask_elems = use_constant_mask ? 0 : static_cast<std::size_t>(2 * R + 1);
    return (image_elems + mask_elems) * sizeof(float);
}

std::size_t vertical_shared_memory_size(long long R, unsigned int threads_per_block, bool use_constant_mask) {
    const unsigned int tile_height = threads_per_block / tile_width;
    const std::size_t image_elems = static_cast<std::size_t>(tile_width) * (tile_height + 2 * R);
    const std::size_t mask_elems = use_constant_mask ? 0 : static_cast<std::size_t>(2 * R + 1);
    return (image_elems + mask_elems) * sizeof(float);
}

}  // namespace

cudaError_t convolution_upload_mask(const float *mask,
                                    long long R,
                                    cudaMemcpyKind mask_copy_kind,
                                    cudaStream_t stream) {
    if (mask == nullptr || R < 0 || R > CONVOLUTION_MAX_RADIUS) {
        return cudaErrorInvalidValue;
    }

    const std::size_t mask_bytes = static_cast<std::size_t>(2 * R + 1) * sizeof(float);
    return cudaMemcpyToSymbolAsync(convolution_constant_mask, mask, mask_bytes, 0, mask_copy_kind, stream);
}

cudaError_t horizontal_convolve_channels(const float *image,
                                         const float *mask,
                                         float *output,
                                         long long width,
                                         long long height,
                                         long long channels,
                                         long long R,
                                         long long pitch,
                                         long long output_pitch,
                                         unsigned int threads_per_block,
                                         cudaStream_t stream,
                                         cudaMemcpyKind mask_copy_kind) {
    if (image == nullptr || output == nullptr || mask == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = validate_launch_args(width, height, channels, R, pitch, output_pitch, threads_per_block);
    if (status != cudaSuccess) {
        return status;
    }

    bool use_constant_mask = false;
    status = maybe_upload_mask(mask, R, mask_copy_kind, stream, &use_constant_mask);
    if (status != cudaSuccess) {
        return status;
    }

    if (compile_use_shared_memory) {
        const dim3 grid = shared_grid(width, height, channels, threads_per_block);
        const std::size_t shared_memory_size = horizontal_shared_memory_size(R, threads_per_block, use_constant_mask);
        if (use_constant_mask) {
            horizontal_convolve_shared_channels_kernel<true><<<grid, threads_per_block, shared_memory_size, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        } else {
            horizontal_convolve_shared_channels_kernel<false><<<grid, threads_per_block, shared_memory_size, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        }
    } else {
        const unsigned int number_of_blocks = global_grid_x(width, height, channels, threads_per_block);
        if (use_constant_mask) {
            horizontal_convolve_global_channels_kernel<true><<<number_of_blocks, threads_per_block, 0, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        } else {
            horizontal_convolve_global_channels_kernel<false><<<number_of_blocks, threads_per_block, 0, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        }
    }

    return cudaGetLastError();
}

cudaError_t vertical_convolve_channels(const float *image,
                                       const float *mask,
                                       float *output,
                                       long long width,
                                       long long height,
                                       long long channels,
                                       long long R,
                                       long long pitch,
                                       long long output_pitch,
                                       unsigned int threads_per_block,
                                       cudaStream_t stream,
                                       cudaMemcpyKind mask_copy_kind) {
    if (image == nullptr || output == nullptr || mask == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = validate_launch_args(width, height, channels, R, pitch, output_pitch, threads_per_block);
    if (status != cudaSuccess) {
        return status;
    }

    bool use_constant_mask = false;
    status = maybe_upload_mask(mask, R, mask_copy_kind, stream, &use_constant_mask);
    if (status != cudaSuccess) {
        return status;
    }

    if (compile_use_shared_memory) {
        const dim3 grid = shared_grid(width, height, channels, threads_per_block);
        const std::size_t shared_memory_size = vertical_shared_memory_size(R, threads_per_block, use_constant_mask);
        if (use_constant_mask) {
            vertical_convolve_shared_channels_kernel<true><<<grid, threads_per_block, shared_memory_size, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        } else {
            vertical_convolve_shared_channels_kernel<false><<<grid, threads_per_block, shared_memory_size, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        }
    } else {
        const unsigned int number_of_blocks = global_grid_x(width, height, channels, threads_per_block);
        if (use_constant_mask) {
            vertical_convolve_global_channels_kernel<true><<<number_of_blocks, threads_per_block, 0, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        } else {
            vertical_convolve_global_channels_kernel<false><<<number_of_blocks, threads_per_block, 0, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
        }
    }

    return cudaGetLastError();
}

cudaError_t gaussian_blur_separable_channels(const float *image,
                                             const float *mask,
                                             float *temp,
                                             float *output,
                                             long long width,
                                             long long height,
                                             long long channels,
                                             long long R,
                                             long long image_pitch,
                                             long long temp_pitch,
                                             long long output_pitch,
                                             unsigned int threads_per_block,
                                             cudaStream_t stream,
                                             cudaMemcpyKind mask_copy_kind) {
    cudaError_t status = horizontal_convolve_channels(image, mask, temp, width, height, channels, R, image_pitch, temp_pitch,
                                                      threads_per_block, stream, mask_copy_kind);
    if (status != cudaSuccess) {
        return status;
    }

    return vertical_convolve_channels(temp, mask, output, width, height, channels, R, temp_pitch, output_pitch,
                                      threads_per_block, stream, mask_copy_kind);
}

cudaError_t gaussian_blur_fused_channels(const float *image,
                                         const float *mask,
                                         float *output,
                                         long long width,
                                         long long height,
                                         long long channels,
                                         long long R,
                                         long long pitch,
                                         long long output_pitch,
                                         unsigned int threads_per_block,
                                         cudaStream_t stream,
                                         cudaMemcpyKind mask_copy_kind) {
#if CONVOLUTION_ENABLE_FUSED_KERNEL
    if (image == nullptr || output == nullptr || mask == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = validate_launch_args(width, height, channels, R, pitch, output_pitch, threads_per_block);
    if (status != cudaSuccess) {
        return status;
    }

    bool use_constant_mask = false;
    status = maybe_upload_mask(mask, R, mask_copy_kind, stream, &use_constant_mask);
    if (status != cudaSuccess) {
        return status;
    }

    const unsigned int number_of_blocks = global_grid_x(width, height, channels, threads_per_block);
    if (use_constant_mask) {
        fused_gaussian_convolve_global_channels_kernel<true><<<number_of_blocks, threads_per_block, 0, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
    } else {
        fused_gaussian_convolve_global_channels_kernel<false><<<number_of_blocks, threads_per_block, 0, stream>>>(image, output, width, height, channels, pitch, output_pitch, mask, R);
    }

    return cudaGetLastError();
#else
    return cudaErrorNotSupported;
#endif
}

// Compatibility kernels and wrappers for dim_len x dim_len single-channel code.
__global__ void horizontal_convolve_kernel(const float *image,
                                           float *output,
                                           long long dim_len,
                                           long long pitch,
                                           long long output_pitch,
                                           const float *mask,
                                           long long R) {
    const long long thread_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long pixel_count = dim_len * dim_len;

    if (thread_idx >= pixel_count) {
        return;
    }

    const long long row_idx = thread_idx / dim_len;
    const long long column_idx = thread_idx - row_idx * dim_len;

    float sum = 0.0f;
#pragma unroll 1
    for (long long j = -R; j <= R; ++j) {
        const long long image_column_idx = clamp_index(column_idx + j, dim_len);
        sum += image_row_ptr(image, pitch, row_idx)[image_column_idx] * mask[R - j];
    }

    output_row_ptr(output, output_pitch, row_idx)[column_idx] = sum;
}

__global__ void vertical_convolve_kernel(const float *image,
                                         float *output,
                                         long long dim_len,
                                         long long pitch,
                                         long long output_pitch,
                                         const float *mask,
                                         long long R) {
    const long long thread_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long pixel_count = dim_len * dim_len;

    if (thread_idx >= pixel_count) {
        return;
    }

    const long long row_idx = thread_idx / dim_len;
    const long long column_idx = thread_idx - row_idx * dim_len;

    float sum = 0.0f;
#pragma unroll 1
    for (long long j = -R; j <= R; ++j) {
        const long long image_row_idx = clamp_index(row_idx + j, dim_len);
        sum += image_row_ptr(image, pitch, image_row_idx)[column_idx] * mask[R - j];
    }

    output_row_ptr(output, output_pitch, row_idx)[column_idx] = sum;
}

__global__ void fused_gaussian_convolve_kernel(const float *image,
                                               float *output,
                                               long long dim_len,
                                               long long pitch,
                                               long long output_pitch,
                                               const float *mask,
                                               long long R) {
    const long long thread_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long pixel_count = dim_len * dim_len;

    if (thread_idx >= pixel_count) {
        return;
    }

    const long long row_idx = thread_idx / dim_len;
    const long long column_idx = thread_idx - row_idx * dim_len;

    float sum = 0.0f;
#pragma unroll 1
    for (long long y = -R; y <= R; ++y) {
        const long long image_row_idx = clamp_index(row_idx + y, dim_len);
        const float coefficient_y = mask[R - y];
#pragma unroll 1
        for (long long x = -R; x <= R; ++x) {
            const long long image_column_idx = clamp_index(column_idx + x, dim_len);
            const float coefficient_x = mask[R - x];
            sum += image_row_ptr(image, pitch, image_row_idx)[image_column_idx] * coefficient_x * coefficient_y;
        }
    }

    output_row_ptr(output, output_pitch, row_idx)[column_idx] = sum;
}

cudaError_t horizontal_convolve(const float *image,
                                const float *mask,
                                float *output,
                                long long dim_len,
                                long long R,
                                long long pitch,
                                long long output_pitch,
                                unsigned int threads_per_block,
                                cudaStream_t stream,
                                cudaMemcpyKind mask_copy_kind) {
    return horizontal_convolve_channels(image, mask, output, dim_len, dim_len, 1, R, pitch, output_pitch,
                                        threads_per_block, stream, mask_copy_kind);
}

cudaError_t vertical_convolve(const float *image,
                              const float *mask,
                              float *output,
                              long long dim_len,
                              long long R,
                              long long pitch,
                              long long output_pitch,
                              unsigned int threads_per_block,
                              cudaStream_t stream,
                              cudaMemcpyKind mask_copy_kind) {
    return vertical_convolve_channels(image, mask, output, dim_len, dim_len, 1, R, pitch, output_pitch,
                                      threads_per_block, stream, mask_copy_kind);
}

cudaError_t gaussian_blur_separable(const float *image,
                                    const float *mask,
                                    float *temp,
                                    float *output,
                                    long long dim_len,
                                    long long R,
                                    long long image_pitch,
                                    long long temp_pitch,
                                    long long output_pitch,
                                    unsigned int threads_per_block,
                                    cudaStream_t stream,
                                    cudaMemcpyKind mask_copy_kind) {
    return gaussian_blur_separable_channels(image, mask, temp, output, dim_len, dim_len, 1, R,
                                            image_pitch, temp_pitch, output_pitch, threads_per_block,
                                            stream, mask_copy_kind);
}

cudaError_t gaussian_blur_fused(const float *image,
                                const float *mask,
                                float *output,
                                long long dim_len,
                                long long R,
                                long long pitch,
                                long long output_pitch,
                                unsigned int threads_per_block,
                                cudaStream_t stream,
                                cudaMemcpyKind mask_copy_kind) {
    return gaussian_blur_fused_channels(image, mask, output, dim_len, dim_len, 1, R,
                                        pitch, output_pitch, threads_per_block, stream, mask_copy_kind);
}

__host__ void convolve(const float *image,
                       const float *mask,
                       float *output,
                       long long dim_len,
                       long long R,
                       long long pitch,
                       unsigned int threads_per_block) {
    const long long output_pitch = dim_len * static_cast<long long>(sizeof(float));
    (void) horizontal_convolve(image, mask, output, dim_len, R, pitch, output_pitch, threads_per_block, 0,
                               cudaMemcpyDeviceToDevice);
}
