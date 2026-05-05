#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef CONVOLUTION_TILE_WIDTH
#define CONVOLUTION_TILE_WIDTH 32
#endif

#ifndef CONVOLUTION_MAX_RADIUS
#define CONVOLUTION_MAX_RADIUS 64
#endif

#ifndef CONVOLUTION_USE_SHARED_MEMORY
#define CONVOLUTION_USE_SHARED_MEMORY 1
#endif

#ifndef CONVOLUTION_USE_CONSTANT_MASK
#define CONVOLUTION_USE_CONSTANT_MASK 1
#endif

#ifndef CONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED
#define CONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED 1
#endif

#ifndef CONVOLUTION_ENABLE_FUSED_KERNEL
#define CONVOLUTION_ENABLE_FUSED_KERNEL 1
#endif

// Computes one horizontal pass of a separable 2D convolution over a dim_len x dim_len image.
// image and output may be pitched allocations. pitch and output_pitch are byte strides.
// mask has length (2 * R + 1).
__global__ void horizontal_convolve_kernel(const float *image,
                                           float *output,
                                           long long dim_len,
                                           long long pitch,
                                           long long output_pitch,
                                           const float *mask,
                                           long long R);

// Computes one vertical pass of a separable 2D convolution over a dim_len x dim_len image.
// image and output may be pitched allocations. pitch and output_pitch are byte strides.
// mask has length (2 * R + 1).
__global__ void vertical_convolve_kernel(const float *image,
                                         float *output,
                                         long long dim_len,
                                         long long pitch,
                                         long long output_pitch,
                                         const float *mask,
                                         long long R);

// Fused separable Gaussian pass over a dim_len x dim_len image.
__global__ void fused_gaussian_convolve_kernel(const float *image,
                                               float *output,
                                               long long dim_len,
                                               long long pitch,
                                               long long output_pitch,
                                               const float *mask,
                                               long long R);

// Uploads a device or host mask to the constant-memory mask buffer used by the launch wrappers.
// Use cudaMemcpyDeviceToDevice when mask is already in device memory, or cudaMemcpyHostToDevice
// when mask is in host memory.
cudaError_t convolution_upload_mask(const float *mask,
                                    long long R,
                                    cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice,
                                    cudaStream_t stream = 0);

cudaError_t horizontal_convolve(const float *image,
                                const float *mask,
                                float *output,
                                long long dim_len,
                                long long R,
                                long long pitch,
                                long long output_pitch,
                                unsigned int threads_per_block,
                                cudaStream_t stream = 0,
                                cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

cudaError_t vertical_convolve(const float *image,
                              const float *mask,
                              float *output,
                              long long dim_len,
                              long long R,
                              long long pitch,
                              long long output_pitch,
                              unsigned int threads_per_block,
                              cudaStream_t stream = 0,
                              cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

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
                                    cudaStream_t stream = 0,
                                    cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

cudaError_t gaussian_blur_fused(const float *image,
                                const float *mask,
                                float *output,
                                long long dim_len,
                                long long R,
                                long long pitch,
                                long long output_pitch,
                                unsigned int threads_per_block,
                                cudaStream_t stream = 0,
                                cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

// Rectangular/multichannel versions used by the Gaussian blur JPEG input benchmark.
// The image layout is planar: channel 0 rows, then channel 1 rows, etc.
// Each channel plane has height rows, and each row uses the provided byte pitch.
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
                                         cudaStream_t stream = 0,
                                         cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

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
                                       cudaStream_t stream = 0,
                                       cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

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
                                             cudaStream_t stream = 0,
                                             cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

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
                                         cudaStream_t stream = 0,
                                         cudaMemcpyKind mask_copy_kind = cudaMemcpyDeviceToDevice);

// Backward-compatible wrapper for your original horizontal pass naming.
// pitch is the input image byte stride. output is treated as contiguous rows.
__host__ void convolve(const float *image,
                       const float *mask,
                       float *output,
                       long long dim_len,
                       long long R,
                       long long pitch,
                       unsigned int threads_per_block);

#endif
