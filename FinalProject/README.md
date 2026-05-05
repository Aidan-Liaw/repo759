# Gaussian blur CUDA latency benchmark

This package replaces the original `task2.cu` 1D convolution test with a JPEG-backed Gaussian blur latency benchmark.

Files:

- `task2.cu`: benchmark harness. `main.cu` is an identical copy for build systems that expect `main.cu`.
- `convolution.cu` / `convolution.cuh`: separable/fused Gaussian blur kernels with rectangular image and planar RGB/Y support.

## Basic build

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o task2
```

Run:

```bash
./task2 input.jpg 3 256 100
# argv: <image.jpg> <R> <threads_per_block> [test_iterations] [sigma] [gpu_device_index]
```

`stdout` prints:

1. mean milliseconds per tested Gaussian blur invocation
2. the last output value, to make scripted tests harder to dead-code away

Configuration details are printed on `stderr`.

## Useful compile-time switches

Colour versus B/W:

```bash
-DTASK2_USE_RGB=1   # planar RGB decode through nvJPEG
-DTASK2_USE_RGB=0   # single-channel luma/Y decode through nvJPEG
```

Average-latency-oriented options:

```bash
-DTASK2_USE_PINNED_HOST_MEMORY=1
-DTASK2_USE_PITCHED_LINEAR_MEMORY=1
-DTASK2_USE_CUDA_STREAMS=1
-DCONVOLUTION_USE_SHARED_MEMORY=1
-DCONVOLUTION_USE_CONSTANT_MASK=1
-DCONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED=1
```

Determinism-oriented options:

```bash
-DTASK2_USE_CUDA_GRAPHS=1
-DTASK2_USE_GREEN_CONTEXTS=1
-DTASK2_GREEN_CONTEXT_API=0      # 0 auto, 1 Runtime API, 2 Driver API
-DTASK2_GREEN_CONTEXT_SM_COUNT=8
-DTASK2_USE_FUSED_GAUSSIAN=1
```

Background contention test:

```bash
-DTASK2_USE_BUSY_WAIT_COMPETITOR=1
-DTASK2_BUSY_WAIT_MILLISECONDS=1000
-DTASK2_BUSY_WAIT_BLOCKS_PER_SM=4
```

MIG/MPS labeling switches:

```bash
-DTASK2_ASSUME_MIG=1
-DTASK2_ASSUME_MPS=1
```

These labels do not enable MIG or MPS. Use them only so your logs record which external GPU environment you ran under.

## Example variants

Baseline separable blur:

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o blur_baseline
```

CUDA Graphs, shared-memory separable blur:

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o blur_graph \
  -DTASK2_USE_CUDA_GRAPHS=1 \
  -DCONVOLUTION_USE_SHARED_MEMORY=1 \
  -DCONVOLUTION_USE_CONSTANT_MASK=1 \
  -DCONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED=1
```

Runtime Green Contexts, if compiling with CUDA 13.1+ headers:

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o blur_gc_runtime \
  -DTASK2_USE_GREEN_CONTEXTS=1 \
  -DTASK2_GREEN_CONTEXT_API=1 \
  -DTASK2_GREEN_CONTEXT_SM_COUNT=8
```

Driver Green Contexts fallback:

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o blur_gc_driver \
  -DTASK2_USE_GREEN_CONTEXTS=1 \
  -DTASK2_GREEN_CONTEXT_API=2 \
  -DTASK2_GREEN_CONTEXT_SM_COUNT=8
```

Green Contexts with a background busy-wait competitor:

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o blur_gc_busy \
  -DTASK2_USE_GREEN_CONTEXTS=1 \
  -DTASK2_GREEN_CONTEXT_API=0 \
  -DTASK2_USE_BUSY_WAIT_COMPETITOR=1 \
  -DTASK2_BUSY_WAIT_MILLISECONDS=1000
```

Busy-wait-only process for MPS/MIG external experiments:

```bash
nvcc -std=c++17 -O3 task2.cu convolution.cu -lnvjpeg -lcuda -o busy_only \
  -DTASK2_RUN_BUSY_WAIT_ONLY=1
./busy_only 0
```

Run a separate `task2` process while `busy_only` is active to test MPS/MIG process isolation rather than only same-process stream contention.
