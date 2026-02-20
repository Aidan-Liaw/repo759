cmake_minimum_required(VERSION 3.0)
project(HW04 CUDA)

set(CMAKE_CUDA_STANDARD 17)

set_target_properties(HW04 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(task1 task1.cu
        matmul.cu matmul.cuh)

add_executable(task2 task2.cu
        stencil.cu stencil.cuh)
