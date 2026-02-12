cmake_minimum_required(VERSION 3.0)
project(HW03 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(task1 task1.cu)

add_executable(task2 task2.cu)

add_executable(task3 vscale.cu vscale.cuh)

set_target_properties(task1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(task2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(task3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
