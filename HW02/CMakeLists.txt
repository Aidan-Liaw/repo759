cmake_minimum_required(VERSION 3.30)
project(HW02)

set(CMAKE_CXX_STANDARD 20)

add_executable(HW02 main.cpp
        matmul.cpp
        matmul.h
        convolution.cpp
        convolution.h)

add_executable(task1 task1.cpp
        scan.cpp
        scan.h)
