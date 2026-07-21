cmake_minimum_required(VERSION 3.16)

project(EZLLM_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ezllm
    src/main.cpp
    src/Tensor.cpp
    src/LayerBlock.cpp
    src/TensorUtils.cpp
    src/Model.cpp
    src/Tensor4D/MathOps.cpp
    src/Tensor4D/Tensor4D.cpp
    src/Tensor4D/Tensor4DUtils.cpp
    src/Tensor4D/SlicedTensor4D.cpp
)

target_include_directories(ezllm PRIVATE include)

target_compile_definitions(ezllm PRIVATE
    EZLLM_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(ezllm PRIVATE
        /W4
        /utf-8

        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/arch:AVX2>
    )
else()
    target_compile_options(ezllm PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -finput-charset=UTF-8
        -fexec-charset=UTF-8

        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-mavx2>
        $<$<CONFIG:Release>:-mfma>
    )
endif()