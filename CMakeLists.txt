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
)

target_include_directories(ezllm PRIVATE include)

target_compile_definitions(ezllm PRIVATE
    EZLLM_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(ezllm PRIVATE /W4
        -O3
        -march=native
    )
else()
    target_compile_options(ezllm PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
endif()