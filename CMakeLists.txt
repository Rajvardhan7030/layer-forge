cmake_minimum_required(VERSION 3.18)
project(layer-forge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/W4 /OX)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Directories
include_directories(include)

# Core Library
file(GLOB_RECURSE CORE_SOURCES src/core/*.cpp)
file(GLOB_RECURSE MODEL_SOURCES src/model/*.cpp)
file(GLOB_RECURSE LAYER_SOURCES src/layers/*.cpp)
file(GLOB_RECURSE SCHEDULER_SOURCES src/scheduler/*.cpp)
file(GLOB_RECURSE KV_CACHE_SOURCES src/kv_cache/*.cpp)
file(GLOB_RECURSE SAMPLER_SOURCES src/sampler/*.cpp)
file(GLOB_RECURSE TOKENIZER_SOURCES src/tokenizer/*.cpp)

add_library(layer-forge_core STATIC
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${LAYER_SOURCES}
    ${SCHEDULER_SOURCES}
    ${KV_CACHE_SOURCES}
    ${SAMPLER_SOURCES}
    ${TOKENIZER_SOURCES}
)

# Placeholder for main executable/testing
add_executable(forge_cli src/main.cpp)
target_link_libraries(forge_cli PRIVATE layer-forge_core)

# TODO: Add pybind11 for python bindings
# find_package(pybind11 REQUIRED)
# pybind11_add_module(friday_engine_cpp python/bindings.cpp)
# target_link_libraries(friday_engine_cpp PRIVATE layer-forge_core)
