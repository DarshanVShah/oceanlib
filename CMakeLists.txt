cmake_minimum_required(VERSION 3.20)
project(oceanlib VERSION 0.1.0 LANGUAGES CXX)

option(OCEAN_BUILD_TESTS    "Build oceanlib tests"      ON)
option(OCEAN_BUILD_BENCH    "Build oceanlib benchmarks" ON)
option(OCEAN_BUILD_EXAMPLES "Build oceanlib examples"   OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Core library. Zero external dependencies, by design.
# ---------------------------------------------------------------------------
add_library(ocean STATIC
  src/core/rng.cpp
)
add_library(ocean::ocean ALIAS ocean)

target_include_directories(ocean
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(ocean PUBLIC cxx_std_20)

# Determinism policy: strict IEEE semantics, never fast-math. Reassociation or
# contraction of float ops would make results depend on compiler version and
# optimisation level, which breaks promise #3 (same seed + time => same waves).
if(MSVC)
  target_compile_options(ocean PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(ocean PRIVATE
    -Wall -Wextra -Wpedantic
    -fno-fast-math -ffp-contract=off)
endif()

if(OCEAN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(OCEAN_BUILD_BENCH)
  add_subdirectory(bench)
endif()
if(OCEAN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
