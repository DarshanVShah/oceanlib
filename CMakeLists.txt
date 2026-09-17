cmake_minimum_required(VERSION 3.20)
project(oceanlib VERSION 0.1.0 LANGUAGES CXX C)

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
  src/core/fft.cpp
  src/core/cpu_features.cpp
  src/core/fft_kernel_scalar.cpp
  src/core/fft_kernel_sse2.cpp
  src/core/fft_kernel_avx2.cpp
  src/core/fft_kernel_neon.cpp
  src/core/spectrum.cpp
  src/core/evolve.cpp
  src/core/thread_pool.cpp
  src/ocean.cpp
  src/c_api.cpp
)
add_library(ocean::ocean ALIAS ocean)

target_include_directories(ocean
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(ocean PUBLIC cxx_std_20)

# The AVX2 kernel is the only file built with AVX2 enabled. Everything else
# stays at the architectural baseline, so the binary still boots on a CPU
# without AVX2 and reaches this code only through the runtime dispatch in
# cpu_features.cpp. Compiling the whole library with /arch:AVX2 would let the
# compiler emit AVX2 anywhere it liked and produce a binary that crashes on
# older hardware - which would break promise #1.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64|i[3-6]86|X86)")
  if(MSVC)
    set_source_files_properties(src/core/fft_kernel_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/core/fft_kernel_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2")
  endif()
endif()

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
