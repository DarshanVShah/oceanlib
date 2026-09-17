# doctest is fetched only for tests; the core library stays dependency-free.
include(FetchContent)
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(doctest)

add_executable(ocean_tests
  test_main.cpp
  alloc_probe.cpp
  test_rng.cpp
  test_api.cpp
  test_fft.cpp
  test_spectrum.cpp
  test_simulation.cpp
  test_query.cpp
  test_threading.cpp
  test_simd.cpp
  test_c_api.cpp
  c_header_check.c
)
target_link_libraries(ocean_tests PRIVATE ocean doctest::doctest)
target_include_directories(ocean_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

# The C file is built as C on purpose: a "C API" header that is only ever
# included from C++ drifts into C++-only constructs within a release or two.
set_source_files_properties(c_header_check.c PROPERTIES LANGUAGE C)

if(MSVC)
  target_compile_options(ocean_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>
    $<$<COMPILE_LANGUAGE:C>:/W4>)
else()
  target_compile_options(ocean_tests PRIVATE -Wall -Wextra -fno-fast-math -ffp-contract=off)
endif()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(ocean_tests)
