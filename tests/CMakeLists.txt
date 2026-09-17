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
)
target_link_libraries(ocean_tests PRIVATE ocean doctest::doctest)
target_include_directories(ocean_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(ocean_tests PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(ocean_tests PRIVATE -Wall -Wextra -fno-fast-math -ffp-contract=off)
endif()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(ocean_tests)
