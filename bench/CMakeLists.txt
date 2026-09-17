# Benchmarks link against the internal headers on purpose: a per-stage
# breakdown is what tells us where to spend optimisation effort, and the
# public API deliberately does not expose it.
add_executable(ocean_bench bench_update.cpp)
target_link_libraries(ocean_bench PRIVATE ocean)
target_include_directories(ocean_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(ocean_bench PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(ocean_bench PRIVATE
    -Wall -Wextra -fno-fast-math -ffp-contract=off)
endif()
