# Benchmarks link against the internal headers on purpose: a per-stage
# breakdown is what tells us where to spend optimisation effort, and the
# public API deliberately does not expose it.
add_executable(ocean_bench bench_update.cpp)
add_executable(ocean_bench_interaction bench_interaction.cpp)
target_link_libraries(ocean_bench PRIVATE ocean)
target_link_libraries(ocean_bench_interaction PRIVATE ocean)
target_include_directories(ocean_bench_interaction PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(ocean_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

foreach(t ocean_bench ocean_bench_interaction)
  if(MSVC)
    target_compile_options(${t} PRIVATE /W4 /permissive- /fp:precise)
  else()
    target_compile_options(${t} PRIVATE
      -Wall -Wextra -fno-fast-math -ffp-contract=off)
  endif()
endforeach()
