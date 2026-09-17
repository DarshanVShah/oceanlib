# Examples use only the public API and the public include directory - no
# access to src/. That is deliberate: if an example cannot be written against
# include/ocean alone, the public API is missing something.
add_executable(ocean_example_basic basic.cpp)
target_link_libraries(ocean_example_basic PRIVATE ocean)

if(MSVC)
  target_compile_options(ocean_example_basic PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(ocean_example_basic PRIVATE -Wall -Wextra)
endif()
