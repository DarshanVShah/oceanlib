# Converts a .spv binary into a C header holding a uint32_t array.
#
# Invoked in script mode by a custom command:
#   cmake -DSPV_IN=... -DHDR_OUT=... -DVAR_NAME=... -P EmbedSpirv.cmake
#
# Embedding rather than shipping .spv files next to the executable removes
# runtime asset paths, which are the usual reason a demo fails to launch from a
# different working directory - and makes it impossible to run a shader that
# does not match the binary it was built with.

file(READ "${SPV_IN}" hex_data HEX)
string(LENGTH "${hex_data}" hex_len)
math(EXPR word_count "${hex_len} / 8")

set(body "")
set(i 0)
while(i LESS word_count)
  math(EXPR offset "${i} * 8")
  string(SUBSTRING "${hex_data}" ${offset} 8 word)
  # file(READ ... HEX) yields bytes in file order; SPIR-V words are
  # little-endian, so reverse the four bytes to rebuild each word.
  string(SUBSTRING "${word}" 0 2 b0)
  string(SUBSTRING "${word}" 2 2 b1)
  string(SUBSTRING "${word}" 4 2 b2)
  string(SUBSTRING "${word}" 6 2 b3)
  string(APPEND body "    0x${b3}${b2}${b1}${b0}u,")
  math(EXPR line_break "${i} % 4")
  if(line_break EQUAL 3)
    string(APPEND body "\n")
  else()
    string(APPEND body " ")
  endif()
  math(EXPR i "${i} + 1")
endwhile()

get_filename_component(src_name "${SPV_IN}" NAME)
file(WRITE "${HDR_OUT}"
"// Generated from ${src_name}. Do not edit - regenerated on every build.
#pragma once

#include <cstdint>

static const uint32_t ${VAR_NAME}[] = {
${body}
};
")
