# Vulkan viewer.
#
# This is the ONLY part of the project with external dependencies. The core
# library must keep building with no Vulkan SDK present, which is why this
# lives behind OCEAN_BUILD_VIEWER and under examples/ rather than beside the
# library. The precedent is doctest: an optional target may have dependencies,
# the library may not.

find_package(Vulkan REQUIRED)

include(FetchContent)
set(GLFW_BUILD_DOCS      OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL         OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glfw)

# --- shader compilation -------------------------------------------------
find_program(OCEAN_GLSLC NAMES glslc
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" ${Vulkan_GLSLC_EXECUTABLE})
if(NOT OCEAN_GLSLC)
  message(FATAL_ERROR
    "glslc not found. It ships with the Vulkan SDK; set VULKAN_SDK or disable "
    "OCEAN_BUILD_VIEWER.")
endif()

set(OCEAN_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${OCEAN_SHADER_DIR}")

set(OCEAN_SHADER_HEADERS "")
file(GLOB OCEAN_SHADER_INCLUDES
     "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")

function(ocean_add_shader src var)
  get_filename_component(name "${src}" NAME)
  string(REPLACE "." "_" stem "${name}")
  set(spv "${OCEAN_SHADER_DIR}/${stem}.spv")
  set(hdr "${OCEAN_SHADER_DIR}/${stem}.h")
  add_custom_command(
    OUTPUT "${hdr}"
    # -O for size/speed; --target-env fixes the SPIR-V version so the same
    # source cannot compile differently on someone else's SDK.
    COMMAND "${OCEAN_GLSLC}" -O --target-env=vulkan1.2
            "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o "${spv}"
    COMMAND "${CMAKE_COMMAND}" -DSPV_IN=${spv} -DHDR_OUT=${hdr}
            -DVAR_NAME=${var}
            -P "${CMAKE_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
    # Every shared .glsl is a dependency of every shader, because they are
    # reached by #include and glslc resolves those itself - so nothing in the
    # DEPENDS list would otherwise mention common.glsl.
    #
    # This was a real bug, not a precaution: editing common.glsl produced
    # "ninja: no work to do" and the old SPIR-V kept running, so a shader
    # change appeared to have no effect. Depending on the whole set is
    # slightly over-broad - touching one include rebuilds all four shaders -
    # which for four small files is a far better trade than a stale binary
    # that silently ignores an edit.
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
            ${OCEAN_SHADER_INCLUDES}
            "${CMAKE_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
    COMMENT "Compiling and embedding ${name}"
    VERBATIM)
  set(OCEAN_SHADER_HEADERS ${OCEAN_SHADER_HEADERS} "${hdr}" PARENT_SCOPE)
endfunction()

ocean_add_shader(shaders/ocean.vert kOceanVertSpv)
set(ALL_HEADERS ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/ocean.frag kOceanFragSpv)
list(APPEND ALL_HEADERS ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/sky.vert kSkyVertSpv)
list(APPEND ALL_HEADERS ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/object.vert kObjectVertSpv)
set(ALL_HEADERS ${ALL_HEADERS} ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/object.frag kObjectFragSpv)
set(ALL_HEADERS ${ALL_HEADERS} ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/sky.frag kSkyFragSpv)
list(APPEND ALL_HEADERS ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/post.vert kPostVertSpv)
list(APPEND ALL_HEADERS ${OCEAN_SHADER_HEADERS})
ocean_add_shader(shaders/post.frag kPostFragSpv)
list(APPEND ALL_HEADERS ${OCEAN_SHADER_HEADERS})

add_custom_target(ocean_viewer_shaders DEPENDS ${ALL_HEADERS})

# --- boat model ----------------------------------------------------------
#
# The boat's real geometry (a Viking faering, assets/gislinge_viking_boat.glb)
# is baked into a plain header of vertex/index arrays at build time by a small
# host tool - not loaded at runtime - so the viewer still ships no asset file
# to find at launch (the same reasoning as embedding the shaders above,
# applied to a real mesh instead of a procedural one). See tools/gltf_bake.cpp
# for what it does and does not support.
add_executable(gltf_bake tools/gltf_bake.cpp)

set(OCEAN_BOAT_GLB "${CMAKE_SOURCE_DIR}/assets/gislinge_viking_boat (1).glb")
set(OCEAN_BOAT_HDR "${OCEAN_SHADER_DIR}/boat_mesh.h")
add_custom_command(
  OUTPUT "${OCEAN_BOAT_HDR}"
  COMMAND gltf_bake "${OCEAN_BOAT_GLB}" "${OCEAN_BOAT_HDR}"
  DEPENDS gltf_bake "${OCEAN_BOAT_GLB}"
  COMMENT "Baking boat_mesh.h from the Viking boat glb"
  VERBATIM)
add_custom_target(ocean_viewer_boat_mesh DEPENDS "${OCEAN_BOAT_HDR}")

# --- the viewer ---------------------------------------------------------
add_executable(ocean_viewer
  main.cpp
  vk_context.cpp
  ocean_view.cpp
)
add_dependencies(ocean_viewer ocean_viewer_shaders ocean_viewer_boat_mesh)
target_link_libraries(ocean_viewer PRIVATE ocean Vulkan::Vulkan glfw)
target_include_directories(ocean_viewer PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${OCEAN_SHADER_DIR}")

if(MSVC)
  # The screenshot writer uses fopen, which MSVC deprecates in favour of its
  # non-portable fopen_s. The viewer is cross-platform, so we keep fopen.
  target_compile_definitions(ocean_viewer PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(ocean_viewer PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(ocean_viewer PRIVATE -Wall -Wextra)
endif()
