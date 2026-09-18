// Everything specific to drawing oceanlib's output.
//
// Demonstrates 3 cascades (ADR-020) summed in the shader at every pixel, over
// a geometry-clipmap mesh (ADR-021) instead of a single uniform-resolution
// tiled grid: concentric rings, each twice the previous ring's cell size,
// centred on the camera and re-snapped every frame. The core integration is
// unchanged in kind from the single-mesh version: upload each cascade's two
// buffers as RGBA32F textures, and sample them in the vertex shader - the
// clipmap only changes how many vertices ask for that sample and where.
#pragma once

#include "clipmap.hpp"
#include "vk_context.hpp"
#include "vk_math.hpp"

#include "ocean/cascade.hpp"
#include "ocean/ocean.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace viewer {

inline constexpr std::size_t kMaxCascades = 3;

// Must match the `Globals` block in shaders/common.glsl exactly.
//
// std140 rules: every member here is a vec4 or mat4, both of which are
// 16-byte aligned, so the C++ and GLSL layouts agree without padding members.
// Using a vec3 or a bare float would silently introduce padding on one side
// only - the classic uniform-buffer bug.
struct Globals {
    vkm::Mat4 view_proj;
    vkm::Mat4 inv_view_proj;
    float     cam_pos[4];
    float     sun_dir[4];
    float     cascade_patch[4];  // x,y,z = patch_length of cascades 0,1,2; w = cascade grid resolution N
    float     params[4];         // x = time, y = vertical FOV (radians),
                                  // z = viewport height (pixels), w unused -
                                  // y,z feed the vertex shader's screen-space
                                  // subpixel test for the cascade fade (ADR-021)
    float     shading[4];        // foam strength, exposure, fog density, unused
};

// Pushed once per ring draw. Matches the `RingPush` block in ocean.vert.
struct RingPush {
    float offset_x     = 0.0f;
    float offset_z     = 0.0f;
    float cell_size    = 1.0f;
    float morph_start  = 0.6f;  // fraction of the ring's own half-extent where the geomorph blend begins
};

class OceanView {
public:
    // `levels` must have between 1 and kMaxCascades entries. `layout`
    // describes the clipmap: ring count and ring 0's (finest) cell size.
    bool init(VkContext& ctx, const std::vector<ocean::OceanDesc>& levels,
              const RingLayout& layout);
    void shutdown(VkContext& ctx);

    // Copies every cascade level's buffers into this frame's textures,
    // re-places every clipmap ring around globals.cam_pos, and records the
    // draw. `frame` selects which set of per-frame resources to use.
    void record(VkContext& ctx, VkCommandBuffer cmd, std::uint32_t image_index,
                std::uint32_t frame, const ocean::CascadeStack& stack,
                const Globals& globals);

    void set_wireframe(bool on) { wireframe_ = on; }
    [[nodiscard]] bool wireframe() const { return wireframe_; }

    // Exact triangle count for the CURRENT frame's draw (varies by at most a
    // handful of triangles frame to frame as stitch bands are regenerated,
    // though their triangle COUNT is fixed - only vertex positions change).
    [[nodiscard]] std::uint32_t triangle_count() const { return triangle_count_; }

private:
    bool create_mesh(VkContext& ctx);
    bool create_textures(VkContext& ctx);
    bool create_descriptors(VkContext& ctx);
    bool create_pipelines(VkContext& ctx);

    std::size_t   level_count_      = 0;
    std::array<std::uint32_t, kMaxCascades> level_sizes_{};
    bool          wireframe_       = false;
    std::uint32_t triangle_count_  = 0;

    RingLayout ring_layout_{};

    // Static (built once at init): the shared vertex buffer and the two
    // index buffers every ring's main draw reuses (ADR-021).
    VkBuffer       clip_vertex_buffer_   = VK_NULL_HANDLE;
    VkDeviceMemory clip_vertex_memory_   = VK_NULL_HANDLE;
    VkBuffer       clip_solid_indices_   = VK_NULL_HANDLE;
    VkDeviceMemory clip_solid_memory_    = VK_NULL_HANDLE;
    std::uint32_t  clip_solid_count_     = 0;
    VkBuffer       clip_annulus_indices_ = VK_NULL_HANDLE;
    VkDeviceMemory clip_annulus_memory_  = VK_NULL_HANDLE;
    std::uint32_t  clip_annulus_count_   = 0;

    // Static: the stitch band's topology never changes (ADR-021), only its
    // vertex positions do, every frame.
    VkBuffer       stitch_index_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory stitch_index_memory_ = VK_NULL_HANDLE;
    std::uint32_t  stitch_index_count_  = 0;

    // Reused every frame: update() rewrites this in place, then its
    // .vertices are memcpy'd into whichever frame-in-flight's mapped GPU
    // buffer is currently safe to write - never reallocated after build().
    StitchBand stitch_scratch_;

    struct LevelTextures {
        VkImage        displacement        = VK_NULL_HANDLE;
        VkDeviceMemory displacement_memory = VK_NULL_HANDLE;
        VkImageView    displacement_view   = VK_NULL_HANDLE;
        VkImage        normal              = VK_NULL_HANDLE;
        VkDeviceMemory normal_memory       = VK_NULL_HANDLE;
        VkImageView    normal_view         = VK_NULL_HANDLE;
        bool           initialised         = false;
    };

    struct FrameResources {
        std::array<LevelTextures, kMaxCascades> levels{};

        VkBuffer       staging        = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        void*          staging_mapped = nullptr;
        std::array<VkDeviceSize, kMaxCascades> staging_offset{};

        VkBuffer        uniform        = VK_NULL_HANDLE;
        VkDeviceMemory  uniform_memory = VK_NULL_HANDLE;
        void*           uniform_mapped = nullptr;
        VkDescriptorSet descriptor     = VK_NULL_HANDLE;

        // Host-visible, persistently mapped, sized once at init to fit
        // StitchBand::vertices - rewritten every frame, never reallocated.
        VkBuffer        stitch_vertex        = VK_NULL_HANDLE;
        VkDeviceMemory  stitch_vertex_memory = VK_NULL_HANDLE;
        void*           stitch_vertex_mapped = nullptr;
    };
    FrameResources frames_[kFramesInFlight]{};

    VkImage        dummy_displacement_        = VK_NULL_HANDLE;
    VkDeviceMemory dummy_displacement_memory_  = VK_NULL_HANDLE;
    VkImageView    dummy_displacement_view_    = VK_NULL_HANDLE;
    VkImage        dummy_normal_               = VK_NULL_HANDLE;
    VkDeviceMemory dummy_normal_memory_        = VK_NULL_HANDLE;
    VkImageView    dummy_normal_view_          = VK_NULL_HANDLE;

    VkSampler             sampler_        = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_     = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            ocean_pipeline_  = VK_NULL_HANDLE;
    VkPipeline            ocean_wire_pipeline_ = VK_NULL_HANDLE;
    VkPipeline            sky_pipeline_    = VK_NULL_HANDLE;
};

}  // namespace viewer
