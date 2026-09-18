// Everything specific to drawing oceanlib's output.
//
// Demonstrates 3 cascades (ADR-020): a far/large scale, a mid scale, and a
// near/fine scale, summed in the shader at every pixel. The viewer fixes the
// count at 3 (kMaxCascades below) to keep the descriptor layout and shader
// loops simple - the library's own CascadeStack is not limited to 3, this is
// a demo-only simplification.
//
// The core integration is unchanged in kind from the single-cascade version:
// upload each cascade's two buffers as RGBA32F textures, and sample them -
// just three times instead of once, summed per ADR-020's "additive sum, no
// distance weighting" decision.
#pragma once

#include "vk_context.hpp"
#include "vk_math.hpp"

#include "ocean/cascade.hpp"
#include "ocean/foam.hpp"
#include "ocean/interaction.hpp"
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
    float     cascade_patch[4];  // x,y,z = patch_length of cascades 0,1,2;
                                 // w = tiles per side of the outer mesh
    float     params[4];         // x = time, y = mesh resolution, z,w unused
    float     shading[4];        // foam strength, exposure, fog density, unused
    float     cascade_texel[4];  // x,y,z = world size of one texel per
                                 // cascade; w = world units per pixel per
                                 // metre of distance
    float     interaction[4];    // x,y = world low corner of the interaction
                                 // field; z = its extent; w = isolate flag
};

class OceanView {
public:
    // `levels` must have between 1 and kMaxCascades entries; `mesh_resolution`
    // is quads per tile edge; `tiles` is the number of copies of cascade 0's
    // (the largest scale's) patch per side of the outer mesh (odd, centred on
    // the origin).
    bool init(VkContext& ctx, const std::vector<ocean::OceanDesc>& levels,
              std::uint32_t mesh_resolution, std::uint32_t tiles,
              std::uint32_t interaction_size);
    void shutdown(VkContext& ctx);

    // Copies every cascade level's buffers into this frame's textures and
    // records the draw. `frame` selects which set of per-frame resources to
    // use.
    void record(VkContext& ctx, VkCommandBuffer cmd, std::uint32_t image_index,
                std::uint32_t frame, const ocean::CascadeStack& stack,
                const ocean::InteractionField& field,
                const std::vector<const ocean::FoamField*>& foam,
                const Globals& globals);

    void set_wireframe(bool on) { wireframe_ = on; }
    [[nodiscard]] bool wireframe() const { return wireframe_; }
    [[nodiscard]] std::uint32_t triangle_count() const
    {
        return mesh_resolution_ * mesh_resolution_ * 2u * tiles_ * tiles_;
    }

private:
    bool create_mesh(VkContext& ctx);
    bool create_textures(VkContext& ctx);
    bool create_descriptors(VkContext& ctx);
    bool create_pipelines(VkContext& ctx);

    std::size_t   level_count_      = 0;
    std::array<std::uint32_t, kMaxCascades> level_sizes_{};
    std::uint32_t interaction_size_ = 0;
    std::uint32_t mesh_resolution_ = 0;
    std::uint32_t tiles_           = 0;
    std::uint32_t index_count_     = 0;
    bool          wireframe_       = false;

    VkBuffer       vertex_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertex_memory_ = VK_NULL_HANDLE;
    VkBuffer       index_buffer_  = VK_NULL_HANDLE;
    VkDeviceMemory index_memory_  = VK_NULL_HANDLE;

    // One set of textures per frame in flight, per cascade level.
    //
    // A single set would race: we only wait on the fence for THIS frame index,
    // so the other in-flight submission may still be sampling the textures
    // while we overwrite them. Double-buffering is cheaper and simpler than
    // the extra barriers a single set would need.
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

        // One staging buffer holding every level's displacement+normal data
        // back to back, persistently mapped. Sized to the SUM of all levels'
        // byte counts, since levels may have different resolutions.
        VkBuffer       staging        = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        void*          staging_mapped = nullptr;
        std::array<VkDeviceSize, kMaxCascades> staging_offset{};

        // The interaction field: one more RGBA32F texture, uploaded from the
        // same persistently-mapped staging buffer.
        VkImage        interaction             = VK_NULL_HANDLE;
        VkDeviceMemory interaction_memory      = VK_NULL_HANDLE;
        VkImageView    interaction_view        = VK_NULL_HANDLE;
        VkDeviceSize   interaction_offset      = 0;
        bool           interaction_initialised = false;

        VkBuffer        uniform        = VK_NULL_HANDLE;
        VkDeviceMemory  uniform_memory = VK_NULL_HANDLE;
        void*           uniform_mapped = nullptr;
        VkDescriptorSet descriptor     = VK_NULL_HANDLE;
    };
    FrameResources frames_[kFramesInFlight]{};

    // A single 1x1 texture pair (displacement all-zero, normal +Y unit)
    // that unused cascade slots (when fewer than kMaxCascades levels are
    // active) are bound to instead of aliasing level 0's real texture.
    // Without this, an unused slot pointed at level 0 would double- or
    // triple-count level 0's contribution in the shader's sum, rather
    // than contributing nothing - a real bug caught before it ever ran,
    // by tracing through what --cascades 1 would actually sample.
    VkImage        dummy_displacement_        = VK_NULL_HANDLE;
    VkDeviceMemory dummy_displacement_memory_  = VK_NULL_HANDLE;
    VkImageView    dummy_displacement_view_    = VK_NULL_HANDLE;
    VkImage        dummy_normal_               = VK_NULL_HANDLE;
    VkDeviceMemory dummy_normal_memory_        = VK_NULL_HANDLE;
    VkImageView    dummy_normal_view_          = VK_NULL_HANDLE;

    VkSampler             sampler_        = VK_NULL_HANDLE;

    // A second sampler, CLAMP_TO_BORDER with a transparent-black border, for
    // the interaction field. The cascade sampler REPEATs because those fields
    // really are periodic; this one is not, and reusing the repeating sampler
    // would tile one splash across the whole ocean.
    VkSampler             clamp_sampler_  = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_     = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            ocean_pipeline_  = VK_NULL_HANDLE;
    VkPipeline            ocean_wire_pipeline_ = VK_NULL_HANDLE;
    VkPipeline            sky_pipeline_    = VK_NULL_HANDLE;
};

}  // namespace viewer
