// Everything specific to drawing oceanlib's output.
//
// The whole integration is: upload two buffers as RGBA32F textures, and sample
// them. If this file is longer than you expected, note that almost all of it is
// Vulkan object creation - the part that actually consumes the library is
// `upload_ocean_buffers`, and it is two memcpys and two image copies.
#pragma once

#include "vk_context.hpp"
#include "vk_math.hpp"

#include "ocean/ocean.hpp"

#include <cstdint>
#include <vector>

namespace viewer {

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
    float     params[4];   // patch_length, tiles, time, mesh resolution
    float     shading[4];  // foam strength, exposure, fog density, choppiness
};

class OceanView {
public:
    // `mesh_resolution` is the number of quads per tile edge; `tiles` is the
    // number of patch copies per side (odd, centred on the origin).
    bool init(VkContext& ctx, std::uint32_t ocean_size,
              std::uint32_t mesh_resolution, std::uint32_t tiles);
    void shutdown(VkContext& ctx);

    // Copies the library's buffers into this frame's textures and records the
    // draw. `frame` selects which set of per-frame resources to use.
    void record(VkContext& ctx, VkCommandBuffer cmd, std::uint32_t image_index,
                std::uint32_t frame, const ocean::Buffers& buffers,
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

    std::uint32_t ocean_size_      = 0;
    std::uint32_t mesh_resolution_ = 0;
    std::uint32_t tiles_           = 0;
    std::uint32_t index_count_     = 0;
    bool          wireframe_       = false;

    VkBuffer       vertex_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertex_memory_ = VK_NULL_HANDLE;
    VkBuffer       index_buffer_  = VK_NULL_HANDLE;
    VkDeviceMemory index_memory_  = VK_NULL_HANDLE;

    // One set of textures per frame in flight.
    //
    // A single set would race: we only wait on the fence for THIS frame index,
    // so the other in-flight submission may still be sampling the textures
    // while we overwrite them. Double-buffering is cheaper and simpler than
    // the extra barriers a single set would need, and at 256^2 it costs 2 MB.
    struct FrameResources {
        VkImage        displacement        = VK_NULL_HANDLE;
        VkDeviceMemory displacement_memory = VK_NULL_HANDLE;
        VkImageView    displacement_view   = VK_NULL_HANDLE;
        VkImage        normal              = VK_NULL_HANDLE;
        VkDeviceMemory normal_memory       = VK_NULL_HANDLE;
        VkImageView    normal_view         = VK_NULL_HANDLE;

        VkBuffer       staging        = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        void*          staging_mapped = nullptr;

        VkBuffer        uniform        = VK_NULL_HANDLE;
        VkDeviceMemory  uniform_memory = VK_NULL_HANDLE;
        void*           uniform_mapped = nullptr;
        VkDescriptorSet descriptor     = VK_NULL_HANDLE;

        bool textures_initialised = false;
    };
    FrameResources frames_[kFramesInFlight]{};

    VkSampler             sampler_        = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_     = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            ocean_pipeline_  = VK_NULL_HANDLE;
    VkPipeline            ocean_wire_pipeline_ = VK_NULL_HANDLE;
    VkPipeline            sky_pipeline_    = VK_NULL_HANDLE;
};

}  // namespace viewer
