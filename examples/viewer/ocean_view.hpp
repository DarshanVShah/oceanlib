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

#include "clipmap.hpp"
#include "props.hpp"
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
    float     shading[4];        // foam strength, exposure, fog density,
                                 // choppiness
    float     cascade_texel[4];  // x,y,z = world size of one texel per
                                 // cascade; w = world units per pixel per
                                 // metre of distance
    float     interaction[4];    // x,y = world low corner of the interaction
                                 // field; z = its extent; w = isolate flag
    float     water[4];          // x = camera depth below the surface in
                                 // metres, positive when submerged;
                                 // y = tone curve for the post pass
    float     slope_var[4];      // x,y,z = mean-square slope of cascades 0,1,2
                                 // (both axes summed); w = the BRDF's own base
                                 // roughness, as alpha

    // --- hull carve (ADR-028) --------------------------------------------
    // The ocean is one continuous sheet, so it runs straight through an open
    // boat and fills it. These describe the hull's waterline section in its
    // own space, and the shader discards sea inside it.
    vkm::Mat4 hull_inv_model;    // world -> hull local
    float     hull_section[256]; // signed z bounds (lo, hi) on a 16 x 8
                                 // (station, level) grid. MUST be read as
                                 // vec4[64] in GLSL: std140 gives a float[] a
                                 // 16-byte stride, so a float[256] there would
                                 // be 4 KB, not 1 KB.
    float     hull_params[4];    // x = hull half-length, 0 disables the carve;
                                 // y = keel local Y, z = hull top local Y,
                                 // w = unused
};

class OceanView {
public:
    // `levels` must have between 1 and kMaxCascades entries; `mesh_resolution`
    // is quads per tile edge; `tiles` is the number of copies of cascade 0's
    // (the largest scale's) patch per side of the outer mesh (odd, centred on
    // the origin).
    // `rings` is the clipmap's ring count and `base_cell` ring 0's cell size
    // in metres; together they set both the near detail and the far range
    // (ADR-025). They are independent of cascade scale by design.
    // `half_textures` picks RGBA16F over RGBA32F for the cascade and
    // interaction textures, halving the per-frame upload (ADR-026). Exposed
    // rather than hardcoded so the two can be measured against each other.
    bool init(VkContext& ctx, const std::vector<ocean::OceanDesc>& levels,
              std::uint32_t rings, float base_cell,
              std::uint32_t interaction_size, bool half_textures);
    void shutdown(VkContext& ctx);

    // Copies every cascade level's buffers into this frame's textures and
    // records the draw. `frame` selects which set of per-frame resources to
    // use. `props` are drawn on top of the ocean: the boat and any rocks in
    // flight or resting.
    void record(VkContext& ctx, VkCommandBuffer cmd, std::uint32_t image_index,
                std::uint32_t frame, const ocean::CascadeStack& stack,
                const ocean::InteractionField& field,
                const std::vector<const ocean::FoamField*>& foam,
                const Globals& globals,
                const std::vector<PropInstance>& props);

    void set_wireframe(bool on) { wireframe_ = on; }
    [[nodiscard]] bool wireframe() const { return wireframe_; }
    // Ring 0 is solid, every other ring is an annulus, and each outer ring
    // also draws a stitch band against its finer neighbour.
    [[nodiscard]] std::uint32_t triangle_count() const
    {
        const std::uint32_t outer = (ring_layout_.ring_count > 0)
                                        ? ring_layout_.ring_count - 1 : 0;
        return clip_solid_count_ / 3 + outer * (clip_annulus_count_ / 3) +
               outer * (stitch_index_count_ / 3);
    }

    // The far edge of the outermost ring, in metres - what the clipmap
    // actually covers.
    [[nodiscard]] float world_extent() const
    {
        return (ring_layout_.ring_count > 0)
                   ? ring_layout_.footprint(ring_layout_.ring_count - 1) : 0.0f;
    }
    [[nodiscard]] const RingLayout& rings() const { return ring_layout_; }

    // What one frame actually pushes across the bus, and what the CPU spends
    // preparing it. The upload was the largest span in the frame after the
    // clipmap landed, and neither half of its cost is guessable.
    [[nodiscard]] VkDeviceSize upload_bytes() const { return upload_bytes_; }
    [[nodiscard]] const GpuStat& staging_stat() const { return staging_stat_; }
    void reset_staging_stat() { staging_stat_ = GpuStat{"staging"}; }
    [[nodiscard]] bool half_textures() const { return half_textures_; }

private:
    bool create_mesh(VkContext& ctx);
    bool create_props(VkContext& ctx);
    bool create_textures(VkContext& ctx);
    bool create_descriptors(VkContext& ctx);
    bool create_pipelines(VkContext& ctx);

    std::size_t   level_count_      = 0;
    std::array<std::uint32_t, kMaxCascades> level_sizes_{};
    std::uint32_t interaction_size_ = 0;
    bool          wireframe_       = false;

    // --- upload format (ADR-026) -----------------------------------------
    // RGBA16F halves the bytes crossing the bus; RGBA32F is what the library
    // hands over untouched. `comp_bytes_` is the size of ONE component, and it
    // is the single number every piece of staging arithmetic reads, so the two
    // paths differ in exactly one place.
    bool          half_textures_ = true;
    VkFormat      texel_format_  = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkDeviceSize  comp_bytes_    = 2;
    VkDeviceSize  upload_bytes_  = 0;
    GpuStat       staging_stat_{"staging"};

    // --- clipmap (ADR-025) ------------------------------------------------
    RingLayout  ring_layout_{};
    StitchBand  stitch_scratch_{};   // CPU side; rewritten in place each frame

    // One shared local-space vertex grid, drawn once per ring with a different
    // (offset, cell size) pushed per draw. Two index buffers over it: the
    // solid grid for ring 0 and the holed one for every ring beyond it.
    VkBuffer       clip_vertex_buffer_  = VK_NULL_HANDLE;
    VkDeviceMemory clip_vertex_memory_  = VK_NULL_HANDLE;
    VkBuffer       clip_solid_indices_  = VK_NULL_HANDLE;
    VkDeviceMemory clip_solid_memory_   = VK_NULL_HANDLE;
    VkBuffer       clip_annulus_indices_ = VK_NULL_HANDLE;
    VkDeviceMemory clip_annulus_memory_  = VK_NULL_HANDLE;
    std::uint32_t  clip_solid_count_    = 0;
    std::uint32_t  clip_annulus_count_  = 0;

    // The stitch band's topology never changes, so its indices are uploaded
    // once and only its positions are rewritten per frame.
    VkBuffer       stitch_index_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory stitch_index_memory_ = VK_NULL_HANDLE;
    std::uint32_t  stitch_index_count_  = 0;
    VkDeviceSize   stitch_vertex_bytes_ = 0;   // one band's worth

    // The box and rock meshes (props.hpp), sharing one vertex/index buffer
    // pair the way the ocean grid does. One PropMeshRange per PropMesh value.
    std::array<PropMeshRange, static_cast<std::size_t>(PropMesh::Count)> prop_ranges_{};
    VkBuffer       prop_vertex_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory prop_vertex_memory_ = VK_NULL_HANDLE;
    VkBuffer       prop_index_buffer_  = VK_NULL_HANDLE;
    VkDeviceMemory prop_index_memory_  = VK_NULL_HANDLE;

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

        // The interaction field: one more texture in the same format as the
        // cascades, uploaded from the same persistently-mapped staging buffer.
        VkImage        interaction             = VK_NULL_HANDLE;
        VkDeviceMemory interaction_memory      = VK_NULL_HANDLE;
        VkImageView    interaction_view        = VK_NULL_HANDLE;
        VkDeviceSize   interaction_offset      = 0;
        bool           interaction_initialised = false;

        // One region per outer ring: each ring's stitch is regenerated against
        // its own finer neighbour, so they cannot share a buffer within a
        // frame the way a single scratch region could.
        VkBuffer       stitch_vertex        = VK_NULL_HANDLE;
        VkDeviceMemory stitch_vertex_memory = VK_NULL_HANDLE;
        void*          stitch_vertex_mapped = nullptr;

        VkBuffer        uniform        = VK_NULL_HANDLE;
        VkDeviceMemory  uniform_memory = VK_NULL_HANDLE;
        void*           uniform_mapped = nullptr;
        VkDescriptorSet descriptor     = VK_NULL_HANDLE;

        // The post pass reads the HDR target through its own single-binding
        // set. Separate from `descriptor` rather than one more binding on it,
        // because the scene pipelines bind that set while rendering INTO the
        // HDR image - a set that also names that image as a sampled texture
        // would be a read/write hazard on the same resource in the same pass.
        VkDescriptorSet post_descriptor = VK_NULL_HANDLE;

        // One set per thing the blur passes read: [0] is the scene, [i+1] is
        // bloom level i. Downsample step i reads blur_src[i], upsample step i
        // reads blur_src[i + 2].
        std::array<VkDescriptorSet, kBloomLevels + 1> blur_src{};
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
    VkPipeline            prop_pipeline_   = VK_NULL_HANDLE;

    // --- post ------------------------------------------------------------
    VkDescriptorSetLayout post_set_layout_      = VK_NULL_HANDLE;
    VkPipelineLayout      post_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            post_pipeline_        = VK_NULL_HANDLE;
    // CLAMP_TO_EDGE, unlike either sampler above: a fullscreen pass reads its
    // own resolution exactly, and the wrap mode only matters once bloom starts
    // sampling off the edge - where repeating would wrap the bright side of
    // the image onto the dark one.
    VkSampler             post_sampler_         = VK_NULL_HANDLE;

    // --- bloom -----------------------------------------------------------
    // One binding, the source image; shared by both blur directions.
    VkDescriptorSetLayout blur_set_layout_      = VK_NULL_HANDLE;
    VkPipelineLayout      blur_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            bloom_down_pipeline_  = VK_NULL_HANDLE;
    VkPipeline            bloom_up_pipeline_    = VK_NULL_HANDLE;
    float                 bloom_intensity_      = 0.0f;

public:
    // How much of the composite is the blurred image rather than the sharp
    // one. 0 disables the chain entirely, skipping its passes.
    void set_bloom(float intensity) { bloom_intensity_ = intensity; }
};

}  // namespace viewer
