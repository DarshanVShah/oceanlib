#include "ocean_view.hpp"

#include <array>

#include "object_frag.h"
#include "object_vert.h"
#include "ocean_frag.h"
#include "ocean_vert.h"
#include "bloom_down_frag.h"
#include "bloom_up_frag.h"
#include "post_frag.h"
#include "post_vert.h"
#include "sky_frag.h"
#include "sky_vert.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace viewer {
namespace {

VkShaderModule make_module(VkDevice device, const std::uint32_t* code,
                           std::size_t bytes)
{
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode    = code;
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &ci, nullptr, &m);
    return m;
}

// Uploads `data` into a fresh DEVICE_LOCAL buffer via a throwaway staging
// buffer. Used for every buffer that is written once at startup and read many
// times per frame after - the grid mesh, and the box/rock prop meshes.
void upload_device_local(VkContext& ctx, const void* data, VkDeviceSize size,
                         VkBufferUsageFlags usage, VkBuffer& buf,
                         VkDeviceMemory& mem)
{
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    ctx.create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      staging, staging_mem);
    void* mapped = nullptr;
    vkMapMemory(ctx.device, staging_mem, 0, size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(ctx.device, staging_mem);

    ctx.create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);

    VkCommandBuffer cmd = ctx.begin_one_shot();
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, staging, buf, 1, &copy);
    ctx.end_one_shot(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, staging_mem, nullptr);
}

// Push constants are the leading bytes of PropInstance (model, scale, color,
// surf) - everything except `mesh`, which selects the draw call rather than
// travelling to the shader. offsetof rather than a hardcoded size so adding a
// field to PropInstance cannot silently leave the range behind; 112 bytes as
// this stands, against the 128 every Vulkan implementation must offer.
constexpr VkDeviceSize kPropPushSize = offsetof(PropInstance, mesh);
static_assert(kPropPushSize <= 128,
              "prop push constants must fit the guaranteed minimum range");

}  // namespace

bool OceanView::init(VkContext& ctx, const std::vector<ocean::OceanDesc>& levels,
                     std::uint32_t rings, float base_cell,
                     std::uint32_t interaction_size)
{
    ring_layout_.ring_count =
        (rings < 1u) ? 1u : ((rings > kMaxRings) ? kMaxRings : rings);
    ring_layout_.base_cell_size = (base_cell > 0.0f) ? base_cell : 0.5f;
    interaction_size_ = interaction_size;
    level_count_ = levels.size();
    if (level_count_ == 0 || level_count_ > kMaxCascades) {
        std::fprintf(stderr,
                     "[viewer] level count must be between 1 and %zu\n",
                     kMaxCascades);
        return false;
    }
    for (std::size_t i = 0; i < level_count_; ++i) {
        level_sizes_[i] = levels[i].size;
    }
    if (!create_mesh(ctx)) return false;
    if (!create_props(ctx)) return false;
    if (!create_textures(ctx)) return false;
    if (!create_descriptors(ctx)) return false;
    return create_pipelines(ctx);
}

// ---------------------------------------------------------------------------
// Grid mesh
// ---------------------------------------------------------------------------

bool OceanView::create_mesh(VkContext& ctx)
{
    // The clipmap's shared local grid (ADR-025). Two floats per vertex and
    // nothing else - position, normal and foam all come from the cascade
    // textures at draw time, so the rings cost no per-frame vertex traffic at
    // all. Only the stitch bands are rewritten each frame, and only their
    // positions; their topology is fixed.
    const ClipmapMesh mesh = ClipmapMesh::build();
    clip_solid_count_   = static_cast<std::uint32_t>(mesh.solid_indices.size());
    clip_annulus_count_ = static_cast<std::uint32_t>(mesh.annulus_indices.size());

    // Device-local: none of this changes, so one staged copy at startup buys
    // the fastest possible reads forever after.
    upload_device_local(ctx, mesh.vertices.data(),
                        mesh.vertices.size() * sizeof(float),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        clip_vertex_buffer_, clip_vertex_memory_);
    upload_device_local(ctx, mesh.solid_indices.data(),
                        mesh.solid_indices.size() * sizeof(std::uint32_t),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        clip_solid_indices_, clip_solid_memory_);
    upload_device_local(ctx, mesh.annulus_indices.data(),
                        mesh.annulus_indices.size() * sizeof(std::uint32_t),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        clip_annulus_indices_, clip_annulus_memory_);

    stitch_scratch_      = StitchBand::build();
    stitch_index_count_  = static_cast<std::uint32_t>(stitch_scratch_.indices.size());
    stitch_vertex_bytes_ =
        static_cast<VkDeviceSize>(stitch_scratch_.vertices.size()) * sizeof(float);
    if (stitch_index_count_ > 0) {
        upload_device_local(ctx, stitch_scratch_.indices.data(),
                            stitch_scratch_.indices.size() * sizeof(std::uint32_t),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            stitch_index_buffer_, stitch_index_memory_);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Prop meshes (boat, rock)
// ---------------------------------------------------------------------------

bool OceanView::create_props(VkContext& ctx)
{
    std::vector<PropVertex>    vtx;
    std::vector<std::uint32_t> idx;
    append_box(vtx, idx, prop_ranges_[static_cast<std::size_t>(PropMesh::Box)]);
    append_rock(vtx, idx, prop_ranges_[static_cast<std::size_t>(PropMesh::Rock)]);
    append_boat_hull(vtx, idx, prop_ranges_[static_cast<std::size_t>(PropMesh::BoatHull)]);
    append_boat_sail(vtx, idx, prop_ranges_[static_cast<std::size_t>(PropMesh::BoatSail)]);

    upload_device_local(ctx, vtx.data(), vtx.size() * sizeof(PropVertex),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, prop_vertex_buffer_,
                        prop_vertex_memory_);
    upload_device_local(ctx, idx.data(), idx.size() * sizeof(std::uint32_t),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, prop_index_buffer_,
                        prop_index_memory_);
    return true;
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

bool OceanView::create_textures(VkContext& ctx)
{
    auto make_image = [&](std::uint32_t size, VkImage& image,
                          VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        // R32G32B32A32_SFLOAT because that is literally what the library
        // hands us. No conversion, no quantisation, no repacking - which was
        // the entire point of choosing that output layout.
        ci.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent        = {size, size, 1};
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(ctx.device, &ci, nullptr, &image);

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(ctx.device, image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex =
            ctx.find_memory_type(req.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(ctx.device, &ai, nullptr, &memory);
        vkBindImageMemory(ctx.device, image, memory, 0);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image            = image;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = ci.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &view);
    };

    for (auto& f : frames_) {
        // Staging buffer holds every level's displacement+normal data back to
        // back, persistently mapped, sized to the SUM of all levels' byte
        // counts since levels may have different resolutions.
        VkDeviceSize total_bytes = 0;
        for (std::size_t i = 0; i < level_count_; ++i) {
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(level_sizes_[i]) *
                                       level_sizes_[i] * 4 * sizeof(float);
            f.staging_offset[i] = total_bytes;  // displacement starts here
            total_bytes += bytes;                // normal follows immediately after
            total_bytes += bytes;
        }
        f.interaction_offset = total_bytes;
        total_bytes += static_cast<VkDeviceSize>(interaction_size_) *
                       interaction_size_ * 4 * sizeof(float);

        for (std::size_t i = 0; i < level_count_; ++i) {
            make_image(level_sizes_[i], f.levels[i].displacement,
                      f.levels[i].displacement_memory, f.levels[i].displacement_view);
            make_image(level_sizes_[i], f.levels[i].normal,
                      f.levels[i].normal_memory, f.levels[i].normal_view);
        }
        make_image(interaction_size_, f.interaction, f.interaction_memory,
                   f.interaction_view);

        ctx.create_buffer(total_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          f.staging, f.staging_memory);
        vkMapMemory(ctx.device, f.staging_memory, 0, total_bytes, 0,
                    &f.staging_mapped);

        // One stitch region per OUTER ring. Host-visible and persistently
        // mapped, because unlike the rings themselves these positions change
        // every frame - each ring snaps to the camera independently, so the
        // gap its stitch has to close is a continuously varying offset rather
        // than one of a few discrete cases.
        const std::uint32_t outer_rings = (ring_layout_.ring_count > 0)
                                              ? ring_layout_.ring_count - 1 : 0;
        if (outer_rings > 0 && stitch_vertex_bytes_ > 0) {
            const VkDeviceSize bytes = stitch_vertex_bytes_ * outer_rings;
            ctx.create_buffer(bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              f.stitch_vertex, f.stitch_vertex_memory);
            vkMapMemory(ctx.device, f.stitch_vertex_memory, 0, bytes, 0,
                        &f.stitch_vertex_mapped);
        }

        ctx.create_buffer(sizeof(Globals), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          f.uniform, f.uniform_memory);
        vkMapMemory(ctx.device, f.uniform_memory, 0, sizeof(Globals), 0,
                    &f.uniform_mapped);
    }

    // Dummy 1x1 pair for unused cascade slots (see the header comment on
    // dummy_displacement_). Uploaded once via a one-shot staged copy, exactly
    // like the mesh buffers - it never changes after this.
    {
        make_image(1, dummy_displacement_, dummy_displacement_memory_,
                  dummy_displacement_view_);
        make_image(1, dummy_normal_, dummy_normal_memory_, dummy_normal_view_);

        const float zero_disp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float up_normal[4] = {0.0f, 1.0f, 0.0f, 1.0f};  // +Y, unfolded Jacobian

        auto upload_texel = [&](VkImage image, const float texel[4]) {
            VkBuffer staging = VK_NULL_HANDLE;
            VkDeviceMemory staging_mem = VK_NULL_HANDLE;
            const VkDeviceSize bytes = 4 * sizeof(float);
            ctx.create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              staging, staging_mem);
            void* mapped = nullptr;
            vkMapMemory(ctx.device, staging_mem, 0, bytes, 0, &mapped);
            std::memcpy(mapped, texel, static_cast<std::size_t>(bytes));
            vkUnmapMemory(ctx.device, staging_mem);

            VkCommandBuffer cmd = ctx.begin_one_shot();
            transition_image(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent      = {1, 1, 1};
            vkCmdCopyBufferToImage(cmd, staging, image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            transition_image(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            ctx.end_one_shot(cmd);

            vkDestroyBuffer(ctx.device, staging, nullptr);
            vkFreeMemory(ctx.device, staging_mem, nullptr);
        };
        upload_texel(dummy_displacement_, zero_disp);
        upload_texel(dummy_normal_, up_normal);
    }

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    // REPEAT is what makes the tiling free: each cascade's FFT surface is
    // exactly periodic with its OWN patch_length, so a wrapped sample at
    // u = 37.4 is genuinely that cascade's correct value, not an
    // approximation. CLAMP would produce a visible seam at every period.
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.maxLod       = 0.0f;
    vkCreateSampler(ctx.device, &sci, nullptr, &sampler_);

    // The interaction field is NOT periodic, so it clamps to a transparent
    // black border instead of repeating. Getting this wrong is not subtle once
    // you see it - one splash tiles across the entire ocean - but it is very
    // easy to get wrong by reusing the sampler that is already to hand.
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    vkCreateSampler(ctx.device, &sci, nullptr, &clamp_sampler_);
    return true;
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

bool OceanView::create_descriptors(VkContext& ctx)
{
    // Binding 0: the Globals UBO. Bindings 1..6: displacement/normal pairs for
    // up to 3 cascades (1,2 = level 0; 3,4 = level 1; 5,6 = level 2), matching
    // the fixed bindings the shaders declare in common.glsl / ocean.vert /
    // ocean.frag. Unused levels (when level_count_ < 3) still get a
    // descriptor written pointing at level 0's textures, so the shader's
    // fixed 3-cascade loop can read a harmless, valid image rather than an
    // unbound slot - simpler than making the shader loop count dynamic for a
    // demo that only ever runs with 1-3 fixed cascades.
    // Binding 7 is the interaction field, read by BOTH stages: the vertex
    // shader adds its height, the fragment shader adds its slope.
    constexpr std::uint32_t kInteractionBinding =
        1 + 2 * static_cast<std::uint32_t>(kMaxCascades);
    constexpr std::uint32_t kBindingCount = kInteractionBinding + 1;
    VkDescriptorSetLayoutBinding bindings[kBindingCount]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    for (std::uint32_t lvl = 0; lvl < kMaxCascades; ++lvl) {
        const std::uint32_t disp_binding = 1 + lvl * 2;
        const std::uint32_t norm_binding = 2 + lvl * 2;

        bindings[disp_binding].binding         = disp_binding;
        bindings[disp_binding].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[disp_binding].descriptorCount = 1;
        // Sampled in the VERTEX stage - this is the vertex texture fetch that
        // makes GPU-side displacement possible.
        // VERTEX for the displacement itself, FRAGMENT because foam lives in
        // that texture's alpha and has to be sampled per pixel - see the note
        // in ocean.frag.
        bindings[disp_binding].stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[norm_binding].binding         = norm_binding;
        bindings[norm_binding].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[norm_binding].descriptorCount = 1;
        bindings[norm_binding].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    bindings[kInteractionBinding].binding         = kInteractionBinding;
    bindings[kInteractionBinding].descriptorType  =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[kInteractionBinding].descriptorCount = 1;
    bindings[kInteractionBinding].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = kBindingCount;
    lci.pBindings    = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &set_layout_);

    // The post pass's layout: the HDR scene target, plus level 0 of the bloom
    // chain (which by composite time holds every level summed into it).
    VkDescriptorSetLayoutBinding post_bindings[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
        post_bindings[i].binding         = i;
        post_bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        post_bindings[i].descriptorCount = 1;
        post_bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo post_lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    post_lci.bindingCount = 2;
    post_lci.pBindings    = post_bindings;
    vkCreateDescriptorSetLayout(ctx.device, &post_lci, nullptr, &post_set_layout_);

    // The blur passes read exactly one image, whichever step they are on.
    VkDescriptorSetLayoutCreateInfo blur_lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    blur_lci.bindingCount = 1;
    blur_lci.pBindings    = &post_bindings[0];
    vkCreateDescriptorSetLayout(ctx.device, &blur_lci, nullptr, &blur_set_layout_);

    VkSamplerCreateInfo post_sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    post_sci.magFilter    = VK_FILTER_LINEAR;
    post_sci.minFilter    = VK_FILTER_LINEAR;
    post_sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    post_sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    post_sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    post_sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkCreateSampler(ctx.device, &post_sci, nullptr, &post_sampler_);

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Per frame: the cascade textures and the interaction field, the post
    // pass's two, and one per blur source.
    sizes[1].descriptorCount =
        kFramesInFlight * (2 * static_cast<std::uint32_t>(kMaxCascades) + 1
                           + 2 + (kBloomLevels + 1));

    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets       = kFramesInFlight * (2 + kBloomLevels + 1);
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = sizes;
    vkCreateDescriptorPool(ctx.device, &pci, nullptr, &descriptor_pool_);

    for (auto& f : frames_) {
        VkDescriptorSetAllocateInfo ai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = descriptor_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &set_layout_;
        vkAllocateDescriptorSets(ctx.device, &ai, &f.descriptor);

        // Allocated here, but WRITTEN every frame in record(). The HDR target
        // is recreated whenever the swapchain is, so a view written once at
        // startup would dangle after the first window resize - and rewriting
        // one descriptor per frame is cheaper than detecting the resize.
        VkDescriptorSetAllocateInfo pai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        pai.descriptorPool     = descriptor_pool_;
        pai.descriptorSetCount = 1;
        pai.pSetLayouts        = &post_set_layout_;
        vkAllocateDescriptorSets(ctx.device, &pai, &f.post_descriptor);

        for (VkDescriptorSet& set : f.blur_src) {
            VkDescriptorSetAllocateInfo bai{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            bai.descriptorPool     = descriptor_pool_;
            bai.descriptorSetCount = 1;
            bai.pSetLayouts        = &blur_set_layout_;
            vkAllocateDescriptorSets(ctx.device, &bai, &set);
        }

        VkDescriptorBufferInfo ubo{f.uniform, 0, sizeof(Globals)};

        VkDescriptorImageInfo disp_info[kMaxCascades]{};
        VkDescriptorImageInfo norm_info[kMaxCascades]{};
        for (std::uint32_t lvl = 0; lvl < kMaxCascades; ++lvl) {
            // Levels beyond level_count_ are bound to the dummy 1x1 pair, NOT
            // to level 0's real textures: aliasing level 0 would make its
            // contribution to the shader's sum count 2x or 3x instead of
            // once. The dummy pair (all-zero displacement, +Y normal)
            // contributes exactly nothing to height/offset and a neutral
            // normal to the sum, matching "this cascade does not exist".
            if (lvl < level_count_) {
                disp_info[lvl] = {sampler_, f.levels[lvl].displacement_view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                norm_info[lvl] = {sampler_, f.levels[lvl].normal_view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            } else {
                disp_info[lvl] = {sampler_, dummy_displacement_view_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                norm_info[lvl] = {sampler_, dummy_normal_view_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }

        VkWriteDescriptorSet writes[kBindingCount]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = f.descriptor;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo     = &ubo;

        for (std::uint32_t lvl = 0; lvl < kMaxCascades; ++lvl) {
            const std::uint32_t disp_binding = 1 + lvl * 2;
            const std::uint32_t norm_binding = 2 + lvl * 2;

            writes[disp_binding].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[disp_binding].dstSet          = f.descriptor;
            writes[disp_binding].dstBinding      = disp_binding;
            writes[disp_binding].descriptorCount = 1;
            writes[disp_binding].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[disp_binding].pImageInfo      = &disp_info[lvl];

            writes[norm_binding].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[norm_binding].dstSet          = f.descriptor;
            writes[norm_binding].dstBinding      = norm_binding;
            writes[norm_binding].descriptorCount = 1;
            writes[norm_binding].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[norm_binding].pImageInfo      = &norm_info[lvl];
        }

        const VkDescriptorImageInfo inter_info{
            clamp_sampler_, f.interaction_view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[kInteractionBinding].sType  = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[kInteractionBinding].dstSet = f.descriptor;
        writes[kInteractionBinding].dstBinding      = kInteractionBinding;
        writes[kInteractionBinding].descriptorCount = 1;
        writes[kInteractionBinding].descriptorType  =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[kInteractionBinding].pImageInfo = &inter_info;

        vkUpdateDescriptorSets(ctx.device, kBindingCount, writes, 0, nullptr);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

bool OceanView::create_pipelines(VkContext& ctx)
{
    // The prop pipeline's push constants (object.vert/object.frag's `Push`
    // block: model, scale, color, surf). Declaring the range on the shared
    // layout costs the ocean/sky pipelines nothing - they never push into it.
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_range.offset     = 0;
    push_range.size       = static_cast<std::uint32_t>(kPropPushSize);

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &set_layout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &push_range;
    vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipeline_layout_);

    // The post pass takes its parameters as a push constant rather than
    // through the Globals block, so its layout needs neither that buffer nor
    // any of the cascade textures - just the one image it reads.
    VkPushConstantRange post_push{};
    post_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    post_push.offset     = 0;
    post_push.size       = sizeof(float) * 4;

    VkPipelineLayoutCreateInfo post_plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    post_plci.setLayoutCount         = 1;
    post_plci.pSetLayouts            = &post_set_layout_;
    post_plci.pushConstantRangeCount = 1;
    post_plci.pPushConstantRanges    = &post_push;
    vkCreatePipelineLayout(ctx.device, &post_plci, nullptr, &post_pipeline_layout_);

    VkPipelineLayoutCreateInfo blur_plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    blur_plci.setLayoutCount         = 1;
    blur_plci.pSetLayouts            = &blur_set_layout_;
    blur_plci.pushConstantRangeCount = 1;
    blur_plci.pPushConstantRanges    = &post_push;
    vkCreatePipelineLayout(ctx.device, &blur_plci, nullptr, &blur_pipeline_layout_);

    VkShaderModule ocean_vs  = make_module(ctx.device, kOceanVertSpv, sizeof(kOceanVertSpv));
    VkShaderModule ocean_fs  = make_module(ctx.device, kOceanFragSpv, sizeof(kOceanFragSpv));
    VkShaderModule sky_vs    = make_module(ctx.device, kSkyVertSpv, sizeof(kSkyVertSpv));
    VkShaderModule sky_fs    = make_module(ctx.device, kSkyFragSpv, sizeof(kSkyFragSpv));
    VkShaderModule object_vs = make_module(ctx.device, kObjectVertSpv, sizeof(kObjectVertSpv));
    VkShaderModule object_fs = make_module(ctx.device, kObjectFragSpv, sizeof(kObjectFragSpv));
    VkShaderModule post_vs   = make_module(ctx.device, kPostVertSpv, sizeof(kPostVertSpv));
    VkShaderModule post_fs   = make_module(ctx.device, kPostFragSpv, sizeof(kPostFragSpv));
    VkShaderModule bdown_fs  = make_module(ctx.device, kBloomDownFragSpv, sizeof(kBloomDownFragSpv));
    VkShaderModule bup_fs    = make_module(ctx.device, kBloomUpFragSpv, sizeof(kBloomUpFragSpv));

    // Dynamic rendering: the pipeline is told its attachment formats directly,
    // with no VkRenderPass and no VkFramebuffer object anywhere in this file.
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount    = 1;
    // The SCENE pipelines write the HDR target, not the swapchain. Only the
    // post pipeline below is built against the swapchain format.
    rendering.pColorAttachmentFormats = &ctx.hdr_format;
    rendering.depthAttachmentFormat   = ctx.depth_format;

    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount  = 1;

    // Viewport and scissor are dynamic so a window resize does not require
    // rebuilding every pipeline.
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                             VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dynamic_states;

    // The SCENE pipelines rasterise at the device's chosen sample count. The
    // blur and post pipelines below are reset to 1, because they run after the
    // resolve and never see more than one sample per pixel.
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = ctx.msaa_samples;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_attachment;

    auto build = [&](VkShaderModule vs, VkShaderModule fs,
                     const VkVertexInputBindingDescription* bindings,
                     std::uint32_t binding_count,
                     const VkVertexInputAttributeDescription* attributes,
                     std::uint32_t attribute_count, bool depth_test,
                     VkPolygonMode polygon_mode, VkPipelineLayout layout,
                     VkPipeline& out) {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName  = "main";

        VkPipelineVertexInputStateCreateInfo vertex_input{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount   = binding_count;
        vertex_input.pVertexBindingDescriptions      = bindings;
        vertex_input.vertexAttributeDescriptionCount = attribute_count;
        vertex_input.pVertexAttributeDescriptions    = attributes;

        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = polygon_mode;
        raster.lineWidth   = 1.0f;
        // No back-face culling on the water. A choppy surface folds over
        // itself at breaking crests, so culled triangles would punch visible
        // holes exactly where the most interesting geometry is.
        raster.cullMode  = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable  = depth_test ? VK_TRUE : VK_FALSE;
        depth.depthWriteEnable = depth_test ? VK_TRUE : VK_FALSE;
        depth.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkGraphicsPipelineCreateInfo ci{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.pNext               = &rendering;
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vertex_input;
        ci.pInputAssemblyState = &assembly;
        ci.pViewportState      = &viewport;
        ci.pRasterizationState = &raster;
        ci.pMultisampleState   = &multisample;
        ci.pDepthStencilState  = &depth;
        ci.pColorBlendState    = &blend;
        ci.pDynamicState       = &dynamic;
        ci.layout              = layout;

        return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci,
                                         nullptr, &out) == VK_SUCCESS;
    };

    const VkVertexInputBindingDescription grid_binding{
        0, sizeof(float) * 2, VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription grid_attribute{
        0, 0, VK_FORMAT_R32G32_SFLOAT, 0};

    const VkVertexInputBindingDescription prop_binding{
        0, sizeof(PropVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription prop_attributes[2] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
         static_cast<std::uint32_t>(offsetof(PropVertex, px))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
         static_cast<std::uint32_t>(offsetof(PropVertex, nx))},
    };

    bool ok = true;
    // The sky writes no depth, so the ocean always draws over it regardless of
    // the order the depth buffer would otherwise imply.
    ok &= build(sky_vs, sky_fs, nullptr, 0, nullptr, 0, false,
               VK_POLYGON_MODE_FILL, pipeline_layout_, sky_pipeline_);
    ok &= build(ocean_vs, ocean_fs, &grid_binding, 1, &grid_attribute, 1, true,
               VK_POLYGON_MODE_FILL, pipeline_layout_, ocean_pipeline_);
    ok &= build(ocean_vs, ocean_fs, &grid_binding, 1, &grid_attribute, 1, true,
               VK_POLYGON_MODE_LINE, pipeline_layout_, ocean_wire_pipeline_);
    // Depth-tested like the ocean, but never wireframed - see ocean.frag's
    // note that props are shaded double-sided, which the shared cullMode =
    // NONE inside `build` already gives them for free.
    ok &= build(object_vs, object_fs, &prop_binding, 1, prop_attributes, 2,
               true, VK_POLYGON_MODE_FILL, pipeline_layout_, prop_pipeline_);

    // The blur passes write bloom levels, which carry the HDR format, and like
    // the post pass have no depth attachment at all - nor any multisampling,
    // since they consume the already-resolved image.
    rendering.depthAttachmentFormat  = VK_FORMAT_UNDEFINED;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ok &= build(post_vs, bdown_fs, nullptr, 0, nullptr, 0, false,
               VK_POLYGON_MODE_FILL, blur_pipeline_layout_, bloom_down_pipeline_);

    // The upsample ADDS to the level it writes, and the blend state is what
    // does the adding - the shader emits only its own contribution. That is
    // what makes the chain progressive: each level ends up holding its own
    // gathered light plus everything coarser, so level 0 alone carries the
    // whole multi-scale glow and the composite needs one texture read.
    blend_attachment.blendEnable         = VK_TRUE;
    blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
    blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    ok &= build(post_vs, bup_fs, nullptr, 0, nullptr, 0, false,
               VK_POLYGON_MODE_FILL, blur_pipeline_layout_, bloom_up_pipeline_);
    blend_attachment.blendEnable = VK_FALSE;

    // The post pipeline is the one that writes the swapchain.
    rendering.pColorAttachmentFormats = &ctx.swapchain_format;
    ok &= build(post_vs, post_fs, nullptr, 0, nullptr, 0, false,
               VK_POLYGON_MODE_FILL, post_pipeline_layout_, post_pipeline_);

    vkDestroyShaderModule(ctx.device, post_vs, nullptr);
    vkDestroyShaderModule(ctx.device, post_fs, nullptr);
    vkDestroyShaderModule(ctx.device, bdown_fs, nullptr);
    vkDestroyShaderModule(ctx.device, bup_fs, nullptr);
    vkDestroyShaderModule(ctx.device, ocean_vs, nullptr);
    vkDestroyShaderModule(ctx.device, ocean_fs, nullptr);
    vkDestroyShaderModule(ctx.device, sky_vs, nullptr);
    vkDestroyShaderModule(ctx.device, sky_fs, nullptr);
    vkDestroyShaderModule(ctx.device, object_vs, nullptr);
    vkDestroyShaderModule(ctx.device, object_fs, nullptr);

    if (!ok) std::fprintf(stderr, "[vulkan] pipeline creation failed\n");
    return ok;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------

void OceanView::record(VkContext& ctx, VkCommandBuffer cmd,
                       std::uint32_t image_index, std::uint32_t frame,
                       const ocean::CascadeStack& stack,
                       const ocean::InteractionField& field,
                       const std::vector<const ocean::FoamField*>& foam,
                       const Globals& globals,
                       const std::vector<PropInstance>& props)
{
    FrameResources& f = frames_[frame];

    // THE ENTIRE INTEGRATION WITH THE LIBRARY IS THESE MEMCPYS, ONE PAIR PER
    // CASCADE LEVEL.
    //
    // No repacking, no per-component conversion, no interleave pass, for any
    // level: each cascade's output buffers are already laid out exactly as
    // RGBA32F textures (ADR-004), regardless of that level's own resolution.
    for (std::size_t i = 0; i < level_count_; ++i) {
        const ocean::Buffers b = stack.buffers(i);
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(b.size) * b.size *
                                   4 * sizeof(float);
        auto* dst = static_cast<std::uint8_t*>(f.staging_mapped) + f.staging_offset[i];
        std::memcpy(dst, b.displacement, static_cast<std::size_t>(bytes));
        std::memcpy(dst + bytes, b.normal, static_cast<std::size_t>(bytes));

        // Overwrite the displacement texture's ALPHA channel with the
        // persistent foam.
        //
        // displacement.w already carries the FFT's instantaneous Jacobian
        // foam, and every shader that reads foam reads it from there. Swapping
        // the value in the staging copy therefore upgrades the whole pipeline
        // to advected, persistent foam without a new texture, a new binding or
        // a single line of shader change - and it cannot desynchronise, since
        // both come from the same cascade level at the same instant.
        //
        // The library's own buffer is untouched; this writes into the staging
        // copy the GPU is about to receive.
        if (i < foam.size() && foam[i] != nullptr) {
            const float* src = foam[i]->data();
            auto* texels = reinterpret_cast<float*>(dst);
            const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;
            for (std::size_t c = 0; c < cells; ++c) texels[4 * c + 3] = src[c];
        }
    }

    // One more memcpy for the interaction field - same story as the cascades,
    // because InteractionBuffers uses the same RGBA32F-shaped layout.
    {
        const ocean::InteractionBuffers ib = field.buffers();
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(ib.size) * ib.size *
                                   4 * sizeof(float);
        std::memcpy(static_cast<std::uint8_t*>(f.staging_mapped) +
                        f.interaction_offset,
                    ib.field, static_cast<std::size_t>(bytes));
    }

    std::memcpy(f.uniform_mapped, &globals, sizeof(Globals));

    for (std::size_t i = 0; i < level_count_; ++i) {
        LevelTextures& lvl = f.levels[i];
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(level_sizes_[i]) *
                                   level_sizes_[i] * 4 * sizeof(float);

        // Transition both textures for the upload. The source layout is
        // UNDEFINED on the very first use (nothing to preserve) and
        // SHADER_READ afterwards.
        const VkImageLayout old_layout = lvl.initialised
                                             ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                             : VK_IMAGE_LAYOUT_UNDEFINED;
        const VkPipelineStageFlags2 src_stage =
            lvl.initialised ? VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                           : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        const VkAccessFlags2 src_access =
            lvl.initialised ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;

        for (VkImage image : {lvl.displacement, lvl.normal}) {
            transition_image(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, old_layout,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, src_stage,
                             src_access, VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent      = {level_sizes_[i], level_sizes_[i], 1};

        copy.bufferOffset = f.staging_offset[i];
        vkCmdCopyBufferToImage(cmd, f.staging, lvl.displacement,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        copy.bufferOffset = f.staging_offset[i] + bytes;
        vkCmdCopyBufferToImage(cmd, f.staging, lvl.normal,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        for (VkImage image : {lvl.displacement, lvl.normal}) {
            transition_image(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
        lvl.initialised = true;
    }

    {
        const VkImageLayout old_layout =
            f.interaction_initialised ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                      : VK_IMAGE_LAYOUT_UNDEFINED;
        const VkPipelineStageFlags2 src_stage =
            f.interaction_initialised ? VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                      : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        const VkAccessFlags2 src_access =
            f.interaction_initialised ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;

        transition_image(cmd, f.interaction, VK_IMAGE_ASPECT_COLOR_BIT,
                         old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         src_stage, src_access, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent      = {interaction_size_, interaction_size_, 1};
        copy.bufferOffset     = f.interaction_offset;
        vkCmdCopyBufferToImage(cmd, f.staging, f.interaction,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        transition_image(cmd, f.interaction, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        f.interaction_initialised = true;
    }

    // Everything above is staging-buffer to image copies: the per-frame cost of
    // handing the library's output to the GPU. Worth timing on its own, since
    // it scales with cascade resolution rather than with anything on screen.
    ctx.gpu_mark(cmd, "upload");

    // --- scene pass, into the HDR target ----------------------------------
    //
    // UNDEFINED as the old layout every frame: the whole attachment is cleared
    // below, so there is nothing in it worth preserving and telling the driver
    // so is free.
    const bool multisampled = (ctx.msaa_samples != VK_SAMPLE_COUNT_1_BIT);

    transition_image(cmd, ctx.hdr_image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    if (multisampled) {
        transition_image(cmd, ctx.hdr_ms_image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    // With MSAA the scene is drawn into the multisampled image and the resolve
    // writes the single-sample one; without it, straight into the latter.
    color.imageView   = multisampled ? ctx.hdr_ms_view : ctx.hdr_view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (multisampled) {
        color.resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
        color.resolveImageView   = ctx.hdr_view;
        color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // DONT_CARE for the multisampled image itself: the resolve is what anyone
    // downstream reads, so storing the per-sample data would be writing a
    // buffer nothing opens - and on a tiler it is what keeps the samples in
    // tile memory rather than spilling them to RAM.
    color.storeOp     = multisampled ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                     : VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView   = ctx.depth_view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil = {1.0f, 0};

    transition_image(cmd, ctx.depth_image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea           = {{0, 0}, ctx.extent};
    rendering.layerCount           = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments    = &color;
    rendering.pDepthAttachment     = &depth;

    vkCmdBeginRendering(cmd, &rendering);

    VkViewport vp{0.0f, 0.0f, static_cast<float>(ctx.extent.width),
                  static_cast<float>(ctx.extent.height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, ctx.extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &f.descriptor, 0, nullptr);

    // Sky first: three vertices, no buffers, generated from gl_VertexIndex.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    ctx.gpu_mark(cmd, "sky");

    // Ocean: one instance per tile of cascade 0's (the largest scale's) patch.
    // Cascades 1 and 2 are sampled through their own wrapped UVs inside that
    // same instanced mesh - see ocean.vert.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      wireframe_ ? ocean_wire_pipeline_ : ocean_pipeline_);

    // Every ring is placed independently against the camera, each snapped to
    // its OWN cell size - the finest snap available at that scale, which is
    // what stops the near water swimming as the camera moves. The cost is that
    // neighbouring rings are then generally NOT aligned with each other, and
    // the stitch bands below are what close the resulting gap.
    std::array<RingPlacement, kMaxRings> placements{};
    ring_layout_.place(globals.cam_pos[0], globals.cam_pos[2], placements);

    const VkDeviceSize clip_offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &clip_vertex_buffer_, &clip_offset);

    auto push_ring = [&](std::uint32_t r) {
        struct RingPush {
            float offset_x, offset_z, cell_size, morph_start;
        } push{placements[r].world_x, placements[r].world_z,
               placements[r].cell_size,
               // The outermost ring has nothing coarser to hand off to, so it
               // must not morph - morphing toward a ring that is not there
               // would pull its outer edge inward and expose the horizon.
               (r + 1 < ring_layout_.ring_count) ? 0.6f : 1.0f};
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(push), &push);
    };

    // Ring 0 is solid; it is the only one with no hole, because there is no
    // finer ring to nest inside it.
    push_ring(0);
    vkCmdBindIndexBuffer(cmd, clip_solid_indices_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, clip_solid_count_, 1, 0, 0, 0);

    for (std::uint32_t r = 1; r < ring_layout_.ring_count; ++r) {
        push_ring(r);
        vkCmdBindVertexBuffers(cmd, 0, 1, &clip_vertex_buffer_, &clip_offset);
        vkCmdBindIndexBuffer(cmd, clip_annulus_indices_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, clip_annulus_count_, 1, 0, 0, 0);

        // The stitch is built in THIS ring's local space against the previous
        // ring's actual outer edge, so it uses the same push constants the
        // annulus above already bound.
        if (f.stitch_vertex_mapped != nullptr && stitch_index_count_ > 0) {
            stitch_scratch_.update(placements[r], placements[r - 1]);
            const VkDeviceSize region = stitch_vertex_bytes_ * (r - 1);
            std::memcpy(static_cast<std::uint8_t*>(f.stitch_vertex_mapped) + region,
                        stitch_scratch_.vertices.data(),
                        static_cast<std::size_t>(stitch_vertex_bytes_));
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.stitch_vertex, &region);
            vkCmdBindIndexBuffer(cmd, stitch_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, stitch_index_count_, 1, 0, 0, 0);
        }
    }
    ctx.gpu_mark(cmd, "ocean");

    // Props: the boat and any rocks, one push-constant update and one indexed
    // draw per instance. Counts here are always small (a boat's three boxes
    // plus a handful of rocks in flight), so there is no batching to be won -
    // the push constant IS the per-instance data.
    if (!props.empty()) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, prop_pipeline_);
        const VkDeviceSize prop_offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &prop_vertex_buffer_, &prop_offset);
        vkCmdBindIndexBuffer(cmd, prop_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        for (const PropInstance& p : props) {
            const PropMeshRange& range =
                prop_ranges_[static_cast<std::size_t>(p.mesh)];
            vkCmdPushConstants(cmd, pipeline_layout_,
                              VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                              0, static_cast<std::uint32_t>(kPropPushSize), &p);
            vkCmdDrawIndexed(cmd, range.index_count, 1, range.first_index,
                             range.vertex_base, 0);
        }
    }
    ctx.gpu_mark(cmd, "props");

    vkCmdEndRendering(cmd);

    // --- post pass, HDR target to swapchain -------------------------------
    //
    // The scene's writes have to be visible to the fragment shader that is
    // about to sample them, which is what this barrier is for as much as the
    // layout change: without the COLOR_ATTACHMENT_WRITE -> SHADER_SAMPLED_READ
    // dependency the post pass could read an attachment the colour pipe has
    // not finished flushing.
    transition_image(cmd, ctx.hdr_image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // Written every frame rather than once at startup: these views are
    // recreated with the swapchain, so a descriptor written once would dangle
    // after the first resize. Safe to update here because begin_frame has
    // already waited on this frame slot's fence, so nothing is reading them.
    const VkDescriptorImageInfo scene_info{
        post_sampler_, ctx.hdr_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo bloom0_info{
        post_sampler_, ctx.bloom[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkWriteDescriptorSet post_writes[2]{};
    post_writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    post_writes[0].dstSet          = f.post_descriptor;
    post_writes[0].dstBinding      = 0;
    post_writes[0].descriptorCount = 1;
    post_writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    post_writes[0].pImageInfo      = &scene_info;
    post_writes[1]                 = post_writes[0];
    post_writes[1].dstBinding      = 1;
    post_writes[1].pImageInfo      = &bloom0_info;
    vkUpdateDescriptorSets(ctx.device, 2, post_writes, 0, nullptr);

    // One set per blur source: [0] the scene, [i + 1] bloom level i.
    VkDescriptorImageInfo blur_info[kBloomLevels + 1]{};
    VkWriteDescriptorSet  blur_writes[kBloomLevels + 1]{};
    for (std::uint32_t i = 0; i < kBloomLevels + 1; ++i) {
        blur_info[i] = {post_sampler_,
                        (i == 0) ? ctx.hdr_view : ctx.bloom[i - 1].view,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        blur_writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        blur_writes[i].dstSet          = f.blur_src[i];
        blur_writes[i].dstBinding      = 0;
        blur_writes[i].descriptorCount = 1;
        blur_writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        blur_writes[i].pImageInfo      = &blur_info[i];
    }
    vkUpdateDescriptorSets(ctx.device, kBloomLevels + 1, blur_writes, 0, nullptr);

    // --- bloom chain -------------------------------------------------------
    //
    // Down first, gathering light into successively smaller images, then back
    // up, adding each level into the one above it. By the time this finishes,
    // level 0 holds every scale summed, so the composite reads one texture.
    //
    // At intensity 0 the whole thing is skipped rather than run and multiplied
    // by nothing - it is ten render passes, and the point of a quality tier is
    // that turning a feature off stops paying for it.
    if (bloom_intensity_ > 0.0f) {
        auto blur_pass = [&](VkPipeline pipeline, VkImageView dst,
                             VkExtent2D dst_extent, VkExtent2D src_extent,
                             VkDescriptorSet src_set, bool load, bool first,
                             float radius) {
            VkRenderingAttachmentInfo att{
                VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView   = dst;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            // LOAD on the way up, because the blend adds to what is there;
            // DONT_CARE on the way down, because every pixel is overwritten.
            att.loadOp      = load ? VK_ATTACHMENT_LOAD_OP_LOAD
                                   : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea           = {{0, 0}, dst_extent};
            ri.layerCount           = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments    = &att;

            vkCmdBeginRendering(cmd, &ri);
            const VkViewport bvp{0.0f, 0.0f,
                                 static_cast<float>(dst_extent.width),
                                 static_cast<float>(dst_extent.height),
                                 0.0f, 1.0f};
            const VkRect2D bsc{{0, 0}, dst_extent};
            vkCmdSetViewport(cmd, 0, 1, &bvp);
            vkCmdSetScissor(cmd, 0, 1, &bsc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    blur_pipeline_layout_, 0, 1, &src_set, 0,
                                    nullptr);
            // The filter works in SOURCE texels, so the offsets it needs are
            // the reciprocal of the source resolution, not the destination's.
            const float params[4] = {1.0f / static_cast<float>(src_extent.width),
                                     1.0f / static_cast<float>(src_extent.height),
                                     first ? 1.0f : 0.0f, radius};
            vkCmdPushConstants(cmd, blur_pipeline_layout_,
                               VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(params),
                               params);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        };

        for (std::uint32_t i = 0; i < kBloomLevels; ++i) {
            // UNDEFINED: this level is fully overwritten below, so last
            // frame's contents are not worth a preserving transition.
            transition_image(cmd, ctx.bloom[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

            blur_pass(bloom_down_pipeline_, ctx.bloom[i].view,
                      ctx.bloom[i].extent,
                      (i == 0) ? ctx.extent : ctx.bloom[i - 1].extent,
                      f.blur_src[i], false, i == 0, 1.0f);

            transition_image(cmd, ctx.bloom[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        for (std::int32_t i = static_cast<std::int32_t>(kBloomLevels) - 2;
             i >= 0; --i) {
            const auto lvl = static_cast<std::uint32_t>(i);
            // SHADER_READ this time, not UNDEFINED: the blend adds into what
            // the downsample already put here, so it must be preserved.
            transition_image(cmd, ctx.bloom[lvl].image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);

            blur_pass(bloom_up_pipeline_, ctx.bloom[lvl].view,
                      ctx.bloom[lvl].extent, ctx.bloom[lvl + 1].extent,
                      f.blur_src[lvl + 2], true, false, 1.5f);

            transition_image(cmd, ctx.bloom[lvl].image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    } else {
        // Bloom off: the chain never ran, so level 0 is still UNDEFINED while
        // the post pass's descriptor declares it SHADER_READ_ONLY. The shader
        // skips the fetch, but the layout has to be legal regardless - and an
        // undefined image can read back as Inf or NaN, which mix() would
        // propagate even at a weight of zero.
        transition_image(cmd, ctx.bloom[0].image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    ctx.gpu_mark(cmd, "bloom");

    transition_image(cmd, ctx.images[image_index], VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo present{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    present.imageView   = ctx.image_views[image_index];
    present.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // DONT_CARE, not CLEAR: the fullscreen triangle covers every pixel, so
    // clearing first would be writing the whole attachment twice.
    present.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    present.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo post_pass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    post_pass.renderArea           = {{0, 0}, ctx.extent};
    post_pass.layerCount           = 1;
    post_pass.colorAttachmentCount = 1;
    post_pass.pColorAttachments    = &present;

    vkCmdBeginRendering(cmd, &post_pass);
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, post_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            post_pipeline_layout_, 0, 1, &f.post_descriptor,
                            0, nullptr);

    // Exposure travels with the pass that uses it. It was previously read from
    // the Globals block by every material shader, which meant three shaders
    // had to agree about it; now exactly one does.
    // x exposure, y tone curve, z bloom intensity (unused yet), w reserved.
    const float post_params[4] = {globals.shading[1], globals.water[1],
                                  bloom_intensity_, 0.0f};
    vkCmdPushConstants(cmd, post_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(post_params), post_params);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    ctx.gpu_mark(cmd, "post");

    transition_image(cmd, ctx.images[image_index], VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
}

void OceanView::shutdown(VkContext& ctx)
{
    if (ctx.device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(ctx.device);

    for (auto& f : frames_) {
        if (f.staging_mapped) vkUnmapMemory(ctx.device, f.staging_memory);
        if (f.uniform_mapped) vkUnmapMemory(ctx.device, f.uniform_memory);
        vkDestroyBuffer(ctx.device, f.staging, nullptr);
        vkFreeMemory(ctx.device, f.staging_memory, nullptr);
        vkDestroyBuffer(ctx.device, f.uniform, nullptr);
        vkFreeMemory(ctx.device, f.uniform_memory, nullptr);
        for (auto& lvl : f.levels) {
            if (lvl.displacement == VK_NULL_HANDLE) continue;
            vkDestroyImageView(ctx.device, lvl.displacement_view, nullptr);
            vkDestroyImage(ctx.device, lvl.displacement, nullptr);
            vkFreeMemory(ctx.device, lvl.displacement_memory, nullptr);
            vkDestroyImageView(ctx.device, lvl.normal_view, nullptr);
            vkDestroyImage(ctx.device, lvl.normal, nullptr);
            vkFreeMemory(ctx.device, lvl.normal_memory, nullptr);
        }
        if (f.interaction != VK_NULL_HANDLE) {
            vkDestroyImageView(ctx.device, f.interaction_view, nullptr);
            vkDestroyImage(ctx.device, f.interaction, nullptr);
            vkFreeMemory(ctx.device, f.interaction_memory, nullptr);
        }
    }

    vkDestroyImageView(ctx.device, dummy_displacement_view_, nullptr);
    vkDestroyImage(ctx.device, dummy_displacement_, nullptr);
    vkFreeMemory(ctx.device, dummy_displacement_memory_, nullptr);
    vkDestroyImageView(ctx.device, dummy_normal_view_, nullptr);
    vkDestroyImage(ctx.device, dummy_normal_, nullptr);
    vkFreeMemory(ctx.device, dummy_normal_memory_, nullptr);

    vkDestroySampler(ctx.device, sampler_, nullptr);
    vkDestroySampler(ctx.device, clamp_sampler_, nullptr);
    vkDestroyPipeline(ctx.device, ocean_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, ocean_wire_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, sky_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, prop_pipeline_, nullptr);
    if (post_pipeline_) vkDestroyPipeline(ctx.device, post_pipeline_, nullptr);
    if (post_pipeline_layout_)
        vkDestroyPipelineLayout(ctx.device, post_pipeline_layout_, nullptr);
    if (post_set_layout_)
        vkDestroyDescriptorSetLayout(ctx.device, post_set_layout_, nullptr);
    if (bloom_down_pipeline_) vkDestroyPipeline(ctx.device, bloom_down_pipeline_, nullptr);
    if (bloom_up_pipeline_) vkDestroyPipeline(ctx.device, bloom_up_pipeline_, nullptr);
    if (blur_pipeline_layout_)
        vkDestroyPipelineLayout(ctx.device, blur_pipeline_layout_, nullptr);
    if (blur_set_layout_)
        vkDestroyDescriptorSetLayout(ctx.device, blur_set_layout_, nullptr);
    if (post_sampler_) vkDestroySampler(ctx.device, post_sampler_, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout_, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, set_layout_, nullptr);
    for (VkBuffer b : {clip_vertex_buffer_, clip_solid_indices_,
                       clip_annulus_indices_, stitch_index_buffer_}) {
        if (b) vkDestroyBuffer(ctx.device, b, nullptr);
    }
    for (VkDeviceMemory m : {clip_vertex_memory_, clip_solid_memory_,
                             clip_annulus_memory_, stitch_index_memory_}) {
        if (m) vkFreeMemory(ctx.device, m, nullptr);
    }
    for (FrameResources& fr : frames_) {
        if (fr.stitch_vertex) {
            vkDestroyBuffer(ctx.device, fr.stitch_vertex, nullptr);
            vkFreeMemory(ctx.device, fr.stitch_vertex_memory, nullptr);
        }
    }
    vkDestroyBuffer(ctx.device, prop_vertex_buffer_, nullptr);
    vkFreeMemory(ctx.device, prop_vertex_memory_, nullptr);
    vkDestroyBuffer(ctx.device, prop_index_buffer_, nullptr);
    vkFreeMemory(ctx.device, prop_index_memory_, nullptr);
}

}  // namespace viewer
