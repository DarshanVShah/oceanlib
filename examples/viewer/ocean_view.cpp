#include "ocean_view.hpp"

#include "ocean_frag.h"
#include "ocean_vert.h"
#include "sky_frag.h"
#include "sky_vert.h"

#include <algorithm>
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

}  // namespace

bool OceanView::init(VkContext& ctx, const std::vector<ocean::OceanDesc>& levels,
                     const RingLayout& layout)
{
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
    ring_layout_ = layout;
    ring_layout_.ring_count =
        std::clamp<std::uint32_t>(ring_layout_.ring_count, 1, kMaxRings);

    if (!create_mesh(ctx)) return false;
    if (!create_textures(ctx)) return false;
    if (!create_descriptors(ctx)) return false;
    return create_pipelines(ctx);
}

// ---------------------------------------------------------------------------
// Clipmap mesh
// ---------------------------------------------------------------------------

namespace {

// Stages `data` through a temporary host-visible buffer into a new
// device-local one. Startup-only (mirrors the single upload lambda the old
// single-tile mesh used).
bool upload_device_local(VkContext& ctx, const void* data, VkDeviceSize size,
                         VkBufferUsageFlags usage, VkBuffer& buf,
                         VkDeviceMemory& mem)
{
    if (size == 0) return true;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    if (!ctx.create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           staging, staging_mem)) {
        return false;
    }
    void* mapped = nullptr;
    vkMapMemory(ctx.device, staging_mem, 0, size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(ctx.device, staging_mem);

    const bool ok = ctx.create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);
    if (ok) {
        VkCommandBuffer cmd = ctx.begin_one_shot();
        VkBufferCopy copy{0, 0, size};
        vkCmdCopyBuffer(cmd, staging, buf, 1, &copy);
        ctx.end_one_shot(cmd);
    }

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, staging_mem, nullptr);
    return ok;
}

}  // namespace

bool OceanView::create_mesh(VkContext& ctx)
{
    // Vertex/index generation is pure host-side math, verified independently
    // of Vulkan (clipmap.hpp/.cpp) - this function only uploads the result.
    const ClipmapMesh mesh = ClipmapMesh::build();

    if (!upload_device_local(ctx, mesh.vertices.data(),
                             mesh.vertices.size() * sizeof(float),
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             clip_vertex_buffer_, clip_vertex_memory_)) {
        return false;
    }
    if (!upload_device_local(ctx, mesh.solid_indices.data(),
                             mesh.solid_indices.size() * sizeof(std::uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             clip_solid_indices_, clip_solid_memory_)) {
        return false;
    }
    clip_solid_count_ = static_cast<std::uint32_t>(mesh.solid_indices.size());

    if (!upload_device_local(ctx, mesh.annulus_indices.data(),
                             mesh.annulus_indices.size() * sizeof(std::uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             clip_annulus_indices_, clip_annulus_memory_)) {
        return false;
    }
    clip_annulus_count_ = static_cast<std::uint32_t>(mesh.annulus_indices.size());

    // The stitch band's topology is static (ADR-021); only build() runs at
    // init. update() (called every frame in record()) only ever rewrites
    // stitch_scratch_.vertices, never touches .indices.
    stitch_scratch_ = StitchBand::build();
    if (!upload_device_local(ctx, stitch_scratch_.indices.data(),
                             stitch_scratch_.indices.size() * sizeof(std::uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             stitch_index_buffer_, stitch_index_memory_)) {
        return false;
    }
    stitch_index_count_ = static_cast<std::uint32_t>(stitch_scratch_.indices.size());

    // Triangle count for a frame with every ring present: one solid centre,
    // (ring_count - 1) annuli, (ring_count - 1) stitch bands (ring 0 has no
    // finer neighbour to stitch against).
    triangle_count_ =
        clip_solid_count_ / 3 +
        (ring_layout_.ring_count - 1) * (clip_annulus_count_ / 3) +
        (ring_layout_.ring_count - 1) * (stitch_index_count_ / 3);
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

    const VkDeviceSize stitch_bytes =
        static_cast<VkDeviceSize>(stitch_scratch_.vertices.size()) * sizeof(float);
    // One stitch region per OUTER ring (ring_count - 1 of them), each big
    // enough for one full StitchBand - see record() for why every ring's
    // stitch needs its own region rather than sharing one.
    const VkDeviceSize stitch_total_bytes =
        stitch_bytes * static_cast<VkDeviceSize>(ring_layout_.ring_count > 0
                                                      ? ring_layout_.ring_count - 1
                                                      : 0);

    for (auto& f : frames_) {
        VkDeviceSize total_bytes = 0;
        for (std::size_t i = 0; i < level_count_; ++i) {
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(level_sizes_[i]) *
                                       level_sizes_[i] * 4 * sizeof(float);
            f.staging_offset[i] = total_bytes;
            total_bytes += bytes;
            total_bytes += bytes;
        }

        for (std::size_t i = 0; i < level_count_; ++i) {
            make_image(level_sizes_[i], f.levels[i].displacement,
                      f.levels[i].displacement_memory, f.levels[i].displacement_view);
            make_image(level_sizes_[i], f.levels[i].normal,
                      f.levels[i].normal_memory, f.levels[i].normal_view);
        }

        ctx.create_buffer(total_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          f.staging, f.staging_memory);
        vkMapMemory(ctx.device, f.staging_memory, 0, total_bytes, 0,
                    &f.staging_mapped);

        ctx.create_buffer(sizeof(Globals), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          f.uniform, f.uniform_memory);
        vkMapMemory(ctx.device, f.uniform_memory, 0, sizeof(Globals), 0,
                    &f.uniform_mapped);

        if (stitch_total_bytes > 0) {
            ctx.create_buffer(stitch_total_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              f.stitch_vertex, f.stitch_vertex_memory);
            vkMapMemory(ctx.device, f.stitch_vertex_memory, 0, stitch_total_bytes,
                        0, &f.stitch_vertex_mapped);
        }
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
    constexpr std::uint32_t kBindingCount = 1 + 2 * static_cast<std::uint32_t>(kMaxCascades);
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
        bindings[disp_binding].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[norm_binding].binding         = norm_binding;
        bindings[norm_binding].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[norm_binding].descriptorCount = 1;
        bindings[norm_binding].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = kBindingCount;
    lci.pBindings    = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &set_layout_);

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = kFramesInFlight * 2 * static_cast<std::uint32_t>(kMaxCascades);

    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets       = kFramesInFlight;
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

        vkUpdateDescriptorSets(ctx.device, kBindingCount, writes, 0, nullptr);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

bool OceanView::create_pipelines(VkContext& ctx)
{
    // RingPush is pushed once per ring draw (offset, cell size, morph start)
    // - see ocean.vert's matching `RingPush` block.
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push_range.offset     = 0;
    push_range.size       = sizeof(RingPush);

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &set_layout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &push_range;
    vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipeline_layout_);

    VkShaderModule ocean_vs = make_module(ctx.device, kOceanVertSpv, sizeof(kOceanVertSpv));
    VkShaderModule ocean_fs = make_module(ctx.device, kOceanFragSpv, sizeof(kOceanFragSpv));
    VkShaderModule sky_vs   = make_module(ctx.device, kSkyVertSpv, sizeof(kSkyVertSpv));
    VkShaderModule sky_fs   = make_module(ctx.device, kSkyFragSpv, sizeof(kSkyFragSpv));

    // Dynamic rendering: the pipeline is told its attachment formats directly,
    // with no VkRenderPass and no VkFramebuffer object anywhere in this file.
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount    = 1;
    rendering.pColorAttachmentFormats = &ctx.swapchain_format;
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

    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_attachment;

    auto build = [&](VkShaderModule vs, VkShaderModule fs, bool has_vertex_input,
                     bool depth_test, VkPolygonMode polygon_mode,
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

        // Local (x, z), same 2-float layout the clipmap grid and the stitch
        // band both share - so no pipeline or binding change is needed
        // between drawing a ring's main mesh and drawing its stitch band.
        VkVertexInputBindingDescription binding{0, sizeof(float) * 2,
                                                VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attribute{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};

        VkPipelineVertexInputStateCreateInfo vertex_input{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        if (has_vertex_input) {
            vertex_input.vertexBindingDescriptionCount   = 1;
            vertex_input.pVertexBindingDescriptions      = &binding;
            vertex_input.vertexAttributeDescriptionCount = 1;
            vertex_input.pVertexAttributeDescriptions    = &attribute;
        }

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
        ci.layout              = pipeline_layout_;

        return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci,
                                         nullptr, &out) == VK_SUCCESS;
    };

    bool ok = true;
    // The sky writes no depth, so the ocean always draws over it regardless of
    // the order the depth buffer would otherwise imply.
    ok &= build(sky_vs, sky_fs, false, false, VK_POLYGON_MODE_FILL, sky_pipeline_);
    ok &= build(ocean_vs, ocean_fs, true, true, VK_POLYGON_MODE_FILL,
                ocean_pipeline_);
    ok &= build(ocean_vs, ocean_fs, true, true, VK_POLYGON_MODE_LINE,
                ocean_wire_pipeline_);

    vkDestroyShaderModule(ctx.device, ocean_vs, nullptr);
    vkDestroyShaderModule(ctx.device, ocean_fs, nullptr);
    vkDestroyShaderModule(ctx.device, sky_vs, nullptr);
    vkDestroyShaderModule(ctx.device, sky_fs, nullptr);

    if (!ok) std::fprintf(stderr, "[vulkan] pipeline creation failed\n");
    return ok;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------

void OceanView::record(VkContext& ctx, VkCommandBuffer cmd,
                       std::uint32_t image_index, std::uint32_t frame,
                       const ocean::CascadeStack& stack, const Globals& globals)
{
    FrameResources& f = frames_[frame];

    // THE ENTIRE INTEGRATION WITH THE LIBRARY IS THESE MEMCPYS, ONE PAIR PER
    // CASCADE LEVEL.
    for (std::size_t i = 0; i < level_count_; ++i) {
        const ocean::Buffers b = stack.buffers(i);
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(b.size) * b.size *
                                   4 * sizeof(float);
        auto* dst = static_cast<std::uint8_t*>(f.staging_mapped) + f.staging_offset[i];
        std::memcpy(dst, b.displacement, static_cast<std::size_t>(bytes));
        std::memcpy(dst + bytes, b.normal, static_cast<std::size_t>(bytes));
    }

    std::memcpy(f.uniform_mapped, &globals, sizeof(Globals));

    // --- clipmap ring placement -------------------------------------------
    //
    // Every ring is re-snapped to the camera's CURRENT position every frame;
    // this is cheap (a handful of rounds and multiplies) and is what keeps
    // each ring following at the finest precision its own scale allows
    // (ADR-021). Fills a fixed array - see RingLayout::place - so this does
    // not allocate.
    std::array<RingPlacement, kMaxRings> placements{};
    ring_layout_.place(globals.cam_pos[0], globals.cam_pos[2], placements);

    const VkDeviceSize stitch_bytes =
        static_cast<VkDeviceSize>(stitch_scratch_.vertices.size()) * sizeof(float);
    for (std::uint32_t r = 1; r < ring_layout_.ring_count; ++r) {
        // Regenerate this ring's stitch band against its next-finer
        // neighbour, then copy into ITS OWN region of the per-frame buffer -
        // every ring needs a distinct region, since all these memcpys happen
        // during recording, before the GPU executes any of the draws that
        // will read them (see clipmap.hpp's StitchBand comment).
        stitch_scratch_.update(placements[r], placements[r - 1]);
        auto* dst = static_cast<std::uint8_t*>(f.stitch_vertex_mapped) +
                    static_cast<VkDeviceSize>(r - 1) * stitch_bytes;
        std::memcpy(dst, stitch_scratch_.vertices.data(),
                   static_cast<std::size_t>(stitch_bytes));
    }

    for (std::size_t i = 0; i < level_count_; ++i) {
        LevelTextures& lvl = f.levels[i];
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(level_sizes_[i]) *
                                   level_sizes_[i] * 4 * sizeof(float);

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

    // --- render ----------------------------------------------------------
    transition_image(cmd, ctx.images[image_index], VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView   = ctx.image_views[image_index];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
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

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      wireframe_ ? ocean_wire_pipeline_ : ocean_pipeline_);

    // Ring 0: solid, no hole, no stitch (nothing finer to stitch against).
    {
        RingPush push{};
        push.offset_x    = placements[0].world_x;
        push.offset_z    = placements[0].world_z;
        push.cell_size   = placements[0].cell_size;
        push.morph_start = (ring_layout_.ring_count > 1) ? 0.6f : 1.0f;
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(push), &push);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &clip_vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(cmd, clip_solid_indices_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, clip_solid_count_, 1, 0, 0, 0);
    }

    // Rings 1..N-1: annulus (main clipmap vertex buffer) plus that ring's
    // freshly-updated stitch band (its own small per-frame buffer region).
    for (std::uint32_t r = 1; r < ring_layout_.ring_count; ++r) {
        RingPush push{};
        push.offset_x    = placements[r].world_x;
        push.offset_z    = placements[r].world_z;
        push.cell_size   = placements[r].cell_size;
        push.morph_start = (r + 1 < ring_layout_.ring_count) ? 0.6f : 1.0f;
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(push), &push);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &clip_vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(cmd, clip_annulus_indices_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, clip_annulus_count_, 1, 0, 0, 0);

        // The stitch band is drawn against ring (r-1)'s ACTUAL boundary, so
        // it must use ring r's OWN transform (it is built in ring r's local
        // space - see clipmap.hpp) - the same push constants as the annulus
        // above are still bound, no change needed.
        const VkDeviceSize stitch_offset =
            static_cast<VkDeviceSize>(r - 1) * stitch_bytes;
        vkCmdBindVertexBuffers(cmd, 0, 1, &f.stitch_vertex, &stitch_offset);
        vkCmdBindIndexBuffer(cmd, stitch_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, stitch_index_count_, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

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
        if (f.stitch_vertex_mapped) vkUnmapMemory(ctx.device, f.stitch_vertex_memory);
        vkDestroyBuffer(ctx.device, f.staging, nullptr);
        vkFreeMemory(ctx.device, f.staging_memory, nullptr);
        vkDestroyBuffer(ctx.device, f.uniform, nullptr);
        vkFreeMemory(ctx.device, f.uniform_memory, nullptr);
        if (f.stitch_vertex != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx.device, f.stitch_vertex, nullptr);
            vkFreeMemory(ctx.device, f.stitch_vertex_memory, nullptr);
        }
        for (auto& lvl : f.levels) {
            if (lvl.displacement == VK_NULL_HANDLE) continue;
            vkDestroyImageView(ctx.device, lvl.displacement_view, nullptr);
            vkDestroyImage(ctx.device, lvl.displacement, nullptr);
            vkFreeMemory(ctx.device, lvl.displacement_memory, nullptr);
            vkDestroyImageView(ctx.device, lvl.normal_view, nullptr);
            vkDestroyImage(ctx.device, lvl.normal, nullptr);
            vkFreeMemory(ctx.device, lvl.normal_memory, nullptr);
        }
    }

    vkDestroyImageView(ctx.device, dummy_displacement_view_, nullptr);
    vkDestroyImage(ctx.device, dummy_displacement_, nullptr);
    vkFreeMemory(ctx.device, dummy_displacement_memory_, nullptr);
    vkDestroyImageView(ctx.device, dummy_normal_view_, nullptr);
    vkDestroyImage(ctx.device, dummy_normal_, nullptr);
    vkFreeMemory(ctx.device, dummy_normal_memory_, nullptr);

    vkDestroySampler(ctx.device, sampler_, nullptr);
    vkDestroyPipeline(ctx.device, ocean_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, ocean_wire_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, sky_pipeline_, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout_, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, set_layout_, nullptr);
    vkDestroyBuffer(ctx.device, clip_vertex_buffer_, nullptr);
    vkFreeMemory(ctx.device, clip_vertex_memory_, nullptr);
    vkDestroyBuffer(ctx.device, clip_solid_indices_, nullptr);
    vkFreeMemory(ctx.device, clip_solid_memory_, nullptr);
    vkDestroyBuffer(ctx.device, clip_annulus_indices_, nullptr);
    vkFreeMemory(ctx.device, clip_annulus_memory_, nullptr);
    vkDestroyBuffer(ctx.device, stitch_index_buffer_, nullptr);
    vkFreeMemory(ctx.device, stitch_index_memory_, nullptr);
}

}  // namespace viewer
