#include "ocean_view.hpp"

#include "ocean_frag.h"
#include "ocean_vert.h"
#include "sky_frag.h"
#include "sky_vert.h"

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

bool OceanView::init(VkContext& ctx, std::uint32_t ocean_size,
                     std::uint32_t mesh_resolution, std::uint32_t tiles)
{
    ocean_size_      = ocean_size;
    mesh_resolution_ = mesh_resolution;
    tiles_           = tiles;

    if (!create_mesh(ctx)) return false;
    if (!create_textures(ctx)) return false;
    if (!create_descriptors(ctx)) return false;
    return create_pipelines(ctx);
}

// ---------------------------------------------------------------------------
// Grid mesh
// ---------------------------------------------------------------------------

bool OceanView::create_mesh(VkContext& ctx)
{
    // A flat unit grid in [0,1]^2. Two floats per vertex and nothing else -
    // position, normal and foam all come from the textures at draw time, so
    // there is no per-frame vertex traffic at all.
    const std::uint32_t verts_per_side = mesh_resolution_ + 1;
    std::vector<float> vertices;
    vertices.reserve(static_cast<std::size_t>(verts_per_side) * verts_per_side * 2);
    for (std::uint32_t z = 0; z < verts_per_side; ++z) {
        for (std::uint32_t x = 0; x < verts_per_side; ++x) {
            vertices.push_back(static_cast<float>(x) / mesh_resolution_);
            vertices.push_back(static_cast<float>(z) / mesh_resolution_);
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(static_cast<std::size_t>(mesh_resolution_) *
                    mesh_resolution_ * 6);
    for (std::uint32_t z = 0; z < mesh_resolution_; ++z) {
        for (std::uint32_t x = 0; x < mesh_resolution_; ++x) {
            const std::uint32_t i0 = z * verts_per_side + x;
            const std::uint32_t i1 = i0 + 1;
            const std::uint32_t i2 = i0 + verts_per_side;
            const std::uint32_t i3 = i2 + 1;
            // Counter-clockwise when viewed from +Y, matching the front face
            // set in the pipeline.
            indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
            indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
        }
    }
    index_count_ = static_cast<std::uint32_t>(indices.size());

    auto upload = [&](const void* data, VkDeviceSize size,
                      VkBufferUsageFlags usage, VkBuffer& buf,
                      VkDeviceMemory& mem) {
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

        // Device-local for the mesh: it never changes, so paying one staged
        // copy at startup buys the fastest possible reads forever after.
        ctx.create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);

        VkCommandBuffer cmd = ctx.begin_one_shot();
        VkBufferCopy copy{0, 0, size};
        vkCmdCopyBuffer(cmd, staging, buf, 1, &copy);
        ctx.end_one_shot(cmd);

        vkDestroyBuffer(ctx.device, staging, nullptr);
        vkFreeMemory(ctx.device, staging_mem, nullptr);
    };

    upload(vertices.data(), vertices.size() * sizeof(float),
           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer_, vertex_memory_);
    upload(indices.data(), indices.size() * sizeof(std::uint32_t),
           VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer_, index_memory_);
    return true;
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

bool OceanView::create_textures(VkContext& ctx)
{
    const VkDeviceSize texture_bytes =
        static_cast<VkDeviceSize>(ocean_size_) * ocean_size_ * 4 * sizeof(float);

    auto make_image = [&](VkImage& image, VkDeviceMemory& memory,
                          VkImageView& view) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        // R32G32B32A32_SFLOAT because that is literally what the library
        // hands us. No conversion, no quantisation, no repacking - which was
        // the entire point of choosing that output layout.
        ci.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent        = {ocean_size_, ocean_size_, 1};
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
        make_image(f.displacement, f.displacement_memory, f.displacement_view);
        make_image(f.normal, f.normal_memory, f.normal_view);

        // One staging buffer holding both textures back to back, persistently
        // mapped. Mapping and unmapping every frame would be pure overhead,
        // and HOST_COHERENT removes the need for explicit flushes.
        ctx.create_buffer(texture_bytes * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          f.staging, f.staging_memory);
        vkMapMemory(ctx.device, f.staging_memory, 0, texture_bytes * 2, 0,
                    &f.staging_mapped);

        ctx.create_buffer(sizeof(Globals), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          f.uniform, f.uniform_memory);
        vkMapMemory(ctx.device, f.uniform_memory, 0, sizeof(Globals), 0,
                    &f.uniform_mapped);
    }

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    // REPEAT is what makes the tiling free: the FFT surface is exactly
    // periodic, so a wrapped sample at u = 1.05 is genuinely the right value,
    // not an approximation. CLAMP would produce a visible seam at every tile.
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
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding         = 1;  // displacement
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    // Sampled in the VERTEX stage - this is the vertex texture fetch that
    // makes GPU-side displacement possible.
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[2].binding         = 2;  // normal + jacobian
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 3;
    lci.pBindings    = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &set_layout_);

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = kFramesInFlight;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = kFramesInFlight * 2;

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
        VkDescriptorImageInfo disp{sampler_, f.displacement_view,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo nrm{sampler_, f.normal_view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = f.descriptor;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo    = &ubo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo     = &disp;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo     = &nrm;

        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

bool OceanView::create_pipelines(VkContext& ctx)
{
    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &set_layout_;
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
                       const ocean::Buffers& buffers, const Globals& globals)
{
    FrameResources& f = frames_[frame];

    const VkDeviceSize texture_bytes =
        static_cast<VkDeviceSize>(ocean_size_) * ocean_size_ * 4 * sizeof(float);

    // THE ENTIRE INTEGRATION WITH THE LIBRARY IS THESE TWO MEMCPYS.
    //
    // No repacking, no per-component conversion, no interleave pass. The
    // library's output buffers are already laid out exactly as RGBA32F
    // textures, which is what ADR-004 bought.
    std::memcpy(f.staging_mapped, buffers.displacement,
                static_cast<std::size_t>(texture_bytes));
    std::memcpy(static_cast<std::uint8_t*>(f.staging_mapped) + texture_bytes,
                buffers.normal, static_cast<std::size_t>(texture_bytes));

    std::memcpy(f.uniform_mapped, &globals, sizeof(Globals));

    // Transition both textures for the upload. The source layout is UNDEFINED
    // on the very first use (nothing to preserve) and SHADER_READ afterwards.
    const VkImageLayout old_layout = f.textures_initialised
                                         ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                         : VK_IMAGE_LAYOUT_UNDEFINED;
    const VkPipelineStageFlags2 src_stage =
        f.textures_initialised ? VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                               : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    const VkAccessFlags2 src_access =
        f.textures_initialised ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;

    for (VkImage image : {f.displacement, f.normal}) {
        transition_image(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, old_layout,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, src_stage,
                         src_access, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {ocean_size_, ocean_size_, 1};

    copy.bufferOffset = 0;
    vkCmdCopyBufferToImage(cmd, f.staging, f.displacement,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    copy.bufferOffset = texture_bytes;
    vkCmdCopyBufferToImage(cmd, f.staging, f.normal,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    for (VkImage image : {f.displacement, f.normal}) {
        transition_image(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    f.textures_initialised = true;

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

    // Ocean: one instance per tile. The instance index becomes the tile offset
    // in the vertex shader, so the whole tiled field is a single draw call.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      wireframe_ ? ocean_wire_pipeline_ : ocean_pipeline_);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer_, &offset);
    vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, index_count_, tiles_ * tiles_, 0, 0, 0);

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
        vkDestroyBuffer(ctx.device, f.staging, nullptr);
        vkFreeMemory(ctx.device, f.staging_memory, nullptr);
        vkDestroyBuffer(ctx.device, f.uniform, nullptr);
        vkFreeMemory(ctx.device, f.uniform_memory, nullptr);
        vkDestroyImageView(ctx.device, f.displacement_view, nullptr);
        vkDestroyImage(ctx.device, f.displacement, nullptr);
        vkFreeMemory(ctx.device, f.displacement_memory, nullptr);
        vkDestroyImageView(ctx.device, f.normal_view, nullptr);
        vkDestroyImage(ctx.device, f.normal, nullptr);
        vkFreeMemory(ctx.device, f.normal_memory, nullptr);
    }

    vkDestroySampler(ctx.device, sampler_, nullptr);
    vkDestroyPipeline(ctx.device, ocean_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, ocean_wire_pipeline_, nullptr);
    vkDestroyPipeline(ctx.device, sky_pipeline_, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout_, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, set_layout_, nullptr);
    vkDestroyBuffer(ctx.device, vertex_buffer_, nullptr);
    vkFreeMemory(ctx.device, vertex_memory_, nullptr);
    vkDestroyBuffer(ctx.device, index_buffer_, nullptr);
    vkFreeMemory(ctx.device, index_memory_, nullptr);
}

}  // namespace viewer
