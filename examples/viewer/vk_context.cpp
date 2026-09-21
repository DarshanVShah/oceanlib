#include "vk_context.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace viewer {
namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
               VkDebugUtilsMessageTypeFlagsEXT,
               const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[vulkan] %s\n", data->pMessage);
    }
    return VK_FALSE;  // never abort the call being validated
}

bool check(VkResult r, const char* what)
{
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vulkan] %s failed (VkResult %d)\n", what,
                     static_cast<int>(r));
        return false;
    }
    return true;
}

}  // namespace

void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageAspectFlags aspect, VkImageLayout old_layout,
                      VkImageLayout new_layout, VkPipelineStageFlags2 src_stage,
                      VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                      VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask     = src_stage;
    barrier.srcAccessMask    = src_access;
    barrier.dstStageMask     = dst_stage;
    barrier.dstAccessMask    = dst_access;
    barrier.oldLayout        = old_layout;
    barrier.newLayout        = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image            = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

bool VkContext::init(GLFWwindow* w, bool enable_validation,
                     VkSampleCountFlagBits requested_msaa)
{
    window = w;
    msaa_samples = requested_msaa;
    if (!create_instance(enable_validation)) return false;
    if (!check(glfwCreateWindowSurface(instance, window, nullptr, &surface),
               "glfwCreateWindowSurface")) {
        return false;
    }
    if (!pick_physical_device()) return false;
    if (!create_device()) return false;
    if (!create_swapchain()) return false;
    // Both attachments have to agree, so the usable set is the intersection.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);
        const VkSampleCountFlags usable =
            props.limits.framebufferColorSampleCounts &
            props.limits.framebufferDepthSampleCounts;

        for (VkSampleCountFlagBits b : {VK_SAMPLE_COUNT_8_BIT,
                                        VK_SAMPLE_COUNT_4_BIT,
                                        VK_SAMPLE_COUNT_2_BIT}) {
            if (usable & b) { max_msaa_ = b; break; }
        }
        if (!(usable & msaa_samples)) {
            std::fprintf(stderr,
                         "MSAA: %ux not supported, falling back to %ux\n",
                         static_cast<unsigned>(msaa_samples),
                         static_cast<unsigned>(max_msaa_));
            msaa_samples = max_msaa_;
        }
    }

    if (!create_depth_resources()) return false;
    if (!create_hdr_target()) return false;
    if (!create_bloom_chain()) return false;
    if (!create_frames()) return false;
    create_timestamp_pool();
    return true;
}

bool VkContext::create_instance(bool enable_validation)
{
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "oceanlib viewer";
    app.pEngineName      = "oceanlib";
    // 1.3 for dynamic rendering and synchronization2 in core. Both remove a
    // large amount of boilerplate - render passes, framebuffers, and the old
    // barrier API - and every GPU that can run this demo supports 1.3.
    app.apiVersion       = VK_API_VERSION_1_3;

    std::uint32_t glfw_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_count);
    std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_count);

    std::vector<const char*> layers;
    if (enable_validation) {
        std::uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        for (const auto& l : available) {
            if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                validation_enabled = true;
                break;
            }
        }
        if (!validation_enabled) {
            std::fprintf(stderr,
                         "[vulkan] validation layer not installed; continuing "
                         "without it\n");
        }
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = static_cast<std::uint32_t>(layers.size());
    ci.ppEnabledLayerNames     = layers.data();

    if (!check(vkCreateInstance(&ci, nullptr, &instance), "vkCreateInstance")) {
        return false;
    }

    if (validation_enabled) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT dci{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            dci.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debug_callback;
            create(instance, &dci, nullptr, &debug_messenger);
        }
    }
    return true;
}

bool VkContext::pick_physical_device()
{
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        std::fprintf(stderr, "[vulkan] no Vulkan-capable device found\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int best_score = -1;

    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;

        // Must have a queue family that both renders and presents. Separate
        // graphics and present families exist on some hardware, but handling
        // that adds ownership transfers for no benefit on any GPU this demo
        // will realistically meet.
        std::uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> families(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, families.data());

        int found = -1;
        for (std::uint32_t i = 0; i < qcount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present);
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                found = static_cast<int>(i);
                break;
            }
        }
        if (found < 0) continue;

        // Prefer discrete: this is a laptop with an integrated GPU too, and
        // landing on the iGPU would make the benchmark numbers meaningless.
        int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                        ? 1000
                        : 10;
        if (score > best_score) {
            best_score   = score;
            best         = dev;
            queue_family = static_cast<std::uint32_t>(found);
        }
    }

    if (best == VK_NULL_HANDLE) {
        std::fprintf(stderr,
                     "[vulkan] no device supports Vulkan 1.3 with a "
                     "graphics+present queue\n");
        return false;
    }

    physical = best;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    std::printf("GPU: %s\n", props.deviceName);
    return true;
}

bool VkContext::create_device()
{
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan13Features f13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &f13;
    // Wireframe mode is a debugging aid worth having in a geometry-heavy demo.
    features.features.fillModeNonSolid = VK_TRUE;
    features.features.samplerAnisotropy = VK_TRUE;

    const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext                   = &features;
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.enabledExtensionCount   = 1;
    ci.ppEnabledExtensionNames = device_extensions;

    if (!check(vkCreateDevice(physical, &ci, nullptr, &device),
               "vkCreateDevice")) {
        return false;
    }
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family;
    return check(vkCreateCommandPool(device, &pci, nullptr, &command_pool),
                 "vkCreateCommandPool");
}

bool VkContext::create_swapchain()
{
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);

    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count,
                                         formats.data());

    // Prefer an sRGB target: the shader tone maps and gamma-encodes into a
    // UNORM target, so we pick the plain UNORM form and do the transfer
    // function ourselves. Picking an _SRGB format here would apply it twice
    // and wash the image out.
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    swapchain_format = chosen.format;

    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        extent.width = std::clamp(static_cast<std::uint32_t>(w),
                                  caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<std::uint32_t>(h),
                                   caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface          = surface;
    ci.minImageCount    = image_count;
    ci.imageFormat      = chosen.format;
    ci.imageColorSpace  = chosen.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    // TRANSFER_SRC so --screenshot can copy the presented image back.
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // FIFO is the only mode guaranteed present, and v-sync is what we want for
    // a viewer anyway - an uncapped frame rate would just heat the laptop.
    ci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped          = VK_TRUE;

    if (!check(vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain),
               "vkCreateSwapchainKHR")) {
        return false;
    }

    vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
    images.resize(count);
    vkGetSwapchainImagesKHR(device, swapchain, &count, images.data());

    image_views.resize(count);
    render_finished.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image            = images[i];
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = swapchain_format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!check(vkCreateImageView(device, &vci, nullptr, &image_views[i]),
                   "vkCreateImageView")) {
            return false;
        }
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device, &sci, nullptr, &render_finished[i]);
    }
    return true;
}

bool VkContext::create_depth_resources()
{
    // D32_SFLOAT is mandatory-supported as a depth attachment, so no format
    // hunting is needed. No stencil: nothing here uses one.
    depth_format = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = depth_format;
    ici.extent      = {extent.width, extent.height, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = msaa_samples;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (!check(vkCreateImage(device, &ici, nullptr, &depth_image),
               "vkCreateImage(depth)")) {
        return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, depth_image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!check(vkAllocateMemory(device, &ai, nullptr, &depth_memory),
               "vkAllocateMemory(depth)")) {
        return false;
    }
    vkBindImageMemory(device, depth_image, depth_memory, 0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = depth_image;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = depth_format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    return check(vkCreateImageView(device, &vci, nullptr, &depth_view),
                 "vkCreateImageView(depth)");
}

bool VkContext::create_hdr_target()
{
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = hdr_format;
    ici.extent        = {extent.width, extent.height, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // Written as an attachment by the scene, read as a texture by the post
    // pass. Both usages have to be declared up front or the second one is
    // undefined behaviour that happens to work on the driver you tested.
    ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &ici, nullptr, &hdr_image) != VK_SUCCESS) return false;

    // The multisampled colour target, when there is one. TRANSIENT_ATTACHMENT
    // alongside COLOR_ATTACHMENT because nothing ever samples it: it is
    // written, resolved into hdr_image, and discarded within one pass, which
    // on a tiler lets the driver keep it in tile memory and never write it to
    // RAM at all. That is most of why MSAA is affordable on mobile.
    if (msaa_samples != VK_SAMPLE_COUNT_1_BIT) {
        VkImageCreateInfo mci = ici;
        mci.samples = msaa_samples;
        mci.usage   = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        if (vkCreateImage(device, &mci, nullptr, &hdr_ms_image) != VK_SUCCESS)
            return false;

        VkMemoryRequirements mreq{};
        vkGetImageMemoryRequirements(device, hdr_ms_image, &mreq);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize  = mreq.size;
        mai.memoryTypeIndex = find_memory_type(mreq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &mai, nullptr, &hdr_ms_memory) != VK_SUCCESS)
            return false;
        vkBindImageMemory(device, hdr_ms_image, hdr_ms_memory, 0);

        VkImageViewCreateInfo mvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        mvi.image            = hdr_ms_image;
        mvi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        mvi.format           = hdr_format;
        mvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &mvi, nullptr, &hdr_ms_view) != VK_SUCCESS)
            return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, hdr_image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &ai, nullptr, &hdr_memory) != VK_SUCCESS) return false;
    vkBindImageMemory(device, hdr_image, hdr_memory, 0);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image            = hdr_image;
    vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vi.format           = hdr_format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(device, &vi, nullptr, &hdr_view) == VK_SUCCESS;
}

bool VkContext::create_bloom_chain()
{
    for (std::uint32_t i = 0; i < kBloomLevels; ++i) {
        BloomLevel& lvl = bloom[i];
        // Level 0 is half resolution, and each one after that halves again.
        // Clamped at 1 so a very small or very thin window cannot ask for a
        // zero-sized image, which is not a legal extent.
        lvl.extent.width  = (extent.width  >> (i + 1)) ? (extent.width  >> (i + 1)) : 1u;
        lvl.extent.height = (extent.height >> (i + 1)) ? (extent.height >> (i + 1)) : 1u;

        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = hdr_format;
        ici.extent        = {lvl.extent.width, lvl.extent.height, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        // Each level is written as an attachment on the way down and sampled
        // on the way back up, so both usages are needed on every level.
        ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &lvl.image) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, lvl.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &lvl.memory) != VK_SUCCESS) return false;
        vkBindImageMemory(device, lvl.image, lvl.memory, 0);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image            = lvl.image;
        vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vi.format           = hdr_format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &vi, nullptr, &lvl.view) != VK_SUCCESS) return false;
    }
    return true;
}

bool VkContext::create_frames()
{
    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkCommandBufferAllocateInfo ai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool        = command_pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (!check(vkAllocateCommandBuffers(device, &ai, &frames[i].cmd),
                   "vkAllocateCommandBuffers")) {
            return false;
        }

        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device, &sci, nullptr, &frames[i].image_available);

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        // Signalled, so the very first frame does not deadlock waiting for
        // work that was never submitted.
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fci, nullptr, &frames[i].in_flight);
    }
    return true;
}

// ---------------------------------------------------------------------------
// GPU timing
// ---------------------------------------------------------------------------

void VkContext::create_timestamp_pool()
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    timestamp_period_ns_ = props.limits.timestampPeriod;

    // Two separate capabilities, and both have to hold. A device can report a
    // timestamp period while the queue family we actually submit to writes
    // zero valid bits, which would leave every reading garbage rather than
    // failing loudly - so the queue family is checked, not just the device.
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                             families.data());

    const std::uint32_t valid_bits =
        (queue_family < family_count) ? families[queue_family].timestampValidBits : 0;

    if (timestamp_period_ns_ <= 0.0f || valid_bits == 0) {
        std::fprintf(stderr,
                     "GPU timing unavailable: timestampPeriod %.3f, "
                     "timestampValidBits %u on queue family %u\n",
                     timestamp_period_ns_, valid_bits, queue_family);
        return;
    }

    // Only the low `valid_bits` of each value are defined. Shifting by 64 is
    // undefined behaviour in C++, so the full-width case is spelled out.
    timestamp_mask_ = (valid_bits >= 64)
                          ? ~std::uint64_t{0}
                          : ((std::uint64_t{1} << valid_bits) - 1);

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = kFramesInFlight * kMaxGpuMarks;
    if (vkCreateQueryPool(device, &qpi, nullptr, &timestamp_pool_) != VK_SUCCESS) {
        std::fprintf(stderr, "GPU timing unavailable: query pool creation failed\n");
        return;
    }

    gpu_supported_ = true;
}

void VkContext::gpu_begin(VkCommandBuffer cmd)
{
    if (!gpu_supported_) return;
    GpuFrameMarks& m = gpu_marks_[frame_index];
    m.count = 0;

    // The whole slice is reset even though only `count` of it gets written:
    // an unwritten query is UNAVAILABLE rather than stale, and reading one is
    // undefined. Resetting outside a render pass is required, which is why
    // this belongs in begin_frame and not beside the first draw.
    vkCmdResetQueryPool(cmd, timestamp_pool_, frame_index * kMaxGpuMarks,
                        kMaxGpuMarks);

    // BOTTOM_OF_PIPE for the zero point: "after everything submitted so far
    // has finished". TOP_OF_PIPE would be satisfied the instant the command
    // reaches the front of the queue, which measures submission, not work.
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         timestamp_pool_, frame_index * kMaxGpuMarks);
    m.names[m.count] = nullptr;
    ++m.count;
}

void VkContext::gpu_mark(VkCommandBuffer cmd, const char* name)
{
    if (!gpu_supported_) return;
    GpuFrameMarks& m = gpu_marks_[frame_index];
    if (m.count == 0 || m.count >= kMaxGpuMarks) return;   // no begin, or full

    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         timestamp_pool_,
                         frame_index * kMaxGpuMarks + m.count);
    m.names[m.count] = name;
    ++m.count;
    m.recorded = true;
}

void VkContext::gpu_collect(std::uint32_t slot)
{
    if (!gpu_supported_) return;
    GpuFrameMarks& m = gpu_marks_[slot];
    if (!m.recorded || m.count < 2) return;

    std::uint64_t ticks[kMaxGpuMarks]{};
    // No WAIT bit: begin_frame has already waited on this slot's fence, so the
    // results are there. Asking the driver to wait as well would be a second,
    // redundant stall - and if they somehow are not ready, a dropped sample is
    // the right outcome for a statistic, not a stalled frame.
    const VkResult r = vkGetQueryPoolResults(
        device, timestamp_pool_, slot * kMaxGpuMarks, m.count,
        sizeof(ticks), ticks, sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return;

    gpu_spans_.clear();
    const double ns_per_tick = static_cast<double>(timestamp_period_ns_);
    for (std::uint32_t i = 1; i < m.count; ++i) {
        const std::uint64_t a = ticks[i - 1] & timestamp_mask_;
        const std::uint64_t b = ticks[i] & timestamp_mask_;
        // Masked subtraction in unsigned arithmetic, so a counter that wrapped
        // between the two marks still yields the true positive interval.
        const std::uint64_t d = (b - a) & timestamp_mask_;
        gpu_spans_.push_back({m.names[i], static_cast<double>(d) * ns_per_tick * 1e-6});
    }

    const std::uint64_t first = ticks[0] & timestamp_mask_;
    const std::uint64_t last  = ticks[m.count - 1] & timestamp_mask_;
    gpu_total_ms_ = static_cast<double>((last - first) & timestamp_mask_) *
                    ns_per_tick * 1e-6;

    // Accumulate. The span list is a fixed sequence for a given build, so an
    // index is a stable key; the name is carried along only for printing.
    if (gpu_stats_.size() != gpu_spans_.size()) {
        gpu_stats_.assign(gpu_spans_.size(), GpuStat{});
    }
    auto accumulate = [](GpuStat& st, const char* name, double ms) {
        st.name = name;
        if (st.count == 0) { st.min = ms; st.max = ms; }
        else { st.min = (ms < st.min) ? ms : st.min;
               st.max = (ms > st.max) ? ms : st.max; }
        st.sum += ms;
        ++st.count;
    };
    for (std::size_t i = 0; i < gpu_spans_.size(); ++i) {
        accumulate(gpu_stats_[i], gpu_spans_[i].name, gpu_spans_[i].ms);
    }
    accumulate(gpu_total_stat_, "total", gpu_total_ms_);
}

void VkContext::gpu_reset_stats()
{
    gpu_stats_.clear();
    gpu_total_stat_ = GpuStat{};
}

bool VkContext::begin_frame(std::uint32_t& image_index, VkCommandBuffer& cmd)
{
    Frame& f = frames[frame_index];
    vkWaitForFences(device, 1, &f.in_flight, VK_TRUE, UINT64_MAX);

    // This slot's previous submission has now completed, so its timestamps are
    // readable without stalling anything.
    gpu_collect(frame_index);

    const VkResult acq = vkAcquireNextImageKHR(
        device, swapchain, UINT64_MAX, f.image_available, VK_NULL_HANDLE,
        &image_index);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        return false;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return false;

    // Reset only after we know we are going to submit; resetting before the
    // acquire would leave the fence unsignalled on the early-out path above
    // and the next wait would hang.
    vkResetFences(device, 1, &f.in_flight);
    vkResetCommandBuffer(f.cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);
    gpu_begin(f.cmd);

    cmd = f.cmd;
    return true;
}

void VkContext::end_frame(std::uint32_t image_index)
{
    Frame& f = frames[frame_index];
    vkEndCommandBuffer(f.cmd);

    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = f.image_available;
    // Wait only at the point the attachment is actually written. Waiting at
    // TOP_OF_PIPE instead would serialise vertex work behind the acquire for
    // no reason.
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = render_finished[image_index];
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

    VkCommandBufferSubmitInfo cbi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = f.cmd;

    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount   = 1;
    submit.pWaitSemaphoreInfos      = &wait;
    submit.commandBufferInfoCount   = 1;
    submit.pCommandBufferInfos      = &cbi;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos    = &signal;

    vkQueueSubmit2(queue, 1, &submit, f.in_flight);

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &render_finished[image_index];
    present.swapchainCount     = 1;
    present.pSwapchains        = &swapchain;
    present.pImageIndices      = &image_index;

    const VkResult r = vkQueuePresentKHR(queue, &present);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        recreate_swapchain();
    }

    frame_index = (frame_index + 1) % kFramesInFlight;
}

void VkContext::destroy_swapchain()
{
    if (depth_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, depth_view, nullptr);
        vkDestroyImage(device, depth_image, nullptr);
        vkFreeMemory(device, depth_memory, nullptr);
        depth_view = VK_NULL_HANDLE;
    }
    if (hdr_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, hdr_view, nullptr);
        vkDestroyImage(device, hdr_image, nullptr);
        vkFreeMemory(device, hdr_memory, nullptr);
        hdr_view = VK_NULL_HANDLE;
    }
    if (hdr_ms_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, hdr_ms_view, nullptr);
        vkDestroyImage(device, hdr_ms_image, nullptr);
        vkFreeMemory(device, hdr_ms_memory, nullptr);
        hdr_ms_view = VK_NULL_HANDLE;
    }
    for (BloomLevel& lvl : bloom) {
        if (lvl.view == VK_NULL_HANDLE) continue;
        vkDestroyImageView(device, lvl.view, nullptr);
        vkDestroyImage(device, lvl.image, nullptr);
        vkFreeMemory(device, lvl.memory, nullptr);
        lvl = BloomLevel{};
    }
    for (VkSemaphore s : render_finished) vkDestroySemaphore(device, s, nullptr);
    for (VkImageView v : image_views) vkDestroyImageView(device, v, nullptr);
    render_finished.clear();
    image_views.clear();
    images.clear();
    if (swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

void VkContext::recreate_swapchain()
{
    // A minimised window reports a zero-sized framebuffer, which is not a
    // legal swapchain extent. Block until it comes back rather than failing.
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    while (w == 0 || h == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &w, &h);
    }

    vkDeviceWaitIdle(device);
    destroy_swapchain();
    create_swapchain();
    create_depth_resources();
    create_hdr_target();
    create_bloom_chain();
}

void VkContext::shutdown()
{
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);

    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (frames[i].image_available)
            vkDestroySemaphore(device, frames[i].image_available, nullptr);
        if (frames[i].in_flight)
            vkDestroyFence(device, frames[i].in_flight, nullptr);
    }
    destroy_swapchain();
    if (timestamp_pool_) vkDestroyQueryPool(device, timestamp_pool_, nullptr);
    if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;

    if (debug_messenger != VK_NULL_HANDLE) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(instance, debug_messenger, nullptr);
    }
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
}

std::uint32_t VkContext::find_memory_type(std::uint32_t type_bits,
                                          VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physical, &mem);
    for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    std::fprintf(stderr, "[vulkan] no memory type satisfies the request\n");
    return 0;
}

bool VkContext::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer& buffer,
                              VkDeviceMemory& memory) const
{
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(device, &ci, nullptr, &buffer), "vkCreateBuffer")) {
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
    if (!check(vkAllocateMemory(device, &ai, nullptr, &memory),
               "vkAllocateMemory")) {
        return false;
    }
    vkBindBufferMemory(device, buffer, memory, 0);
    return true;
}

VkCommandBuffer VkContext::begin_one_shot() const
{
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = command_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VkContext::end_one_shot(VkCommandBuffer cmd) const
{
    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos    = &cbi;

    vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);  // startup only, so a full stall is fine
    vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

}  // namespace viewer
