// Vulkan bootstrap: instance, device, swapchain, per-frame synchronisation.
//
// Nothing here knows about oceans. Kept separate so ocean_view.cpp can be read
// as "what it takes to draw the library's output" without 600 lines of
// unrelated setup in the way.
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#include <vector>

namespace viewer {

// Two frames in flight: the CPU records frame N+1 while the GPU is still
// working on N. Three buys little once the CPU side is this cheap, and costs
// an extra frame of input latency.
inline constexpr std::uint32_t kFramesInFlight = 2;

struct Frame {
    VkCommandBuffer cmd            = VK_NULL_HANDLE;
    VkSemaphore     image_available = VK_NULL_HANDLE;
    VkFence         in_flight       = VK_NULL_HANDLE;
};

class VkContext {
public:
    // `gpu_preference`: "discrete" (default if null - a laptop's iGPU would
    // otherwise make benchmark numbers meaningless by accident), "integrated",
    // or a substring of the device name to match (e.g. "UHD"), so a specific
    // GPU can be selected for a before/after comparison.
    bool init(GLFWwindow* window, bool enable_validation,
              const char* gpu_preference = nullptr);
    void shutdown();

    // Acquires the next swapchain image and begins recording. Returns false if
    // the swapchain needed recreating, in which case the caller should skip
    // the frame.
    bool begin_frame(std::uint32_t& image_index, VkCommandBuffer& cmd);
    void end_frame(std::uint32_t image_index);

    // Writes a GPU timestamp for the frame currently being recorded (slot
    // `frame_index`); `start` selects which of the two queries that slot
    // uses. Cheap - one vkCmdWriteTimestamp2 - safe to call every frame.
    void write_timestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits2 stage,
                         bool start);

    // GPU time, in milliseconds, that the timestamps written for frame slot
    // `frame_index` bounded - valid only after that slot's fence has been
    // waited on (begin_frame already does this before recording reuses the
    // slot), so this reads the PREVIOUS frame that used this slot, exactly
    // like the fence wait it piggybacks on.
    [[nodiscard]] double last_gpu_ms() const;

    void recreate_swapchain();
    void wait_idle() const { vkDeviceWaitIdle(device); }

    // --- small allocation helpers ---------------------------------------
    //
    // A real engine would use VMA. For a viewer that makes a fixed handful of
    // allocations at startup, one vkAllocateMemory each is clearer than
    // pulling in an allocator, and the limit on allocation count is nowhere
    // near being a problem.
    std::uint32_t find_memory_type(std::uint32_t type_bits,
                                   VkMemoryPropertyFlags props) const;
    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props, VkBuffer& buffer,
                       VkDeviceMemory& memory) const;

    // Runs a short command buffer and waits for it. Startup only.
    VkCommandBuffer begin_one_shot() const;
    void            end_one_shot(VkCommandBuffer cmd) const;

    GLFWwindow*      window       = nullptr;
    VkInstance       instance     = VK_NULL_HANDLE;
    VkSurfaceKHR     surface      = VK_NULL_HANDLE;
    VkPhysicalDevice physical     = VK_NULL_HANDLE;
    VkDevice         device       = VK_NULL_HANDLE;
    VkQueue          queue        = VK_NULL_HANDLE;
    std::uint32_t    queue_family = 0;

    VkSwapchainKHR           swapchain        = VK_NULL_HANDLE;
    VkFormat                 swapchain_format = VK_FORMAT_UNDEFINED;
    VkExtent2D               extent{};
    std::vector<VkImage>     images;
    std::vector<VkImageView> image_views;
    // One per swapchain image, not per frame-in-flight: the semaphore a
    // present waits on must belong to the image being presented, or a
    // fast-recycling swapchain can wait on a semaphore that is still pending.
    std::vector<VkSemaphore> render_finished;

    VkFormat       depth_format = VK_FORMAT_UNDEFINED;
    VkImage        depth_image  = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView    depth_view   = VK_NULL_HANDLE;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    Frame         frames[kFramesInFlight]{};
    std::uint32_t frame_index = 0;

    bool validation_enabled = false;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

    // 2 queries per frame-in-flight slot (start, end); nanoseconds per tick
    // comes from VkPhysicalDeviceLimits and varies by vendor, so every
    // conversion to milliseconds must go through it rather than assuming 1.
    VkQueryPool timestamp_pool   = VK_NULL_HANDLE;
    float       timestamp_period_ns = 1.0f;

private:
    bool create_instance(bool enable_validation);
    bool pick_physical_device(const char* gpu_preference);
    bool create_device();
    bool create_swapchain();
    void destroy_swapchain();
    bool create_depth_resources();
    bool create_frames();
};

// Records a layout transition with synchronization2. Dependency scopes are
// passed explicitly rather than hidden behind an "auto" helper, because
// getting them wrong is the single most common source of Vulkan bugs that
// only appear on a different vendor's driver.
void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageAspectFlags aspect, VkImageLayout old_layout,
                      VkImageLayout new_layout, VkPipelineStageFlags2 src_stage,
                      VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                      VkAccessFlags2 dst_access);

}  // namespace viewer
