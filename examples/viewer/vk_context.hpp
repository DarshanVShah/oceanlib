// Vulkan bootstrap: instance, device, swapchain, per-frame synchronisation.
//
// Nothing here knows about oceans. Kept separate so ocean_view.cpp can be read
// as "what it takes to draw the library's output" without 600 lines of
// unrelated setup in the way.
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
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

// --- GPU timing -----------------------------------------------------------

// Most timestamps any one frame may write. Eight is well past what the viewer
// needs and costs 8 * 8 bytes per frame in flight, so there is no reason to
// tune it.
inline constexpr std::uint32_t kMaxGpuMarks = 8;

// One measured interval: the time from the previous mark to this one.
struct GpuSpan {
    const char* name = nullptr;
    double      ms   = 0.0;
};

// The same interval accumulated over many frames.
//
// A single frame's timestamp is not a measurement of anything. Measured on
// this viewer, the upload span alone came back as 2.29, 0.61 and 2.51 ms on
// three identical runs - a 4x spread - because one sample catches whatever
// clock state and queue overlap that particular frame happened to be in.
// Reporting the mean with its range attached is the difference between a
// number and a number someone could act on.
struct GpuStat {
    const char*   name  = nullptr;
    double        sum   = 0.0;
    double        min   = 0.0;
    double        max   = 0.0;
    std::uint64_t count = 0;
    [[nodiscard]] double mean() const { return count ? sum / static_cast<double>(count) : 0.0; }
};

class VkContext {
public:
    bool init(GLFWwindow* window, bool enable_validation);
    void shutdown();

    // Acquires the next swapchain image and begins recording. Returns false if
    // the swapchain needed recreating, in which case the caller should skip
    // the frame.
    bool begin_frame(std::uint32_t& image_index, VkCommandBuffer& cmd);
    void end_frame(std::uint32_t image_index);

    void recreate_swapchain();
    void wait_idle() const { vkDeviceWaitIdle(device); }

    // --- GPU timing ------------------------------------------------------
    //
    // Timestamps go into this frame's own slice of one query pool and are read
    // back kFramesInFlight frames later, when begin_frame() has already waited
    // on that slot's fence. Reading them any sooner would mean blocking on the
    // GPU to measure the GPU, which changes the thing being measured - the
    // results are simply two frames stale instead, which for a running average
    // on screen is no difference at all.
    //
    // A mark is a point, and the span it names is the interval from the
    // previous mark to it. gpu_begin() lays down the unnamed zero point.
    void gpu_begin(VkCommandBuffer cmd);
    void gpu_mark(VkCommandBuffer cmd, const char* name);

    // Spans from the most recently completed frame. Empty until one has
    // finished, and empty for good on a device whose queue cannot timestamp.
    [[nodiscard]] const std::vector<GpuSpan>& gpu_spans() const { return gpu_spans_; }
    [[nodiscard]] double gpu_total_ms() const { return gpu_total_ms_; }
    [[nodiscard]] bool   gpu_timing_supported() const { return gpu_supported_; }

    // Accumulated across every frame since the last reset. Reset after the
    // warm-up frames so first-frame allocation and shader compilation do not
    // sit inside the average.
    [[nodiscard]] const std::vector<GpuStat>& gpu_stats() const { return gpu_stats_; }
    [[nodiscard]] const GpuStat& gpu_total_stat() const { return gpu_total_stat_; }
    void gpu_reset_stats();

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

    // The scene's real render target: linear HDR, tonemapped later by the post
    // pass rather than by each material shader.
    //
    // fp16 rather than the swapchain's 8-bit UNORM because the values written
    // here are radiance, not colour. Sun glitter on water runs orders of
    // magnitude above the diffuse sea around it, and clamping that to 1.0 at
    // the point it is generated throws away exactly the range bloom and a
    // filmic curve exist to use. fp16 also costs half the bandwidth of fp32
    // for a dynamic range nothing in this scene comes close to exhausting.
    //
    // Sized to the swapchain, so it lives and dies with it.
    VkFormat       hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkImage        hdr_image  = VK_NULL_HANDLE;
    VkDeviceMemory hdr_memory = VK_NULL_HANDLE;
    VkImageView    hdr_view   = VK_NULL_HANDLE;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    Frame         frames[kFramesInFlight]{};
    std::uint32_t frame_index = 0;

    bool validation_enabled = false;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

    // --- GPU timing state -------------------------------------------------
    VkQueryPool timestamp_pool_ = VK_NULL_HANDLE;
    // Nanoseconds per tick, from the device limits. Vendors differ by orders
    // of magnitude here, so a raw tick delta means nothing on its own.
    float         timestamp_period_ns_ = 0.0f;
    // Not every queue family can timestamp, and those that can may implement
    // fewer than 64 valid bits - the rest are undefined and must be masked off
    // before subtracting, or a wrap looks like a wildly negative interval.
    std::uint64_t timestamp_mask_ = 0;
    bool          gpu_supported_  = false;

    struct GpuFrameMarks {
        std::array<const char*, kMaxGpuMarks> names{};
        std::uint32_t count    = 0;
        bool          recorded = false;   // has this slot ever been submitted
    };
    std::array<GpuFrameMarks, kFramesInFlight> gpu_marks_{};
    std::vector<GpuSpan> gpu_spans_;
    double               gpu_total_ms_ = 0.0;
    std::vector<GpuStat> gpu_stats_;
    GpuStat              gpu_total_stat_{};

    void gpu_collect(std::uint32_t slot);

private:
    bool create_instance(bool enable_validation);
    bool pick_physical_device();
    bool create_device();
    bool create_swapchain();
    void destroy_swapchain();
    bool create_depth_resources();
    bool create_hdr_target();
    bool create_frames();
    void create_timestamp_pool();
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
