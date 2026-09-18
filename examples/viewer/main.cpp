// oceanlib Vulkan viewer.
//
//   WASD / QE   move          mouse (hold RMB)  look
//   Shift       move faster   Tab               wireframe
//   Space       pause time    Esc               quit
//   1 / 2       choppiness    R                 reset camera
//
//   --size N          ocean grid resolution (default 256)
//   --rings N         clipmap ring count, 1-8 (default 6; ADR-021)
//   --cell C          finest ring's world metres per mesh cell (default 0.5)
//   --cascades N      number of cascade scales, 1-3 (default 3; ADR-020)
//   --wind U          wind speed in m/s (default 12)
//   --depth D         water depth in metres for shallow-water dispersion
//                     (default 0 = deep water; try 3-8 for visibly shoaled waves)
//   --chop C          choppiness / Tessendorf lambda (default 1.0)
//   --foam F          Jacobian threshold for foam (default 0.6)
//   --wireframe       start in wireframe mode
//   --simd L          force a SIMD level: scalar|sse2|avx2|neon (default: native)
//   --threads N       cap the thread pool (default: one per hardware thread) -
//                     use 1 to simulate a single-core machine
//   --screenshot P    render a few frames, write P as BMP, exit
//   --frames N        frames to render before the screenshot (default 90)
//   --no-validation   skip the Vulkan validation layer
//   --gpu G           select the GPU: discrete|integrated, or a name
//                     substring e.g. "UHD" (default: prefer discrete)

#include "ocean_view.hpp"
#include "vk_context.hpp"
#include "vk_math.hpp"

#include "ocean/cascade.hpp"
#include "ocean/ocean.h"    // for ocean_simd_level(); the viewer
                            // deliberately uses only public headers
#include "ocean/ocean.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    std::uint32_t size   = 256;
    std::uint32_t rings  = 6;
    float         cell   = 0.5f;
    float         wind   = 12.0f;
    float         depth  = 0.0f;   // <= 0 = deep water
    float         chop   = 1.0f;
    float         foam   = 0.6f;
    bool          wireframe = false;
    std::string   screenshot;
    std::string   simd;           // "" = native max; else scalar|sse2|avx2|neon
    std::uint32_t threads = 0;    // 0 = one worker per hardware thread
    std::uint32_t cascades = 3;   // 1-3, see ADR-020
    int           frames = 90;
    bool          validation = true;
    std::string   gpu;            // "" = prefer discrete (default);
                                  // "discrete" | "integrated" | a name substring
};

Options parse_args(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (a == "--size")             o.size  = std::strtoul(next(), nullptr, 10);
        else if (a == "--rings")       o.rings = std::strtoul(next(), nullptr, 10);
        else if (a == "--cell")        o.cell  = std::strtof(next(), nullptr);
        else if (a == "--wind")        o.wind  = std::strtof(next(), nullptr);
        else if (a == "--depth")       o.depth = std::strtof(next(), nullptr);
        else if (a == "--chop")        o.chop  = std::strtof(next(), nullptr);
        else if (a == "--foam")        o.foam  = std::strtof(next(), nullptr);
        else if (a == "--wireframe")   o.wireframe = true;
        else if (a == "--simd")        o.simd = next();
        else if (a == "--threads")     o.threads = std::strtoul(next(), nullptr, 10);
        else if (a == "--cascades")    o.cascades = std::strtoul(next(), nullptr, 10);
        else if (a == "--screenshot")  o.screenshot = next();
        else if (a == "--frames")      o.frames = std::atoi(next());
        else if (a == "--no-validation") o.validation = false;
        else if (a == "--gpu")          o.gpu = next();
    }
    if (o.rings < 1) o.rings = 1;
    if (o.rings > viewer::kMaxRings) o.rings = viewer::kMaxRings;
    if (o.cascades < 1) o.cascades = 1;
    if (o.cascades > 3) o.cascades = 3;
    return o;
}

// --- camera --------------------------------------------------------------

struct Camera {
    vkm::Vec3 position{0.0f, 18.0f, 60.0f};
    float yaw   = -1.6f;   // radians, looking along -Z
    float pitch = -0.18f;

    [[nodiscard]] vkm::Vec3 forward() const
    {
        return vkm::normalize({std::cos(pitch) * std::cos(yaw), std::sin(pitch),
                               std::cos(pitch) * std::sin(yaw)});
    }
    [[nodiscard]] vkm::Vec3 right() const
    {
        return vkm::normalize(vkm::cross(forward(), {0.0f, 1.0f, 0.0f}));
    }
    void reset() { *this = Camera{}; }
};

// --- screenshot ----------------------------------------------------------

// Writes a 24-bit BMP. BMP because it needs no compression library, so the
// viewer keeps its dependency list at exactly GLFW and Vulkan.
bool write_bmp(const char* path, const std::uint8_t* bgra, std::uint32_t width,
               std::uint32_t height)
{
    // BMP rows are 4-byte aligned and stored bottom-up.
    const std::uint32_t row_bytes = ((width * 3u + 3u) / 4u) * 4u;
    const std::uint32_t pixel_bytes = row_bytes * height;
    const std::uint32_t file_bytes = 54u + pixel_bytes;

    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s for writing\n", path);
        return false;
    }

    std::uint8_t header[54]{};
    header[0] = 'B'; header[1] = 'M';
    std::memcpy(&header[2], &file_bytes, 4);
    const std::uint32_t offset = 54;
    std::memcpy(&header[10], &offset, 4);
    const std::uint32_t dib = 40;
    std::memcpy(&header[14], &dib, 4);
    std::memcpy(&header[18], &width, 4);
    std::memcpy(&header[22], &height, 4);
    const std::uint16_t planes = 1, bpp = 24;
    std::memcpy(&header[26], &planes, 2);
    std::memcpy(&header[28], &bpp, 2);
    std::memcpy(&header[34], &pixel_bytes, 4);
    std::fwrite(header, 1, sizeof(header), f);

    std::vector<std::uint8_t> row(row_bytes, 0);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* src = bgra + static_cast<std::size_t>(height - 1 - y) *
                                             width * 4;
        for (std::uint32_t x = 0; x < width; ++x) {
            // Swapchain format is B8G8R8A8_UNORM and BMP stores BGR, so the
            // channels already line up; only the alpha is dropped.
            row[x * 3 + 0] = src[x * 4 + 0];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + 2];
        }
        std::fwrite(row.data(), 1, row_bytes, f);
    }
    std::fclose(f);
    return true;
}

bool capture_frame(viewer::VkContext& ctx, std::uint32_t image_index,
                   const char* path)
{
    ctx.wait_idle();

    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(ctx.extent.width) * ctx.extent.height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!ctx.create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           buffer, memory)) {
        return false;
    }

    VkCommandBuffer cmd = ctx.begin_one_shot();
    // The image is in PRESENT_SRC after the frame that drew it.
    viewer::transition_image(cmd, ctx.images[image_index],
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                             VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {ctx.extent.width, ctx.extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, ctx.images[image_index],
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                           &copy);

    viewer::transition_image(cmd, ctx.images[image_index],
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                             VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_TRANSFER_READ_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    ctx.end_one_shot(cmd);

    void* mapped = nullptr;
    vkMapMemory(ctx.device, memory, 0, bytes, 0, &mapped);
    const bool ok = write_bmp(path, static_cast<const std::uint8_t*>(mapped),
                              ctx.extent.width, ctx.extent.height);
    vkUnmapMemory(ctx.device, memory);

    vkDestroyBuffer(ctx.device, buffer, nullptr);
    vkFreeMemory(ctx.device, memory, nullptr);
    return ok;
}

// --- input state ---------------------------------------------------------

struct Input {
    Camera camera;
    double last_x = 0.0, last_y = 0.0;
    bool   looking = false;
    bool   paused = false;
    bool   wireframe = false;
    float  choppiness = 1.0f;
};

void key_callback(GLFWwindow* w, int key, int, int action, int)
{
    if (action != GLFW_PRESS) return;
    auto* in = static_cast<Input*>(glfwGetWindowUserPointer(w));
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
        case GLFW_KEY_TAB:    in->wireframe = !in->wireframe; break;
        case GLFW_KEY_SPACE:  in->paused = !in->paused; break;
        case GLFW_KEY_R:      in->camera.reset(); break;
        case GLFW_KEY_1:      in->choppiness = std::fmax(0.0f, in->choppiness - 0.25f); break;
        case GLFW_KEY_2:      in->choppiness = std::fmin(4.0f, in->choppiness + 0.25f); break;
        default: break;
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const Options opt = parse_args(argc, argv);

    if (glfwInit() != GLFW_TRUE) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    if (glfwVulkanSupported() != GLFW_TRUE) {
        std::fprintf(stderr, "no Vulkan loader found\n");
        glfwTerminate();
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window =
        glfwCreateWindow(1600, 900, "oceanlib viewer", nullptr, nullptr);
    if (window == nullptr) {
        std::fprintf(stderr, "window creation failed\n");
        glfwTerminate();
        return 1;
    }

    Input input;
    input.choppiness = opt.chop;
    input.wireframe  = opt.wireframe;
    glfwSetWindowUserPointer(window, &input);
    glfwSetKeyCallback(window, key_callback);

    viewer::VkContext ctx;
    if (!ctx.init(window, opt.validation,
                  opt.gpu.empty() ? nullptr : opt.gpu.c_str())) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // --- the library --------------------------------------------------------
    //
    // Three cascades (ADR-020): a far scale for big swells, a mid scale, and a
    // near scale for fine ripples, each an ordinary OceanDesc - the only new
    // concept is that CascadeStack sums several of them. --wind/--depth/
    // --chop/--foam apply uniformly (the same physical sea state, viewed at
    // three different wavelength bands); patch_length and small_wave_cutoff
    // are what actually separate the scales, and seeds are distinct per level
    // deliberately (see CascadeStack's class comment on correlated seeds).
    if (!opt.simd.empty()) {
        const char* installed = ocean_force_simd_level(opt.simd.c_str());
        std::printf("requested SIMD level '%s' -> installed '%s'\n",
                    opt.simd.c_str(), installed);
    }

    struct CascadeSpec { float patch_length; float cutoff; std::uint64_t seed; };
    static constexpr CascadeSpec kSpecs[3] = {
        {800.0f, 15.0f, 1337},  // far:  big swells
        {150.0f,  4.0f, 1338},  // mid
        { 25.0f,  0.5f, 1339},  // near: fine ripples
    };

    std::vector<ocean::OceanDesc> levels;
    for (std::uint32_t i = 0; i < opt.cascades; ++i) {
        ocean::OceanDesc d;
        d.size                    = opt.size;
        d.patch_length            = kSpecs[i].patch_length;
        d.seed                    = kSpecs[i].seed;
        d.choppiness              = input.choppiness;
        d.spectrum.wind_speed     = opt.wind;
        d.spectrum.depth          = opt.depth;
        d.spectrum.fetch          = 100000.0f;
        d.spectrum.wind_direction = 0.4f;
        d.spectrum.small_wave_cutoff = kSpecs[i].cutoff;
        d.foam_threshold          = opt.foam;
        d.thread_count            = opt.threads;
        levels.push_back(d);
    }

    ocean::CascadeStack stack{std::span<const ocean::OceanDesc>(levels)};
    float active_choppiness = input.choppiness;

    char depth_desc[64];
    if (opt.depth > 0.0f) {
        std::snprintf(depth_desc, sizeof(depth_desc), "%.1f m (shallow)", opt.depth);
    } else {
        std::snprintf(depth_desc, sizeof(depth_desc), "infinite (deep water)");
    }
    std::printf("ocean: %u cascade(s), %ux%u each, patches", opt.cascades, opt.size, opt.size);
    for (std::uint32_t i = 0; i < opt.cascades; ++i) std::printf(" %.0fm", kSpecs[i].patch_length);
    std::printf(", wind %.1f m/s, depth %s, SIMD %s\n",
                opt.wind, depth_desc, ocean_simd_level());

    viewer::RingLayout ring_layout;
    ring_layout.ring_count     = opt.rings;
    ring_layout.base_cell_size = opt.cell;

    viewer::OceanView view;
    if (!view.init(ctx, levels, ring_layout)) {
        ctx.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    std::printf("clipmap: %u rings, %.2f m finest cell, %.0f m outermost footprint, %.2fM triangles/frame\n",
                opt.rings, opt.cell,
                ring_layout.footprint(opt.rings - 1),
                view.triangle_count() / 1.0e6);

    const vkm::Vec3 sun = vkm::normalize({-0.45f, 0.38f, -0.80f});

    double sim_time = 0.0;
    auto   last     = std::chrono::steady_clock::now();
    double ocean_ms_avg = 0.0;
    double gpu_ms_avg = 0.0;
    double fps_avg = 0.0;
    int    frame_counter = 0;
    std::uint32_t last_image = 0;

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();

        const auto now = std::chrono::steady_clock::now();
        const double wall_dt =
            std::chrono::duration<double>(now - last).count();
        last = now;

        // In --screenshot mode, advance by a FIXED virtual timestep instead of
        // measured wall-clock time. Otherwise the captured frame depends on
        // how fast this particular machine happened to render the warmup
        // frames - the same --frames N would land at a different simulated
        // time on a fast GPU than on a slow one, making screenshots useless
        // for comparing configurations (or for any kind of visual regression
        // testing across machines). A fixed 1/60s virtual step makes
        // --screenshot fully reproducible: same flags, same seed, same pixels,
        // regardless of how long each frame actually took to compute.
        const double dt = opt.screenshot.empty() ? wall_dt : (1.0 / 60.0);
        if (!input.paused) sim_time += dt;

        // --- camera -------------------------------------------------------
        const bool rmb =
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        double mx = 0.0, my = 0.0;
        glfwGetCursorPos(window, &mx, &my);
        if (rmb && !input.looking) {
            input.last_x = mx;
            input.last_y = my;
        }
        if (rmb) {
            input.camera.yaw   += static_cast<float>((mx - input.last_x) * 0.0032);
            input.camera.pitch -= static_cast<float>((my - input.last_y) * 0.0032);
            input.camera.pitch = std::fmax(-1.45f, std::fmin(1.45f, input.camera.pitch));
            input.last_x = mx;
            input.last_y = my;
        }
        input.looking = rmb;

        const float speed =
            (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 120.0f : 28.0f) *
            static_cast<float>(dt);
        const vkm::Vec3 fwd = input.camera.forward();
        const vkm::Vec3 rgt = input.camera.right();
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) input.camera.position += fwd * speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) input.camera.position += fwd * -speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) input.camera.position += rgt * speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) input.camera.position += rgt * -speed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) input.camera.position.y += speed;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) input.camera.position.y -= speed;

        // --- simulate -----------------------------------------------------
        //
        // Choppiness is fixed at construction, so changing it means rebuilding
        // the simulation. That reallocates, which is exactly why it is a key
        // press and not a per-frame parameter: the library's contract is that
        // update() never allocates, and honouring that means some things are
        // construction-time decisions.
        if (input.choppiness != active_choppiness) {
            for (auto& lvl : levels) lvl.choppiness = input.choppiness;
            ctx.wait_idle();
            stack = ocean::CascadeStack{std::span<const ocean::OceanDesc>(levels)};
            active_choppiness = input.choppiness;
        }

        const auto sim_start = std::chrono::steady_clock::now();
        stack.update(sim_time);
        const double ocean_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - sim_start).count();

        view.set_wireframe(input.wireframe);

        // --- draw ---------------------------------------------------------
        std::uint32_t image_index = 0;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (!ctx.begin_frame(image_index, cmd)) continue;
        last_image = image_index;

        // Read back BEFORE view.record() below resets and rewrites this
        // frame slot's timestamp pair - begin_frame() just waited on this
        // slot's fence, so the PREVIOUS frame that used it (kFramesInFlight
        // frames ago) is guaranteed complete and its GPU time is valid now.
        const double gpu_ms = ctx.last_gpu_ms();

        const float aspect = static_cast<float>(ctx.extent.width) /
                             static_cast<float>(ctx.extent.height);
        const vkm::Mat4 proj = vkm::perspective(1.05f, aspect, 0.3f, 8000.0f);
        const vkm::Mat4 viewm = vkm::look_at(
            input.camera.position, input.camera.position + fwd, {0.0f, 1.0f, 0.0f});

        viewer::Globals globals{};
        globals.view_proj     = proj * viewm;
        globals.inv_view_proj = vkm::inverse(globals.view_proj);
        globals.cam_pos[0] = input.camera.position.x;
        globals.cam_pos[1] = input.camera.position.y;
        globals.cam_pos[2] = input.camera.position.z;
        globals.sun_dir[0] = sun.x;
        globals.sun_dir[1] = sun.y;
        globals.sun_dir[2] = sun.z;
        globals.cascade_patch[0] = levels[0].patch_length;
        globals.cascade_patch[1] = (levels.size() > 1) ? levels[1].patch_length : levels[0].patch_length;
        globals.cascade_patch[2] = (levels.size() > 2) ? levels[2].patch_length : levels[0].patch_length;
        globals.cascade_patch[3] = static_cast<float>(opt.size);  // cascade grid resolution N
        globals.params[0]  = static_cast<float>(sim_time);
        globals.params[1]  = 0.0f;
        globals.params[2]  = 0.0f;
        globals.params[3]  = 0.0f;
        globals.shading[0] = 1.0f;     // foam strength
        globals.shading[1] = 1.15f;    // exposure
        // Fog density: tuned so far geometry (out to the clipmap's outermost
        // ring footprint) fogs into raw sky_color before its sharp sun-glare
        // term can wash out a wide band of near-horizon pixels at once - see
        // ADR-020's viewer notes. Re-tuned for the clipmap in ADR-021.
        globals.shading[2] = 0.0015f; // fog density
        globals.shading[3] = input.choppiness;

        view.record(ctx, cmd, image_index, ctx.frame_index, stack, globals);
        ctx.end_frame(image_index);

        // --- HUD ----------------------------------------------------------
        ocean_ms_avg = ocean_ms_avg * 0.95 + ocean_ms * 0.05;
        // gpu_ms is 0 for the first couple of frames (see last_gpu_ms), which
        // would drag the running average down artificially - skip those.
        if (gpu_ms > 0.0) gpu_ms_avg = gpu_ms_avg * 0.95 + gpu_ms * 0.05;
        fps_avg = fps_avg * 0.95 + (dt > 0.0 ? 1.0 / dt : 0.0) * 0.05;
        if (++frame_counter % 15 == 0) {
            const float h = stack.height_at(input.camera.position.x,
                                            input.camera.position.z);
            char title[256];
            std::snprintf(title, sizeof(title),
                          "oceanlib  |  %.0f fps  |  ocean %.2f ms  |  gpu %.2f ms  |  "
                          "%ux%u  |  chop %.2f  |  water under camera %+.2f m%s",
                          fps_avg, ocean_ms_avg, gpu_ms_avg, opt.size, opt.size,
                          input.choppiness, h, input.paused ? "  [PAUSED]" : "");
            glfwSetWindowTitle(window, title);
        }

        if (!opt.screenshot.empty() && frame_counter >= opt.frames) {
            if (capture_frame(ctx, last_image, opt.screenshot.c_str())) {
                std::printf("wrote %s (%ux%u)\n", opt.screenshot.c_str(),
                            ctx.extent.width, ctx.extent.height);
            }
            std::printf("gpu frame time: %.3f ms (exponential moving average, decay 0.95)  |  %u triangles/frame\n",
                        gpu_ms_avg, view.triangle_count());
            break;
        }
    }

    view.shutdown(ctx);
    ctx.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
