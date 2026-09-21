// oceanlib Vulkan viewer.
//
//   WASD / QE   move          mouse (hold RMB)  look
//   Shift       move faster   Tab               wireframe
//   Space       pause time    Esc               quit
//   1 / 2       choppiness    R                 reset camera
//
//   --size N          ocean grid resolution (default 256)
//   --mesh M          mesh quads per tile edge (default 256)
//   --tiles T         tiles per side of the outer (far-cascade) mesh, odd (default 7)
//   --cascades N      number of cascade scales, 1-3 (default 3; ADR-020)
//   --interaction N   interaction field resolution, power of two (default 256)
//   --cam-x/y/z V     camera position; --cam-yaw/--cam-pitch V its heading
//   --isolate         start with the interaction field shown on its own
//   --no-foam-advect  fall back to the instantaneous Jacobian foam
//   --no-detail-fade  keep every cascade at full strength to the horizon
//   --no-slope-var    stop the faded cascades' slope variance from feeding the
//                     specular lobe, leaving the distant sea a mirror (ADR-023)
//   --foam-decay R    foam decay rate, 1/s (default 0.30, half-life 2.3 s)
//   --foam-gain G     how fast breaking injects foam, 1/s (default 4)
//   --foam-advect S   multiplier on the advecting current (default 1)
//   --turbidity T     Preetham sky turbidity: 2 arctic, 3 clear, 6 hazy
//   --sun-elev R      sun elevation in radians above the horizon
//   --sun-azim R      sun azimuth in radians, 0 = +X
//   --splash N        drop a scripted rock at the focus point on frame N, so
//                     --screenshot can capture ripples without a mouse. The
//                     position and frame are fixed, and --screenshot already
//                     advances on a fixed virtual timestep, so the captured
//                     image is reproducible across machines.
//
// Interaction (ADR-021):
//   LEFT CLICK        throw a rock from the camera; it arcs under gravity and
//                     splashes where it lands on the DISPLACED surface
//   - / =             impulse strength      [ / ]   impulse radius
//
// A boat (floaters.hpp) floats nearby, its hull sampling and reacting to the
// same displaced surface the renderer draws - see Boat's class comment.
//   I                 show the interaction field on its own, without the swell
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

#include "floaters.hpp"
#include "ocean_view.hpp"
#include "vk_context.hpp"
#include "vk_math.hpp"

#include "ocean/cascade.hpp"
#include "ocean/foam.hpp"
#include "ocean/interaction.hpp"
#include "ocean/ocean.h"    // for ocean_simd_level(); the viewer
                            // deliberately uses only public headers
#include "ocean/ocean.hpp"

#include <algorithm>
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
    std::uint32_t mesh   = 256;
    std::uint32_t tiles  = 7;
    float         wind   = 12.0f;
    float         depth  = 0.0f;   // <= 0 = deep water
    float         chop   = 1.0f;
    // 0.6 produced literally zero foam at ordinary choppiness - the Jacobian
    // simply never falls that far - which meant the foam path was dead code in
    // the demo. Measured: at threshold 0.6 coverage is 0.0000; at 0.85 it is a
    // few per cent, concentrated on breaking crests, which is what foam is.
    float         foam   = 0.85f;
    bool          wireframe = false;
    std::string   screenshot;
    std::string   simd;           // "" = native max; else scalar|sse2|avx2|neon
    std::uint32_t threads = 0;    // 0 = one worker per hardware thread
    std::uint32_t cascades = 3;   // 1-3, see ADR-020
    std::uint32_t interaction = 256;  // interaction field resolution (ADR-021)
    int           splash = -1;    // frame to inject a scripted impulse on, -1 = never
    bool          isolate = false;  // start with the interaction field shown alone
    // Camera placement, so a capture can be framed from the command line.
    float         cam_x = 0.0f, cam_y = 18.0f, cam_z = 60.0f;
    float         cam_yaw = -1.6f, cam_pitch = -0.18f;
    bool          cam_set = false;
    float         impulse = 0.45f, impulse_radius = 0.55f;
    bool          no_foam_advect = false;
    bool          no_detail_fade = false;
    bool          no_slope_var   = false;
    // Steady-state coverage under a sustained source is source_gain/decay, so
    // this ratio is the knob that decides whether foam reads as whitecaps or
    // as milk. 4/0.3 = 13x saturated the entire ocean; 1.5/0.4 = 3.75x lets a
    // genuinely breaking crest reach white while a briefly-folding one stays
    // faint.
    float         foam_decay  = 0.40f;
    float         foam_gain   = 1.50f;
    float         foam_advect = 1.0f;
    // Preetham turbidity: 2 is arctic-clear, 3 clear, 6 hazy, 10 murky.
    float         turbidity = 2.6f;
    float         sky_scale = 0.05f;
    float         sun_elev = 0.38f;    // radians above the horizon
    float         sun_azim = -2.08f;   // radians, 0 = +X
    int           frames = 90;
    bool          validation = true;
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
        else if (a == "--mesh")        o.mesh  = std::strtoul(next(), nullptr, 10);
        else if (a == "--tiles")       o.tiles = std::strtoul(next(), nullptr, 10);
        else if (a == "--wind")        o.wind  = std::strtof(next(), nullptr);
        else if (a == "--depth")       o.depth = std::strtof(next(), nullptr);
        else if (a == "--chop")        o.chop  = std::strtof(next(), nullptr);
        else if (a == "--foam")        o.foam  = std::strtof(next(), nullptr);
        else if (a == "--wireframe")   o.wireframe = true;
        else if (a == "--simd")        o.simd = next();
        else if (a == "--threads")     o.threads = std::strtoul(next(), nullptr, 10);
        else if (a == "--cascades")    o.cascades = std::strtoul(next(), nullptr, 10);
        else if (a == "--interaction") o.interaction = std::strtoul(next(), nullptr, 10);
        else if (a == "--splash")     o.splash = std::atoi(next());
        else if (a == "--isolate")    o.isolate = true;
        else if (a == "--impulse")        o.impulse = std::strtof(next(), nullptr);
        else if (a == "--impulse-radius") o.impulse_radius = std::strtof(next(), nullptr);
        else if (a == "--no-foam-advect") o.no_foam_advect = true;
        else if (a == "--no-detail-fade") o.no_detail_fade = true;
        else if (a == "--no-slope-var")   o.no_slope_var = true;
        else if (a == "--foam-decay")  o.foam_decay  = std::strtof(next(), nullptr);
        else if (a == "--foam-gain")   o.foam_gain   = std::strtof(next(), nullptr);
        else if (a == "--foam-advect") o.foam_advect = std::strtof(next(), nullptr);
        else if (a == "--turbidity")  o.turbidity = std::strtof(next(), nullptr);
        else if (a == "--sky-scale")  o.sky_scale = std::strtof(next(), nullptr);
        else if (a == "--sun-elev")   o.sun_elev  = std::strtof(next(), nullptr);
        else if (a == "--sun-azim")   o.sun_azim  = std::strtof(next(), nullptr);
        else if (a == "--cam-x")     { o.cam_x = std::strtof(next(), nullptr); o.cam_set = true; }
        else if (a == "--cam-y")     { o.cam_y = std::strtof(next(), nullptr); o.cam_set = true; }
        else if (a == "--cam-z")     { o.cam_z = std::strtof(next(), nullptr); o.cam_set = true; }
        else if (a == "--cam-yaw")   { o.cam_yaw = std::strtof(next(), nullptr); o.cam_set = true; }
        else if (a == "--cam-pitch") { o.cam_pitch = std::strtof(next(), nullptr); o.cam_set = true; }
        else if (a == "--screenshot")  o.screenshot = next();
        else if (a == "--frames")      o.frames = std::atoi(next());
        else if (a == "--no-validation") o.validation = false;
    }
    if (o.tiles % 2 == 0) ++o.tiles;  // must be odd to centre on the origin
    if (o.cascades < 1) o.cascades = 1;
    if (o.cascades > 3) o.cascades = 3;
    return o;
}

// --- sub-pixel slope statistics -------------------------------------------

// Mean-square slope of one cascade: the average of (dh/dx)^2 + (dh/dz)^2 over
// its whole patch, both axes summed.
//
// This is the number the BRDF needs for the detail it is NOT drawing. When
// detail_fade() drops a cascade because its texels have fallen below a pixel,
// the waves in that band do not stop existing - they stop being RESOLVED. Their
// geometry is correctly discarded; their slope variance is not the renderer's
// to throw away, because that variance is exactly what makes a distant sea
// look like a broad sheen rather than a hard mirror. Feeding it back in as
// roughness is the whole of ADR-023.
//
// Measured once per sea state rather than per frame, and that is not an
// optimisation but a property of the model: for a stationary Gaussian sea the
// slope variance is a spectral integral, the sum of k^2 S(k) over all k, which
// carries no time dependence. The sea surface moves; its slope statistics do
// not. So this is re-measured only when the stack is rebuilt - on a choppiness
// change, which does alter the displaced surface's slopes.
//
// Measured from the library's NORMAL BUFFER rather than from the spectrum in
// closed form, deliberately. The published normals are the exact
// displaced-surface normals with choppy displacement included (ADR-005); a
// closed-form spectral integral would describe the undisplaced Gaussian
// surface instead, and disagree with the surface actually on screen. Agreeing
// with what is drawn is the entire point of the exercise.
float mean_square_slope(const ocean::Buffers& b)
{
    if (b.normal == nullptr || b.size == 0) return 0.0f;

    const std::size_t n = static_cast<std::size_t>(b.size) * b.size;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float ny  = b.normal[i * 4 + 1];
        const float inv = 1.0f / std::max(ny, 1e-4f);
        const double sx = -static_cast<double>(b.normal[i * 4 + 0]) * inv;
        const double sz = -static_cast<double>(b.normal[i * 4 + 2]) * inv;
        sum += sx * sx + sz * sz;
    }
    return static_cast<float>(sum / static_cast<double>(n));
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

// --- ray marching against the displaced surface ---------------------------

// A conservative bound on |surface height| in metres.
//
// Used only to bracket the march. Too small and the ray can start below a
// crest and miss it; too large and the bracket is wider than it needs to be
// and costs a few extra steps. It is a constant rather than a measured
// per-frame maximum because scanning every cascade's height channel each frame
// would cost more than the march it is meant to accelerate, and the march is
// only ever driven by a mouse click.
constexpr float kSurfaceBound = 14.0f;

struct RayHit {
    vkm::Vec3 point{};
    bool      hit = false;
    int       steps = 0;
};

// March a ray against the DISPLACED ocean surface - not the flat y = 0 plane.
//
// ALGORITHM. The surface is single-valued in (x, z) almost everywhere, so
// define f(t) = ray(t).y - surface_height(ray(t).xz). f starts positive (the
// ray begins above the water) and we look for its first sign change, then
// refine.
//
// The march uses a step proportional to distance travelled rather than a fixed
// one, because perspective means a step that is one pixel wide near the camera
// is many pixels wide far away: a fixed world step wastes hundreds of samples
// in the foreground to pay for the horizon. The floor keeps the step from
// collapsing to nothing right in front of the eye.
//
// Refinement is the ILLINOIS method (modified regula falsi) rather than plain
// bisection or a secant step. It keeps the bracket, so it cannot diverge the
// way a secant can on a function this bumpy, while converging superlinearly
// instead of bisection's one bit per iteration. The halving of the retained
// endpoint's value is what stops the classic regula-falsi stall where one end
// never moves.
//
// FAILURE CASES, all of them real at grazing angles:
//
//  - A ray nearly parallel to the surface covers a huge horizontal distance
//    per unit of vertical drop, so between two samples it can pass clean over
//    a crest and come down the far side. The click then lands on a wave much
//    further away than the one under the cursor. Rejecting |D.y| below a
//    threshold bounds how bad this gets, at the cost of refusing near-horizon
//    clicks outright - which is better than silently dropping a rock 4 km away.
//  - A choppy surface genuinely folds at breaking crests, where it is not a
//    function of (x, z) at all. There f is multi-valued and "the first
//    crossing" is only approximately meaningful. This is the same limit
//    ADR-011's fixed-point inversion has, inherited.
//  - Far from the camera, float precision in both the ray parameterisation and
//    the surface query becomes comparable to the step, so the refinement stops
//    buying accuracy. The march distance is capped for that reason.
//
// The scalable fix for the first case is a max-mipmap of the height field with
// cone or hierarchical stepping, which can take provably safe long strides.
// That is a real renderer's answer; for a demo driven by mouse clicks, a
// distance-proportional march with a bounded step count is the honest trade.
RayHit raymarch_ocean(const ocean::WaterSurface& water, vkm::Vec3 origin,
                      vkm::Vec3 dir, float max_dist)
{
    RayHit out;
    dir = vkm::normalize(dir);

    // Looking up, or so close to horizontal that the failure above dominates.
    if (dir.y > -0.02f) return out;

    auto f_at = [&](float t) {
        const vkm::Vec3 p = origin + dir * t;
        return p.y - water.height_at(p.x, p.z);
    };

    // Bracket: enter at the top of the possible surface band, leave at its
    // bottom. Skipping the empty sky above the waves is free accuracy.
    float t0 = 0.0f;
    if (origin.y > kSurfaceBound) t0 = (origin.y - kSurfaceBound) / -dir.y;
    float t1 = (origin.y + kSurfaceBound) / -dir.y;
    if (t1 > max_dist) t1 = max_dist;
    if (t0 >= t1) return out;

    float ta = t0, fa = f_at(ta);
    if (fa < 0.0f) {                 // already under water at the entry point
        out.point = origin + dir * ta;
        out.hit   = true;
        return out;
    }

    constexpr int kMaxSteps = 512;
    float tb = ta, fb = fa;
    bool bracketed = false;
    for (int i = 0; i < kMaxSteps && ta < t1; ++i) {
        const float step = std::fmax(0.20f, ta * 0.01f);
        tb = std::fmin(ta + step, t1);
        fb = f_at(tb);
        out.steps = i + 1;
        if (fb <= 0.0f) { bracketed = true; break; }
        ta = tb; fa = fb;
    }
    if (!bracketed) return out;

    // Illinois refinement. 40 iterations is far more than needed - it
    // converges in about 6 - but each one is a single surface query and this
    // runs once per click, not once per pixel.
    for (int i = 0; i < 40; ++i) {
        const float denom = fb - fa;
        if (std::fabs(denom) < 1e-12f) break;
        float tc = (ta * fb - tb * fa) / denom;
        if (!(tc > std::fmin(ta, tb) && tc < std::fmax(ta, tb))) {
            tc = 0.5f * (ta + tb);       // fall back to bisection if it strays
        }
        const float fc = f_at(tc);
        if (std::fabs(fc) < 1e-4f) { ta = tc; break; }
        if ((fc < 0.0f) == (fb < 0.0f)) {
            tb = tc; fb = fc; fa *= 0.5f;
        } else {
            ta = tc; fa = fc; fb *= 0.5f;
        }
    }

    out.point = origin + dir * (0.5f * (ta + tb));
    out.hit   = true;
    return out;
}

// Unproject a pixel into a world-space ray.
vkm::Vec3 pixel_ray(const vkm::Mat4& inv_view_proj, double mx, double my,
                    std::uint32_t width, std::uint32_t height)
{
    // Vulkan NDC: x and y both in [-1, 1] with y pointing DOWN, which matches
    // GLFW's cursor origin at the top-left, so no flip is needed here.
    const float nx = 2.0f * static_cast<float>(mx) / static_cast<float>(width) - 1.0f;
    const float ny = 2.0f * static_cast<float>(my) / static_cast<float>(height) - 1.0f;

    auto unproject = [&](float z) {
        float v[4] = {nx, ny, z, 1.0f};
        float r[4] = {0, 0, 0, 0};
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                r[row] += inv_view_proj.m[col][row] * v[col];
            }
        }
        const float inv_w = 1.0f / r[3];
        return vkm::Vec3{r[0] * inv_w, r[1] * inv_w, r[2] * inv_w};
    };
    // Vulkan depth range is [0, 1], so the near plane is z = 0.
    return vkm::normalize(unproject(1.0f) - unproject(0.0f));
}

// --- input state ---------------------------------------------------------

struct Input {
    Camera camera;
    double last_x = 0.0, last_y = 0.0;
    bool   looking = false;
    bool   paused = false;
    bool   wireframe = false;
    float  choppiness = 1.0f;

    // Interaction debug controls.
    float  impulse_strength = 0.45f;
    float  impulse_radius   = 0.55f;
    bool   isolate          = false;
    bool   click_pending    = false;
    double click_x = 0.0, click_y = 0.0;
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
        // Interaction debug controls.
        case GLFW_KEY_I:      in->isolate = !in->isolate; break;
        case GLFW_KEY_MINUS:
            in->impulse_strength = std::fmax(0.02f, in->impulse_strength - 0.05f); break;
        case GLFW_KEY_EQUAL:
            in->impulse_strength = std::fmin(4.0f, in->impulse_strength + 0.05f); break;
        case GLFW_KEY_LEFT_BRACKET:
            in->impulse_radius = std::fmax(0.15f, in->impulse_radius - 0.05f); break;
        case GLFW_KEY_RIGHT_BRACKET:
            in->impulse_radius = std::fmin(3.0f, in->impulse_radius + 0.05f); break;
        default: break;
    }
}

void mouse_button_callback(GLFWwindow* w, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
    auto* in = static_cast<Input*>(glfwGetWindowUserPointer(w));
    // The click is only RECORDED here. Resolving it needs the view-projection
    // matrix and the ocean state, neither of which exists inside a GLFW
    // callback, so the actual ray march happens in the frame loop.
    glfwGetCursorPos(w, &in->click_x, &in->click_y);
    in->click_pending = true;
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
    input.isolate          = opt.isolate;
    input.impulse_strength = opt.impulse;
    input.impulse_radius   = opt.impulse_radius;
    if (opt.cam_set) {
        input.camera.position = {opt.cam_x, opt.cam_y, opt.cam_z};
        input.camera.yaw      = opt.cam_yaw;
        input.camera.pitch    = opt.cam_pitch;
    }
    input.wireframe  = opt.wireframe;
    glfwSetWindowUserPointer(window, &input);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    viewer::VkContext ctx;
    if (!ctx.init(window, opt.validation)) {
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
        // Persistent foam advects with the surface current, so the cascades
        // have to produce one. This is what the extra transforms buy.
        d.compute_velocity        = !opt.no_foam_advect;
        levels.push_back(d);
    }

    ocean::CascadeStack stack{std::span<const ocean::OceanDesc>(levels)};
    float active_choppiness = input.choppiness;

    // The local interaction field (ADR-021). 64 m across at 256^2 gives 0.25 m
    // cells, so the accurate band runs from about 0.5 m to 4 m wavelengths -
    // exactly the range a rock splash produces.
    ocean::InteractionDesc idesc;
    idesc.size         = opt.interaction;
    idesc.extent       = 64.0f;
    idesc.damping      = 0.25f;
    idesc.absorb_cells = 20;
    idesc.thread_count = opt.threads;
    ocean::InteractionField field{idesc};
    ocean::WaterSurface water{stack, field};

    // Warm the cascades to t=0 so the boat below can spawn already resting on
    // the REAL surface, before the frame loop's own first stack.update() runs.
    stack.update(0.0);

    // Slope variance per cascade, for the roughness the detail fade would
    // otherwise discard (ADR-023). Time-invariant for a given sea state, so it
    // is measured here and refreshed only when the stack is rebuilt.
    std::vector<float> slope_var(levels.size(), 0.0f);
    auto measure_slope_var = [&]() {
        for (std::size_t i = 0; i < levels.size(); ++i) {
            slope_var[i] = mean_square_slope(stack.buffers(i));
        }
    };
    measure_slope_var();
    if (opt.no_slope_var) {
        std::printf("sub-pixel slope variance: OFF (--no-slope-var) - unresolved "
                    "cascades contribute no roughness\n");
    }
    for (std::size_t i = 0; i < levels.size(); ++i) {
        std::printf("  cascade %zu mean-square slope %.5f "
                    "(rms slope %.2f deg) -> roughness alpha %.4f when unresolved\n",
                    i, slope_var[i],
                    std::atan(std::sqrt(slope_var[i] * 0.5f)) * 57.2958f,
                    std::sqrt(slope_var[i]));
    }

    // Props: a floating boat, and rocks thrown by left-clicking the water
    // (floaters.hpp). Placed ahead of the default camera so it is visible on
    // launch without having to go looking for it.
    viewer::Boat boat;
    boat.reset(-10.0f, -18.0f);
    // reset() cannot know the wave height on its own (it takes no water
    // surface), so it leaves position.y at 0 - fine for a flat sea, but on a
    // real swell that can be a metre or more off. Snapping it here removes
    // what used to read as the boat "getting submerged" the instant the
    // viewer launched: it was really the buoyancy spring yanking the hull
    // down from y=0 to wherever the actual local wave happened to be.
    boat.position.y = water.height_at(boat.position.x, boat.position.z);
    viewer::RockThrower rock_thrower;

    // One foam field per cascade, because foam is born from each scale's own
    // Jacobian and rides that scale's own current. The shader already combines
    // the levels with a screen blend, so nothing downstream changes.
    std::vector<ocean::FoamField> foam_fields;
    std::vector<const ocean::FoamField*> foam_ptrs;
    if (!opt.no_foam_advect) {
        ocean::FoamDesc fdesc;
        fdesc.decay        = opt.foam_decay;
        fdesc.source_gain  = opt.foam_gain;
        fdesc.advect_scale = opt.foam_advect;
        fdesc.thread_count = opt.threads;
        foam_fields.reserve(levels.size());
        for (std::size_t i = 0; i < levels.size(); ++i) {
            foam_fields.emplace_back(stack.level(i), fdesc);
        }
        for (auto& f : foam_fields) foam_ptrs.push_back(&f);
        std::printf("foam: persistent and advected, decay %.2f /s "
                    "(half-life %.1f s), gain %.1f /s, advect scale %.2f\n",
                    opt.foam_decay, 0.6931 / opt.foam_decay, opt.foam_gain,
                    opt.foam_advect);
    } else {
        std::printf("foam: instantaneous Jacobian threshold only "
                    "(--no-foam-advect)\n");
    }

    std::printf("interaction: %ux%u over %.0f m (dx = %.3f m), kernel P=%u, "
                "dt = %.4f s (limit %.4f s), dispersion error %.1f%%\n",
                idesc.size, idesc.size, idesc.extent,
                idesc.extent / idesc.size, idesc.kernel_radius,
                field.fixed_dt(), field.stable_dt_limit(),
                100.0f * field.kernel_dispersion_error());

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

    viewer::OceanView view;
    if (!view.init(ctx, levels, opt.mesh, opt.tiles, idesc.size)) {
        ctx.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    std::printf("mesh: %u quads/tile, %ux%u tiles (of the far cascade), %.2fM triangles/frame\n",
                opt.mesh, opt.tiles, opt.tiles,
                view.triangle_count() / 1.0e6);

    // Sun placed from elevation/azimuth rather than a hardcoded vector, so the
    // Preetham sky can actually be driven through a day - which is most of the
    // reason for having an analytic sky rather than a baked one.
    const vkm::Vec3 sun = vkm::normalize(
        {std::cos(opt.sun_elev) * std::cos(opt.sun_azim), std::sin(opt.sun_elev),
         std::cos(opt.sun_elev) * std::sin(opt.sun_azim)});

    double sim_time = 0.0;
    auto   last     = std::chrono::steady_clock::now();
    double ocean_ms_avg = 0.0;
    double inter_ms_avg = 0.0;
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
            // Choppy displacement changes the displaced surface's slopes, so
            // the variance the BRDF is standing in for has changed too. One
            // update is needed first: a freshly built stack has no state yet.
            stack.update(sim_time);
            measure_slope_var();
        }

        const auto sim_start = std::chrono::steady_clock::now();
        stack.update(sim_time);
        const double ocean_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - sim_start).count();

        // --- interaction --------------------------------------------------
        //
        // Recentre on the camera every frame. The move snaps to whole cells,
        // so it is an exact shift with nothing resampled - there is no jolt to
        // smooth over. Done BEFORE resolving the click so the impulse lands in
        // the grid the renderer will actually sample this frame.
        // Recentre on what the camera is LOOKING AT, not on where it is.
        //
        // The field is 64 m across, which is the right size for resolving
        // ripples (0.25 m cells) but small against a scene several kilometres
        // wide. Anchoring it to the camera puts it under the viewer's feet,
        // where nothing is happening, while the water they are actually
        // looking at - a hundred metres ahead - falls outside the grid and
        // silently swallows every impulse aimed at it. Anchoring it to the
        // view ray's meeting with the mean water level puts the resolution
        // where the attention is.
        const vkm::Vec3 cam_fwd = input.camera.forward();
        float focus_t = 60.0f;
        if (cam_fwd.y < -0.01f) focus_t = input.camera.position.y / -cam_fwd.y;
        focus_t = std::fmax(8.0f, std::fmin(focus_t, 90.0f));
        const vkm::Vec3 focus = input.camera.position + cam_fwd * focus_t;
        field.recenter(focus.x, focus.z);

        // Scripted splash for reproducible captures: same frame, same place,
        // same fixed virtual timestep, therefore the same pixels anywhere.
        if (opt.splash >= 0 && frame_counter == opt.splash) {
            ocean::Disturbance d;
            d.world_x    = focus.x;
            d.world_z    = focus.z;
            d.radius     = input.impulse_radius;
            d.strength   = input.impulse_strength;
            d.velocity_y = -4.0f * input.impulse_strength;
            field.add(d);
            std::printf("scripted splash at (%.1f, %.1f) on frame %d\n", focus.x, focus.z, opt.splash);
        }

        if (input.click_pending) {
            input.click_pending = false;

            // The view-projection from LAST frame is the one the user was
            // looking at when they clicked, and it is what is rebuilt below;
            // rebuilding it here keeps the ray consistent with the pixels that
            // were on screen.
            const float aspect_now = static_cast<float>(ctx.extent.width) /
                                     static_cast<float>(ctx.extent.height);
            const vkm::Mat4 proj_now =
                vkm::perspective(1.05f, aspect_now, 0.3f, 8000.0f);
            const vkm::Mat4 view_now =
                vkm::look_at(input.camera.position,
                             input.camera.position + input.camera.forward(),
                             {0.0f, 1.0f, 0.0f});
            const vkm::Mat4 inv_vp = vkm::inverse(proj_now * view_now);

            const vkm::Vec3 dir = pixel_ray(inv_vp, input.click_x, input.click_y,
                                            ctx.extent.width, ctx.extent.height);
            const RayHit h = raymarch_ocean(water, input.camera.position, dir,
                                            2000.0f);
            if (h.hit) {
                // Throw an actual rock rather than stamping an instant
                // disturbance: it leaves the camera, arcs under gravity, and
                // the splash below is added only once it lands (see
                // RockThrower::update) - so the ray march finds WHERE to aim,
                // not where to splash.
                rock_thrower.throw_at(input.camera.position, h.point, 0.22f);
                std::printf("threw a rock at (%.2f, %.2f, %.2f) after %d "
                            "march steps\n",
                            h.point.x, h.point.y, h.point.z, h.steps);
            } else {
                std::printf("no surface hit (ray too shallow, or past the "
                            "2 km march cap)\n");
            }
        }

        // --- props ----------------------------------------------------------
        //
        // Both read the CURRENT water state (this frame's cascades plus
        // whatever the interaction field still holds from last frame), so
        // they must run after stack.update() and before field.update() folds
        // this frame's new splashes in - the same ordering the click handling
        // above already relies on.
        // Drop the warm-up out of the GPU averages: the first frames pay for
        // pipeline creation, first-touch allocation and a cold clock state,
        // none of which is what a steady-state measurement is asking about.
        if (frame_counter == 60) ctx.gpu_reset_stats();

        boat.update(water, input.paused ? 0.0f : static_cast<float>(dt));

        // The hull pushes back on the water it is floating in. Boat::update
        // works out how hard from the probes it already took, and leaves the
        // disturbances here to submit - the same immediate-mode contract the
        // rock splashes below use, and the reason InteractionField takes
        // sources rather than handing out handles.
        //
        // These are Continuous sources, so they are re-submitted every frame
        // by design; the library scales them by the timestep, which is what
        // keeps the wake the same at any frame rate. They are only resolved
        // while the boat is inside the 64 m interaction field, which follows
        // what the camera is looking at - look at the boat and its wake is
        // there, which is exactly when it can be seen.
        for (const ocean::Disturbance& d : boat.wake()) field.add(d);

        for (const viewer::RockThrower::Splash& s :
             rock_thrower.update(water, input.paused ? 0.0f : static_cast<float>(dt))) {
            ocean::Disturbance d;
            d.world_x  = s.x;
            d.world_z  = s.z;
            d.radius   = input.impulse_radius;
            d.strength = input.impulse_strength;
            // The rock's own impact speed drives the velocity term, clamped
            // to the same range the debug slider produces - a real impact,
            // not an arbitrary constant, but still bounded against the field's
            // stability limits.
            d.velocity_y = std::clamp(-s.impact_speed, -8.0f, -0.5f);
            d.kind       = ocean::SourceKind::Impulse;
            field.add(d);
            std::printf("rock landed at (%.2f, %.2f), impact speed %.2f m/s\n",
                        s.x, s.z, s.impact_speed);
        }

        // Foam reads the ocean state, so it is stepped AFTER the stack has
        // advanced and never before.
        const auto foam_start = std::chrono::steady_clock::now();
        for (auto& f : foam_fields) {
            f.update(input.paused ? 0.0f : static_cast<float>(dt));
        }
        const double foam_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - foam_start).count();

        const auto inter_start = std::chrono::steady_clock::now();
        field.update(input.paused ? 0.0f : static_cast<float>(dt));
        const double inter_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - inter_start).count();

        view.set_wireframe(input.wireframe);

        // --- draw ---------------------------------------------------------
        std::uint32_t image_index = 0;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (!ctx.begin_frame(image_index, cmd)) continue;
        last_image = image_index;

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
        globals.cascade_patch[3] = static_cast<float>(opt.tiles);
        // A cascade the viewer is not running contributes no variance, so the
        // spare slots stay zero rather than repeating cascade 0's.
        for (std::size_t i = 0; i < 3; ++i) {
            globals.slope_var[i] = (!opt.no_slope_var && i < slope_var.size())
                                       ? slope_var[i] : 0.0f;
        }
        // Base roughness: what the BRDF uses where every cascade is fully
        // resolved and there is no unresolved detail to stand in for. Water
        // itself is very smooth at that point - this is close to a mirror, and
        // it is the value the specular lobe carried as a bare constant before
        // the sub-pixel term existed.
        globals.slope_var[3] = 0.055f * 0.055f;
        globals.params[0]  = static_cast<float>(sim_time);
        globals.params[1]  = static_cast<float>(opt.mesh);
        globals.params[2]  = opt.turbidity;
        globals.params[3]  = opt.sky_scale;
        globals.shading[0] = 1.0f;     // foam strength
        globals.shading[1] = 1.15f;    // exposure
        // Fog density was tuned for a 200 m single-patch ocean (visible extent
        // ~1400 m at 7 tiles). The far cascade is now 800 m (~5600 m visible),
        // so it is scaled up to match - see ADR-020's viewer notes: without
        // this, far geometry fogs into raw sky_color (including its sharp
        // sun-glare term) too slowly, and a wide band of near-horizon pixels
        // can all catch the glare spike at once, washing out the sky.
        globals.shading[2] = 0.0015f; // fog density
        globals.shading[3] = input.choppiness;

        // Detail fade needs each cascade's texel size and the angular size of
        // one pixel. tan(fov/2)*2/height is the world size a pixel covers per
        // metre of distance, which is what turns a texel size into a
        // texels-per-pixel ratio in the shader.
        for (std::uint32_t i = 0; i < 3; ++i) {
            const std::size_t lvl = std::min<std::size_t>(i, levels.size() - 1);
            globals.cascade_texel[i] =
                levels[lvl].patch_length / static_cast<float>(levels[lvl].size);
        }
        // 0 makes the shader's texels-per-pixel ratio diverge, so every
        // cascade stays fully on - the A/B control for the fade.
        globals.cascade_texel[3] =
            opt.no_detail_fade
                ? 0.0f
                : 2.0f * std::tan(0.5f * 1.05f) /
                      static_cast<float>(ctx.extent.height);

        // How far the camera is below the water, using the COMBINED surface so
        // that swimming under a wake counts. Queried once per frame, not per
        // pixel - the shader only needs the depth, and the surface it is under
        // is the same everywhere in the frame.
        const float surf_y = water.height_at(input.camera.position.x,
                                             input.camera.position.z);
        globals.water[0] = surf_y - input.camera.position.y;
        globals.water[1] = 0.0f;
        globals.water[2] = 0.0f;
        globals.water[3] = 0.0f;

        const ocean::InteractionBuffers ib = field.buffers();
        globals.interaction[0] = ib.origin_x;
        globals.interaction[1] = ib.origin_z;
        globals.interaction[2] = ib.extent;
        globals.interaction[3] = input.isolate ? 1.0f : 0.0f;

        std::vector<viewer::PropInstance> props;
        boat.append(props);
        rock_thrower.append(props);

        view.record(ctx, cmd, image_index, ctx.frame_index, stack, field,
                    foam_ptrs, globals, props);
        ctx.end_frame(image_index);

        // --- HUD ----------------------------------------------------------
        ocean_ms_avg =
            ocean_ms_avg * 0.95 + (ocean_ms + inter_ms + foam_ms) * 0.05;
        inter_ms_avg = inter_ms_avg * 0.95 + inter_ms * 0.05;
        fps_avg = fps_avg * 0.95 + (dt > 0.0 ? 1.0 / dt : 0.0) * 0.05;
        if (++frame_counter % 15 == 0) {
            // Queried through WaterSurface, so the number in the title bar
            // includes the wake - the whole point of the combined query.
            const float h = surf_y;

            // GPU cost, per pass, measured on the device rather than inferred
            // from the frame rate - which on a vsynced swapchain measures the
            // display and nothing else. Two frames stale by construction (see
            // VkContext::gpu_collect), which no running average can tell.
            char gpu_desc[160] = "";
            if (ctx.gpu_timing_supported() && !ctx.gpu_spans().empty()) {
                int n = std::snprintf(gpu_desc, sizeof(gpu_desc), "  |  gpu %.2f ms (",
                                      ctx.gpu_total_ms());
                const auto& spans = ctx.gpu_spans();
                for (std::size_t i = 0; i < spans.size() && n > 0 &&
                                        n < static_cast<int>(sizeof(gpu_desc)); ++i) {
                    n += std::snprintf(gpu_desc + n, sizeof(gpu_desc) - n, "%s%s %.2f",
                                       i ? " " : "",
                                       spans[i].name ? spans[i].name : "?", spans[i].ms);
                }
                if (n > 0 && n < static_cast<int>(sizeof(gpu_desc))) {
                    std::snprintf(gpu_desc + n, sizeof(gpu_desc) - n, ")");
                }
            }

            char title[512];
            std::snprintf(title, sizeof(title),
                          "oceanlib  |  %.0f fps  |  sim %.2f ms (wake %.2f)%s  |  "
                          "%ux%u  |  chop %.2f  |  water %+.2f m  |  "
                          "impulse %.2f m / r %.2f m  |  rocks in flight %zu%s%s",
                          fps_avg, ocean_ms_avg, inter_ms_avg, gpu_desc,
                          opt.size, opt.size,
                          input.choppiness, h,
                          input.impulse_strength, input.impulse_radius,
                          rock_thrower.in_flight(),
                          input.isolate ? "  [WAKE ONLY]" : "",
                          input.paused ? "  [PAUSED]" : "");
            glfwSetWindowTitle(window, title);
        }

        if (!opt.screenshot.empty() && frame_counter >= opt.frames) {
            // The GPU breakdown on the way out, so a headless capture run is
            // also a measurement run - the title bar is no use to a script.
            if (ctx.gpu_timing_supported() && !ctx.gpu_stats().empty()) {
                const viewer::GpuStat& tot = ctx.gpu_total_stat();
                std::printf("gpu over %llu frames (mean [min..max] ms):\n",
                            static_cast<unsigned long long>(tot.count));
                std::printf("  %-8s %6.3f  [%6.3f .. %6.3f]\n", "total",
                            tot.mean(), tot.min, tot.max);
                for (const viewer::GpuStat& st : ctx.gpu_stats()) {
                    std::printf("  %-8s %6.3f  [%6.3f .. %6.3f]\n",
                                st.name ? st.name : "?", st.mean(), st.min, st.max);
                }
            }
            for (std::size_t i = 0; i < foam_fields.size(); ++i) {
                const ocean::Buffers lb = stack.buffers(i);
                double instant = 0.0;
                const std::size_t cells =
                    static_cast<std::size_t>(lb.size) * lb.size;
                for (std::size_t c = 0; c < cells; ++c) {
                    instant += lb.displacement[4 * c + 3];
                }
                std::printf("  cascade %zu foam coverage: instantaneous %.4f, "
                            "persistent %.4f\n", i, instant / cells,
                            foam_fields[i].coverage());
            }
            if (capture_frame(ctx, last_image, opt.screenshot.c_str())) {
                std::printf("wrote %s (%ux%u)\n", opt.screenshot.c_str(),
                            ctx.extent.width, ctx.extent.height);
            }
            break;
        }
    }

    view.shutdown(ctx);
    ctx.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
