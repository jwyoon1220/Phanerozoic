#pragma once
// Temporal upscaling / anti-aliasing (plan §2.8): NVIDIA DLSS Super Resolution and DLAA
// through the NGX SDK (Linux path: NGX directly; Streamline is Windows-only).
//  * "Highest accessible version": the newest DLSS runtime the build ships (SDK
//    PHX_DLSS_VERSION) with the Default render preset, so the driver's/OTA's newest
//    model (transformer presets today) is used; availability is queried at run time
//    and the engine falls back to native rendering.
//  * Inputs: pre-exposed HDR colour, reverse-Z depth and camera motion vectors at render
//    resolution, sub-pixel jitter (Halton 2,3). Output: HDR at display resolution.
#include "engine/core/Types.hpp"
#include "engine/gpu/Resources.hpp"

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace phx {
class VulkanContext;

enum class UpscaleMode : u8 { Native = 0, DLAA, Quality, Balanced, Performance, UltraPerformance, Count };
const char* upscaleModeName(UpscaleMode m);
bool parseUpscaleMode(const char* text, UpscaleMode& out);

struct UpscaleInputs {
    const Image* color = nullptr;  // render resolution, SHADER_READ_ONLY_OPTIMAL
    const Image* depth = nullptr;  // render resolution, D32 reverse-Z
    const Image* motion = nullptr; // render resolution, xy = previous − current (UV units)
    const Image* output = nullptr; // display resolution, GENERAL, storage
    VkFormat colorFormat = VK_FORMAT_UNDEFINED, depthFormat = VK_FORMAT_UNDEFINED, motionFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D render{}, display{};
    glm::vec2 jitterPx{0.0f}; // sample offset from the pixel centre, render pixels (+x right, +y down)
    bool reset = false;
    f32 frameMs = 16.7f;
};

class Upscaler {
public:
    // Extensions NGX needs, to be enabled when the instance / device are created.
    static void instanceExtensions(std::vector<const char*>& out);
    static void deviceExtensions(std::vector<const char*>& out);

    explicit Upscaler(VulkanContext& ctx);
    ~Upscaler();
    Upscaler(const Upscaler&) = delete;
    Upscaler& operator=(const Upscaler&) = delete;

    [[nodiscard]] bool dlssAvailable() const { return available_; }
    [[nodiscard]] const std::string& status() const { return status_; }

    // Render resolution for a mode (NGX optimal settings); display for Native.
    [[nodiscard]] VkExtent2D renderExtent(UpscaleMode mode, VkExtent2D display) const;

    // Records DLSS into `cmd` (creates the feature for this mode/resolution on first use).
    // Returns false if DLSS could not run (caller should then present the input directly).
    bool evaluate(VkCommandBuffer cmd, UpscaleMode mode, const UpscaleInputs& in);
    // Drops the feature; call with the GPU idle (resize, mode change).
    void releaseFeature();

private:
    struct Ngx;
    VulkanContext& ctx_;
    std::unique_ptr<Ngx> ngx_;
    bool available_ = false;
    std::string status_ = "DLSS not compiled in";
};

// Halton(2,3) sub-pixel jitter in [-0.5, 0.5]² for a frame; the phase count grows with
// the upscale factor (≥ 8 samples per output pixel).
glm::vec2 jitterHalton(u64 frame, VkExtent2D render, VkExtent2D display);

} // namespace phx
