#pragma once
#include "engine/core/TripleBuffer.hpp"
#include "engine/gpu/Resources.hpp"
#include "engine/gpu/Swapchain.hpp"
#include "engine/gpu/Vk.hpp"
#include "engine/jobs/JobSystem.hpp"
#include "engine/render/FrameSnapshot.hpp"

#include <atomic>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace phx {
class VulkanContext;
class SkyRenderer;
class TerrainRenderer;
class HeightField;
class GpuProfiler;
class TerrainShadows;
class HiZ;
class VegetationRenderer;
class CascadedShadows;
class PerfOverlay;
class Upscaler;

struct RendererConfig {
    PresentPolicy present = PresentPolicy::Vsync;
    f64 starEpochYear = 2026.0;
    std::string screenshotPath; // used when FrameSnapshot::screenshotRequest changes
    const HeightField* heights = nullptr; // shared with the supervisor (read-only, thread-safe)
};

// Render thread (plan §2.1): the only thread that touches VkQueue submit/present
// and the swapchain. Command recording is fanned out to workers, each using its
// own VkCommandPool per frame-in-flight; this thread gathers and submits.
class Renderer {
public:
    static constexpr u32 kFramesInFlight = 2;

    Renderer(VulkanContext& ctx, jobs::JobSystem& jobs, TripleBuffer<FrameSnapshot>& snapshots, RendererConfig cfg);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void start();
    void stop();
    [[nodiscard]] u32 screenshotsWritten() const { return screenshotsWritten_.load(std::memory_order_acquire); }

private:
    struct ThreadCommandPool {
        VkCommandPool pool = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> buffers;
        u32 used = 0;
    };
    struct FrameSlot {
        VkSemaphore imageAcquired = VK_NULL_HANDLE;
        u64 timelineValue = 0;
        std::vector<ThreadCommandPool> pools; // indexed by JobSystem::threadIndex()
    };

    void threadMain();
    void createFrameResources();
    void destroyFrameResources();
    void createTargets();
    void destroyTargets();
    bool renderFrame(const FrameSnapshot& snap);
    void recreateSwapchain(Extent2D extent);
    void setUpscaleMode(u8 mode);
    void waitTimeline(u64 value);
    VkCommandBuffer beginCommandBuffer(FrameSlot& slot);
    void recordFrame(VkCommandBuffer cmd, u32 imageIndex, u32 slot, u64 frameAddr, bool capture);
    void saveScreenshot();

    VulkanContext& ctx_;
    jobs::JobSystem& jobs_;
    TripleBuffer<FrameSnapshot>& snapshots_;
    RendererConfig cfg_;

    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<Bindless> bindless_;
    std::unique_ptr<SkyRenderer> sky_;
    std::unique_ptr<TerrainRenderer> terrain_;
    std::unique_ptr<GpuProfiler> profiler_;
    std::unique_ptr<TerrainShadows> shadows_;
    std::unique_ptr<HiZ> hiz_;
    std::unique_ptr<VegetationRenderer> vegetation_;
    std::unique_ptr<CascadedShadows> csm_;
    std::unique_ptr<PerfOverlay> overlay_;
    // Temporal upscaling (DLSS SR / DLAA): scene at render resolution → output resolution.
    std::unique_ptr<Upscaler> upscaler_;
    u8 mode_ = 0;                   // UpscaleMode
    VkExtent2D renderExtent_{};
    Image hdrOut_;                  // upscaled HDR (output resolution), stars drawn on top
    Image motion_;                  // camera motion vectors (render resolution)
    VkPipeline motionPipe_ = VK_NULL_HANDLE;
    glm::vec2 jitter_{0.0f};
    bool resetHistory_ = true;
    f32 lastFrameMs_ = 16.7f;
    glm::dvec3 lastCamera_{0.0};
    [[nodiscard]] bool upscaling() const;
    VkFormat overlayFormat_ = VK_FORMAT_UNDEFINED;
    bool overlayVisible_ = false;
    f64 lastCpu_[5]{};
    Image hdr_;
    Image depth_;
    VkPipeline tonemapPipe_ = VK_NULL_HANDLE;
    VkFormat tonemapFormat_ = VK_FORMAT_UNDEFINED;
    Buffer readback_;
    u32 lastScreenshotRequest_ = 0;
    std::atomic<u32> screenshotsWritten_{0};

    FrameSlot frames_[kFramesInFlight];
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    u64 frameNumber_ = 0;

    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace phx
