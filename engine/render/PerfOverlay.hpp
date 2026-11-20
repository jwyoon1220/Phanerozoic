#pragma once
// Performance overlay (top right, F3) and the on-foot HUD (bottom left): frame rate and frame-time graph, render-thread
// CPU breakdown, GPU pass timings, process RAM and VRAM. Text from stb_easy_font is
// built on the render thread into a host-visible quad buffer and drawn in one call at
// the end of the tonemap pass.
#include "engine/gpu/Resources.hpp"
#include "engine/render/FrameSnapshot.hpp"

#include <array>
#include <string>
#include <vector>

namespace phx {
class GpuProfiler;

struct PerfCpuTimes {   // render thread, ms
    f64 gpuWait = 0, acquire = 0, update = 0, record = 0, submit = 0;
};
struct PerfWorldStats {
    u32 terrainDrawn = 0, terrainResident = 0, terrainPending = 0;
    u32 natureInstances = 0, natureCells = 0;
    u32 workers = 0;
    u64 jobsExecuted = 0; // cumulative
};

class PerfOverlay {
public:
    PerfOverlay(VulkanContext& ctx, Bindless& bindless, VkFormat colorFormat, u32 framesInFlight);
    ~PerfOverlay();

    void recordFrameInterval(f64 ms);                                   // once per presented frame
    // Builds this slot's quads (render thread, before recording).
    void build(u32 slot, VkExtent2D extent, bool perf, const PlayerHud& hud, const PerfCpuTimes& cpu, const GpuProfiler& gpu,
               const PerfWorldStats& world);
    void record(VkCommandBuffer cmd, u32 slot, VkExtent2D extent);     // inside the swapchain pass

private:
    struct Vertex {
        f32 x, y, z;
        u8 rgba[4];
    };
    void quad(f32 x0, f32 y0, f32 x1, f32 y1, u32 rgba);
    void text(f32 x, f32 y, const std::string& s, u32 rgba);
    [[nodiscard]] f32 textWidth(const std::string& s) const;
    void buildHud(VkExtent2D extent, const PlayerHud& hud);

    VulkanContext& ctx_;
    Bindless& bindless_;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    std::vector<Buffer> buffers_;
    std::vector<u32> quadCounts_;
    std::vector<Vertex> verts_;
    f32 scale_ = 2.0f;

    // Frame-time history (graph) and 1 s statistics window.
    std::array<f32, 240> history_{};
    u32 historyHead_ = 0;
    std::vector<f32> window_;
    f64 windowElapsed_ = 0;
    f64 fps_ = 0, avgMs_ = 0, p99Ms_ = 0, maxMs_ = 0, minMs_ = 0;
    // Slow-changing values, refreshed every 0.5 s.
    f64 sampleClock_ = 0;
    u64 rssBytes_ = 0;
    u64 vramUsage_ = 0, vramBudget_ = 0, vmaBytes_ = 0;
    u64 lastJobs_ = 0;
    f64 jobsPerSecond_ = 0;
    PerfCpuTimes cpuSmooth_{};
};

} // namespace phx
