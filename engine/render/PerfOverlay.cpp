#include "engine/render/PerfOverlay.hpp"

#include "engine/gpu/GpuProfiler.hpp"
#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/SkyRenderer.hpp"

#include <stb_easy_font.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <format>
#include <unistd.h>

namespace phx {
namespace {
constexpr u32 kMaxQuads = 24000;

u32 rgba(u8 r, u8 g, u8 b, u8 a = 255) { return u32(r) | u32(g) << 8 | u32(b) << 16 | u32(a) << 24; }
const u32 kWhite = rgba(235, 235, 235), kGrey = rgba(150, 155, 160), kGreen = rgba(110, 220, 120),
          kYellow = rgba(240, 200, 80), kRed = rgba(240, 90, 80), kCyan = rgba(120, 200, 240);

u32 frameColor(f64 ms) { return ms <= 16.7 ? kGreen : (ms <= 33.4 ? kYellow : kRed); }

u64 readRss() {
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long size = 0, resident = 0;
    const int n = std::fscanf(f, "%lu %lu", &size, &resident);
    std::fclose(f);
    return n == 2 ? u64(resident) * u64(sysconf(_SC_PAGESIZE)) : 0;
}
std::string bytes(u64 b) {
    return b >= (1ull << 30) ? std::format("{:.2f} GiB", f64(b) / f64(1ull << 30)) : std::format("{:.0f} MiB", f64(b) / f64(1ull << 20));
}
} // namespace

PerfOverlay::PerfOverlay(VulkanContext& ctx, Bindless& bindless, VkFormat colorFormat, u32 framesInFlight)
    : ctx_(ctx), bindless_(bindless) {
    GraphicsPipelineDesc d{"overlay.vert", "overlay.frag", colorFormat};
    d.alphaBlend = true;
    pipe_ = createGraphicsPipeline(ctx, bindless, d);
    for (u32 i = 0; i < framesInFlight; ++i)
        buffers_.push_back(createBuffer(ctx, VkDeviceSize(kMaxQuads) * 4 * sizeof(Vertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        MemoryUsage::HostUpload));
    quadCounts_.assign(framesInFlight, 0);
    verts_.reserve(std::size_t(kMaxQuads) * 4);
    window_.reserve(1024);
}

PerfOverlay::~PerfOverlay() {
    vkDestroyPipeline(ctx_.device, pipe_, nullptr);
    for (Buffer& b : buffers_) destroyBuffer(ctx_, b);
}

void PerfOverlay::recordFrameInterval(f64 ms) {
    history_[historyHead_] = f32(ms);
    historyHead_ = (historyHead_ + 1) % u32(history_.size());
    window_.push_back(f32(ms));
    windowElapsed_ += ms;
    sampleClock_ += ms;
    if (windowElapsed_ >= 1000.0) {
        std::sort(window_.begin(), window_.end());
        fps_ = f64(window_.size()) * 1000.0 / windowElapsed_;
        avgMs_ = windowElapsed_ / f64(window_.size());
        p99Ms_ = window_[std::min(window_.size() - 1, window_.size() * 99 / 100)];
        maxMs_ = window_.back();
        minMs_ = window_.front();
        window_.clear();
        windowElapsed_ = 0;
    }
}

void PerfOverlay::quad(f32 x0, f32 y0, f32 x1, f32 y1, u32 c) {
    if (verts_.size() / 4 >= kMaxQuads) return;
    Vertex v{0, 0, 0, {u8(c), u8(c >> 8), u8(c >> 16), u8(c >> 24)}};
    for (auto [x, y] : {std::pair{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}) {
        v.x = x, v.y = y;
        verts_.push_back(v);
    }
}

f32 PerfOverlay::textWidth(const std::string& s) const { return f32(stb_easy_font_width(const_cast<char*>(s.c_str()))) * scale_; }

void PerfOverlay::text(f32 x, f32 y, const std::string& s, u32 c) {
    static thread_local std::vector<Vertex> tmp(std::size_t(8192) * 4);
    u8 col[4] = {u8(c), u8(c >> 8), u8(c >> 16), u8(c >> 24)};
    const int quads = stb_easy_font_print(0, 0, const_cast<char*>(s.c_str()), col, tmp.data(), int(tmp.size() * sizeof(Vertex)));
    // 1 px drop shadow, then the glyphs (stb emits 4 vertices per quad).
    for (int pass = 0; pass < 2; ++pass)
        for (int q = 0; q < quads && verts_.size() / 4 < kMaxQuads; ++q)
            for (int k = 0; k < 4; ++k) {
                Vertex v = tmp[std::size_t(q) * 4 + k];
                v.x = x + v.x * scale_ + (pass == 0 ? scale_ * 0.5f : 0.0f);
                v.y = y + v.y * scale_ + (pass == 0 ? scale_ * 0.5f : 0.0f);
                if (pass == 0) v.rgba[0] = v.rgba[1] = v.rgba[2] = 0, v.rgba[3] = 200;
                verts_.push_back(v);
            }
}

void PerfOverlay::build(u32 slot, VkExtent2D extent, bool perf, const PlayerHud& hud, const PerfCpuTimes& cpu,
                        const GpuProfiler& gpu, const PerfWorldStats& world) {
    verts_.clear();
    scale_ = std::max(1.0f, std::round(f32(extent.height) / 720.0f));
    auto smooth = [](f64& a, f64 b) { a += (b - a) * 0.05; };
    smooth(cpuSmooth_.gpuWait, cpu.gpuWait), smooth(cpuSmooth_.acquire, cpu.acquire), smooth(cpuSmooth_.update, cpu.update);
    smooth(cpuSmooth_.record, cpu.record), smooth(cpuSmooth_.submit, cpu.submit);

    if (sampleClock_ >= 500.0 || rssBytes_ == 0) {
        rssBytes_ = readRss();
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(ctx_.allocator, budgets);
        const VkPhysicalDeviceMemoryProperties* mp = nullptr;
        vmaGetMemoryProperties(ctx_.allocator, &mp);
        vramUsage_ = vramBudget_ = vmaBytes_ = 0;
        for (u32 h = 0; h < mp->memoryHeapCount; ++h)
            if (mp->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                vramUsage_ += budgets[h].usage;
                vramBudget_ += budgets[h].budget;
                vmaBytes_ += budgets[h].statistics.blockBytes;
            }
        jobsPerSecond_ = f64(world.jobsExecuted - lastJobs_) * 1000.0 / std::max(sampleClock_, 1.0);
        lastJobs_ = world.jobsExecuted;
        sampleClock_ = 0;
    }

    if (hud.visible) buildHud(extent, hud);
    if (perf) {
    // ---- lines (label, value, colour)
    struct Line {
        std::string s;
        u32 color;
    };
    std::vector<Line> lines;
    lines.push_back({std::format("{:.0f} FPS  {:.2f} ms", fps_, avgMs_), frameColor(avgMs_)});
    lines.push_back({std::format("min {:.1f}  p99 {:.1f}  max {:.1f} ms", minMs_, p99Ms_, maxMs_), frameColor(p99Ms_)});
    const f64 cpuBusy = cpuSmooth_.update + cpuSmooth_.record + cpuSmooth_.submit;
    lines.push_back({std::format("CPU render {:.2f} ms", cpuBusy), kWhite});
    lines.push_back({std::format("upd {:.2f} rec {:.2f} sub {:.2f}", cpuSmooth_.update, cpuSmooth_.record, cpuSmooth_.submit), kGrey});
    lines.push_back({std::format("wait gpu {:.2f} acq {:.2f}", cpuSmooth_.gpuWait, cpuSmooth_.acquire), kGrey});
    f64 gpuTotal = 0;
    for (const auto& [n, ms] : gpu.smoothed()) gpuTotal += ms;
    lines.push_back({std::format("GPU {:.2f} ms", gpuTotal), frameColor(gpuTotal)});
    for (const auto& [n, ms] : gpu.smoothed())
        if (ms >= 0.01) lines.push_back({std::format("{} {:.2f}", n, ms), kGrey});
    lines.push_back({std::format("RAM {}", bytes(rssBytes_)), kCyan});
    if (ctx_.caps.memoryBudget)
        lines.push_back({std::format("VRAM {} / {}", bytes(vramUsage_), bytes(vramBudget_)),
                         vramUsage_ > vramBudget_ * 9 / 10 ? kRed : kCyan});
    lines.push_back({std::format("VRAM engine {}", bytes(vmaBytes_)), kGrey});
    lines.push_back({std::format("tiles {} / {} ({} pend)", world.terrainDrawn, world.terrainResident, world.terrainPending), kGrey});
    lines.push_back({std::format("nature {}k in {} cells", world.natureInstances / 1000, world.natureCells), kGrey});
    lines.push_back({std::format("workers {}  jobs {:.0f}/s", world.workers, jobsPerSecond_), kGrey});

    // ---- layout (top right)
    const f32 pad = 6.0f * scale_, lineH = 10.0f * scale_;
    const f32 graphW = 240.0f * scale_ / 2.0f * 1.0f, graphH = 40.0f * scale_;
    f32 width = graphW;
    for (const Line& l : lines) width = std::max(width, textWidth(l.s));
    const f32 x1 = f32(extent.width) - pad, x0 = x1 - width - 2 * pad, y0 = pad;
    const f32 y1 = y0 + pad + graphH + pad + f32(lines.size()) * lineH + pad;
    quad(x0, y0, x1, y1, rgba(10, 12, 16, 150));

    // Frame-time graph: one bar per frame, 0–33 ms, lines at 16.7 and 33.3 ms.
    const f32 gx0 = x0 + pad, gy1 = y0 + pad + graphH, gw = width;
    const u32 n = u32(history_.size());
    const f32 barW = gw / f32(n);
    for (u32 i = 0; i < n; ++i) {
        const f32 ms = history_[(historyHead_ + i) % n];
        if (ms <= 0) continue;
        const f32 h = std::min(ms / 33.3f, 1.0f) * graphH;
        quad(gx0 + f32(i) * barW, gy1 - h, gx0 + f32(i + 1) * barW, gy1, frameColor(ms));
    }
    quad(gx0, gy1 - graphH * 0.5f, gx0 + gw, gy1 - graphH * 0.5f + 1, rgba(255, 255, 255, 90));
    quad(gx0, gy1 - graphH, gx0 + gw, gy1 - graphH + 1, rgba(255, 255, 255, 60));

    f32 y = gy1 + pad;
    for (const Line& l : lines) {
        text(x1 - pad - textWidth(l.s), y, l.s, l.color); // right aligned
        y += lineH;
    }

    }

    quadCounts_[slot] = u32(verts_.size() / 4);
    std::memcpy(buffers_[slot].mapped, verts_.data(), verts_.size() * sizeof(Vertex));
    vmaFlushAllocation(ctx_.allocator, buffers_[slot].allocation, 0, VK_WHOLE_SIZE);
}

void PerfOverlay::buildHud(VkExtent2D extent, const PlayerHud& h) {
    const f32 pad = 6.0f * scale_, lineH = 10.0f * scale_;
    const f32 barW = 120.0f * scale_, barH = 5.0f * scale_, labelW = 50.0f * scale_;
    struct Bar {
        const char* label;
        f32 value;
        u32 color;
    };
    const Bar bars[] = {{"HEALTH", h.health, rgba(220, 70, 60)},
                        {"STAMINA", h.stamina, rgba(235, 200, 80)},
                        {"FOOD", h.energy, rgba(200, 140, 80)},
                        {"WATER", h.water, rgba(90, 170, 240)}};
    const std::string status = std::format("{}  {:.1f} m/s  slope {:.0f} deg  {:.0f} W", h.mode, h.speed, h.slopeDeg, h.powerW);
    const std::string temps = std::format("air {:.1f} C   body {:.1f} C", h.airTempC, h.coreTempC);
    const f32 w = std::max({labelW + barW, textWidth(status), textWidth(temps)});
    const f32 x0 = pad, y1 = f32(extent.height) - pad;
    const f32 y0 = y1 - (pad * 2 + 4 * lineH + 2 * lineH);
    quad(x0, y0, x0 + w + 2 * pad, y1, rgba(10, 12, 16, 140));
    f32 y = y0 + pad;
    for (const Bar& b : bars) {
        text(x0 + pad, y, b.label, kGrey);
        const f32 bx = x0 + pad + labelW, by = y + (lineH - barH) * 0.5f - scale_;
        quad(bx, by, bx + barW, by + barH, rgba(255, 255, 255, 40));
        const f32 v = std::clamp(b.value, 0.0f, 100.0f) / 100.0f;
        quad(bx, by, bx + barW * v, by + barH, v < 0.2f ? kRed : b.color);
        y += lineH;
    }
    text(x0 + pad, y, status, kWhite);
    text(x0 + pad, y + lineH, temps, h.coreTempC > 39.0f || h.coreTempC < 35.8f ? kRed : kGrey);
    // Centre dot.
    const f32 cx = f32(extent.width) * 0.5f, cy = f32(extent.height) * 0.5f, r = scale_;
    quad(cx - r, cy - r, cx + r, cy + r, rgba(255, 255, 255, 160));
}

void PerfOverlay::record(VkCommandBuffer cmd, u32 slot, VkExtent2D extent) {
    if (!quadCounts_[slot]) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0, nullptr);
    PushConstants pc{0, buffers_[slot].address, extent.width, extent.height};
    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd, quadCounts_[slot] * 6, 1, 0, 0);
}

} // namespace phx
