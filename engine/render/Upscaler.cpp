#include "engine/render/Upscaler.hpp"

#include "engine/core/Log.hpp"
#include "engine/gpu/VulkanContext.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>

#ifdef PHX_HAS_DLSS
#include <nvsdk_ngx_helpers_vk.h>
#endif

namespace phx {

const char* upscaleModeName(UpscaleMode m) {
    switch (m) {
    case UpscaleMode::Native: return "native";
    case UpscaleMode::DLAA: return "DLAA";
    case UpscaleMode::Quality: return "DLSS Quality";
    case UpscaleMode::Balanced: return "DLSS Balanced";
    case UpscaleMode::Performance: return "DLSS Performance";
    case UpscaleMode::UltraPerformance: return "DLSS Ultra Performance";
    default: return "?";
    }
}

bool parseUpscaleMode(const char* t, UpscaleMode& out) {
    struct Entry {
        const char* name;
        UpscaleMode mode;
    };
    static constexpr Entry kNames[] = {{"off", UpscaleMode::Native},          {"native", UpscaleMode::Native},
                                       {"dlaa", UpscaleMode::DLAA},           {"quality", UpscaleMode::Quality},
                                       {"balanced", UpscaleMode::Balanced},   {"performance", UpscaleMode::Performance},
                                       {"ultra", UpscaleMode::UltraPerformance}};
    for (const Entry& e : kNames)
        if (!std::strcmp(t, e.name)) return out = e.mode, true;
    return false;
}

glm::vec2 jitterHalton(u64 frame, VkExtent2D render, VkExtent2D display) {
    auto halton = [](u32 i, u32 b) {
        f32 f = 1.0f, r = 0.0f;
        for (; i > 0; i /= b) f /= f32(b), r += f * f32(i % b);
        return r;
    };
    const f32 ratio = f32(display.width) / f32(std::max(render.width, 1u));
    const u32 phases = std::max(8u, u32(std::ceil(8.0f * ratio * ratio)));
    const u32 i = u32(frame % phases) + 1;
    return {halton(i, 2) - 0.5f, halton(i, 3) - 0.5f};
}

#ifdef PHX_HAS_DLSS

namespace {
// Project id for NGX_ENGINE_TYPE_CUSTOM (any stable GUID identifies the application).
constexpr const char* kProjectId = "6f3c1d2a-8e4b-4c7d-9a51-0b2e7f9c3d18";

NVSDK_NGX_PerfQuality_Value perfQuality(UpscaleMode m) {
    switch (m) {
    case UpscaleMode::DLAA: return NVSDK_NGX_PerfQuality_Value_DLAA;
    case UpscaleMode::Quality: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    case UpscaleMode::Balanced: return NVSDK_NGX_PerfQuality_Value_Balanced;
    case UpscaleMode::Performance: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    case UpscaleMode::UltraPerformance: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    default: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }
}

// NGX_DLSS_GET_OPTIMAL_SETTINGS from nvsdk_ngx_helpers.h, which cannot be included on
// Linux (it pulls in the D3D and CUDA helpers unconditionally).
bool optimalRenderSize(NVSDK_NGX_Parameter* p, VkExtent2D display, NVSDK_NGX_PerfQuality_Value q, VkExtent2D& out) {
    void* callback = nullptr;
    NVSDK_NGX_Parameter_GetVoidPointer(p, NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback, &callback);
    if (!callback) return false;
    NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_Width, display.width);
    NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_Height, display.height);
    NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_PerfQualityValue, q);
    NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_RTXValue, false);
    if (NVSDK_NGX_FAILED(reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback>(callback)(p))) return false;
    unsigned w = 0, h = 0;
    NVSDK_NGX_Parameter_GetUI(p, NVSDK_NGX_Parameter_OutWidth, &w);
    NVSDK_NGX_Parameter_GetUI(p, NVSDK_NGX_Parameter_OutHeight, &h);
    if (!w || !h) return false;
    out = {w, h};
    return true;
}

std::wstring executableDir() {
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return (ec ? std::filesystem::current_path() : exe.parent_path()).wstring();
}
} // namespace

struct Upscaler::Ngx {
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    UpscaleMode featureMode = UpscaleMode::Native;
    VkExtent2D featureRender{}, featureDisplay{};
    std::wstring appDir;
    std::vector<const wchar_t*> paths;
    NVSDK_NGX_FeatureCommonInfo common{};
};

void Upscaler::instanceExtensions(std::vector<const char*>& out) {
    unsigned int ni = 0, nd = 0;
    const char** inst = nullptr;
    const char** dev = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &inst, &nd, &dev))) return;
    for (unsigned i = 0; i < ni; ++i) out.push_back(inst[i]);
}

void Upscaler::deviceExtensions(std::vector<const char*>& out) {
    unsigned int ni = 0, nd = 0;
    const char** inst = nullptr;
    const char** dev = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &inst, &nd, &dev))) return;
    for (unsigned i = 0; i < nd; ++i) out.push_back(dev[i]);
}

Upscaler::Upscaler(VulkanContext& ctx) : ctx_(ctx), ngx_(std::make_unique<Ngx>()) {
    ngx_->appDir = executableDir();
    ngx_->paths = {ngx_->appDir.c_str()};
    ngx_->common.PathListInfo.Path = ngx_->paths.data();
    ngx_->common.PathListInfo.Length = u32(ngx_->paths.size());
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", ngx_->appDir.c_str(), ctx.instance, ctx.physicalDevice, ctx.device,
        vkGetInstanceProcAddr, vkGetDeviceProcAddr, &ngx_->common, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        status_ = std::format("NGX init failed (0x{:08x}): not an NVIDIA RTX GPU or driver without NGX", u32(r));
        log::warn("{}", status_);
        return;
    }
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&ngx_->params);
    if (NVSDK_NGX_FAILED(r) || !ngx_->params) {
        status_ = std::format("NGX capability query failed (0x{:08x})", u32(r));
        log::warn("{}", status_);
        return;
    }
    int supported = 0, needsDriver = 0;
    unsigned int minMajor = 0, minMinor = 0;
    NVSDK_NGX_Parameter_GetI(ngx_->params, NVSDK_NGX_Parameter_SuperSampling_Available, &supported);
    NVSDK_NGX_Parameter_GetI(ngx_->params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver);
    NVSDK_NGX_Parameter_GetUI(ngx_->params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minMajor);
    NVSDK_NGX_Parameter_GetUI(ngx_->params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minMinor);
    if (!supported) {
        status_ = needsDriver ? std::format("DLSS needs driver ≥ {}.{}", minMajor, minMinor)
                              : std::string("DLSS not supported on this GPU");
        log::warn("{}", status_);
        return;
    }
    available_ = true;
    status_ = std::format("DLSS SR/DLAA available (SDK {}, runtime from {})", PHX_DLSS_VERSION,
                          std::filesystem::path(ngx_->appDir).string());
    log::info("{}", status_);
}

Upscaler::~Upscaler() {
    releaseFeature();
    if (ngx_ && ngx_->params) NVSDK_NGX_VULKAN_DestroyParameters(ngx_->params);
    if (available_ || (ngx_ && ngx_->params)) NVSDK_NGX_VULKAN_Shutdown1(ctx_.device);
}

VkExtent2D Upscaler::renderExtent(UpscaleMode mode, VkExtent2D display) const {
    if (!available_ || mode == UpscaleMode::Native) return display;
    VkExtent2D r;
    return optimalRenderSize(ngx_->params, display, perfQuality(mode), r) ? r : display;
}

void Upscaler::releaseFeature() {
    if (ngx_ && ngx_->feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(ngx_->feature);
        ngx_->feature = nullptr;
    }
}

bool Upscaler::evaluate(VkCommandBuffer cmd, UpscaleMode mode, const UpscaleInputs& in) {
    if (!available_ || mode == UpscaleMode::Native) return false;
    Ngx& n = *ngx_;
    const bool sameShape = n.feature && n.featureMode == mode && n.featureRender.width == in.render.width &&
                           n.featureRender.height == in.render.height && n.featureDisplay.width == in.display.width &&
                           n.featureDisplay.height == in.display.height;
    if (!sameShape) {
        releaseFeature(); // the caller recreated its targets with the GPU idle
        NVSDK_NGX_DLSS_Create_Params cp{};
        cp.Feature.InWidth = in.render.width;
        cp.Feature.InHeight = in.render.height;
        cp.Feature.InTargetWidth = in.display.width;
        cp.Feature.InTargetHeight = in.display.height;
        cp.Feature.InPerfQualityValue = perfQuality(mode);
        // Pre-exposed HDR, reverse-Z depth, low-resolution unjittered motion vectors;
        // DLSS adapts exposure itself (our tonemapper exposes after it).
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                  NVSDK_NGX_DLSS_Feature_Flags_DepthInverted | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        // Default presets: the newest model the installed runtime/driver provides.
        for (const char* p : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                              NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                              NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance})
            NVSDK_NGX_Parameter_SetUI(n.params, p, NVSDK_NGX_DLSS_Hint_Render_Preset_Default);
        const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(ctx_.device, cmd, 1, 1, &n.feature, n.params, &cp);
        if (NVSDK_NGX_FAILED(r) || !n.feature) {
            log::error("DLSS feature creation failed (0x{:08x}); falling back to native", u32(r));
            available_ = false;
            n.feature = nullptr;
            return false;
        }
        n.featureMode = mode, n.featureRender = in.render, n.featureDisplay = in.display;
        log::info("DLSS feature: {} {}x{} → {}x{}", upscaleModeName(mode), in.render.width, in.render.height, in.display.width,
                  in.display.height);
    }

    const VkImageSubresourceRange color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageSubresourceRange depth{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    NVSDK_NGX_Resource_VK colorRes = NVSDK_NGX_Create_ImageView_Resource_VK(in.color->view, in.color->image, color, in.colorFormat,
                                                                            in.render.width, in.render.height, false);
    NVSDK_NGX_Resource_VK depthRes = NVSDK_NGX_Create_ImageView_Resource_VK(in.depth->view, in.depth->image, depth, in.depthFormat,
                                                                            in.render.width, in.render.height, false);
    NVSDK_NGX_Resource_VK mvRes = NVSDK_NGX_Create_ImageView_Resource_VK(in.motion->view, in.motion->image, color, in.motionFormat,
                                                                         in.render.width, in.render.height, false);
    NVSDK_NGX_Resource_VK outRes = NVSDK_NGX_Create_ImageView_Resource_VK(in.output->view, in.output->image, color, in.colorFormat,
                                                                          in.display.width, in.display.height, true);
    NVSDK_NGX_VK_DLSS_Eval_Params ep{};
    ep.Feature.pInColor = &colorRes;
    ep.Feature.pInOutput = &outRes;
    ep.pInDepth = &depthRes;
    ep.pInMotionVectors = &mvRes;
    ep.InJitterOffsetX = in.jitterPx.x;
    ep.InJitterOffsetY = in.jitterPx.y;
    ep.InRenderSubrectDimensions = {in.render.width, in.render.height};
    ep.InReset = in.reset ? 1 : 0;
    ep.InMVScaleX = f32(in.render.width); // motion vectors are stored in UV units
    ep.InMVScaleY = f32(in.render.height);
    ep.InFrameTimeDeltaInMsec = in.frameMs;
    const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, n.feature, n.params, &ep);
    if (NVSDK_NGX_FAILED(r)) {
        static bool warned = false;
        if (!warned) log::error("DLSS evaluate failed (0x{:08x})", u32(r)), warned = true;
        return false;
    }
    return true;
}

#else // !PHX_HAS_DLSS

struct Upscaler::Ngx {};
void Upscaler::instanceExtensions(std::vector<const char*>&) {}
void Upscaler::deviceExtensions(std::vector<const char*>&) {}
Upscaler::Upscaler(VulkanContext& ctx) : ctx_(ctx) {}
Upscaler::~Upscaler() = default;
VkExtent2D Upscaler::renderExtent(UpscaleMode, VkExtent2D display) const { return display; }
bool Upscaler::evaluate(VkCommandBuffer, UpscaleMode, const UpscaleInputs&) { return false; }
void Upscaler::releaseFeature() {}

#endif

} // namespace phx
