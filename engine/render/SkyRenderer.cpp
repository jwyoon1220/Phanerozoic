#include "engine/render/SkyRenderer.hpp"

#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/CascadedShadows.hpp"
#include "engine/render/TerrainShadows.hpp"
#include "engine/world/HeightField.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstring>

#ifndef PHX_DATA_DIR
#define PHX_DATA_DIR "data"
#endif

namespace phx {
namespace {
constexpr VkExtent2D kTransmittanceSize{256, 64};
constexpr VkExtent2D kMultiScatterSize{32, 32};
constexpr VkExtent2D kSkyViewSize{192, 108};
constexpr f32 kStarLimitMag = 8.0f;

u32 groups(u32 n) { return (n + 7) / 8; }

// Reverse-Z, infinite far plane, Vulkan clip space (y down).
glm::mat4 infiniteReverseZ(f32 fovY, f32 aspect, f32 zNear) {
    const f32 f = 1.0f / std::tan(fovY * 0.5f);
    glm::mat4 p(0.0f);
    p[0][0] = f / aspect;
    p[1][1] = -f;
    p[2][3] = -1.0f;
    p[3][2] = zNear;
    return p;
}

void computeToFragmentBarrier(VkCommandBuffer cmd, VkImage image) {
    imageBarrier(cmd, image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
}
} // namespace

SkyRenderer::SkyRenderer(VulkanContext& ctx, Bindless& bindless, u32 framesInFlight, f64 starEpochYear)
    : ctx_(ctx), bindless_(bindless) {
    const VkImageUsageFlags lutUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    transmittance_ = createImage2D(ctx, kTransmittanceSize, kHdrFormat, lutUsage);
    multiScatter_ = createImage2D(ctx, kMultiScatterSize, kHdrFormat, lutUsage);
    skyViewSun_ = createImage2D(ctx, kSkyViewSize, kHdrFormat, lutUsage);
    skyViewMoon_ = createImage2D(ctx, kSkyViewSize, kHdrFormat, lutUsage);
    // LUTs live in GENERAL: written by compute, sampled everywhere.
    const std::pair<u32, Image*> luts[] = {{kTexTransmittance, &transmittance_},
                                           {kTexMultiScatter, &multiScatter_},
                                           {kTexSkyViewSun, &skyViewSun_},
                                           {kTexSkyViewMoon, &skyViewMoon_}};
    for (auto [slot, img] : luts) {
        bindless.setSampled(slot, img->view, VK_IMAGE_LAYOUT_GENERAL);
        bindless.setStorage(slot, img->view);
    }

    transmittancePipe_ = createComputePipeline(ctx, bindless, "transmittance.comp");
    multiScatterPipe_ = createComputePipeline(ctx, bindless, "multiscatter.comp");
    skyViewPipe_ = createComputePipeline(ctx, bindless, "skyview.comp");
    // Sky and stars are drawn after terrain at depth 0 (infinity) with EQUAL: only
    // pixels the terrain did not cover are shaded, and mountains occlude stars.
    GraphicsPipelineDesc skyDesc{"fullscreen.vert", "sky.frag", kHdrFormat, false};
    skyDesc.depthFormat = kDepthFormat;
    skyDesc.depthTest = true;
    skyDesc.depthCompare = VK_COMPARE_OP_EQUAL;
    skyPipe_ = createGraphicsPipeline(ctx, bindless, skyDesc);
    GraphicsPipelineDesc starDesc = skyDesc;
    starDesc.vertex = "stars.vert";
    starDesc.fragment = "stars.frag";
    starDesc.additiveBlend = true;
    starsPipe_ = createGraphicsPipeline(ctx, bindless, starDesc);

    for (u32 i = 0; i < framesInFlight; ++i) {
        frameUniforms_.push_back(createBuffer(ctx, sizeof(FrameUniforms), 0, MemoryUsage::HostUpload));
        planets_.push_back(createBuffer(ctx, sizeof(astro::GpuStar) * astro::kPlanetCount,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::HostUpload));
    }

    loadStars(starEpochYear);
    precomputeStaticLuts();
}

SkyRenderer::~SkyRenderer() {
    for (VkPipeline p : {transmittancePipe_, multiScatterPipe_, skyViewPipe_, skyPipe_, starsPipe_})
        vkDestroyPipeline(ctx_.device, p, nullptr);
    for (Buffer& b : frameUniforms_) destroyBuffer(ctx_, b);
    for (Buffer& b : planets_) destroyBuffer(ctx_, b);
    destroyBuffer(ctx_, stars_);
    for (Image* img : {&transmittance_, &multiScatter_, &skyViewSun_, &skyViewMoon_}) destroyImage(ctx_, *img);
}

void SkyRenderer::loadStars(f64 year) {
    std::vector<astro::StarRecord> records;
    const std::string path = std::string(PHX_DATA_DIR) + "/stars.bin";
    std::vector<astro::GpuStar> gpu;
    if (astro::loadStarRecords(path, records)) {
        gpu = astro::buildGpuStars(records, year, kStarLimitMag);
        log::info("Stars: {} of {} catalog entries (V <= {:.1f}), proper motion applied to {:.2f}", gpu.size(),
                  records.size(), kStarLimitMag, year);
    } else {
        log::warn("Star catalog {} not found — sky will have no stars (build target phx_star_data)", path);
        gpu.push_back({}); // keep a valid buffer
    }
    starCount_ = records.empty() ? 0 : u32(gpu.size());
    const VkDeviceSize bytes = gpu.size() * sizeof(astro::GpuStar);
    stars_ = createBuffer(ctx_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          MemoryUsage::DeviceLocal);
    ImmediateSubmitter(ctx_).uploadBuffer(stars_, gpu.data(), bytes);
}

void SkyRenderer::precomputeStaticLuts() {
    ImmediateSubmitter(ctx_).run([&](VkCommandBuffer cmd) {
        for (Image* img : {&transmittance_, &multiScatter_, &skyViewSun_, &skyViewMoon_})
            imageBarrier(cmd, img->image, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        PushConstants pc{};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindless_.pipelineLayout, 0, 1, &bindless_.set,
                                0, nullptr);
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, transmittancePipe_);
        vkCmdDispatch(cmd, groups(kTransmittanceSize.width), groups(kTransmittanceSize.height), 1);
        computeToFragmentBarrier(cmd, transmittance_.image);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, multiScatterPipe_);
        vkCmdDispatch(cmd, groups(kMultiScatterSize.width), groups(kMultiScatterSize.height), 1);
        computeToFragmentBarrier(cmd, multiScatter_.image);
    });
    log::info("Atmosphere LUTs precomputed (transmittance {}x{}, multi-scattering {}x{})", kTransmittanceSize.width,
              kTransmittanceSize.height, kMultiScatterSize.width, kMultiScatterSize.height);
}

u64 SkyRenderer::updateFrame(u32 slot, const FrameSnapshot& snap, VkExtent2D extent, u64 frameCounter,
                             const ShadowMapUniforms& shadows, const CascadeUniforms& cascades) {
    const CameraState& cam = snap.camera;
    const glm::vec3 forward{std::sin(cam.yaw) * std::cos(cam.pitch), std::cos(cam.yaw) * std::cos(cam.pitch),
                            std::sin(cam.pitch)};
    const glm::mat4 view = glm::lookAtRH(glm::vec3(0.0f), forward, glm::vec3(0, 0, 1));
    const f32 aspect = f32(extent.width) / f32(extent.height);
    // Near plane from height above ground (supervisor): reverse-Z float keeps depth
    // precision anyway; this only avoids clipping the ground under the viewer's feet.
    const glm::mat4 proj = infiniteReverseZ(cam.fovY, aspect, cam.nearPlane);

    const glm::dmat3 ecefToEnu = ecefToEnuMatrix(cam.positionEcef);
    const f64 altitudeKm = (glm::length(cam.positionEcef) - kPlanetRadiusM) / 1000.0;

    FrameUniforms u{};
    u.viewProj = proj * view;
    u.ecefToEnu = glm::mat4(glm::mat3(ecefToEnu));
    u.viewProjEcef = u.viewProj * u.ecefToEnu;
    viewProjEcef_ = u.viewProjEcef;
    u.cameraEcef = glm::vec4(glm::vec3(cam.positionEcef), f32(kPlanetRadiusM));
    for (int l = 0; l < 2; ++l) {
        u.shadowEnu[l] = shadows.anchorEnu[l];
        u.shadowOffset[l] = shadows.anchorOffset[l];
    }
    u.shadowParams = shadows.params;
    for (u32 c = 0; c < 3; ++c) u.csm[c] = cascades.lightViewProj[c];
    u.csmSplits = cascades.splits;
    u.csmTexel = cascades.texel;
    u.ground = glm::vec4(snap.groundAlbedo, 0.0f);
    {
        // Large modulus keeps float precision (~4 mm) while wrapping only every 65 km;
        // power-of-two texture/noise scales tile it seamlessly.
        auto wrap = [](f64 v) { const f64 m = std::fmod(v, 65536.0); return f32(m < 0 ? m + 65536.0 : m); };
        u.cameraMod = glm::vec4(wrap(cam.positionEcef.x), wrap(cam.positionEcef.y), wrap(cam.positionEcef.z), 65536.0f);
    }
    {
        // Deciduous colour: turning from late September, peak around November, fading by
        // late December (no leaf-fall model yet).
        const f32 doy = snap.sky.dayOfYear;
        const f32 turn = glm::smoothstep(265.f, 310.f, doy) * (1.f - glm::smoothstep(345.f, 362.f, doy));
        const f32 t = f32(std::fmod(snap.simTime, 100000.0));
        const f32 gust = 0.75f + 0.25f * std::sin(t * 0.21f) * std::sin(t * 0.077f + 1.3f);
        u.world = glm::vec4(doy, t, gust, turn);
        u.windDir = glm::vec4(glm::normalize(glm::vec3(0.8f, 0.45f, 0.f)), 0.f); // prevailing from the WSW
        u.cullParams = glm::vec4(cam.nearPlane, f32(extent.height) / (2.f * std::tan(cam.fovY * 0.5f)), f32(extent.width),
                                 f32(extent.height));
    }
    u.invViewProj = glm::inverse(u.viewProj);
    u.icrsToEnu = glm::mat4(snap.sky.icrsToEnu);
    u.sunDir = glm::vec4(snap.sky.sunDir, snap.sky.sunRadius);
    u.moonDir = glm::vec4(snap.sky.moonDir, snap.sky.moonRadius);
    u.moonParams = glm::vec4(snap.sky.moonIlluminance, snap.sky.moonFraction, snap.sky.earthshine, 0.0f);
    u.camera = glm::vec4(snap.exposure, f32(std::fmod(snap.simTime, 3600.0)), f32(altitudeKm),
                         2.0f * std::tan(cam.fovY * 0.5f) / f32(extent.height));
    u.resolution = glm::vec4(extent.width, extent.height, 1.0f / extent.width, 1.0f / extent.height);
    u.tonemap = glm::vec4(snap.adaptationLuminance, f32(frameCounter % 4096), 0, 0);
    std::memcpy(frameUniforms_[slot].mapped, &u, sizeof(u));
    std::memcpy(planets_[slot].mapped, snap.sky.planets, sizeof(snap.sky.planets));
    for (Buffer* b : {&frameUniforms_[slot], &planets_[slot]}) vmaFlushAllocation(ctx_.allocator, b->allocation, 0, VK_WHOLE_SIZE);
    return frameUniforms_[slot].address;
}

void SkyRenderer::recordCompute(VkCommandBuffer cmd, u64 frameAddr) {
    // Previous frame's sky pass may still be sampling the sky-view LUTs.
    for (Image* img : {&skyViewSun_, &skyViewMoon_})
        imageBarrier(cmd, img->image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0,
                            nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skyViewPipe_);
    for (u32 light = 0; light < 2; ++light) {
        PushConstants pc{frameAddr, 0, 0, light};
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups(kSkyViewSize.width), groups(kSkyViewSize.height), 1);
    }
    for (Image* img : {&skyViewSun_, &skyViewMoon_}) computeToFragmentBarrier(cmd, img->image);
}

void SkyRenderer::recordSky(VkCommandBuffer cmd, u64 frameAddr, u32 slot) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0,
                            nullptr);
    PushConstants pc{frameAddr, 0, 0, 0};
    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipe_);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, starsPipe_);
    if (starCount_) {
        pc.stars = stars_.address;
        pc.starCount = starCount_;
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 6, starCount_, 0, 0);
    }
    pc.stars = planets_[slot].address;
    pc.starCount = astro::kPlanetCount;
    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 6, astro::kPlanetCount, 0, 0);
}

} // namespace phx
