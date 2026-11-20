#include "engine/render/TerrainShadows.hpp"

#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/SkyRenderer.hpp"
#include "engine/world/HeightField.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace phx {
namespace {
constexpr u32 kBands = 32; // jobs per map
}

TerrainShadows::TerrainShadows(VulkanContext& ctx, Bindless& bindless, jobs::JobSystem& jobs,
                               const HeightField& heights, u32 framesInFlight)
    : ctx_(ctx), bindless_(bindless), jobs_(jobs), heights_(heights), framesInFlight_(framesInFlight) {
    for (u32 l = 0; l < 2; ++l) {
        Level& lv = levels_[l];
        lv.image = createImage2D(ctx, {kSize, kSize}, VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        lv.staging = createBuffer(ctx, VkDeviceSize(kSize) * kSize * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  MemoryUsage::HostUpload);
        bindless.setSampled(l == 0 ? kTexShadowNear : kTexShadowFar, lv.image.view);
        lv.visibility = createImage2D(ctx, {kSize, kSize}, VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        bindless.setSampled(l == 0 ? kTexShadowVisNear : kTexShadowVisFar, lv.visibility.view, VK_IMAGE_LAYOUT_GENERAL);
        bindless.setStorage(l == 0 ? kStorShadowVisNear : kStorShadowVisFar, lv.visibility.view);
    }
    visPipe_ = createComputePipeline(ctx, bindless, "shadowvis.comp");
}

TerrainShadows::~TerrainShadows() {
    for (Level& lv : levels_) {
        if (lv.generating) jobs_.wait(lv.counter); // jobs write into the staging buffer
        destroyImage(ctx_, lv.image);
        destroyImage(ctx_, lv.visibility);
        destroyBuffer(ctx_, lv.staging);
    }
    vkDestroyPipeline(ctx_.device, visPipe_, nullptr);
}

void TerrainShadows::startGeneration(u32 l, const glm::dvec3& anchor) {
    Level& lv = levels_[l];
    lv.pendingAnchor = anchor;
    lv.generating = true;
    const glm::dmat3 toEcef = glm::transpose(ecefToEnuMatrix(anchor));
    lv.gen = {anchor, toEcef[0], toEcef[1], kExtent[l], kExtent[l] / kSize};
    const GenParams* g = &lv.gen;
    lv.maxHeightBits.store(0, std::memory_order_relaxed);
    std::atomic<u32>* maxBits = &lv.maxHeightBits;
    auto* out = static_cast<f32*>(lv.staging.mapped);
    const HeightField* hf = &heights_;
    for (u32 b = 0; b < kBands; ++b) {
        const u32 y0 = b * kSize / kBands, y1 = (b + 1) * kSize / kBands;
        jobs_.submit({.priority = jobs::Priority::Background, .signal = &lv.counter}, [hf, out, g, y0, y1, maxBits] {
            f32 bandMax = 0.0f;
            for (u32 y = y0; y < y1; ++y)
                for (u32 x = 0; x < kSize; ++x) {
                    const f64 ex = (f64(x) + 0.5) * g->texel - g->extent * 0.5;
                    const f64 ny = g->extent * 0.5 - (f64(y) + 0.5) * g->texel; // row 0 = north
                    const glm::dvec3 dir = glm::normalize(g->anchor + g->east * ex + g->north * ny);
                    const f32 h = f32(std::max(hf->height(dir, g->texel), 0.0));
                    out[std::size_t(y) * kSize + x] = h;
                    bandMax = std::max(bandMax, h);
                }
            // Non-negative floats order like their bit patterns: an integer max suffices.
            const u32 bits = std::bit_cast<u32>(bandMax);
            u32 cur = maxBits->load(std::memory_order_relaxed);
            while (cur < bits && !maxBits->compare_exchange_weak(cur, bits, std::memory_order_relaxed)) {}
        });
    }
}

void TerrainShadows::update(const glm::dvec3& cam, const glm::dvec3& sunEcef, u64 frame) {
    // Visibility depends on the sun direction: recompute after ~0.1° of motion.
    if (glm::dot(sunEcef, lastSun_) < std::cos(0.1 * 3.14159265358979 / 180.0)) {
        lastSun_ = sunEcef;
        visDirty_ = true;
    }
    const glm::dvec3 anchorNow = glm::normalize(cam) * kPlanetRadiusM;
    for (u32 l = 0; l < 2; ++l) {
        Level& lv = levels_[l];
        if (lv.generating && lv.counter.done()) {
            // The copy is recorded in this frame's command buffer ahead of the terrain
            // pass, so this frame's uniforms switch to the new anchor together with it.
            lv.generating = false;
            lv.uploadPending = true;
            lv.anchor = lv.pendingAnchor;
            lv.valid = true;
            lv.maxHeight = std::bit_cast<f32>(lv.maxHeightBits.load(std::memory_order_relaxed));
            visDirty_ = true;
        }
        if (lv.generating || lv.uploadPending || frame < lv.reuseFrame) continue;
        // Re-centre once the camera moved an eighth of the map away from its anchor.
        const f64 moved = glm::length(anchorNow - (lv.valid ? lv.anchor : glm::dvec3(0.0)));
        if (!lv.valid || moved > kExtent[l] / 8.0) startGeneration(l, anchorNow);
    }
    for (Level& lv : levels_)
        if (lv.uploadPending) lv.reuseFrame = frame + framesInFlight_;
}

void TerrainShadows::recordUploads(VkCommandBuffer cmd, u64 frameAddr) {
    for (u32 l = 0; l < 2; ++l) {
        Level& lv = levels_[l];
        const bool first = !layoutsInitialised_;
        if (!lv.uploadPending && !first) continue;
        const VkImageLayout from = first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageBarrier(cmd, lv.image.image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, from,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        if (lv.uploadPending) {
            vmaFlushAllocation(ctx_.allocator, lv.staging.allocation, 0, VK_WHOLE_SIZE);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {kSize, kSize, 1};
            vkCmdCopyBufferToImage(cmd, lv.staging.buffer, lv.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &region);
            lv.uploadPending = false;
        } else {
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, lv.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        }
        imageBarrier(cmd, lv.image.image, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    const bool firstVis = !layoutsInitialised_;
    layoutsInitialised_ = true;

    if (!visDirty_ && !firstVis) return;
    visDirty_ = false;
    for (Level& lv : levels_) // previous frames may still sample the visibility maps
        imageBarrier(cmd, lv.visibility.image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     firstVis ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0,
                            nullptr);
    for (u32 l = 0; l < 2; ++l) {
        PushConstants pc{frameAddr, 0, 0, l, 0};
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, kSize / 8, kSize / 8, 1);
    }
    for (Level& lv : levels_)
        imageBarrier(cmd, lv.visibility.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
}

ShadowMapUniforms TerrainShadows::uniforms(const glm::dvec3& cam) const {
    ShadowMapUniforms u{};
    bool any = false;
    for (u32 l = 0; l < 2; ++l) {
        const Level& lv = levels_[l];
        u.anchorEnu[l] = glm::mat4(glm::mat3(ecefToEnuMatrix(lv.valid ? lv.anchor : cam)));
        const glm::dvec3 off = (lv.valid ? lv.anchor : cam) - cam;
        u.anchorOffset[l] = glm::vec4(glm::vec3(off), lv.valid ? f32(kExtent[l]) : 0.0f);
        any |= lv.valid;
    }
    const f32 maxH = std::max(levels_[0].valid ? levels_[0].maxHeight : 0.0f, levels_[1].valid ? levels_[1].maxHeight : 0.0f);
    u.params = glm::vec4(any ? 1.0f : 0.0f, f32(kExtent[0] / kSize), f32(kExtent[1] / kSize), maxH);
    return u;
}

} // namespace phx
