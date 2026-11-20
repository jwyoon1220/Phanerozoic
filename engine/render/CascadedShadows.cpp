#include "engine/render/CascadedShadows.hpp"

#include <cstdlib>

#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/SkyRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace phx {

CascadedShadows::CascadedShadows(VulkanContext& ctx, Bindless& bindless) : ctx_(ctx), bindless_(bindless) {
    for (u32 c = 0; c < kCascades; ++c) {
        maps_[c] = createImage2D(ctx, {kSize, kSize}, VK_FORMAT_D32_SFLOAT,
                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        bindless.setSampled(kTexCascade0 + c, maps_[c].view, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    }
}

CascadedShadows::~CascadedShadows() {
    for (Image& m : maps_) destroyImage(ctx_, m);
}

void CascadedShadows::update(const glm::dvec3& cam, const glm::dvec3& fwd, const glm::dvec3& right, const glm::dvec3& up,
                             f32 fovY, f32 aspect, const glm::dvec3& sun) {
    const f64 ty = std::tan(f64(fovY) * 0.5), tx = ty * aspect;
    // Light basis (world axes are fine: everything is relative to the camera position).
    const glm::dvec3 lz = glm::normalize(sun);
    const glm::dvec3 lx = glm::normalize(glm::cross(std::abs(lz.z) < 0.99 ? glm::dvec3(0, 0, 1) : glm::dvec3(1, 0, 0), lz));
    const glm::dvec3 ly = glm::cross(lz, lx);
    f64 nearD = 0.0;
    for (u32 c = 0; c < kCascades; ++c) {
        const f64 farD = kSplits[c];
        // Bounding sphere of the frustum slice (radius rounded up: stable across rotation).
        const f64 mid = (nearD + farD) * 0.5;
        const glm::dvec3 centre = fwd * mid;
        const glm::dvec3 corner = fwd * farD + right * (farD * tx) + up * (farD * ty);
        f64 radius = std::ceil(glm::length(corner - centre) * 1.05);
        const f64 texel = 2.0 * radius / kSize;
        // Snap the centre to the texel grid in a world-locked light frame (double).
        const glm::dvec3 world = cam + centre;
        f64 px = glm::dot(world, lx), py = glm::dot(world, ly);
        px = std::floor(px / texel) * texel;
        py = std::floor(py / texel) * texel;
        const f64 pz = glm::dot(world, lz);
        const glm::dvec3 snapped = lx * px + ly * py + lz * pz - cam; // camera-relative
        // Casters up to 3 km towards the sun (crowns on the ridge above), receivers behind.
        const f64 back = 3000.0;
        const glm::dvec3 eye = snapped + lz * back;
        const glm::mat4 view = glm::lookAtRH(glm::vec3(eye), glm::vec3(snapped), glm::vec3(ly));
        const glm::mat4 proj = glm::orthoRH_ZO(f32(-radius), f32(radius), f32(-radius), f32(radius), 1.0f, f32(back + radius));
        u_.lightViewProj[c] = proj * view;
        u_.texel[c] = f32(texel);
        nearD = farD;
    }
    u_.splits = glm::vec4(kSplits[0], kSplits[1], kSplits[2], (enabled_ && glm::dot(sun, glm::normalize(cam)) > -0.05) ? 1.0f : 0.0f);
    u_.texel.w = f32(kSize);
}

void CascadedShadows::begin(VkCommandBuffer cmd, u32 c) {
    imageBarrier(cmd, maps_[c].image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                 initialised_[c] ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    initialised_[c] = true;
    VkRenderingAttachmentInfo dep{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    dep.imageView = maps_[c].view;
    dep.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dep.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    dep.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    dep.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {kSize, kSize}};
    ri.layerCount = 1;
    ri.pDepthAttachment = &dep;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0, 0, f32(kSize), f32(kSize), 0, 1};
    VkRect2D sc{{0, 0}, {kSize, kSize}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

void CascadedShadows::end(VkCommandBuffer cmd, u32 c) {
    vkCmdEndRendering(cmd);
    imageBarrier(cmd, maps_[c].image, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
}

} // namespace phx
