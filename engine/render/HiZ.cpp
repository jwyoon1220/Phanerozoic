#include "engine/render/HiZ.hpp"

#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/SkyRenderer.hpp"

#include <algorithm>

namespace phx {

HiZ::HiZ(VulkanContext& ctx, Bindless& bindless) : ctx_(ctx), bindless_(bindless) {
    pipe_ = createComputePipeline(ctx, bindless, "hiz.comp");
}

HiZ::~HiZ() {
    destroy();
    vkDestroyPipeline(ctx_.device, pipe_, nullptr);
}

void HiZ::destroy() {
    for (u32 i = 0; i < mips_; ++i) vkDestroyImageView(ctx_.device, mipViews_[i], nullptr);
    if (image_) vmaDestroyImage(ctx_.allocator, image_, alloc_);
    image_ = VK_NULL_HANDLE;
    mips_ = 0;
}

void HiZ::resize(VkExtent2D depth) {
    destroy();
    // Mip 0 is half resolution: texel (x, y) of mip m covers depth pixels [x·2^(m+1), (x+1)·2^(m+1)).
    extent_ = {(depth.width + 1) / 2, (depth.height + 1) / 2};
    u32 w = extent_.width, h = extent_.height;
    mips_ = 1;
    while ((w > 1 || h > 1) && mips_ < kMaxMips) w = (w + 1) / 2, h = (h + 1) / 2, ++mips_;

    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R32_SFLOAT;
    ci.extent = {extent_.width, extent_.height, 1};
    ci.mipLevels = mips_;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(ctx_.allocator, &ci, &ai, &image_, &alloc_, nullptr));
    for (u32 m = 0; m < mips_; ++m) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image_;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R32_SFLOAT;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1};
        VK_CHECK(vkCreateImageView(ctx_.device, &vi, nullptr, &mipViews_[m]));
        bindless_.setStorageR32(m, mipViews_[m]);
    }
    initialised_ = false;
}

void HiZ::record(VkCommandBuffer cmd) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    b.oldLayout = initialised_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image_;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips_, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
    initialised_ = true;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0, nullptr);
    u32 w = extent_.width, h = extent_.height;
    for (u32 m = 0; m < mips_; ++m) {
        PushConstants pc{0, 0, 0, m, 0};
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo d2{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        d2.memoryBarrierCount = 1;
        d2.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &d2);
        w = (w + 1) / 2, h = (h + 1) / 2;
    }
}

} // namespace phx
