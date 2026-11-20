#pragma once
// Hierarchical depth (min of reverse-Z depth = farthest) built from this frame's terrain
// depth: terrain is drawn first, so mountains occlude vegetation in the same frame — no
// reprojection, no disocclusion errors (plan §2.7, two-pass culling simplified).
#include "engine/gpu/Resources.hpp"

namespace phx {

class HiZ {
public:
    static constexpr u32 kMaxMips = 16;

    HiZ(VulkanContext& ctx, Bindless& bindless);
    ~HiZ();
    void resize(VkExtent2D depthExtent); // render thread, GPU idle
    // Depth must be in SHADER_READ_ONLY (sampled at kTexDepth); leaves the pyramid in GENERAL.
    void record(VkCommandBuffer cmd);
    [[nodiscard]] u32 mipCount() const { return mips_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }

private:
    void destroy();
    VulkanContext& ctx_;
    Bindless& bindless_;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation alloc_ = VK_NULL_HANDLE;
    VkImageView mipViews_[kMaxMips]{};
    VkExtent2D extent_{};
    u32 mips_ = 0;
    bool initialised_ = false;
};

} // namespace phx
