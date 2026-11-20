#pragma once
#include "engine/gpu/Vk.hpp"
#include "engine/platform/Window.hpp"

#include <vector>

namespace phx {
class VulkanContext;

enum class PresentPolicy { Vsync, LowLatency, Uncapped }; // FIFO / MAILBOX / IMMEDIATE

// Owned and used exclusively by the render thread.
class Swapchain {
public:
    Swapchain(VulkanContext& ctx, Extent2D extent, PresentPolicy policy);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // Caller guarantees the GPU no longer uses the old images (render thread waits idle first).
    void recreate(Extent2D extent);

    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    // One per image: a present may still be reading a semaphore after the
    // frame slot that signalled it has been recycled.
    std::vector<VkSemaphore> renderFinished;

private:
    void create(Extent2D extent, VkSwapchainKHR old);
    void destroyViews();

    VulkanContext& ctx_;
    PresentPolicy policy_;
};

} // namespace phx
