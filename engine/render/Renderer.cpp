#include "engine/render/Renderer.hpp"

#include "engine/core/Png.hpp"
#include "engine/gpu/GpuProfiler.hpp"
#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/SkyRenderer.hpp"
#include "engine/render/TerrainRenderer.hpp"
#include "engine/render/TerrainShadows.hpp"
#include "engine/render/HiZ.hpp"
#include "engine/render/CascadedShadows.hpp"
#include "engine/render/PerfOverlay.hpp"
#include "engine/render/VegetationRenderer.hpp"
#include "engine/world/HeightField.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace phx {

Renderer::Renderer(VulkanContext& ctx, jobs::JobSystem& jobs, TripleBuffer<FrameSnapshot>& snapshots,
                   RendererConfig cfg)
    : ctx_(ctx), jobs_(jobs), snapshots_(snapshots), cfg_(std::move(cfg)) {}

Renderer::~Renderer() { stop(); }

void Renderer::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] { threadMain(); });
}

void Renderer::stop() {
    if (!thread_.joinable()) return;
    running_.store(false, std::memory_order_release);
    thread_.join();
}

void Renderer::createFrameResources() {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    si.pNext = &type;
    VK_CHECK(vkCreateSemaphore(ctx_.device, &si, nullptr, &timeline_));

    const u32 threads = jobs_.threadCount();
    for (FrameSlot& f : frames_) {
        VkSemaphoreCreateInfo bi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(ctx_.device, &bi, nullptr, &f.imageAcquired));
        f.pools.resize(threads);
        for (ThreadCommandPool& p : f.pools) {
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pi.queueFamilyIndex = ctx_.graphics.family;
            VK_CHECK(vkCreateCommandPool(ctx_.device, &pi, nullptr, &p.pool));
        }
    }
}

void Renderer::destroyFrameResources() {
    for (FrameSlot& f : frames_) {
        for (ThreadCommandPool& p : f.pools) vkDestroyCommandPool(ctx_.device, p.pool, nullptr);
        f.pools.clear();
        vkDestroySemaphore(ctx_.device, f.imageAcquired, nullptr);
    }
    vkDestroySemaphore(ctx_.device, timeline_, nullptr);
}

void Renderer::createTargets() {
    // Render resolution = output resolution until the upscaler lands (plan §2.8).
    hdr_ = createImage2D(ctx_, swapchain_->extent, SkyRenderer::kHdrFormat,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    bindless_->setSampled(kTexHdr, hdr_.view);
    depth_ = createImage2D(ctx_, swapchain_->extent, SkyRenderer::kDepthFormat,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    bindless_->setSampled(kTexDepth, depth_.view, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    hiz_->resize(swapchain_->extent);
    if (tonemapFormat_ != swapchain_->format) {
        if (tonemapPipe_) vkDestroyPipeline(ctx_.device, tonemapPipe_, nullptr);
        tonemapPipe_ = createGraphicsPipeline(ctx_, *bindless_, {"fullscreen.vert", "tonemap.frag", swapchain_->format, false});
        tonemapFormat_ = swapchain_->format;
        overlay_.reset();
        overlay_ = std::make_unique<PerfOverlay>(ctx_, *bindless_, swapchain_->format, kFramesInFlight);
    }
    readback_ = createBuffer(ctx_, VkDeviceSize(swapchain_->extent.width) * swapchain_->extent.height * 4,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::HostReadback);
}

void Renderer::destroyTargets() {
    destroyImage(ctx_, hdr_);
    destroyImage(ctx_, depth_);
    destroyBuffer(ctx_, readback_);
}

void Renderer::waitTimeline(u64 value) {
    if (value == 0) return;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline_;
    wi.pValues = &value;
    VK_CHECK(vkWaitSemaphores(ctx_.device, &wi, UINT64_MAX));
}

VkCommandBuffer Renderer::beginCommandBuffer(FrameSlot& slot) {
    ThreadCommandPool& p = slot.pools[jobs::JobSystem::threadIndex()];
    if (p.used == p.buffers.size()) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = p.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(ctx_.device, &ai, &cb));
        p.buffers.push_back(cb);
    }
    VkCommandBuffer cmd = p.buffers[p.used++];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void Renderer::recordFrame(VkCommandBuffer cmd, u32 imageIndex, u32 slot, u64 frameAddr, bool capture) {
    const VkExtent2D extent = swapchain_->extent;

    VkViewport viewport{0, 0, f32(extent.width), f32(extent.height), 0, 1};
    VkRect2D scissor{{0, 0}, extent};

    // 1. Sky-view LUTs (compute) and streamed terrain tiles (copies)
    profiler_->begin(cmd, slot, "skyLUT");
    sky_->recordCompute(cmd, frameAddr);
    profiler_->end(cmd, slot);
    profiler_->begin(cmd, slot, "upload");
    terrain_->recordUploads(cmd, slot);
    vegetation_->recordUploads(cmd, slot);
    profiler_->end(cmd, slot);
    profiler_->begin(cmd, slot, "shadowvis");
    shadows_->recordUploads(cmd, frameAddr);
    profiler_->end(cmd, slot);

    // Tree/rock shadow casters into the cascades (before terrain, which receives them).
    profiler_->begin(cmd, slot, "csm");
    vegetation_->recordShadowCull(cmd, frameAddr, slot);
    vegetation_->recordShadows(cmd, frameAddr, *csm_);
    profiler_->end(cmd, slot);

    // 2. HDR scene: terrain, then sky + stars where depth is still at infinity
    imageBarrier(cmd, hdr_.image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo hdrAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    hdrAtt.imageView = hdr_.view;
    hdrAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    hdrAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // sky covers every pixel
    hdrAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    imageBarrier(cmd, depth_.image, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = depth_.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // read by Hi-Z and the forest pass
    depthAtt.clearValue.depthStencil = {0.0f, 0}; // reverse-Z: 0 = infinitely far
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = scissor;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &hdrAtt;
    ri.pDepthAttachment = &depthAtt;
    vkCmdBeginRendering(cmd, &ri);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    profiler_->begin(cmd, slot, "terrain");
    terrain_->recordDraw(cmd, frameAddr, slot);
    profiler_->end(cmd, slot);
    vkCmdEndRendering(cmd);

    // Hi-Z from the terrain depth, then GPU culling of the forest against it.
    imageBarrier(cmd, depth_.image, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    profiler_->begin(cmd, slot, "hiz+cull");
    hiz_->record(cmd);
    vegetation_->recordCull(cmd, frameAddr, slot);
    profiler_->end(cmd, slot);
    imageBarrier(cmd, depth_.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // Forest, then sky + stars where depth is still at infinity.
    hdrAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    vkCmdBeginRendering(cmd, &ri);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    profiler_->begin(cmd, slot, "forest");
    vegetation_->recordDraw(cmd, frameAddr);
    profiler_->end(cmd, slot);
    profiler_->begin(cmd, slot, "sky+stars");
    sky_->recordSky(cmd, frameAddr, slot);
    profiler_->end(cmd, slot);
    vkCmdEndRendering(cmd);
    ri.pDepthAttachment = nullptr;

    // 3. Tonemap to the swapchain
    imageBarrier(cmd, hdr_.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkImage swapImage = swapchain_->images[imageIndex];
    imageBarrier(cmd, swapImage, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo outAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    outAtt.imageView = swapchain_->views[imageIndex];
    outAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    outAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    outAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ri.pColorAttachments = &outAtt;
    vkCmdBeginRendering(cmd, &ri);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_->pipelineLayout, 0, 1, &bindless_->set, 0,
                            nullptr);
    PushConstants pc{frameAddr, 0, 0, 0};
    vkCmdPushConstants(cmd, bindless_->pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipe_);
    profiler_->begin(cmd, slot, "tonemap");
    vkCmdDraw(cmd, 3, 1, 0, 0);
    profiler_->end(cmd, slot);
    if (overlayVisible_) overlay_->record(cmd, slot, swapchain_->extent);
    vkCmdEndRendering(cmd);

    VkImageLayout finalFrom = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkPipelineStageFlags2 finalStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkAccessFlags2 finalAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (capture) {
        imageBarrier(cmd, swapImage, finalStage, finalAccess, VK_PIPELINE_STAGE_2_COPY_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT, finalFrom, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.buffer, 1, &region);
        finalFrom = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        finalStage = VK_PIPELINE_STAGE_2_COPY_BIT;
        finalAccess = VK_ACCESS_2_NONE;
    }
    imageBarrier(cmd, swapImage, finalStage, finalAccess, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, finalFrom,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

void Renderer::saveScreenshot() {
    const VkExtent2D e = swapchain_->extent;
    vmaInvalidateAllocation(ctx_.allocator, readback_.allocation, 0, VK_WHOLE_SIZE);
    const u8* src = static_cast<const u8*>(readback_.mapped);
    const bool bgr = swapchain_->format == VK_FORMAT_B8G8R8A8_UNORM || swapchain_->format == VK_FORMAT_B8G8R8A8_SRGB;
    std::vector<u8> rgb(std::size_t(e.width) * e.height * 3);
    for (std::size_t i = 0; i < std::size_t(e.width) * e.height; ++i) {
        rgb[i * 3 + 0] = src[i * 4 + (bgr ? 2 : 0)];
        rgb[i * 3 + 1] = src[i * 4 + 1];
        rgb[i * 3 + 2] = src[i * 4 + (bgr ? 0 : 2)];
    }
    const std::string path = cfg_.screenshotPath.empty() ? "screenshot.png" : cfg_.screenshotPath;
    if (writePngRgb(path, e.width, e.height, rgb.data())) log::info("Screenshot saved: {}", path);
    else log::error("Failed to write screenshot {}", path);
    screenshotsWritten_.fetch_add(1, std::memory_order_release);
}

bool Renderer::renderFrame(const FrameSnapshot& snap) {
    using SClock = std::chrono::steady_clock;
    const auto tStart = SClock::now();
    const u32 slotIndex = u32(frameNumber_ % kFramesInFlight);
    FrameSlot& slot = frames_[slotIndex];
    waitTimeline(slot.timelineValue);
    const auto tWaited = SClock::now();

    u32 imageIndex = 0;
    const VkResult acq = vkAcquireNextImageKHR(ctx_.device, swapchain_->handle, UINT64_MAX, slot.imageAcquired,
                                               VK_NULL_HANDLE, &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) return false;
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);
    const auto tAcquired = SClock::now();

    for (ThreadCommandPool& p : slot.pools) {
        if (p.used) VK_CHECK(vkResetCommandPool(ctx_.device, p.pool, 0));
        p.used = 0;
    }
    profiler_->beginFrame(slotIndex);

    const bool capture = snap.screenshotRequest != lastScreenshotRequest_;
    lastScreenshotRequest_ = snap.screenshotRequest;

    // Per-frame CPU work on the render thread: uniforms, terrain LOD selection/streaming.
    {
        const glm::dmat3 enuToEcef = glm::transpose(ecefToEnuMatrix(snap.camera.positionEcef));
        const glm::dvec3 sunEcef = enuToEcef * glm::dvec3(snap.sky.sunDir);
        shadows_->update(snap.camera.positionEcef, sunEcef, frameNumber_);
        const CameraState& c = snap.camera;
        const f64 cy = std::cos(c.yaw), sy = std::sin(c.yaw), cp = std::cos(c.pitch), sp = std::sin(c.pitch);
        const glm::dvec3 fwd = enuToEcef * glm::dvec3(sy * cp, cy * cp, sp);
        const glm::dvec3 right = enuToEcef * glm::dvec3(cy, -sy, 0.0);
        csm_->update(c.positionEcef, fwd, right, glm::cross(right, fwd), c.fovY,
                     f32(swapchain_->extent.width) / f32(swapchain_->extent.height), sunEcef);
    }
    const u64 frameAddr = sky_->updateFrame(slotIndex, snap, swapchain_->extent, frameNumber_,
                                            shadows_->uniforms(snap.camera.positionEcef), csm_->uniforms());
    terrain_->update(slotIndex, snap.camera.positionEcef, sky_->viewProjEcef(), snap.camera.fovY,
                     swapchain_->extent.height, frameNumber_ + 1);
    vegetation_->update(slotIndex, snap.camera.positionEcef, frameNumber_ + 1);

    overlayVisible_ = snap.perfOverlay || snap.hud.visible;
    if (overlayVisible_) {
        const TerrainStats& ts = terrain_->stats();
        const VegStats& vs = vegetation_->stats();
        const PerfCpuTimes cpu{lastCpu_[0], lastCpu_[1], lastCpu_[2], lastCpu_[3], lastCpu_[4]};
        overlay_->build(slotIndex, swapchain_->extent, snap.perfOverlay, snap.hud, cpu, *profiler_,
                        {ts.drawnTiles, ts.residentTiles, ts.pendingTiles, vs.instances, vs.cells, jobs_.workerCount(),
                         jobs_.stats().executed});
    }

    // Recording is a job: workers own per-thread pools. Later passes fan out further.
    VkCommandBuffer recorded = VK_NULL_HANDLE;
    jobs::Counter recordDone;
    jobs_.submit({.priority = jobs::Priority::Critical, .signal = &recordDone}, [&, imageIndex, slotIndex, frameAddr] {
        VkCommandBuffer cmd = beginCommandBuffer(slot);
        recordFrame(cmd, imageIndex, slotIndex, frameAddr, capture);
        VK_CHECK(vkEndCommandBuffer(cmd));
        recorded = cmd;
    });
    const auto tUpdated = SClock::now();
    jobs_.wait(recordDone);
    const auto tRecorded = SClock::now();

    const u64 signalValue = frameNumber_ + 1;
    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitInfo.semaphore = slot.imageAcquired;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signalInfos[2]{{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO},
                                         {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    signalInfos[0].semaphore = swapchain_->renderFinished[imageIndex];
    signalInfos[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signalInfos[1].semaphore = timeline_;
    signalInfos[1].value = signalValue;
    signalInfos[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = recorded;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitInfo;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    submit.signalSemaphoreInfoCount = 2;
    submit.pSignalSemaphoreInfos = signalInfos;
    VK_CHECK(vkQueueSubmit2(ctx_.graphics.queue, 1, &submit, VK_NULL_HANDLE));
    slot.timelineValue = signalValue;
    ++frameNumber_;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &swapchain_->renderFinished[imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_->handle;
    pi.pImageIndices = &imageIndex;
    const VkResult pres = vkQueuePresentKHR(ctx_.graphics.queue, &pi);

    {
        auto ms = [](auto a, auto b) { return std::chrono::duration<f64, std::milli>(b - a).count(); };
        const auto tEnd = SClock::now();
        lastCpu_[0] = ms(tStart, tWaited), lastCpu_[1] = ms(tWaited, tAcquired), lastCpu_[2] = ms(tAcquired, tUpdated);
        lastCpu_[3] = ms(tUpdated, tRecorded), lastCpu_[4] = ms(tRecorded, tEnd);
        static auto lastWarn = SClock::time_point{};
        if (ms(tStart, tEnd) > 20.0 && tEnd - lastWarn > std::chrono::seconds(1) && (lastWarn = tEnd, true))
            log::warn("slow frame {:.1f} ms: gpu-wait {:.1f}, acquire {:.1f}, cpu-update {:.1f}, record {:.1f}, submit+present {:.1f}",
                      ms(tStart, tEnd), ms(tStart, tWaited), ms(tWaited, tAcquired), ms(tAcquired, tUpdated),
                      ms(tUpdated, tRecorded), ms(tRecorded, tEnd));
        // (acquire-bound frames usually mean the compositor throttles an occluded window)
    }
    if (capture) {
        waitTimeline(signalValue);
        saveScreenshot();
    }
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) return false;
    VK_CHECK(pres);
    // Some platforms never report OUT_OF_DATE on resize; compare against the window.
    return snap.framebuffer.width == swapchain_->extent.width && snap.framebuffer.height == swapchain_->extent.height;
}

void Renderer::recreateSwapchain(Extent2D extent) {
    waitTimeline(frameNumber_);
    VK_CHECK(vkQueueWaitIdle(ctx_.graphics.queue));
    destroyTargets();
    swapchain_->recreate(extent);
    createTargets();
}

void Renderer::threadMain() {
    jobs_.registerThread(jobs::ThreadRole::Render);
    createFrameResources();

    while (running_.load(std::memory_order_acquire) && !snapshots_.acquireLatest())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    swapchain_ = std::make_unique<Swapchain>(ctx_, snapshots_.read().framebuffer, cfg_.present);
    bindless_ = std::make_unique<Bindless>(ctx_);
    sky_ = std::make_unique<SkyRenderer>(ctx_, *bindless_, kFramesInFlight, cfg_.starEpochYear);
    terrain_ = std::make_unique<TerrainRenderer>(ctx_, *bindless_, jobs_, *cfg_.heights, kFramesInFlight,
                                                 SkyRenderer::kHdrFormat, SkyRenderer::kDepthFormat);
    profiler_ = std::make_unique<GpuProfiler>(ctx_, kFramesInFlight);
    shadows_ = std::make_unique<TerrainShadows>(ctx_, *bindless_, jobs_, *cfg_.heights, kFramesInFlight);
    hiz_ = std::make_unique<HiZ>(ctx_, *bindless_);
    csm_ = std::make_unique<CascadedShadows>(ctx_, *bindless_);
    vegetation_ = std::make_unique<VegetationRenderer>(ctx_, *bindless_, jobs_, *cfg_.heights, SkyRenderer::kHdrFormat,
                                                       SkyRenderer::kDepthFormat);
    createTargets();

    using Clock = std::chrono::steady_clock;
    auto statsStart = Clock::now();
    u64 statsFrames = 0;
    std::vector<f32> frameMs;          // frame-to-frame intervals of this stats window
    frameMs.reserve(65536);
    auto lastFrame = Clock::now();
    f64 worstUpdateMs = 0;

    while (running_.load(std::memory_order_acquire)) {
        snapshots_.acquireLatest();
        const FrameSnapshot& snap = snapshots_.read();
        if (snap.framebuffer.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
            continue;
        }
        if (!renderFrame(snap)) recreateSwapchain(snap.framebuffer);
        ++statsFrames;

        worstUpdateMs = std::max(worstUpdateMs, terrain_->stats().updateMs);
        const auto now = Clock::now();
        frameMs.push_back(std::chrono::duration<f32, std::milli>(now - lastFrame).count());
        if (overlay_) overlay_->recordFrameInterval(frameMs.back());
        lastFrame = now;
        if (now - statsStart >= std::chrono::seconds(5)) {
            const double secs = std::chrono::duration<double>(now - statsStart).count();
            const auto js = jobs_.stats();
            const TerrainStats& ts = terrain_->stats();
            std::sort(frameMs.begin(), frameMs.end());
            const f32 p99 = frameMs.empty() ? 0.f : frameMs[std::min(frameMs.size() - 1, frameMs.size() * 99 / 100)];
            const f32 worst = frameMs.empty() ? 0.f : frameMs.back();
            log::info("render: {:.1f} fps ({:.3f} ms, p99 {:.2f}, max {:.2f}) | terrain {} drawn, {} resident, {} pending, "
                      "max level {} | jobs {} (steals {})",
                      statsFrames / secs, secs * 1000.0 / double(statsFrames), p99, worst, ts.drawnTiles,
                      ts.residentTiles, ts.pendingTiles, ts.maxLevel, js.executed, js.steals);
            frameMs.clear();
            log::info("terrain: {} tiles generated ({:.2f} ms avg on workers), update {:.3f} ms (last), worst update {:.3f} ms",
                      ts.tilesGenerated, ts.tilesGenerated ? ts.generateMsTotal / f64(ts.tilesGenerated) : 0.0, ts.updateMs,
                      worstUpdateMs);
            worstUpdateMs = 0;
            log::info("{}", profiler_->report());
            const VegStats& vs = vegetation_->stats();
            log::info("forest: {} cells, {} trees resident, {} cells pending", vs.cells, vs.instances, vs.pendingCells);
            statsStart = now;
            statsFrames = 0;
        }
    }

    waitTimeline(frameNumber_);
    vkQueueWaitIdle(ctx_.graphics.queue);
    destroyTargets();
    if (tonemapPipe_) vkDestroyPipeline(ctx_.device, tonemapPipe_, nullptr);
    overlay_.reset();
    profiler_.reset();
    vegetation_.reset();
    csm_.reset();
    hiz_.reset();
    shadows_.reset();
    terrain_.reset();
    sky_.reset();
    bindless_.reset();
    swapchain_.reset();
    destroyFrameResources();
}

} // namespace phx
