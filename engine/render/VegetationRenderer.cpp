#include "engine/render/VegetationRenderer.hpp"

#include "engine/gpu/Textures.hpp"
#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/CascadedShadows.hpp"
#include "engine/render/SkyRenderer.hpp"
#include "engine/world/CubeSphere.hpp"
#include "engine/world/HeightField.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

#ifndef PHX_DATA_DIR
#define PHX_DATA_DIR "data"
#endif
#ifndef PHX_ASSET_DIR
#define PHX_ASSET_DIR "data/raw/polyhaven"
#endif

namespace phx {
namespace {
using namespace cubesphere;

// ---- species configuration ------------------------------------------------------------
enum SpeciesFlags : u32 { kImpostor = 1, kWind = 2, kCastsShadow = 4, kTilt = 8 };
struct SpeciesConfig {
    f32 lod[4];      // LOD0/1/2 end distances, then impostor end (0: none)
    u32 flags;
};
constexpr SpeciesConfig kSpecies[kSpeciesCount] = {
    {{15, 45, 130, 4100}, kImpostor | kWind | kCastsShadow},  // 0 Metasequoia (fir)
    {{15, 45, 130, 4100}, kImpostor | kWind | kCastsShadow},  // 1 Lauraceae
    {{15, 45, 130, 4100}, kImpostor | kWind | kCastsShadow},  // 2 Ginkgo
    {{8, 22, 65, 0}, kWind},                                  // 3 fern
    {{15, 40, 130, 0}, kTilt},                                // 4 mossy stones
    {{30, 100, 450, 0}, kTilt | kCastsShadow},                // 5 boulders
    {{6, 16, 45, 0}, kWind},                                  // 6 grass clumps
    {{12, 35, 160, 0}, kWind},                                // 7 shrubs
    {{15, 45, 400, 0}, kWind | kCastsShadow},                 // 8 dryland tree
};
// Shadow-casting species in bucket order (see kShadowCasters).
constexpr u32 kCasterSpecies[VegetationRenderer::kShadowCasters] = {0, 1, 2, 5, 8};


constexpr u32 kLodCapacity[kTreeMeshLods + 1] = {2048, 8192, 32768, 262144};

u32 cameraBucket(u32 species, u32 lod) {
    return lod < kTreeMeshLods ? species * kTreeMeshLods + lod : kSpeciesCount * kTreeMeshLods + species;
}

// Indirect commands followed by parameters read by the shaders (veg_common.glsl).
struct IndirectBlock {
    VkDrawIndexedIndirectCommand cmds[VegetationRenderer::kBuckets]; // 45 × 20 bytes
    u32 pad0;
    f32 species[kSpeciesCount][3][4]; // [height, radius, impostor extent, flags] [lod0..3] [scaleMul, ...]
    u32 capacity[VegetationRenderer::kBuckets];
};
} // namespace

VegetationRenderer::VegetationRenderer(VulkanContext& ctx, Bindless& bindless, jobs::JobSystem& jobs,
                                       const HeightField& heights, VkFormat colorFormat, VkFormat depthFormat)
    : ctx_(ctx), bindless_(bindless), jobs_(jobs), heights_(heights) {
    // ---- species: baked Poly Haven meshes + PBR textures
    std::vector<TextureRequest> texReqs;
    for (u32 sp = 0; sp < kSpeciesCount; ++sp) {
        if (!loadTreeSpecies(sp, PHX_DATA_DIR, PHX_ASSET_DIR, species_[sp]))
            PHX_FATAL("Species {} ({}) missing (build target phx_tree_data)", sp, treeAssetName(sp));
        for (u32 m = 0; m < species_[sp].materials.size(); ++m) {
            const TreeMaterial& mat = species_[sp].materials[m];
            Image* dst = &materialTextures_[sp][m][0];
            texReqs.push_back({mat.albedo, true, dst + 0});
            texReqs.push_back({mat.normal, false, dst + 1});
            texReqs.push_back({mat.arm, false, dst + 2});
        }
    }
    for (u32 sp = 0; sp < kSpeciesCount; ++sp) dims_[sp] = {species_[sp].height, species_[sp].radius};
    loadTextures(ctx, jobs, texReqs);
    for (u32 sp = 0; sp < kSpeciesCount; ++sp)
        for (u32 m = 0; m < species_[sp].materials.size(); ++m)
            for (u32 k = 0; k < 3; ++k)
                bindless.setMaterial(kTexTreeMaterial0 + (sp * kMaxTreeMaterials + m) * 3 + k, materialTextures_[sp][m][k].view);

    std::vector<TreeVertex> verts;
    std::vector<u32> idx;
    u64 tris[kTreeMeshLods]{};
    for (u32 sp = 0; sp < kSpeciesCount; ++sp)
        for (u32 l = 0; l < kTreeMeshLods; ++l) {
            const TreeMesh& m = species_[sp].lods[l];
            meshes_[sp][l] = {u32(idx.size()), u32(m.indices.size()), u32(verts.size())};
            verts.insert(verts.end(), m.vertices.begin(), m.vertices.end());
            idx.insert(idx.end(), m.indices.begin(), m.indices.end());
            tris[l] += m.indices.size() / 3;
        }
    const u32 quadVertex = u32(verts.size()), quadIndex = u32(idx.size());
    for (glm::vec2 c : {glm::vec2(-1, -1), glm::vec2(1, -1), glm::vec2(1, 1), glm::vec2(-1, 1)}) verts.push_back({{c.x, c.y, 0}, 0, 0, 0});
    for (u32 i : {0u, 1u, 2u, 0u, 2u, 3u}) idx.push_back(i);
    for (u32 sp = 0; sp < kSpeciesCount; ++sp) meshes_[sp][kTreeMeshLods] = {quadIndex, 6, quadVertex};

    vertices_ = createBuffer(ctx, verts.size() * sizeof(TreeVertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MemoryUsage::DeviceLocal);
    indices_ = createBuffer(ctx, idx.size() * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            MemoryUsage::DeviceLocal);
    ImmediateSubmitter up(ctx);
    up.uploadBuffer(vertices_, verts.data(), verts.size() * sizeof(TreeVertex));
    up.uploadBuffer(indices_, idx.data(), idx.size() * 4);
    log::info("Nature (Poly Haven, CC0): {} species, LOD0 {} / LOD1 {} / LOD2 {} triangles in total, {} textures",
              kSpeciesCount, tris[0], tris[1], tris[2], texReqs.size());

    // ---- buckets and the indirect/parameter block
    IndirectBlock blk{};
    u32 base = 0;
    auto addBucket = [&](u32 b, const MeshRange& m, u32 capacity) {
        blk.cmds[b] = {m.indexCount, 0, m.firstIndex, i32(m.vertexOffset), base};
        blk.capacity[b] = capacity;
        base += capacity;
    };
    for (u32 sp = 0; sp < kSpeciesCount; ++sp)
        for (u32 l = 0; l < kTreeMeshLods; ++l) addBucket(cameraBucket(sp, l), meshes_[sp][l], kLodCapacity[l]);
    for (u32 sp = 0; sp < kTreeSpeciesCount; ++sp) addBucket(cameraBucket(sp, kTreeMeshLods), meshes_[sp][kTreeMeshLods], kLodCapacity[3]);
    for (u32 c = 0; c < 3; ++c)
        for (u32 k = 0; k < kShadowCasters; ++k) {
            const u32 sp = kCasterSpecies[k];
            const bool quad = c > 0 && sp < kTreeSpeciesCount; // far cascades: light-facing impostor quads for trees
            addBucket(kCameraBuckets + c * kShadowCasters + k, quad ? meshes_[sp][kTreeMeshLods] : meshes_[sp][2],
                      c == 0 ? 2048 : 65536);
        }
    for (u32 sp = 0; sp < kSpeciesCount; ++sp) {
        const SpeciesConfig& cfg = kSpecies[sp];
        const f32 h = species_[sp].height, r = species_[sp].radius;
        f32 flags = f32(cfg.flags);
        f32* p = &blk.species[sp][0][0];
        p[0] = h, p[1] = r, p[2] = std::max(r, h * 0.5f) * 1.05f, p[3] = flags;
        for (int k = 0; k < 4; ++k) p[4 + k] = cfg.lod[k];
        p[8] = kNatureScaleMul[sp];
    }
    visible_ = createBuffer(ctx, VkDeviceSize(base) * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::DeviceLocal);
    indirect_ = createBuffer(ctx, sizeof(IndirectBlock),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MemoryUsage::DeviceLocal);
    up.uploadBuffer(indirect_, &blk, sizeof(blk));

    pool_ = createBuffer(ctx, VkDeviceSize(kCellSlots) * kMaxNaturePerCell * sizeof(VegInstance),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::DeviceLocal);
    for (u32 i = kCellSlots; i-- > 0;) freeSlots_.push_back(i);
    for (u32 i = 0; i < 2; ++i) {
        cellBuffers_.push_back(createBuffer(ctx, sizeof(CellGpu) * kCellSlots, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::HostUpload));
        staging_.push_back(createBuffer(ctx, VkDeviceSize(kUploadsPerFrame) * kMaxNaturePerCell * sizeof(VegInstance),
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::HostUpload));
    }

    // ---- pipelines
    cullPipe_ = createComputePipeline(ctx, bindless, "vegcull.comp");
    GraphicsPipelineDesc d{"vegmesh.vert", "vegmesh.frag", colorFormat};
    d.depthFormat = depthFormat;
    d.depthTest = true;
    d.depthWrite = true;
    d.depthCompare = VK_COMPARE_OP_GREATER;
    meshPipe_ = createGraphicsPipeline(ctx, bindless, d);
    d.vertex = "vegimpostor.vert";
    d.fragment = "vegimpostor.frag";
    impostorPipe_ = createGraphicsPipeline(ctx, bindless, d);
    GraphicsPipelineDesc sd{"vegshadow.vert", "vegshadow.frag", VK_FORMAT_UNDEFINED};
    sd.depthFormat = VK_FORMAT_D32_SFLOAT;
    sd.depthTest = true;
    sd.depthWrite = true;
    sd.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
    sd.depthBiasConstant = 1.5f;
    sd.depthBiasSlope = 2.0f;
    shadowMeshPipe_ = createGraphicsPipeline(ctx, bindless, sd);
    sd.vertex = "vegshadowimp.vert";
    sd.fragment = "vegshadowimp.frag";
    shadowImpostorPipe_ = createGraphicsPipeline(ctx, bindless, sd);

    maxInFlight_ = std::max(2u, jobs.workerCount() / 3);
    bakeImpostors();
    for (TreeSpeciesData& sp : species_) sp.full = {}; // only needed for the bake
}

VegetationRenderer::~VegetationRenderer() {
    while (inFlight_) {
        std::lock_guard lock(doneMutex_);
        for (Request* r : done_) delete r, --inFlight_;
        done_.clear();
    }
    for (Request* r : readyForUpload_) delete r;
    for (VkPipeline p : {cullPipe_, meshPipe_, impostorPipe_, shadowMeshPipe_, shadowImpostorPipe_}) vkDestroyPipeline(ctx_.device, p, nullptr);
    for (Buffer* b : {&vertices_, &indices_, &pool_, &visible_, &indirect_}) destroyBuffer(ctx_, *b);
    for (Buffer& b : cellBuffers_) destroyBuffer(ctx_, b);
    for (Buffer& b : staging_) destroyBuffer(ctx_, b);
    for (u32 s = 0; s < kTreeSpeciesCount; ++s) {
        destroyImage(ctx_, impostorAlbedo_[s]);
        destroyImage(ctx_, impostorNormal_[s]);
    }
    for (auto& sp : materialTextures_)
        for (auto& mat : sp)
            for (Image& img : mat) destroyImage(ctx_, img);
}

// ------------------------------------------------------------------ impostor bake

void VegetationRenderer::bakeImpostors() {
    const u32 atlas = kImpostorFrames * kImpostorFrameSize;
    GraphicsPipelineDesc d{"vegbake.vert", "vegbake.frag", VK_FORMAT_R8G8B8A8_SRGB};
    d.colorFormat2 = VK_FORMAT_R8G8B8A8_UNORM;
    d.depthFormat = VK_FORMAT_D32_SFLOAT;
    d.depthTest = true;
    d.depthWrite = true;
    d.depthCompare = VK_COMPARE_OP_LESS;
    VkPipeline bake = createGraphicsPipeline(ctx_, bindless_, d);
    Image depth = createImage2D(ctx_, {atlas, atlas}, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    ImmediateSubmitter sub(ctx_);
    for (u32 s = 0; s < kTreeSpeciesCount; ++s) {
        // The impostor shows the full-resolution asset: upload it only for the bake.
        const TreeMesh& full = species_[s].full;
        Buffer fv = createBuffer(ctx_, full.vertices.size() * sizeof(TreeVertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 MemoryUsage::DeviceLocal);
        Buffer fi = createBuffer(ctx_, full.indices.size() * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 MemoryUsage::DeviceLocal);
        sub.uploadBuffer(fv, full.vertices.data(), full.vertices.size() * sizeof(TreeVertex));
        sub.uploadBuffer(fi, full.indices.data(), full.indices.size() * 4);

        impostorAlbedo_[s] = createImage2D(ctx_, {atlas, atlas}, VK_FORMAT_R8G8B8A8_SRGB,
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        impostorNormal_[s] = createImage2D(ctx_, {atlas, atlas}, VK_FORMAT_R8G8B8A8_UNORM,
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        sub.run([&](VkCommandBuffer cmd) {
            for (Image* img : {&impostorAlbedo_[s], &impostorNormal_[s]})
                imageBarrier(cmd, img->image, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            imageBarrier(cmd, depth.image, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            VkRenderingAttachmentInfo col[2]{{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}};
            col[0].imageView = impostorAlbedo_[s].view;
            col[1].imageView = impostorNormal_[s].view;
            for (auto& c : col) {
                c.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                c.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                c.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            }
            VkRenderingAttachmentInfo dep{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            dep.imageView = depth.view;
            dep.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            dep.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            dep.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            dep.clearValue.depthStencil = {1.0f, 0};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {atlas, atlas}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 2;
            ri.pColorAttachments = col;
            ri.pDepthAttachment = &dep;
            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bake);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0, nullptr);
            vkCmdBindIndexBuffer(cmd, fi.buffer, 0, VK_INDEX_TYPE_UINT32);
            for (u32 fy = 0; fy < kImpostorFrames; ++fy)
                for (u32 fx = 0; fx < kImpostorFrames; ++fx) {
                    VkViewport vp{f32(fx * kImpostorFrameSize), f32(fy * kImpostorFrameSize), f32(kImpostorFrameSize),
                                  f32(kImpostorFrameSize), 0, 1};
                    VkRect2D sc{{i32(fx * kImpostorFrameSize), i32(fy * kImpostorFrameSize)}, {kImpostorFrameSize, kImpostorFrameSize}};
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    PushConstants pc{0, 0, s, fy * kImpostorFrames + fx, fv.address, 0, indirect_.address};
                    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
                    vkCmdDrawIndexed(cmd, u32(full.indices.size()), 1, 0, 0, 0);
                }
            vkCmdEndRendering(cmd);
            for (Image* img : {&impostorAlbedo_[s], &impostorNormal_[s]})
                imageBarrier(cmd, img->image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        bindless_.setSampled(kTexImpostorAlbedo0 + s, impostorAlbedo_[s].view);
        bindless_.setSampled(kTexImpostorNormal0 + s, impostorNormal_[s].view);
        destroyBuffer(ctx_, fv);
        destroyBuffer(ctx_, fi);
        species_[s].full = {}; // free the CPU copy
    }
    destroyImage(ctx_, depth);
    vkDestroyPipeline(ctx_.device, bake, nullptr);
    log::info("Impostors baked from full-resolution trees: {} species × {}×{} views at {} px", kTreeSpeciesCount,
              kImpostorFrames, kImpostorFrames, kImpostorFrameSize);
}

// ------------------------------------------------------------------ placement (worker jobs)

// ------------------------------------------------------------------ per frame

void VegetationRenderer::update(u32 slot, const glm::dvec3& cam, u64 frame) {
    (void)frame;
    stats_ = {};
    {
        std::lock_guard lock(doneMutex_);
        for (Request* r : done_) readyForUpload_.push_back(r);
        inFlight_ -= u32(done_.size());
        done_.clear();
    }

    std::unordered_set<u64> wanted;
    const glm::dvec3 up = glm::normalize(cam);
    const glm::dvec3 east = glm::normalize(glm::cross(glm::dvec3(0, 0, 1), up)), north = glm::cross(up, east);
    const glm::dvec3 ground = up * kPlanetRadiusM;
    for (const NatureLayer& L : kNatureLayers)
        for (f64 y = -L.radius; y <= L.radius; y += L.step)
            for (f64 x = -L.radius; x <= L.radius; x += L.step)
                if (x * x + y * y <= L.radius * L.radius) wanted.insert(keyAt(glm::normalize(ground + east * x + north * y), L.level));

    for (auto it = cells_.begin(); it != cells_.end();) {
        if (!wanted.count(it->first) && it->second.ready) {
            freeSlots_.push_back(it->second.slot);
            it = cells_.erase(it);
        } else {
            ++it;
        }
    }
    // Request missing cells: finer layers first (they are nearest), then by distance.
    std::vector<std::pair<f64, u64>> missing;
    for (u64 k : wanted)
        if (!cells_.count(k)) {
            const NodeId n = decode(k);
            const f64 size = 2.0 / f64(1u << n.level);
            const f64 d = glm::length(cubeToSphere(n.face, -1 + (n.x + 0.5) * size, -1 + (n.y + 0.5) * size) * kPlanetRadiusM - cam);
            missing.push_back({d - f64(n.level) * 1000.0, k});
        }
    std::sort(missing.begin(), missing.end());
    for (auto [d, k] : missing) {
        if (inFlight_ >= maxInFlight_ || freeSlots_.empty()) break;
        Cell& c = cells_[k];
        c.slot = freeSlots_.back();
        freeSlots_.pop_back();
        auto* req = new Request{k, c.slot, {}, {}};
        ++inFlight_;
        jobs_.submit({.priority = jobs::Priority::Background}, [this, req] {
            req->center = placeNature(req->key, heights_, dims_, req->instances);
            std::lock_guard lock(doneMutex_);
            done_.push_back(req);
        });
    }

    pendingCopies_.clear();
    pendingCounts_.clear();
    auto* dst = static_cast<VegInstance*>(staging_[slot].mapped);
    u32 n = 0;
    auto it = readyForUpload_.begin();
    for (; it != readyForUpload_.end() && n < kUploadsPerFrame; ++it) {
        Request* r = *it;
        auto c = cells_.find(r->key);
        if (c != cells_.end() && c->second.slot == r->slot) {
            if (!wanted.count(r->key)) {
                freeSlots_.push_back(r->slot);
                cells_.erase(c);
            } else {
                std::memcpy(dst + std::size_t(n) * kMaxNaturePerCell, r->instances.data(), r->instances.size() * sizeof(VegInstance));
                pendingCopies_.push_back({n, r->slot});
                pendingCounts_.push_back(u32(r->instances.size()));
                c->second.ready = true;
                c->second.count = u32(r->instances.size());
                c->second.center = r->center;
                ++n;
            }
        }
        delete r;
    }
    readyForUpload_.erase(readyForUpload_.begin(), it);
    if (n) vmaFlushAllocation(ctx_.allocator, staging_[slot].allocation, 0, VK_WHOLE_SIZE);

    frameCells_.clear();
    for (const auto& [k, c] : cells_) {
        if (!c.ready || c.count == 0) continue;
        const glm::dvec3 off = c.center - cam;
        frameCells_.push_back({{f32(off.x), f32(off.y), f32(off.z)}, c.slot, c.count, {}});
        stats_.instances += c.count;
    }
    std::memcpy(cellBuffers_[slot].mapped, frameCells_.data(), frameCells_.size() * sizeof(CellGpu));
    vmaFlushAllocation(ctx_.allocator, cellBuffers_[slot].allocation, 0, VK_WHOLE_SIZE);
    stats_.cells = u32(frameCells_.size());
    stats_.pendingCells = inFlight_ + u32(readyForUpload_.size());
}

void VegetationRenderer::recordUploads(VkCommandBuffer cmd, u32 slot) {
    if (pendingCopies_.empty()) return;
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
    std::vector<VkBufferCopy> copies;
    for (std::size_t i = 0; i < pendingCopies_.size(); ++i) {
        const auto [si, cs] = pendingCopies_[i];
        copies.push_back({VkDeviceSize(si) * kMaxNaturePerCell * sizeof(VegInstance), VkDeviceSize(cs) * kMaxNaturePerCell * sizeof(VegInstance),
                          std::max<VkDeviceSize>(16, VkDeviceSize(pendingCounts_[i]) * sizeof(VegInstance))});
    }
    vkCmdCopyBuffer(cmd, staging_[slot].buffer, pool_.buffer, u32(copies.size()), copies.data());
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// Modes: 0 camera cull (after Hi-Z), 1 clamp counts, 2 shadow cull.
void VegetationRenderer::dispatchCull(VkCommandBuffer cmd, u64 frameAddr, u32 slot, u32 mode) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0, nullptr);
    if (!frameCells_.empty()) {
        PushConstants pc{frameAddr, cellBuffers_[slot].address, u32(frameCells_.size()), mode, pool_.address, visible_.address,
                         indirect_.address};
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, u32(frameCells_.size()), kMaxNaturePerCell / 256, 1);
    }
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
    PushConstants fin{frameAddr, 0, 0, 1, 0, 0, indirect_.address};
    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(fin), &fin);
    vkCmdDispatch(cmd, 1, 1, 1);
    mb.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void VegetationRenderer::recordShadowCull(VkCommandBuffer cmd, u64 frameAddr, u32 slot) {
    // Reset all draw counts (the previous frame's draws read them).
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
    for (u32 b = 0; b < kBuckets; ++b)
        vkCmdFillBuffer(cmd, indirect_.buffer, b * sizeof(VkDrawIndexedIndirectCommand) + offsetof(VkDrawIndexedIndirectCommand, instanceCount), 4, 0);
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
    dispatchCull(cmd, frameAddr, slot, 2);
}

void VegetationRenderer::recordShadows(VkCommandBuffer cmd, u64 frameAddr, CascadedShadows& csm) {
    constexpr VkDeviceSize stride = sizeof(VkDrawIndexedIndirectCommand);
    for (u32 c = 0; c < 3; ++c) {
        csm.begin(cmd, c);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0, nullptr);
        vkCmdBindIndexBuffer(cmd, indices_.buffer, 0, VK_INDEX_TYPE_UINT32);
        PushConstants pc{frameAddr, visible_.address, 0, c, vertices_.address, 0, indirect_.address};
        vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        const VkDeviceSize first = (kCameraBuckets + c * kShadowCasters) * stride;
        if (c == 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMeshPipe_);
            vkCmdDrawIndexedIndirect(cmd, indirect_.buffer, first, kShadowCasters, stride);
        } else {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowImpostorPipe_);
            vkCmdDrawIndexedIndirect(cmd, indirect_.buffer, first, kTreeSpeciesCount, stride);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMeshPipe_);
            vkCmdDrawIndexedIndirect(cmd, indirect_.buffer, first + kTreeSpeciesCount * stride, kShadowCasters - kTreeSpeciesCount, stride);
        }
        csm.end(cmd, c);
    }
}

void VegetationRenderer::recordCull(VkCommandBuffer cmd, u64 frameAddr, u32 slot) { dispatchCull(cmd, frameAddr, slot, 0); }

void VegetationRenderer::recordDraw(VkCommandBuffer cmd, u64 frameAddr) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0, nullptr);
    vkCmdBindIndexBuffer(cmd, indices_.buffer, 0, VK_INDEX_TYPE_UINT32);
    PushConstants pc{frameAddr, visible_.address, 0, 0, vertices_.address, 0, indirect_.address};
    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    constexpr VkDeviceSize stride = sizeof(VkDrawIndexedIndirectCommand);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipe_);
    vkCmdDrawIndexedIndirect(cmd, indirect_.buffer, 0, kSpeciesCount * kTreeMeshLods, stride);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorPipe_);
    vkCmdDrawIndexedIndirect(cmd, indirect_.buffer, kSpeciesCount * kTreeMeshLods * stride, kTreeSpeciesCount, stride);
}

} // namespace phx
