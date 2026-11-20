#include "engine/render/TerrainRenderer.hpp"

#include "engine/gpu/VulkanContext.hpp"
#include "engine/render/SkyRenderer.hpp"
#include "engine/gpu/Textures.hpp"
#include "engine/world/CubeSphere.hpp"
#include "engine/world/Climate.hpp"
#include "engine/world/HeightField.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace phx {
namespace {
constexpr f64 kPi = 3.14159265358979323846;

using namespace cubesphere;

// Transmittance-based LOD bias (plan §2.7): aerial perspective hides detail with
// distance, so distances are stretched monotonically before the CDLOD range tests.
// The vertex shader applies the same warp to its morph distance (terrain.vert), which
// keeps CDLOD's continuity guarantees: only the mapping from metres to "LOD distance"
// changes. ≈1.0× at 1 km, 1.55× at 15 km, 1.9× at 32 km.
f64 lodDistance(f64 d) { return d * (1.0 + 1.0 * (1.0 - std::exp(-d / 20000.0))); }

u32 packNormal(const glm::dvec3& n, f64 height) {
    // Octahedral encoding, 12 bits per component.
    glm::dvec2 p = glm::dvec2(n.x, n.y) / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    if (n.z < 0) p = (1.0 - glm::abs(glm::dvec2(p.y, p.x))) * glm::dvec2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
    const u32 x = u32(std::lround((p.x * 0.5 + 0.5) * 4095.0)), y = u32(std::lround((p.y * 0.5 + 0.5) * 4095.0));
    const u32 h = u32(std::clamp((height + 4000.0) / 40.0, 0.0, 255.0)); // 40 m classes, −4000…6200 m
    return x | (y << 12) | (h << 24);
}
} // namespace

TerrainRenderer::TerrainRenderer(VulkanContext& ctx, Bindless& bindless, jobs::JobSystem& jobs,
                                 const HeightField& heights, u32 framesInFlight, VkFormat colorFormat,
                                 VkFormat depthFormat)
    : ctx_(ctx), bindless_(bindless), jobs_(jobs), heights_(heights) {
    GraphicsPipelineDesc d{"terrain.vert", "terrain.frag", colorFormat};
    d.depthFormat = depthFormat;
    d.depthTest = true;
    d.depthWrite = true;
    d.depthCompare = VK_COMPARE_OP_GREATER; // reverse-Z
    pipeline_ = createGraphicsPipeline(ctx, bindless, d);

    // Ground materials (Poly Haven, CC0): leaf litter, soil/moss, lichen-covered rock.
    {
#ifndef PHX_ASSET_DIR
#define PHX_ASSET_DIR "data/raw/polyhaven"
#endif
        const char* kGround[kGroundMaterials] = {"forest_leaves_02", "forest_ground_04", "lichen_rock", "rock_surface", "sand_01",
                                                 "dry_ground_rocks", "sparse_grass",     "snow_02",     "forest_leaves_04"};
        std::vector<TextureRequest> reqs;
        for (u32 m = 0; m < kGroundMaterials; ++m) {
            const std::string base = std::string(PHX_ASSET_DIR) + "/" + kGround[m] + "/" + kGround[m];
            reqs.push_back({base + "_diff_2k.jpg", true, &groundTextures_[m][0]});
            reqs.push_back({base + "_nor_gl_2k.jpg", false, &groundTextures_[m][1]});
            reqs.push_back({base + "_arm_2k.jpg", false, &groundTextures_[m][2]});
        }
        loadTextures(ctx, jobs, reqs);
        for (u32 m = 0; m < kGroundMaterials; ++m)
            for (u32 k = 0; k < 3; ++k) bindless.setMaterial(kTexGround0 + m * 3 + k, groundTextures_[m][k].view);
    }

    // Tile generation is long-running background work: never occupy every worker, so the
    // render thread's recording jobs and the OS always find an idle hardware thread.
    maxInFlight_ = std::max(2u, jobs.workerCount() > 2 ? jobs.workerCount() - 2 : 1u);
    if (const char* m = std::getenv("PHX_TERRAIN_MAXLEVEL")) maxLevel_ = std::min<u32>(u32(std::atoi(m)), kMaxLevel);
    pool_ = createBuffer(ctx, VkDeviceSize(kPoolSlots) * kVertsPerTile * sizeof(TerrainVertex),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::DeviceLocal);
    for (u32 i = 0; i < framesInFlight; ++i) {
        instanceBuffers_.push_back(createBuffer(ctx, sizeof(TerrainInstance) * kMaxInstances,
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::HostUpload));
        staging_.push_back(createBuffer(ctx, VkDeviceSize(kUploadsPerFrame) * kVertsPerTile * sizeof(TerrainVertex),
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::HostUpload));
    }
    for (u32 i = kPoolSlots; i-- > 0;) freeSlots_.push_back(i);

    // Shared index buffer: 64×64 quads (diagonal (i,j)–(i+1,j+1), matching the morph
    // rule for odd/odd vertices) followed by the four skirts.
    std::vector<u16> idx;
    auto g = [](u32 i, u32 j) { return u16(j * kGrid + i); };
    for (u32 j = 0; j + 1 < kGrid; ++j)
        for (u32 i = 0; i + 1 < kGrid; ++i) {
            idx.insert(idx.end(), {g(i, j), g(i + 1, j), g(i + 1, j + 1)});
            idx.insert(idx.end(), {g(i, j), g(i + 1, j + 1), g(i, j + 1)});
        }
    auto s = [](u32 edge, u32 k) { return u16(kGridVerts + edge * kGrid + k); };
    for (u32 k = 0; k + 1 < kGrid; ++k) {
        const u16 e[4][2] = {{g(k, 0), g(k + 1, 0)}, {g(k, kGrid - 1), g(k + 1, kGrid - 1)},
                             {g(0, k), g(0, k + 1)}, {g(kGrid - 1, k), g(kGrid - 1, k + 1)}};
        for (u32 edge = 0; edge < 4; ++edge) {
            const u16 a = e[edge][0], b = e[edge][1], sa = s(edge, k), sb = s(edge, k + 1);
            idx.insert(idx.end(), {a, b, sb, a, sb, sa});
        }
    }
    indexCount_ = u32(idx.size());
    indices_ = createBuffer(ctx, idx.size() * sizeof(u16), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            MemoryUsage::DeviceLocal);
    ImmediateSubmitter submitter(ctx);
    submitter.uploadBuffer(indices_, idx.data(), idx.size() * sizeof(u16));

    // Levels 0–2 (126 tiles) are generated up front and stay resident: the selection
    // always has a complete fallback, so the world never shows holes while streaming.
    std::vector<Request> base;
    for (u32 level = 0; level <= 2; ++level)
        for (u32 f = 0; f < 6; ++f)
            for (u32 y = 0; y < (1u << level); ++y)
                for (u32 x = 0; x < (1u << level); ++x) base.push_back({makeKey(f, level, x, y), 0, {}, 0, {}});
    jobs_.parallelFor(u32(base.size()), 1, {.priority = jobs::Priority::Critical},
                      [&](u32 b, u32 e) {
                          for (u32 i = b; i < e; ++i) generate(base[i]);
                      });
    for (Request& r : base) {
        const u32 slot = freeSlots_.back();
        freeSlots_.pop_back();
        Buffer tmp = createBuffer(ctx, kVertsPerTile * sizeof(TerrainVertex), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  MemoryUsage::HostUpload);
        std::memcpy(tmp.mapped, r.vertices.data(), kVertsPerTile * sizeof(TerrainVertex));
        submitter.run([&](VkCommandBuffer cmd) {
            VkBufferCopy c{0, VkDeviceSize(slot) * kVertsPerTile * sizeof(TerrainVertex), kVertsPerTile * sizeof(TerrainVertex)};
            vkCmdCopyBuffer(cmd, tmp.buffer, pool_.buffer, 1, &c);
        });
        destroyBuffer(ctx, tmp);
        Tile& t = tiles_[r.key];
        t.slot = slot;
        t.ready = true;
        t.center = r.center;
        t.radius = r.radius;
        t.lastUsed = ~0ull; // pinned
    }
    log::info("Terrain: {} base tiles resident, pool {} slots ({} MiB)", base.size(), kPoolSlots,
              (pool_.size >> 20));
}

TerrainRenderer::~TerrainRenderer() {
    // Jobs may still be generating: wait for them before freeing requests.
    while (true) {
        collectFinished();
        if (inFlight_ == 0) break;
        std::this_thread::yield();
    }
    for (Request* r : readyForUpload_) delete r;
    vkDestroyPipeline(ctx_.device, pipeline_, nullptr);
    for (auto& m : groundTextures_)
        for (Image& img : m) destroyImage(ctx_, img);
    destroyBuffer(ctx_, pool_);
    destroyBuffer(ctx_, indices_);
    for (Buffer& b : instanceBuffers_) destroyBuffer(ctx_, b);
    for (Buffer& b : staging_) destroyBuffer(ctx_, b);
}

// ------------------------------------------------------------------ tile generation (worker jobs)

void TerrainRenderer::generate(Request& r) const {
    const NodeId n = decode(r.key);
    const f64 size = 2.0 / f64(1u << n.level);
    const f64 u0 = -1.0 + n.x * size, v0 = -1.0 + n.y * size;

    constexpr u32 S = kGrid + 4; // two-sample border for normals
    const f64 spacing = edgeLength(n.level) / 64.0; // selects the DEM mip (no aliasing on coarse tiles)
    thread_local std::vector<glm::dvec3> pos;
    thread_local std::vector<f64> h;
    thread_local std::vector<glm::uvec2> hints;
    pos.resize(S * S);
    h.resize(S * S);
    hints.resize(S * S);
    auto un8 = [](f32 v) { return u32(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    for (u32 j = 0; j < S; ++j)
        for (u32 i = 0; i < S; ++i) {
            const f64 u = u0 + (f64(i) - 2.0) / 64.0 * size, v = v0 + (f64(j) - 2.0) / 64.0 * size;
            const glm::dvec3 dir = cubeToSphere(n.face, u, v);
            const bool interior = i >= 2 && j >= 2 && i < kGrid + 2 && j < kGrid + 2;
            f64 hh;
            if (interior) {
                // Terrain features and biome for the shader (materials, far-field colours).
                const SurfaceInfo si = heights_.surface(dir, spacing);
                hh = si.height;
                const BiomeWeights w = biomeWeights(si.tempC, si.precipMm, si.amplitudeC);
                const f32 ai = si.precipMm / potentialEvapotranspiration(si.tempC, si.amplitudeC);
                const f32 dry = 1.0f - std::clamp((ai - 0.25f) / 0.65f, 0.0f, 1.0f);
                hints[j * S + i] = {un8(si.sand) | un8(si.cliff) << 8 | un8(si.talus) << 16 | un8(w.desert) << 24,
                                    un8(w.forest) | un8(w.conifer) << 8 | un8(w.snow) << 16 | un8(dry) << 24};
            } else {
                hh = heights_.height(dir, spacing);
            }
            h[j * S + i] = hh;
            pos[j * S + i] = dir * (kPlanetRadiusM + std::max(hh, 0.0)); // sea surface at 0
        }

    const glm::dvec3 centerDir = cubeToSphere(n.face, u0 + size / 2, v0 + size / 2);
    const f64 centerH = std::max(heights_.height(centerDir, spacing), 0.0);
    r.center = centerDir * (kPlanetRadiusM + centerH);

    r.vertices.resize(kVertsPerTile);
    f64 radius = 0;
    auto at = [&](i32 i, i32 j) -> const glm::dvec3& { return pos[u32(j + 2) * S + u32(i + 2)]; };
    // Normals from a stencil of ≥ ~2.4 m on the finest level too, so adjacent levels
    // (1.2 m and 2.4 m vertex spacing) shade identically across their seam.
    const i32 k = n.level >= 18 ? 2 : 1;
    for (u32 j = 0; j < kGrid; ++j)
        for (u32 i = 0; i < kGrid; ++i) {
            const glm::dvec3& p = at(i, j);
            const f64 hh = h[(j + 2) * S + (i + 2)];
            const i32 ii = i32(i), jj = i32(j);
            glm::dvec3 nrm = hh < 0 ? glm::normalize(p)
                                    : glm::normalize(glm::cross(at(ii + k, jj) - at(ii - k, jj), at(ii, jj + k) - at(ii, jj - k)));
            const glm::dvec3 rel = p - r.center;
            radius = std::max(radius, glm::length(rel));
            TerrainVertex& v = r.vertices[j * kGrid + i];
            v.pos[0] = f32(rel.x), v.pos[1] = f32(rel.y), v.pos[2] = f32(rel.z);
            v.packed = packNormal(nrm, hh);
            v.hintsA = hints[(j + 2) * S + (i + 2)].x;
            v.hintsB = hints[(j + 2) * S + (i + 2)].y;
        }
    // Skirts hang below the edges and hide cracks against coarser neighbours.
    const f64 skirt = edgeLength(n.level) * 0.03 + 20.0;
    for (u32 edge = 0; edge < 4; ++edge)
        for (u32 k = 0; k < kGrid; ++k) {
            const u32 i = edge < 2 ? k : (edge == 2 ? 0 : kGrid - 1);
            const u32 j = edge < 2 ? (edge == 0 ? 0 : kGrid - 1) : k;
            const glm::dvec3& p = at(i, j);
            const glm::dvec3 rel = p - glm::normalize(p) * skirt - r.center;
            TerrainVertex& v = r.vertices[kGridVerts + edge * kGrid + k];
            v.pos[0] = f32(rel.x), v.pos[1] = f32(rel.y), v.pos[2] = f32(rel.z);
            v.packed = r.vertices[j * kGrid + i].packed;
            v.hintsA = r.vertices[j * kGrid + i].hintsA;
            v.hintsB = r.vertices[j * kGrid + i].hintsB;
        }
    r.radius = radius + skirt;
}

// ------------------------------------------------------------------ residency

// Collects up to `needed` least-recently-used ready tiles not needed this frame, once
// per frame (O(n log k)); allocateSlot() then pops from that list.
void TerrainRenderer::prepareEviction(u64 frame, u32 needed) {
    evictable_.clear();
    if (needed <= freeSlots_.size()) return;
    const std::size_t want = needed - freeSlots_.size();
    for (const auto& [key, t] : tiles_)
        if (t.ready && t.lastUsed < frame) evictable_.push_back({t.lastUsed, key});
    const std::size_t k = std::min(want, evictable_.size());
    std::partial_sort(evictable_.begin(), evictable_.begin() + std::ptrdiff_t(k), evictable_.end());
    evictable_.resize(k);
    std::reverse(evictable_.begin(), evictable_.end()); // oldest at the back
}

u32 TerrainRenderer::allocateSlot() {
    if (!freeSlots_.empty()) {
        const u32 s = freeSlots_.back();
        freeSlots_.pop_back();
        return s;
    }
    while (!evictable_.empty()) {
        const u64 key = evictable_.back().second;
        evictable_.pop_back();
        auto it = tiles_.find(key);
        if (it == tiles_.end() || !it->second.ready) continue;
        const u32 s = it->second.slot;
        tiles_.erase(it);
        return s;
    }
    return ~0u;
}

bool TerrainRenderer::ensureRequested(u64 key, f64 priority) {
    auto it = tiles_.find(key);
    if (it != tiles_.end()) return it->second.ready;
    wanted_.push_back({key, priority});
    return false;
}

void TerrainRenderer::collectFinished() {
    std::lock_guard lock(doneMutex_);
    for (Request* r : done_) readyForUpload_.push_back(r);
    inFlight_ -= u32(done_.size());
    done_.clear();
}

// ------------------------------------------------------------------ selection

void TerrainRenderer::select(u64 key, const glm::dvec3& cam, u64 frame) {
    Tile& t = tiles_[key];
    const NodeId n = decode(key);
    const glm::dvec3 rel = t.center - cam;
    const glm::vec3 relF(rel);

    // Frustum (side planes; reverse-Z with infinite far has no useful far plane).
    for (const glm::vec4& p : frustum_)
        if (glm::dot(glm::vec3(p), relF) + p.w < -f32(t.radius)) return;

    // Horizon: cull nodes entirely inside the occlusion cone of the sea-level sphere —
    // within the cone's angle (shrunk by the node's angular radius) and farther than
    // the tangent distance, every point lies behind the planet.
    const f64 camDist = glm::length(cam);
    const f64 occluder = kPlanetRadiusM - 500.0;
    if (camDist > occluder) {
        const f64 d = glm::length(rel);
        const f64 tangentDist = std::sqrt(camDist * camDist - occluder * occluder);
        if (d - t.radius > tangentDist) {
            const f64 coneAngle = std::asin(occluder / camDist);
            const f64 theta = std::acos(std::clamp(glm::dot(rel / d, -cam / camDist), -1.0, 1.0));
            const f64 beta = std::asin(std::min(1.0, t.radius / d));
            if (theta + beta < coneAngle) return;
        }
    }

    const f64 dist = lodDistance(std::max(0.0, glm::length(rel) - t.radius));
    const f64 split = splitFactor_ * edgeLength(n.level);
    if (n.level < maxLevel_ && dist < split) {
        bool all = true;
        u64 children[4];
        for (u32 c = 0; c < 4; ++c) {
            children[c] = makeKey(n.face, n.level + 1, n.x * 2 + (c & 1), n.y * 2 + (c >> 1));
            all &= ensureRequested(children[c], dist / edgeLength(n.level)); // relative distance: most visible first
        }
        if (all) {
            if (t.lastUsed != ~0ull) t.lastUsed = frame; // interior nodes stay resident too
            for (u64 c : children) select(c, cam, frame);
            return;
        }
    }

    // Draw this node.
    if (t.lastUsed != ~0ull) t.lastUsed = frame;
    if (instances_.size() >= kMaxInstances) return;
    TerrainInstance inst{};
    inst.offset[0] = relF.x, inst.offset[1] = relF.y, inst.offset[2] = relF.z;
    inst.slot = t.slot;
    inst.pad[0] = f32(n.level);
    if (n.level == 0) {
        inst.morphStart = inst.morphEnd = 1e30f;
    } else {
        inst.morphEnd = f32(2.0 * split);
        inst.morphStart = f32(0.75 * 2.0 * split);
    }
    instances_.push_back(inst);
    stats_.maxLevel = std::max(stats_.maxLevel, n.level);
}

void TerrainRenderer::update(u32 slot, const glm::dvec3& cam, const glm::mat4& vp, f32 fovY, u32 viewportHeight,
                             u64 frame) {
    const auto updateStart = std::chrono::steady_clock::now();
    stats_ = {};
    collectFinished();

    // Frustum side planes (Gribb–Hartmann) in camera-relative ECEF.
    const glm::mat4 m = glm::transpose(vp);
    frustum_[0] = m[3] + m[0];
    frustum_[1] = m[3] - m[0];
    frustum_[2] = m[3] + m[1];
    frustum_[3] = m[3] - m[1];
    for (glm::vec4& p : frustum_) p /= glm::length(glm::vec3(p));

    // CDLOD split distance from the screen-space target: vertex spacing ≤ ~4 px
    // (shading detail below that comes from normals / textures, not geometry).
    const f64 pixelsPerRadian = viewportHeight / (2.0 * std::tan(fovY * 0.5));
    splitFactor_ = std::max(2.2, pixelsPerRadian / (64.0 * 4.0));

    instances_.clear();
    wanted_.clear();
    for (u32 f = 0; f < 6; ++f) select(makeKey(f, 0, 0, 0), cam, frame);

    // Request missing children: coarse levels and near tiles first.
    std::sort(wanted_.begin(), wanted_.end(), [](const Candidate& a, const Candidate& b) { return a.priority < b.priority; });
    wanted_.erase(std::unique(wanted_.begin(), wanted_.end(), [](const Candidate& a, const Candidate& b) { return a.key == b.key; }),
                  wanted_.end());
    prepareEviction(frame, std::min<u32>(u32(wanted_.size()), maxInFlight_ - std::min(inFlight_, maxInFlight_)));
    for (const Candidate& c : wanted_) {
        if (inFlight_ >= maxInFlight_) break;
        if (tiles_.count(c.key)) continue;
        const u32 s = allocateSlot();
        if (s == ~0u) break;
        Tile& t = tiles_[c.key];
        t.slot = s;
        t.ready = false;
        auto* req = new Request{c.key, s, {}, 0, {}};
        ++inFlight_;
        jobs_.submit({.priority = jobs::Priority::Background}, [this, req] {
            const auto t0 = std::chrono::steady_clock::now();
            generate(*req);
            generateNs_.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                                  std::memory_order_relaxed);
            generated_.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard lock(doneMutex_);
            done_.push_back(req);
        });
    }

    // Stage finished tiles (budgeted) for upload in this frame's command buffer.
    pendingCopies_.clear();
    u32 n = 0;
    auto* dst = static_cast<TerrainVertex*>(staging_[slot].mapped);
    auto it = readyForUpload_.begin();
    for (; it != readyForUpload_.end() && n < kUploadsPerFrame; ++it) {
        Request* r = *it;
        auto t = tiles_.find(r->key);
        if (t != tiles_.end() && t->second.slot == r->slot) {
            std::memcpy(dst + std::size_t(n) * kVertsPerTile, r->vertices.data(), kVertsPerTile * sizeof(TerrainVertex));
            pendingCopies_.push_back({n, r->slot});
            t->second.ready = true; // copy precedes the draw in this frame's command buffer
            t->second.center = r->center;
            t->second.radius = r->radius;
            t->second.lastUsed = frame;
            ++n;
        }
        delete r;
    }
    readyForUpload_.erase(readyForUpload_.begin(), it);
    if (n) vmaFlushAllocation(ctx_.allocator, staging_[slot].allocation, 0, VK_WHOLE_SIZE);

    // Front-to-back: overlapping ridges are rejected by early depth testing instead of
    // running the (heavy) terrain fragment shader several times per pixel.
    std::sort(instances_.begin(), instances_.end(), [](const TerrainInstance& a, const TerrainInstance& b) {
        return a.offset[0] * a.offset[0] + a.offset[1] * a.offset[1] + a.offset[2] * a.offset[2] <
               b.offset[0] * b.offset[0] + b.offset[1] * b.offset[1] + b.offset[2] * b.offset[2];
    });
    std::memcpy(instanceBuffers_[slot].mapped, instances_.data(), instances_.size() * sizeof(TerrainInstance));
    vmaFlushAllocation(ctx_.allocator, instanceBuffers_[slot].allocation, 0, VK_WHOLE_SIZE);

    stats_.drawnTiles = u32(instances_.size());
    stats_.residentTiles = u32(tiles_.size());
    stats_.pendingTiles = inFlight_ + u32(readyForUpload_.size());
    stats_.uploadsThisFrame = n;
    stats_.tilesGenerated = generated_.load(std::memory_order_relaxed);
    stats_.generateMsTotal = f64(generateNs_.load(std::memory_order_relaxed)) * 1e-6;
    stats_.updateMs = std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - updateStart).count();
}

void TerrainRenderer::recordUploads(VkCommandBuffer cmd, u32 slot) {
    if (pendingCopies_.empty()) return;
    // Earlier frames may still read slots being overwritten (evicted tiles).
    VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    before.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    before.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &before;
    vkCmdPipelineBarrier2(cmd, &dep);

    std::vector<VkBufferCopy> copies;
    constexpr VkDeviceSize tileBytes = kVertsPerTile * sizeof(TerrainVertex);
    for (auto [stagingIndex, poolSlot] : pendingCopies_)
        copies.push_back({stagingIndex * tileBytes, poolSlot * tileBytes, tileBytes});
    vkCmdCopyBuffer(cmd, staging_[slot].buffer, pool_.buffer, u32(copies.size()), copies.data());

    VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    after.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    after.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    after.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    after.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    dep.pMemoryBarriers = &after;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void TerrainRenderer::recordDraw(VkCommandBuffer cmd, u64 frameAddr, u32 slot) {
    if (instances_.empty()) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bindless_.pipelineLayout, 0, 1, &bindless_.set, 0,
                            nullptr);
    static const bool levelDebug = std::getenv("PHX_TERRAIN_LEVELS") != nullptr;
    PushConstants pc{frameAddr, instanceBuffers_[slot].address, u32(instances_.size()), levelDebug ? 1u : 0u, pool_.address};
    vkCmdPushConstants(cmd, bindless_.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdBindIndexBuffer(cmd, indices_.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, indexCount_, u32(instances_.size()), 0, 0, 0);
}

} // namespace phx
