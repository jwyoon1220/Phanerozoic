#pragma once
// Instanced nature (plan M5): trees, understorey ferns and rocks from Poly Haven assets.
//  * Placement layers (cube-sphere cells, filled by worker jobs with the terrain's own
//    height function so instances sit exactly on it):
//      level 14 (~610 m) within 4.2 km: trees
//      level 16 (~150 m) within 450 m: boulders
//      level 18 (~76 m)  within 130 m: ferns, mossy stones
//  * GPU culling: camera pass (frustum + Hi-Z against this frame's terrain depth,
//    per-species LOD distances) and a shadow pass (per cascade, independent of the
//    camera frustum) append to indirect buckets.
//  * Drawing: 3 mesh LODs per species; trees switch to impostors baked from the full
//    asset; shadow casters use LOD2 meshes or light-facing impostor quads.
#include "engine/gpu/Resources.hpp"
#include "engine/jobs/JobSystem.hpp"
#include "engine/world/NaturePlacement.hpp"
#include "engine/world/TreeMeshes.hpp"

#include <glm/glm.hpp>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace phx {
class HeightField;
class CascadedShadows;


struct VegStats {
    u32 cells = 0;
    u32 instances = 0;
    u32 pendingCells = 0;
};

class VegetationRenderer {
public:
    static constexpr u32 kCellSlots = 384;
    static constexpr u32 kMaxPerCell = kMaxNaturePerCell;
    static constexpr u32 kCameraBuckets = kSpeciesCount * kTreeMeshLods + kTreeSpeciesCount; // 30
    static constexpr u32 kShadowCasters = 5; // 3 trees, boulder, dryland tree
    static constexpr u32 kBuckets = kCameraBuckets + 3 * kShadowCasters; // 45
    static constexpr u32 kUploadsPerFrame = 8;
    static constexpr u32 kImpostorFrames = 8;
    static constexpr u32 kImpostorFrameSize = 256;

    VegetationRenderer(VulkanContext& ctx, Bindless& bindless, jobs::JobSystem& jobs, const HeightField& heights,
                       VkFormat colorFormat, VkFormat depthFormat);
    ~VegetationRenderer();

    void update(u32 slot, const glm::dvec3& cameraEcef, u64 frame); // render thread
    void recordUploads(VkCommandBuffer cmd, u32 slot);
    void recordShadowCull(VkCommandBuffer cmd, u64 frameAddr, u32 slot); // before the shadow maps
    void recordShadows(VkCommandBuffer cmd, u64 frameAddr, CascadedShadows& csm);
    void recordCull(VkCommandBuffer cmd, u64 frameAddr, u32 slot);       // after HiZ::record
    void recordDraw(VkCommandBuffer cmd, u64 frameAddr);                 // inside the HDR pass
    [[nodiscard]] const VegStats& stats() const { return stats_; }

private:
    struct Cell {
        u32 slot = ~0u;
        u32 count = 0;
        bool ready = false;
        glm::dvec3 center{0.0};
    };
    struct Request {
        u64 key;
        u32 slot;
        glm::dvec3 center;
        std::vector<VegInstance> instances;
    };
    struct CellGpu {
        f32 offset[3];
        u32 slot;
        u32 count;
        u32 pad[3];
    };

    void bakeImpostors();
    void dispatchCull(VkCommandBuffer cmd, u64 frameAddr, u32 slot, u32 mode);

    VulkanContext& ctx_;
    Bindless& bindless_;
    jobs::JobSystem& jobs_;
    const HeightField& heights_;

    Buffer vertices_, indices_;
    struct MeshRange {
        u32 firstIndex, indexCount, vertexOffset;
    } meshes_[kSpeciesCount][kTreeMeshLods + 1]{}; // last = impostor quad
    TreeSpeciesData species_[kSpeciesCount];
    NatureSpeciesDims dims_[kSpeciesCount];
    Image materialTextures_[kSpeciesCount][kMaxTreeMaterials][3]{};
    Image impostorAlbedo_[kTreeSpeciesCount], impostorNormal_[kTreeSpeciesCount];

    Buffer pool_;
    std::vector<Buffer> cellBuffers_;
    std::vector<Buffer> staging_;
    std::vector<std::pair<u32, u32>> pendingCopies_;
    std::vector<u32> pendingCounts_;
    Buffer visible_, indirect_;

    VkPipeline cullPipe_ = VK_NULL_HANDLE, meshPipe_ = VK_NULL_HANDLE, impostorPipe_ = VK_NULL_HANDLE;
    VkPipeline shadowMeshPipe_ = VK_NULL_HANDLE, shadowImpostorPipe_ = VK_NULL_HANDLE;

    std::unordered_map<u64, Cell> cells_;
    std::vector<u32> freeSlots_;
    std::vector<CellGpu> frameCells_;
    std::mutex doneMutex_;
    std::vector<Request*> done_, readyForUpload_;
    u32 inFlight_ = 0, maxInFlight_ = 4;
    VegStats stats_;
};

} // namespace phx
