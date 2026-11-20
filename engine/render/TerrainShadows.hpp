#pragma once
// Terrain sun shadows out to 32 km (plan §2.7, shadow row 1–32 km).
//
// Two camera-anchored height maps (near: 6 km / 2048², far: 64 km / 2048²) are filled
// by background worker jobs straight into mapped staging memory, copied to R32F
// textures, and ray-marched towards the sun per pixel in terrain.frag. Because the
// march happens every frame, the sun can move freely (time-lapse, eclipses) — unlike
// baked horizon maps, which would need re-baking per sun direction bucket.
#include "engine/gpu/Resources.hpp"
#include "engine/jobs/JobSystem.hpp"

#include <glm/glm.hpp>

#include <atomic>

namespace phx {
class HeightField;

struct ShadowMapUniforms {
    glm::mat4 anchorEnu[2];    // rows east/north/up of each map's anchor (upper 3x3)
    glm::vec4 anchorOffset[2]; // xyz anchor − camera (camera-relative ECEF), w map size [m]
    glm::vec4 params;          // x enabled, y near texel [m], z far texel [m]
};

class TerrainShadows {
public:
    static constexpr u32 kSize = 2048;
    static constexpr f64 kExtent[2] = {6000.0, 64000.0}; // near ≈2.9 m, far ≈31 m texels

    TerrainShadows(VulkanContext& ctx, Bindless& bindless, jobs::JobSystem& jobs, const HeightField& heights,
                   u32 framesInFlight);
    ~TerrainShadows();

    // Render thread, once per frame: start regeneration when the camera left the
    // map's central region, and stage finished maps for upload.
    void update(const glm::dvec3& cameraEcef, const glm::dvec3& sunEcef, u64 frame);
    // Outside render passes: map uploads, then (if needed) the visibility recompute.
    void recordUploads(VkCommandBuffer cmd, u64 frameAddr);
    [[nodiscard]] ShadowMapUniforms uniforms(const glm::dvec3& cameraEcef) const;

private:
    struct GenParams {
        glm::dvec3 anchor, east, north;
        f64 extent = 0, texel = 0;
    };
    struct Level {
        GenParams gen;                 // read by the generation jobs
        Image image;
        Image visibility;              // cached sun visibility (RGBA16F, .r)
        Buffer staging;
        glm::dvec3 anchor{0.0};        // anchor on the sea-level sphere (ECEF)
        glm::dvec3 pendingAnchor{0.0};
        bool valid = false;            // image holds data for `anchor`
        bool generating = false;
        bool uploadPending = false;
        u64 reuseFrame = 0;            // staging may be rewritten from this frame on
        jobs::Counter counter;
        std::atomic<u32> maxHeightBits{0}; // running max of generated heights (float bits, ≥ 0)
        f32 maxHeight = 0.0f;               // of the map currently in the image
    };

    void startGeneration(u32 level, const glm::dvec3& anchor);

    VulkanContext& ctx_;
    Bindless& bindless_;
    jobs::JobSystem& jobs_;
    const HeightField& heights_;
    u32 framesInFlight_;
    Level levels_[2];
    bool layoutsInitialised_ = false;
    VkPipeline visPipe_ = VK_NULL_HANDLE;
    glm::dvec3 lastSun_{0.0};
    bool visDirty_ = true;
};

} // namespace phx
