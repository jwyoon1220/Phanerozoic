#pragma once
#include <cstdlib>
// Cascaded sun shadow maps for vegetation casters (trees, rocks) onto terrain and
// vegetation receivers, out to 700 m. Beyond that, terrain shadows come from the height
// maps (TerrainShadows) and canopy shading is approximated.
//  * 3 cascades (30 / 150 / 700 m), 2048² D32 each, orthographic along the sun
//  * stable cascades: bounding-sphere fit + texel snapping in a world-locked frame
//    (double precision on the CPU), so shadows do not shimmer as the camera moves
#include "engine/gpu/Resources.hpp"

#include <glm/glm.hpp>

namespace phx {

inline constexpr u32 kCascades = 3;

struct CascadeUniforms {
    glm::mat4 lightViewProj[kCascades]; // camera-relative ECEF → light clip (z ∈ [0,1])
    glm::vec4 splits;                    // far distance of cascades 0–2, w enabled
    glm::vec4 texel;                     // world texel size per cascade, w = map size
};

class CascadedShadows {
public:
    static constexpr u32 kSize = 2048;
    static constexpr f32 kSplits[kCascades] = {30.0f, 150.0f, 700.0f};

    CascadedShadows(VulkanContext& ctx, Bindless& bindless);
    ~CascadedShadows();

    // Render thread: fit cascades to the camera (camera-relative ECEF basis given by
    // forward/right/up), sun direction in ECEF.
    void update(const glm::dvec3& cameraEcef, const glm::dvec3& forward, const glm::dvec3& right,
                const glm::dvec3& up, f32 fovY, f32 aspect, const glm::dvec3& sunEcef);
    [[nodiscard]] const CascadeUniforms& uniforms() const { return u_; }

    // Begin/end depth-only rendering into cascade c (layout handling included).
    void begin(VkCommandBuffer cmd, u32 c);
    void end(VkCommandBuffer cmd, u32 c);

private:
    VulkanContext& ctx_;
    Bindless& bindless_;
    Image maps_[kCascades];
    bool initialised_[kCascades]{};
    CascadeUniforms u_{};
    bool enabled_ = !(std::getenv("PHX_CSM") && std::getenv("PHX_CSM")[0] == '0'); // PHX_CSM=0 disables
};

} // namespace phx
