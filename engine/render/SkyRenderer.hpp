#pragma once
// Sky rendering (plan §2.6): Hillaire atmosphere LUTs, sun/moon disks, airglow,
// Milky Way, and ~40k catalog stars + planets as PSF sprites.
#include "engine/gpu/Resources.hpp"
#include "engine/render/FrameSnapshot.hpp"

namespace phx {

// Bindless slots — mirror shaders/common.glsl.
enum BindlessSlot : u32 {
    kTexTransmittance = 0,
    kTexMultiScatter = 1,
    kTexSkyViewSun = 2,
    kTexSkyViewMoon = 3,
    kTexHdr = 4,
    kTexShadowNear = 5,
    kTexShadowFar = 6,
    kTexShadowVisNear = 7,
    kTexShadowVisFar = 8,
    kTexDepth = 9,
    kTexCascade0 = 20, // + cascade
    kTexImpostorAlbedo0 = 10, // + species
    kTexImpostorNormal0 = 13, // + species
    kTexTreeMaterial0 = 32,   // + (species·8 + material)·3 + {albedo, normal, ARM}
    kTexGround0 = 512,        // + material·3 + {albedo, normal, ARM}   (species materials use 32–271)
};
// Storage-image slots (shaders/common.glsl STOR_*); 0–3 are the atmosphere LUTs.
enum StorageSlot : u32 { kStorShadowVisNear = 4, kStorShadowVisFar = 5 };

// Mirrors FrameRef in shaders/common.glsl.
struct FrameUniforms {
    glm::mat4 viewProj;      // ENU, rotation only (sky)
    glm::mat4 viewProjEcef;  // camera-relative ECEF → clip (terrain)
    glm::mat4 ecefToEnu;     // camera's local frame
    glm::mat4 invViewProj;
    glm::mat4 icrsToEnu;
    glm::vec4 sunDir;
    glm::vec4 moonDir;
    glm::vec4 moonParams;
    glm::vec4 camera;
    glm::vec4 resolution;
    glm::vec4 tonemap;
    glm::vec4 cameraEcef;    // xyz metres (float, for altitude only), w planet radius
    glm::mat4 shadowEnu[2];  // terrain shadow height maps (see TerrainShadows)
    glm::vec4 shadowOffset[2];
    glm::vec4 shadowParams;
    glm::vec4 world;         // x day of year (0–365), y wind time [s], z wind strength, w season (0 summer … 1 late autumn)
    glm::vec4 windDir;       // xyz wind direction (camera ENU), w gust phase
    glm::vec4 cullParams;    // x near plane [m], y pixels per radian, z render width, w render height
    glm::vec4 cameraMod;     // xyz camera ECEF mod 65536 m (computed in double: precise texture coordinates)
    glm::mat4 csm[3];        // cascaded shadow maps (CascadedShadows)
    glm::vec4 csmSplits;
    glm::vec4 csmTexel;
    glm::vec4 ground;        // rgb mean ground albedo around the camera (bounce light)
};

struct PushConstants {
    u64 frame = 0;
    u64 stars = 0;
    u32 starCount = 0;
    u32 lightIndex = 0;
    u64 aux = 0;
    u64 aux2 = 0;
    u64 aux3 = 0;
};

class SkyRenderer {
public:
    static constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

    SkyRenderer(VulkanContext& ctx, Bindless& bindless, u32 framesInFlight, f64 starEpochYear);
    ~SkyRenderer();

    // Fills this frame's uniform buffer; returns its device address.
    u64 updateFrame(u32 slot, const FrameSnapshot& snap, VkExtent2D extent, u64 frameCounter,
                    const struct ShadowMapUniforms& shadows, const struct CascadeUniforms& cascades);
    [[nodiscard]] const glm::mat4& viewProjEcef() const { return viewProjEcef_; }

    // Before the HDR render pass: refresh sky-view LUTs (compute).
    void recordCompute(VkCommandBuffer cmd, u64 frameAddr);
    // Inside a dynamic-rendering pass on the HDR target.
    void recordSky(VkCommandBuffer cmd, u64 frameAddr, u32 slot);

    [[nodiscard]] u32 starCount() const { return starCount_; }

private:
    void precomputeStaticLuts();
    void loadStars(f64 year);

    VulkanContext& ctx_;
    Bindless& bindless_;
    Image transmittance_, multiScatter_, skyViewSun_, skyViewMoon_;
    VkPipeline transmittancePipe_ = VK_NULL_HANDLE, multiScatterPipe_ = VK_NULL_HANDLE, skyViewPipe_ = VK_NULL_HANDLE;
    VkPipeline skyPipe_ = VK_NULL_HANDLE, starsPipe_ = VK_NULL_HANDLE;
    Buffer stars_;
    u32 starCount_ = 0;
    std::vector<Buffer> frameUniforms_; // per frame in flight, host-visible
    std::vector<Buffer> planets_;       // per frame in flight, host-visible
    glm::mat4 viewProjEcef_{1.0f};
};

} // namespace phx
