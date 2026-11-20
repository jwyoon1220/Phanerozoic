#pragma once
#include "engine/astro/Astro.hpp"
#include "engine/astro/StarCatalog.hpp"
#include "engine/core/Types.hpp"
#include "engine/platform/Window.hpp"

#include <glm/glm.hpp>

namespace phx {

struct CameraState {
    glm::dvec3 positionEcef{0.0, 0.0, 6.4e6}; // metres, planet centre origin
    f32 yaw = 0.0f;      // azimuth, radians, 0 = north, +east (local ENU frame)
    f32 pitch = 0.0f;    // radians above the horizon
    f32 fovY = 1.0472f;  // radians
    f32 nearPlane = 0.05f; // metres; from height above ground (set by the supervisor)
};

// Sky inputs, already reduced to floats in the observer's ENU frame.
struct SkySnapshot {
    glm::mat3 icrsToEnu{1.0f};
    glm::vec3 sunDir{0, 0, 1};
    f32 sunRadius = 0.00465f;
    glm::vec3 moonDir{0, 1, 0};
    f32 moonRadius = 0.0045f;
    f32 moonIlluminance = 0.0f;
    f32 moonFraction = 0.0f;
    f32 earthshine = 0.0f;
    f32 observerAltitudeKm = 0.0f;
    f32 dayOfYear = 172.0f;  // 0–365 (local), drives deciduous autumn colours
    astro::GpuStar planets[astro::kPlanetCount]{}; // ICRS directions, like stars
};

// Immutable per-tick state handed from the supervisor to the render thread.
// On-foot HUD values (supervisor → overlay).
struct PlayerHud {
    bool visible = false;
    char mode[12] = {};
    f32 health = 100, stamina = 100, energy = 100, water = 100; // %
    f32 coreTempC = 37, airTempC = 20, speed = 0, slopeDeg = 0, powerW = 100;
};

struct FrameSnapshot {
    u64 tick = 0;
    f64 simTime = 0.0;          // real seconds since start
    Extent2D framebuffer;
    CameraState camera;
    SkySnapshot sky;
    f32 exposure = 1.0f;        // pre-exposure multiplier
    glm::vec3 groundAlbedo{0.12f}; // biome mean around the camera (ground-bounce light)
    f32 adaptationLuminance = 0.05f;
    u32 screenshotRequest = 0;  // increments when a capture is wanted
    bool perfOverlay = true;    // F3: performance overlay (top right)
    PlayerHud hud;
};

} // namespace phx
