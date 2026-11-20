#pragma once
#include "engine/core/Types.hpp"
#include "engine/gpu/Swapchain.hpp"

#include <string>

namespace phx {

struct EngineConfig {
    u32 width = 1600;
    u32 height = 900;
    const char* title = "Phanerozoic";
    bool validation = true;
    PresentPolicy present = PresentPolicy::Vsync;
    u32 workerCount = 0;       // 0 = auto
    f64 simHz = 60.0;          // fixed simulation step
    f64 autoCloseSeconds = 0;  // >0: quit after this long (smoke tests)

    // World / scenario
    f64 startJdUT = 0;         // 0 = system clock
    f64 timeRate = 1.0;
    // Highest point of the Huashan massif in the 30 m DEM (South Peak area; the DEM's
    // 30 m posts undercut the granite pinnacle: 2016 m vs. 2154.9 m surveyed).
    f64 startLatDeg = 34.4625;
    f64 startLonDeg = 110.0803;
    f64 startHeightAboveGround = 1.7; // eye height; large values give aerial / orbital views
    f64 viewAzimuthDeg = 180;  // initial camera: azimuth (0 = north), altitude, vertical FOV
    f64 viewAltitudeDeg = 25;
    f64 viewFovDeg = 70;
    std::string screenshotPath = "screenshot.png";
    f64 screenshotAtSeconds = 0; // >0: capture after this many seconds, then quit
    f64 flySpeed = 0;            // >0: scripted flight along the view azimuth at constant AGL (benchmarks)
};

// Reads PHX_* environment overrides: PHX_PRESENT, PHX_VALIDATION, PHX_WORKERS,
// PHX_AUTOCLOSE, PHX_DATE (UTC, YYYY-MM-DDTHH:MM), PHX_TIME_RATE, PHX_VIEW (az,alt,fov),
// PHX_SCREENSHOT (path), PHX_SCREENSHOT_AT (seconds), PHX_POS (lat,lon,heightAboveGround),
// PHX_FLY (m/s, scripted streaming benchmark), PHX_SIZE (WxH window size).
EngineConfig engineConfigFromEnvironment(EngineConfig base = {});

// Owns the thread layout: the calling (main) thread becomes the supervisor.
int runEngine(const EngineConfig& config);

} // namespace phx
