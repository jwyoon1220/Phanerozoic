#include "engine/Engine.hpp"

#include "engine/core/Log.hpp"
#include "engine/core/TripleBuffer.hpp"
#include "engine/game/PhysicsWorld.hpp"
#include "engine/game/Player.hpp"
#include "engine/gpu/VulkanContext.hpp"
#include "engine/jobs/JobSystem.hpp"
#include "engine/platform/Window.hpp"
#include "engine/render/FrameSnapshot.hpp"
#include "engine/render/Renderer.hpp"
#include "engine/render/Upscaler.hpp"
#include "engine/world/Climate.hpp"
#include "engine/world/Environment.hpp"
#include "engine/world/HeightField.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>

namespace phx {
namespace {
using Clock = std::chrono::steady_clock;

#ifndef PHX_DATA_DIR
#define PHX_DATA_DIR "data"
#endif

// Free-fly camera (M3 debug navigation; walking/climbing arrives in M6).
// WASD: move, E/Q: up/down, Shift: ×10. Speed scales with height above ground so
// the same keys work from a ridge walk to low orbit.
void moveCamera(CameraState& cam, const Window& w, const HeightField& hf, f64 dt) {
    const glm::dmat3 toEnu = ecefToEnuMatrix(cam.positionEcef);
    const glm::dmat3 toEcef = glm::transpose(toEnu);
    const f64 cy = std::cos(cam.yaw), sy = std::sin(cam.yaw), cp = std::cos(cam.pitch), sp = std::sin(cam.pitch);
    const glm::dvec3 forward = toEcef * glm::dvec3(sy * cp, cy * cp, sp);
    const glm::dvec3 right = toEcef * glm::dvec3(cy, -sy, 0);
    const glm::dvec3 up = glm::normalize(cam.positionEcef);

    glm::dvec3 move(0.0);
    if (w.keyDown(GLFW_KEY_W)) move += forward;
    if (w.keyDown(GLFW_KEY_S)) move -= forward;
    if (w.keyDown(GLFW_KEY_D)) move += right;
    if (w.keyDown(GLFW_KEY_A)) move -= right;
    if (w.keyDown(GLFW_KEY_E)) move += up;
    if (w.keyDown(GLFW_KEY_Q)) move -= up;

    const f64 ground = kPlanetRadiusM + std::max(hf.height(up), 0.0);
    const f64 agl = glm::length(cam.positionEcef) - ground;
    if (glm::length(move) > 0) {
        f64 speed = std::clamp(agl * 0.6, 4.0, 3.0e6);
        if (w.keyDown(GLFW_KEY_LEFT_SHIFT)) speed *= 10.0;
        cam.positionEcef += glm::normalize(move) * speed * dt;
    }
    // Keep the eye above the surface (terrain or sea).
    const glm::dvec3 newUp = glm::normalize(cam.positionEcef);
    const f64 floor = kPlanetRadiusM + std::max(hf.height(newUp), 0.0) + 1.7;
    if (glm::length(cam.positionEcef) < floor) cam.positionEcef = newUp * floor;
}

void applyCameraInput(CameraState& cam, const InputFrame& in) {
    const f32 radPerPixel = 0.0025f * (cam.fovY / 1.0472f); // slower when zoomed in
    cam.yaw += f32(in.mouseDx) * radPerPixel;
    cam.pitch -= f32(in.mouseDy) * radPerPixel;
    cam.pitch = std::clamp(cam.pitch, -1.55f, 1.55f);
    cam.yaw = std::remainder(cam.yaw, 2.0f * 3.14159265f);
    if (in.scroll != 0.0) cam.fovY = std::clamp(cam.fovY * std::pow(0.9f, f32(in.scroll)), 0.02f, 1.75f);
}
} // namespace

EngineConfig engineConfigFromEnvironment(EngineConfig c) {
    if (const char* p = std::getenv("PHX_PRESENT")) {
        if (!std::strcmp(p, "mailbox")) c.present = PresentPolicy::LowLatency;
        else if (!std::strcmp(p, "immediate")) c.present = PresentPolicy::Uncapped;
        else c.present = PresentPolicy::Vsync;
    }
    if (const char* v = std::getenv("PHX_VALIDATION")) c.validation = std::strcmp(v, "0") != 0;
    if (const char* w = std::getenv("PHX_WORKERS")) c.workerCount = u32(std::strtoul(w, nullptr, 10));
    if (const char* a = std::getenv("PHX_AUTOCLOSE")) c.autoCloseSeconds = std::strtod(a, nullptr);
    if (const char* d = std::getenv("PHX_DATE")) {
        if (!parseIsoUtc(d, c.startJdUT)) log::warn("PHX_DATE '{}' not understood (YYYY-MM-DDTHH:MM, UTC)", d);
    }
    if (const char* r = std::getenv("PHX_TIME_RATE")) c.timeRate = std::strtod(r, nullptr);
    if (const char* v = std::getenv("PHX_VIEW")) {
        f32 az = 180, alt = 30, fov = 60;
        std::sscanf(v, "%f,%f,%f", &az, &alt, &fov);
        c.viewAzimuthDeg = az, c.viewAltitudeDeg = alt, c.viewFovDeg = fov;
    }
    if (const char* s = std::getenv("PHX_SCREENSHOT")) c.screenshotPath = s;
    if (const char* s = std::getenv("PHX_SCREENSHOT_AT")) c.screenshotAtSeconds = std::strtod(s, nullptr);
    if (const char* sz = std::getenv("PHX_SIZE")) std::sscanf(sz, "%ux%u", &c.width, &c.height);
    if (const char* f = std::getenv("PHX_FLY")) c.flySpeed = std::strtod(f, nullptr);
    if (const char* p = std::getenv("PHX_POS"))
        std::sscanf(p, "%lf,%lf,%lf", &c.startLatDeg, &c.startLonDeg, &c.startHeightAboveGround);
    return c;
}

int runEngine(const EngineConfig& cfg) {
    jobs::JobSystem jobs({.workerCount = cfg.workerCount});
    jobs.registerThread(jobs::ThreadRole::Supervisor);
    log::info("Threads: supervisor(main) + render + {} workers", jobs.workerCount());

    Window window(cfg.width, cfg.height, cfg.title);
    VulkanContext gpu(window, cfg.validation);

    // Global procedural terrain on the 1° DEM everywhere; the 30 m Copernicus cells are
    // opt-in (PHX_DEM30=1) since the world is generated uniformly from biome rules.
    const bool dem30 = std::getenv("PHX_DEM30") && std::getenv("PHX_DEM30")[0] == '1';
    const HeightField heights(std::string(PHX_DATA_DIR) + "/dem_global.bin", dem30 ? std::string(PHX_DATA_DIR) + "/dem30" : std::string());
    CameraState camera;
    {
        const glm::dvec3 dir = directionFromLatLon(cfg.startLatDeg, cfg.startLonDeg);
        camera.positionEcef = dir * (kPlanetRadiusM + std::max(heights.height(dir), 0.0) + cfg.startHeightAboveGround);
        camera.yaw = f32(cfg.viewAzimuthDeg * astro::kDeg);
        camera.pitch = f32(cfg.viewAltitudeDeg * astro::kDeg);
        camera.fovY = f32(cfg.viewFovDeg * astro::kDeg);
    }
    auto observerFromCamera = [&] {
        f64 lat, lon;
        latLonFromDirection(glm::normalize(camera.positionEcef), lat, lon);
        return astro::Observer{lat * astro::kDeg, lon * astro::kDeg, glm::length(camera.positionEcef) - kPlanetRadiusM};
    };
    Environment env({.observer = observerFromCamera(), .startJdUT = cfg.startJdUT, .timeRate = cfg.timeRate});
    {
        const astro::Observer o = observerFromCamera();
        log::info("Observer: {:.4f}°N {:.4f}°E, {:.0f} m (ground {:.0f} m), {}", o.latitude / astro::kDeg,
                  o.longitude / astro::kDeg, o.altitudeM,
                  heights.height(glm::normalize(camera.positionEcef)), env.statusLine());
    }

    // ---- on foot (M6): Jolt collision patch + character; F toggles free flight.
    PhysicsWorld physics(heights, jobs);
    const int scriptedWalk = std::getenv("PHX_WALK") ? std::atoi(std::getenv("PHX_WALK")) : 0; // 1 walk, 2 sprint, 3 climb
    auto groundUnder = [&](const glm::dvec3& p) {
        const glm::dvec3 d = glm::normalize(p);
        return d * (kPlanetRadiusM + std::max(heights.height(d, 1.2), 0.0));
    };
    std::unique_ptr<Player> player;
    bool walking = false;
    auto startWalking = [&] {
        const glm::dvec3 feet = groundUnder(camera.positionEcef);
        if (player) player->teleport(feet);
        else player = std::make_unique<Player>(physics, feet);
        walking = true;
        camera.positionEcef = player->eyeEcef();
        log::info("On foot: {} colliders around the player (patch built in {:.1f} ms on a worker)", physics.colliderCount(),
                  physics.lastBuildMs());
    };
    f64 airTemp = 20.0;
    f64 stepUsSum = 0, stepUsMax = 0;
    u64 stepCount = 0;

    TripleBuffer<FrameSnapshot> snapshots;
    const f64 dt = 1.0 / cfg.simHz;
    const auto tickDuration = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<f64>(dt));
    const auto start = Clock::now();
    u64 tick = 0;
    u32 screenshotRequest = 0;
    const bool scriptedShot = !cfg.screenshotPath.empty() && cfg.screenshotAtSeconds > 0;
    const bool scripted = scriptedShot || cfg.autoCloseSeconds > 0; // automated runs ignore input
    // Performance overlay: on for interactive runs, off in captures unless PHX_OVERLAY=1.
    bool perfOverlay = !scriptedShot;
    if ((!scripted && cfg.startHeightAboveGround <= 3.0) || scriptedWalk > 0) startWalking();
    if (walking && !scripted) window.setCursorCaptured(true);
    if (const char* o = std::getenv("PHX_OVERLAY")) perfOverlay = o[0] != '0';
    // Upscaling mode: PHX_DLSS = off | dlaa | quality | balanced | performance | ultra (F4 cycles).
    u8 upscaleMode = u8(UpscaleMode::Quality);
    if (const char* m = std::getenv("PHX_DLSS")) {
        UpscaleMode parsed;
        if (parseUpscaleMode(m, parsed)) upscaleMode = u8(parsed);
        else log::warn("PHX_DLSS '{}' not understood (off|dlaa|quality|balanced|performance|ultra)", m);
    }

    auto publish = [&] {
        const glm::dvec3 up = glm::normalize(camera.positionEcef);
        const f64 agl = glm::length(camera.positionEcef) - kPlanetRadiusM - std::max(heights.height(up), 0.0);
        camera.nearPlane = f32(std::clamp(agl * 0.05, 0.05, 1000.0));
        FrameSnapshot& s = snapshots.writeSlot();
        s.tick = tick;
        s.simTime = f64(tick) * dt;
        s.framebuffer = window.framebufferSize();
        s.camera = camera;
        s.screenshotRequest = screenshotRequest;
        {   // Mean ground albedo around the camera for bounce light (biome mix).
            const SurfaceInfo si = heights.surface(glm::normalize(camera.positionEcef), 30.0);
            const BiomeWeights w = biomeWeights(si.tempC, si.precipMm, si.amplitudeC);
            const f32 ai = si.precipMm / potentialEvapotranspiration(si.tempC, si.amplitudeC);
            const glm::vec3 grass = glm::mix(glm::vec3(0.07f, 0.09f, 0.035f), glm::vec3(0.24f, 0.19f, 0.10f),
                                             1.0f - std::clamp((ai - 0.25f) / 0.65f, 0.0f, 1.0f));
            glm::vec3 a = glm::vec3(0.06f, 0.07f, 0.04f) * w.forest + grass * w.grass + glm::vec3(0.40f, 0.30f, 0.18f) * w.desert +
                          glm::vec3(0.12f, 0.12f, 0.10f) * w.tundra + glm::vec3(0.7f, 0.72f, 0.75f) * w.snow;
            const f32 sum = w.forest + w.grass + w.desert + w.tundra + w.snow;
            s.groundAlbedo = si.height < 0.0 ? glm::vec3(0.05f) : a / std::max(sum, 1e-3f);
        }
        s.perfOverlay = perfOverlay;
        s.upscaleMode = upscaleMode;
        s.hud = {};
        if (walking && player) {
            const Vitals& v = player->vitals();
            PlayerHud& h = s.hud;
            h.visible = true;
            std::snprintf(h.mode, sizeof(h.mode), "%s", locomotionName(player->mode()));
            h.health = f32(v.health), h.stamina = f32(v.stamina);
            h.energy = f32(v.energyKcal / Vitals::kEnergyMax * 100.0), h.water = f32(v.waterL / Vitals::kWaterMax * 100.0);
            h.coreTempC = f32(v.coreTempC), h.airTempC = f32(airTemp), h.speed = f32(player->groundSpeed());
            h.slopeDeg = f32(player->slopeDeg()), h.powerW = f32(v.powerW);
        }
        env.fill(s);
        snapshots.publish();
    };
    env.advance(0.0);
    publish();
    {
        FrameSnapshot probe;
        env.fill(probe);
        log::info("Sky: moon illuminance {:.3g}, adaptation {:.3g}, exposure {:.3g}", env.sky().moonIlluminance,
                  probe.adaptationLuminance, probe.exposure);
    }

    Renderer renderer(gpu, jobs, snapshots,
                      {.present = cfg.present,
                       .starEpochYear = env.decimalYear(),
                       .screenshotPath = cfg.screenshotPath,
                       .heights = &heights});
    renderer.start();

    // Supervisor loop: input -> fixed-step simulation -> publish snapshot -> help / idle.
    auto nextTick = start + tickDuration;
    auto nextTitle = start;
    bool shotRequested = false;
    while (!window.shouldClose()) {
        window.pollEvents();
        // Scripted runs (automatic screenshots) ignore the mouse/keyboard: a pointer
        // passing over the window must not change the captured view.
        InputFrame input = window.drainInput();
        if (scripted) input = {};
        for (int key : input.keysPressed) {
            if (!(walking && key == GLFW_KEY_SPACE) && env.handleKey(key)) continue; // Space = jump while on foot
            if (key == GLFW_KEY_ESCAPE) {
                if (window.cursorCaptured()) window.setCursorCaptured(false);
                else window.requestClose();
            }
            if (key == GLFW_KEY_TAB) window.setCursorCaptured(!window.cursorCaptured());
            if (key == GLFW_KEY_F) {
                if (walking) walking = false;
                else startWalking();
            }
            if (key == GLFW_KEY_F12) ++screenshotRequest;
            if (key == GLFW_KEY_F3) perfOverlay = !perfOverlay;
            if (key == GLFW_KEY_F4) {
                upscaleMode = u8((upscaleMode + 1) % u8(UpscaleMode::Count));
                log::info("Upscaling: {}", upscaleModeName(UpscaleMode(upscaleMode)));
            }
        }
        applyCameraInput(camera, input);

        auto now = Clock::now();
        if (now - nextTick > std::chrono::milliseconds(250)) nextTick = now; // hitch: don't spiral
        bool advanced = false;
        while (now >= nextTick) {
            ++tick;
            if (walking) {
                PlayerControls c;
                c.yaw = camera.yaw;
                if (scriptedWalk > 0) {
                    c.forward = 1.0f;
                    c.sprint = scriptedWalk == 2;
                    c.jumpHeld = scriptedWalk == 3;
                    if (scriptedWalk == 3) { // climb test: head straight up the fall line
                        const glm::dvec3 nEnu = ecefToEnuMatrix(player->feetEcef()) * player->groundNormalEcef();
                        if (std::hypot(nEnu.x, nEnu.y) > 0.05) c.yaw = f32(std::atan2(-nEnu.x, -nEnu.y));
                        camera.yaw = c.yaw;
                    }
                } else {
                    c.forward = f32(window.keyDown(GLFW_KEY_W)) - f32(window.keyDown(GLFW_KEY_S));
                    c.strafe = f32(window.keyDown(GLFW_KEY_D)) - f32(window.keyDown(GLFW_KEY_A));
                    c.sprint = window.keyDown(GLFW_KEY_LEFT_SHIFT);
                    c.jumpHeld = window.keyDown(GLFW_KEY_SPACE);
                }
                {
                    f64 lat, lon;
                    latLonFromDirection(glm::normalize(player->feetEcef()), lat, lon);
                    const f64 doy = std::fmod(env.jdUT() - 2451544.5, 365.2425);
                    airTemp = airTemperatureC(lat, glm::length(player->feetEcef()) - kPlanetRadiusM, doy < 0 ? doy + 365.2425 : doy,
                                              env.sky().sun.enu.z);
                }
                const auto stepStart = Clock::now();
                player->step(dt, c, airTemp);
                stepUsSum += std::chrono::duration<f64, std::micro>(Clock::now() - stepStart).count();
                stepUsMax = std::max(stepUsMax, std::chrono::duration<f64, std::micro>(Clock::now() - stepStart).count());
                ++stepCount;
                camera.positionEcef = player->eyeEcef();
                if (scriptedWalk > 0 && tick % u64(cfg.simHz) == 0) {
                    const Vitals& v = player->vitals();
                    log::info("walk t={:.0f}s {} {:.2f} m/s slope {:.0f}° alt {:.1f} m | P {:.0f} W sta {:.0f} hp {:.0f} "
                              "water {:.3f} L kcal {:.0f} core {:.2f} air {:.1f} | colliders {} | step {:.0f} us avg, {:.0f} max",
                              f64(tick) * dt, locomotionName(player->mode()), player->groundSpeed(), player->slopeDeg(),
                              glm::length(player->feetEcef()) - kPlanetRadiusM, v.powerW, v.stamina, v.health, v.waterL,
                              v.energyKcal, v.coreTempC, airTemp, physics.colliderCount(), stepUsSum / f64(std::max<u64>(stepCount, 1)),
                              stepUsMax);
                    stepUsSum = stepUsMax = 0;
                    stepCount = 0;
                }
            } else if (!scripted) {
                moveCamera(camera, window, heights, dt);
            }
            if (!walking && cfg.flySpeed > 0) {
                // Benchmark flight: constant heading and height above ground.
                const glm::dvec3 up0 = glm::normalize(camera.positionEcef);
                const f64 agl0 = glm::length(camera.positionEcef) - kPlanetRadiusM - std::max(heights.height(up0), 0.0);
                const glm::dmat3 toEcef = glm::transpose(ecefToEnuMatrix(camera.positionEcef));
                const glm::dvec3 heading = toEcef * glm::dvec3(std::sin(camera.yaw), std::cos(camera.yaw), 0.0);
                const glm::dvec3 up1 = glm::normalize(camera.positionEcef + heading * cfg.flySpeed * dt);
                camera.positionEcef = up1 * (kPlanetRadiusM + std::max(heights.height(up1), 0.0) + agl0);
            }
            env.setObserver(observerFromCamera());
            env.advance(dt);
            nextTick += tickDuration;
            advanced = true;
        }
        if (advanced || !input.keysPressed.empty() || input.mouseDx != 0 || input.mouseDy != 0 || input.scroll != 0)
            publish();
        (void)advanced;

        if (now >= nextTitle) {
            const glm::dvec3 up = glm::normalize(camera.positionEcef);
            const f64 agl = glm::length(camera.positionEcef) - kPlanetRadiusM - std::max(heights.height(up), 0.0);
            window.setTitle(std::format("{}  —  {}  |  alt {:.0f} m (AGL {:.0f} m)  |  {}  |  {}  [F walk/fly, Tab mouse, F3 perf, F4 DLSS]",
                                        cfg.title, env.statusLine(), glm::length(camera.positionEcef) - kPlanetRadiusM, agl,
                                        walking ? locomotionName(player->mode()) : "fly", upscaleModeName(UpscaleMode(upscaleMode))));
            nextTitle = now + std::chrono::milliseconds(250);
        }

        const f64 elapsed = std::chrono::duration<f64>(now - start).count();
        if (scriptedShot) {
            if (!shotRequested && elapsed >= cfg.screenshotAtSeconds) {
                ++screenshotRequest;
                shotRequested = true;
                publish();
            }
            if (shotRequested && renderer.screenshotsWritten() > 0) break;
        }
        if (cfg.autoCloseSeconds > 0 && elapsed > cfg.autoCloseSeconds) break;

        // Help with short safe jobs while slack remains, otherwise sleep until the
        // next tick but wake immediately on input (bounded to keep latency ≤ 2 ms).
        jobs.supervisorHelp(nextTick);
        const auto remaining = std::chrono::duration<f64>(nextTick - Clock::now()).count();
        if (remaining > 0.0005) window.waitEvents(std::min(remaining, 0.002));
    }

    renderer.stop();
    log::info("Shutdown after {} ticks", tick);
    return 0;
}

} // namespace phx
