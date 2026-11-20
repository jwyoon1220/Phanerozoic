#include "engine/world/Climate.hpp"

#include "engine/core/Log.hpp"
#include "engine/world/HeightField.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace phx {
namespace {
constexpr f64 kPi = 3.14159265358979323846;

f32 smoothstep(f32 e0, f32 e1, f32 x) {
    const f32 t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
f64 gauss(f64 x, f64 mu, f64 sigma) { return std::exp(-((x - mu) / sigma) * ((x - mu) / sigma)); }


// 0.25° land/sea elevation grid used by the upwind marches.
struct ElevationGrid {
    static constexpr u32 W = 1440, H = 720;
    std::vector<f32> h;
    f32 at(f64 lat, f64 lon) const {
        lon = std::fmod(lon + 540.0, 360.0) - 180.0;
        const i32 x = std::clamp(i32((lon + 180.0) * 4.0), 0, i32(W) - 1);
        const i32 y = std::clamp(i32((lat + 90.0) * 4.0), 0, i32(H) - 1);
        return h[std::size_t(y) * W + x];
    }
};

struct Upwind {
    f64 landKm = 5000; // land distance to a substantial ocean (≥ 500 km fetch)
    f64 barrier = 0;   // highest ground on the way (first 1200 km)
};
// March towards the moisture source: along the latitude circle (dir ±1 = east/west)
// or towards the equator (meridional = true).
Upwind march(const ElevationGrid& g, f64 lat, f64 lon, int dir, bool meridional = false) {
    Upwind u;
    const f64 kmPerDegLon = std::max(111.32 * std::cos(lat * kPi / 180.0), 5.0);
    const f64 toEquator = lat >= 0 ? -1.0 : 1.0;
    constexpr f64 kStep = 25.0, kMax = 5000.0, kFetch = 350.0;
    f64 land = 0, ocean = 0;
    for (f64 d = kStep; d <= kMax; d += kStep) {
        const f32 h = meridional ? g.at(lat + toEquator * d / 111.32, lon) : g.at(lat, lon + dir * d / kmPerDegLon);
        if (meridional && std::abs(lat + toEquator * d / 111.32) < 0.5) break; // reached the equator over land
        if (h <= -70.0f) { // modern coastline: the DEM is shifted to the Eocene sea level (−70 m)
            ocean += kStep;
            if (ocean >= kFetch) {
                u.landKm = land;
                return u;
            }
        } else {
            land += kStep + ocean; // a narrow sea counts as overland transport
            ocean = 0;
            if (d <= 1200.0) u.barrier = std::max(u.barrier, f64(h));
        }
    }
    u.landKm = kMax;
    return u;
}
} // namespace

f32 potentialEvapotranspiration(f32 T, f32 amplitude) {
    return 250.0f + 50.0f * std::max(T, 0.0f) + 12.0f * amplitude * smoothstep(-5.0f, 10.0f, T);
}

BiomeWeights biomeWeights(f32 T, f32 P, f32 amplitude) {
    BiomeWeights w;
    const f32 A = P / potentialEvapotranspiration(T, amplitude);
    const f32 warmest = T + 0.5f * amplitude;
    w.snow = smoothstep(1.0f, -5.0f, warmest);
    w.tundra = smoothstep(10.5f, 7.5f, warmest) * (1.0f - w.snow);
    const f32 vegetated = smoothstep(0.1f, 0.4f, A);
    const f32 closed = smoothstep(0.32f, 0.62f, A);
    w.forest = closed * std::max(0.0f, 1.0f - w.tundra - w.snow);
    w.desert = (1.0f - vegetated) * (1.0f - w.snow);
    w.grass = vegetated * (1.0f - closed) * (1.0f - w.snow);
    w.conifer = smoothstep(11.0f, 3.0f, T);
    w.tropical = smoothstep(16.0f, 22.0f, T);
    return w;
}

const char* dominantBiomeName(const BiomeWeights& w) {
    if (w.snow > 0.5f) return "ice";
    if (w.tundra > 0.5f) return "tundra";
    if (w.desert > 0.5f) return w.tropical > 0.5f ? "hot desert" : "cold desert";
    if (w.forest > 0.5f) {
        if (w.tropical > 0.5f) return "tropical forest";
        return w.conifer > 0.5f ? "taiga / conifer forest" : "temperate forest";
    }
    if (w.tropical > 0.5f) return "savanna";
    return "steppe / grassland";
}

Climate::Climate(const HeightField& heights) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    ElevationGrid g;
    g.h.resize(std::size_t(ElevationGrid::W) * ElevationGrid::H);
    grid_.resize(std::size_t(kWidth) * kHeight);

    const u32 threads = std::max(2u, std::thread::hardware_concurrency());
    auto parallel = [&](u32 rows, auto&& fn) {
        std::vector<std::thread> pool;
        for (u32 t = 0; t < threads; ++t)
            pool.emplace_back([&, t] {
                for (u32 y = t; y < rows; y += threads) fn(y);
            });
        for (std::thread& th : pool) th.join();
    };
    parallel(ElevationGrid::H, [&](u32 y) {
        for (u32 x = 0; x < ElevationGrid::W; ++x)
            g.h[std::size_t(y) * ElevationGrid::W + x] =
                f32(heights.baseHeight(-90.0 + (y + 0.5) / 4.0, -180.0 + (x + 0.5) / 4.0));
    });

    parallel(kHeight, [&](u32 y) {
        const f64 lat = -90.0 + (y + 0.5) * 0.5, alat = std::abs(lat);
        for (u32 x = 0; x < kWidth; ++x) {
            const f64 lon = -180.0 + (x + 0.5) * 0.5;
            const f64 base = std::max(0.0, f64(g.at(lat, lon)));
            ClimateSample& c = grid_[std::size_t(y) * kWidth + x];

            // Zonal precipitation belts (mm/yr): ITCZ, subtropical highs, storm tracks.
            const f64 P0 = 200.0 + 2200.0 * gauss(lat, 0.0, 11.0) + 850.0 * gauss(alat, 48.0, 13.0);
            // Prevailing wind: trades (from the east) below 30°, westerlies to 65°,
            // polar easterlies beyond; blended across the boundaries.
            const f64 westerly = std::clamp((alat - 27.0) / 6.0, 0.0, 1.0) * std::clamp((68.0 - alat) / 6.0, 0.0, 1.0);
            const Upwind east = march(g, lat, lon, +1), west = march(g, lat, lon, -1);
            const f64 tropicsScale = 1500.0 + 1500.0 * std::clamp((25.0 - alat) / 10.0, 0.0, 1.0); // moisture recycling
            auto supply = [&](const Upwind& u) {
                const f64 f = std::exp(-u.landKm / tropicsScale);
                const f64 floor = 0.12 + 0.45 * std::clamp((alat - 48.0) / 14.0, 0.0, 1.0); // weak evaporation far north
                const f64 shadow = std::exp(-std::max(0.0, u.barrier - base - 600.0) / 1200.0);
                return (floor + (1.0 - floor) * f) * shadow;
            };
            f64 P = P0 * (westerly * supply(west) + (1.0 - westerly) * supply(east));
            // East-coast summer monsoons in the subtropics (East Asia, SE USA, E Australia…).
            const f64 monsoon = gauss(alat, 27.0, 12.0);
            const f64 shadowE = std::exp(-std::max(0.0, east.barrier - base - 600.0) / 1200.0);
            P += monsoon * 1400.0 * std::exp(-east.landKm / 1200.0) * shadowE;
            // Equatorward summer monsoons (South/East Asia, Sahel, northern Australia, Mexico…).
            const Upwind south = march(g, lat, lon, 0, true);
            const f64 shadowS = std::exp(-std::max(0.0, south.barrier - base - 600.0) / 1200.0);
            P += gauss(alat, 22.0, 10.0) * 1400.0 * std::exp(-south.landKm / 900.0) * shadowS;
            P *= 1.0 + 0.4 * std::clamp(base / 2500.0, 0.0, 1.0); // orographic uplift
            // Arid cores the circulation rules miss (subsidence, cold upwelling coasts):
            // a short list of corrections rather than a data set.
            struct Core { f64 lat, lon, radiusKm, strength; };
            static constexpr Core kCores[] = {
                {22.0, 47.0, 1300.0, 0.95}, // Arabian peninsula (Rub' al Khali)
                {27.0, 71.0, 450.0, 0.8},   // Thar
                {-24.0, 19.0, 700.0, 0.7},  // Kalahari / Namib
                {-25.5, 132.0, 1100.0, 0.6},// Australian interior
                {30.0, -112.0, 550.0, 0.7}, // Sonoran / Mojave
                {29.0, 58.0, 700.0, 0.7},   // Iranian plateau (Dasht-e Lut)
            };
            for (const Core& k : kCores) {
                const f64 dLat = (lat - k.lat) * 111.32;
                const f64 dLon = std::remainder(lon - k.lon, 360.0) * 111.32 * std::cos(lat * kPi / 180.0);
                P *= 1.0 - k.strength * std::exp(-(dLat * dLat + dLon * dLon) / (k.radiusKm * k.radiusKm));
            }

            // Temperature: zonal sea-level mean; continental interiors colder at high latitude.
            const f64 cont = 1.0 - std::exp(-(westerly * west.landKm + (1.0 - westerly) * east.landKm) / 1500.0);
            const f64 T0 = 27.5 - 0.0068 * lat * lat - 7.0 * cont * std::clamp((alat - 35.0) / 30.0, 0.0, 1.0);

            c.seaLevelTempC = f32(T0);
            c.precipMm = f32(P);
            // Seasonal swing: grows with latitude and distance from the ocean.
            c.amplitudeC = f32((4.0 + 34.0 * cont) * std::clamp(alat / 60.0, 0.15, 1.0) + 2.0);
        }
    });
    rg_.resize(grid_.size() * 4);
    for (std::size_t i = 0; i < grid_.size(); ++i) {
        rg_[i * 4 + 0] = grid_[i].seaLevelTempC;
        rg_[i * 4 + 1] = grid_[i].precipMm * 0.001f; // metres
        rg_[i * 4 + 2] = grid_[i].amplitudeC;
        rg_[i * 4 + 3] = 0.0f;
    }

    auto report = [&](const char* name, f64 lat, f64 lon) {
        const ClimateSample c = sample(lat, lon);
        const f64 h = std::max(0.0, heights.baseHeight(lat, lon));
        const f32 T = temperatureAt(c, h);
        log::info("  {:<14} M {:.2f} base {:5.0f}  T {:5.1f} °C (warmest {:5.1f})  P {:5.0f} mm  AI {:4.2f}  → {}", name,
                  heights.mountainFactor(lat, lon), h, T, T + c.amplitudeC * 0.5f,
                  c.precipMm, c.precipMm / potentialEvapotranspiration(T, c.amplitudeC),
                  dominantBiomeName(biomeWeights(T, c.precipMm, c.amplitudeC)));
    };
    log::info("Climate: 0.5° rule-based grid in {:.0f} ms", std::chrono::duration<f64, std::milli>(Clock::now() - t0).count());
    report("Huashan", 34.46, 110.08);
    report("Sahara", 23.0, 10.0);
    report("Gobi", 43.0, 105.0);
    report("Siberia", 62.0, 100.0);
    report("Amazon", -3.0, -60.0);
    report("W Europe", 50.0, 5.0);
    report("Kazakh steppe", 48.0, 68.0);
    report("Guangxi", 24.0, 109.0);
    report("Greenland", 72.0, -40.0);
    report("Congo", 0.0, 22.0);
    report("Atacama", -24.0, -70.0);
    report("Tibet", 32.0, 88.0);
    report("N Canada", 66.0, -100.0);
    report("India", 23.0, 80.0);
    report("Sahel", 13.0, 5.0);
    report("Great Plains", 40.0, -100.0);
    report("Mediterranean", 38.0, 22.0);
    report("Australia", -25.0, 133.0);
    report("Alps", 46.5, 10.0);
    report("Rockies", 40.0, -106.0);
    report("Himalaya", 28.5, 85.0);
    report("Andes", -15.0, -70.0);
    report("Ural", 60.0, 59.5);
    report("Rub al Khali", 20.0, 50.0);
}

ClimateSample Climate::sample(f64 lat, f64 lon) const {
    const f64 fx = (lon + 180.0) / 0.5 - 0.5, fy = (lat + 90.0) / 0.5 - 0.5;
    const i32 x0 = i32(std::floor(fx)), y0 = i32(std::floor(fy));
    const f32 tx = f32(fx - x0), ty = f32(fy - y0);
    auto at = [&](i32 x, i32 y) -> const ClimateSample& {
        x = ((x % i32(kWidth)) + i32(kWidth)) % i32(kWidth);
        y = std::clamp(y, 0, i32(kHeight) - 1);
        return grid_[std::size_t(y) * kWidth + x];
    };
    const ClimateSample &a = at(x0, y0), &b = at(x0 + 1, y0), &c = at(x0, y0 + 1), &d = at(x0 + 1, y0 + 1);
    auto lerp2 = [&](f32 ClimateSample::*m) {
        return (a.*m * (1 - tx) + b.*m * tx) * (1 - ty) + (c.*m * (1 - tx) + d.*m * tx) * ty;
    };
    return {lerp2(&ClimateSample::seaLevelTempC), lerp2(&ClimateSample::precipMm), lerp2(&ClimateSample::amplitudeC)};
}

} // namespace phx
