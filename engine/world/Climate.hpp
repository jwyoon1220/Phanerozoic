#pragma once
// Rule-based modern climate (plan: biomes from latitude/longitude; Köppen-like, no data
// beyond the global DEM). Computed once on a 0.5° grid at start-up:
//  * temperature: zonal sea-level mean, colder in continental interiors at high
//    latitude, lapse rate applied per query with the actual height;
//  * precipitation: zonal belts (ITCZ, subtropical highs, mid-latitude storm tracks)
//    scaled by the distance to the moisture source *upwind* (trades blow from the
//    east, westerlies from the west), east-coast monsoons in the subtropics, and rain
//    shadows behind mountain barriers;
//  * aridity index = precipitation / potential evapotranspiration.
// Biome weights are continuous functions of (temperature, aridity) so transitions are
// gradual; the same formulas run in shaders (shaders/biome.glsl).
#include "engine/core/Types.hpp"

#include <vector>

namespace phx {
class HeightField;

struct ClimateSample {
    f32 seaLevelTempC = 15; // mean annual, reduced to sea level
    f32 precipMm = 800;     // annual
    f32 amplitudeC = 10;    // warmest minus coldest month (continentality)
};

struct BiomeWeights {
    f32 forest = 0;   // closed tree cover (any)
    f32 conifer = 0;  // share of conifers among trees
    f32 tropical = 0; // warm-climate vegetation
    f32 grass = 0;    // open grass/shrub cover
    f32 desert = 0;   // bare sand/rock
    f32 tundra = 0;   // beyond the tree line (cold)
    f32 snow = 0;     // permanent snow/ice
};
// Potential evapotranspiration (mm/yr): warmth plus hot continental summers.
f32 potentialEvapotranspiration(f32 tempC, f32 amplitudeC);
// tempC: local mean annual (after lapse rate). Tree line: warmest month ≈ 10 °C.
BiomeWeights biomeWeights(f32 tempC, f32 precipMm, f32 amplitudeC);
const char* dominantBiomeName(const BiomeWeights& w);

class Climate {
public:
    static constexpr u32 kWidth = 720, kHeight = 360; // 0.5°

    explicit Climate(const HeightField& heights);
    // Bilinear on the grid. lat/lon in degrees.
    [[nodiscard]] ClimateSample sample(f64 latDeg, f64 lonDeg) const;
    [[nodiscard]] static f32 temperatureAt(const ClimateSample& c, f64 heightM) { return c.seaLevelTempC - 0.0065f * f32(heightM); }
    // RGBA32F texture data (sea-level temperature °C, precipitation m, amplitude °C, 0), row 0 = 90°S.
    [[nodiscard]] const std::vector<f32>& textureData() const { return rg_; }

private:
    std::vector<ClimateSample> grid_;
    std::vector<f32> rg_;
};

} // namespace phx
