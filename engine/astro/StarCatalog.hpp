#pragma once
// Baked star catalog (tools/bake/BakeStars.cpp → stars.bin).
#include "engine/core/Types.hpp"

#include <string>
#include <vector>

namespace phx::astro {

inline constexpr char kStarFileMagic[8] = {'P', 'H', 'X', 'S', 'T', 'A', 'R', '1'};

// On-disk record: ICRS / J2000 unit vector, its time derivative from proper
// motion (rad/yr in the same frame), V magnitude and B−V colour index.
struct StarRecord {
    f32 dir[3];
    f32 dDirDt[3];
    f32 vmag;
    f32 bv;
};
static_assert(sizeof(StarRecord) == 32);

// GPU layout (std430/scalar): direction at the game epoch, flux relative to the
// sun's irradiance outside the atmosphere, linear-sRGB colour with luminance 1.
struct GpuStar {
    f32 dir[3];
    f32 flux;
    f32 color[3];
    f32 pad;
};
static_assert(sizeof(GpuStar) == 32);

bool loadStarRecords(const std::string& path, std::vector<StarRecord>& out);

// Applies proper motion to `decimalYear` and converts photometry. Stars fainter
// than `limitMag` are dropped (they belong to the integrated Milky Way term).
std::vector<GpuStar> buildGpuStars(const std::vector<StarRecord>& records, f64 decimalYear, f32 limitMag);

// Shared helpers (also used for planets)
f32 fluxFromMagnitude(f64 vmag);
void colorFromBV(f32 bv, f32 rgbOut[3]); // linear sRGB, luminance-normalised

} // namespace phx::astro
