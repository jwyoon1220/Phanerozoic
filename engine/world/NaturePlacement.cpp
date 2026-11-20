#include "engine/world/NaturePlacement.hpp"

#include "engine/world/CubeSphere.hpp"
#include "engine/world/Climate.hpp"
#include "engine/world/HeightField.hpp"

#include <algorithm>
#include <cmath>

namespace phx {
namespace {
using namespace cubesphere;
constexpr u32 kMaxPerCell = kMaxNaturePerCell;

u64 splitmix(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
f64 unit(u64& s) {
    s = splitmix(s);
    return f64(s >> 11) * (1.0 / 9007199254740992.0);
}
f64 vhash(i64 x, i64 y, i64 z) {
    return f64(splitmix(u64(x) * 73856093ull ^ u64(y) * 19349663ull ^ u64(z) * 83492791ull) >> 11) * (1.0 / 9007199254740992.0);
}
f64 vnoise(const glm::dvec3& p) {
    const glm::dvec3 i = glm::floor(p), f = p - i, u = f * f * (3.0 - 2.0 * f);
    auto h = [&](int a, int b, int c) { return vhash(i64(i.x) + a, i64(i.y) + b, i64(i.z) + c); };
    return glm::mix(glm::mix(glm::mix(h(0, 0, 0), h(1, 0, 0), u.x), glm::mix(h(0, 1, 0), h(1, 1, 0), u.x), u.y),
                    glm::mix(glm::mix(h(0, 0, 1), h(1, 0, 1), u.x), glm::mix(h(0, 1, 1), h(1, 1, 1), u.x), u.y), u.z);
}

u32 packInstance(u32 species, f64 yaw01, f64 scale01, f64 tint01, f64 tilt01) {
    auto q = [](f64 v, u32 bits) { return u32(std::clamp(v, 0.0, 1.0) * f64((1u << bits) - 1) + 0.5); };
    return species | (q(yaw01, 9) << 4) | (q(scale01, 9) << 13) | (q(tint01, 4) << 22) | (q(tilt01, 6) << 26);
}
} // namespace

glm::dvec3 placeNature(u64 key, const HeightField& heights_, const NatureSpeciesDims* dims, std::vector<VegInstance>& out) {
    const NodeId n = decode(key);
    glm::dvec3 center;
    const f64 size = 2.0 / f64(1u << n.level);
    const f64 u0 = -1.0 + n.x * size, v0 = -1.0 + n.y * size;
    const glm::dvec3 cdir = cubeToSphere(n.face, u0 + size / 2, v0 + size / 2);
    center = cdir * (kPlanetRadiusM + std::max(heights_.height(cdir, 1.2), 0.0));
    out.clear();

    // Ground sample: terrain features + climate → biome, slope (rad), downslope azimuth
    // (yaw so the local Z axis points downhill, see vegmesh.vert tilt).
    struct Ground {
        glm::dvec3 dir;
        f64 h, slope, yaw01;
        SurfaceInfo si;
        BiomeWeights bw;
    };
    auto sample = [&](f64 u, f64 v, Ground& g) {
        g.dir = cubeToSphere(n.face, u, v);
        g.si = heights_.surface(g.dir, 1.2);
        g.h = g.si.height;
        if (g.h < 2.0) return false; // sea and shoreline
        const glm::dvec3 east = glm::normalize(glm::cross(glm::dvec3(0, 0, 1), g.dir));
        const glm::dvec3 north = glm::cross(g.dir, east);
        const glm::dvec3 p = g.dir * kPlanetRadiusM;
        const f64 gx = (heights_.height(glm::normalize(p + east * 3.0), 1.2) - g.h) / 3.0;
        const f64 gy = (heights_.height(glm::normalize(p + north * 3.0), 1.2) - g.h) / 3.0;
        g.slope = std::atan(std::sqrt(gx * gx + gy * gy));
        // downhill D = −∇h; yaw a with Z = sin a·E − cos a·N = D  →  a = atan2(Dx, −Dy)
        const f64 a = std::atan2(-gx, gy);
        g.yaw01 = (a < 0 ? a + 2 * kPi : a) / (2 * kPi);
        g.bw = biomeWeights(g.si.tempC, g.si.precipMm, g.si.amplitudeC);
        return g.bw.snow < 0.6f; // nothing grows or lies loose on permanent snow
    };
    auto place = [&](const Ground& g, u32 species, f64 scale01, f64 sinkM, f64 yaw01, f64 tint01, f64 tilt01) {
        if (out.size() >= kMaxPerCell) return;
        const glm::dvec3 pos = g.dir * (kPlanetRadiusM + g.h - sinkM) - center;
        out.push_back({{f32(pos.x), f32(pos.y), f32(pos.z)}, packInstance(species, yaw01, scale01, tint01, tilt01)});
    };
    auto heightOf = [&](u32 sp, f64 scale01) { return dims[sp].height * (0.5 + scale01) * kNatureScaleMul[sp]; };
    // Tilted rocks: their downhill edge lifts by ~radius·sin(slope); sink that much more.
    auto tiltSink = [&](u32 sp, f64 scale01, f64 slope) { return dims[sp].radius * (0.5 + scale01) * kNatureScaleMul[sp] * std::sin(std::min(slope, 1.4)) * 0.6; };
    auto tiltOf = [](f64 slope, f64 follow) { return std::min(slope, 1.4) / 1.4 * follow; }; // packed tilt (×1.4 rad)
    // Dryness for grass/shrub tint: green when moist, straw when arid.
    auto dryness = [](const Ground& g) {
        const f64 ai = g.si.precipMm / potentialEvapotranspiration(g.si.tempC, g.si.amplitudeC);
        return 1.0 - glm::smoothstep(0.25, 0.9, ai);
    };

    u64 seed = splitmix(key);
    auto grid = [&](u32 G, auto&& body) {
        for (u32 j = 0; j < G; ++j)
            for (u32 i = 0; i < G; ++i) {
                u64 s = splitmix(seed ^ (u64(j) << 32 | i) ^ (u64(G) << 48));
                const f64 u = u0 + (f64(i) + 0.1 + 0.8 * unit(s)) * size / G;
                const f64 v = v0 + (f64(j) + 0.1 + 0.8 * unit(s)) * size / G;
                body(u, v, s);
            }
    };

    if (n.level == kNatureLayers[0].level) {
        // Trees (~6 m candidate grid): canopy cover from the biome, open ground on cliffs,
        // scree and dunes, natural clearings; conifers in cold climates, broadleaf
        // evergreens in the tropics, small dryland trees scattered through savanna.
        grid(100, [&](f64 u, f64 v, u64& s) {
            const f64 accept = unit(s);
            if (accept > 0.62) return;
            Ground g;
            if (!sample(u, v, g) || g.slope > 0.72) return;
            const BiomeWeights& w = g.bw;
            const f64 savanna = w.grass * (0.12 + 0.18 * w.tropical);
            f64 density = 0.62 * (w.forest + savanna + 0.04 * w.grass) * glm::smoothstep(0.72, 0.42, g.slope);
            density *= (1.0 - g.si.cliff) * (1.0 - g.si.sand) * (1.0 - 0.8 * g.si.talus);
            const glm::dvec3 q = g.dir * (kPlanetRadiusM / 420.0);
            density *= glm::smoothstep(0.22, 0.42, vnoise(q) * 0.65 + vnoise(q * 3.1 + 17.0) * 0.35);
            if (accept > density) return;
            const f64 rs = unit(s);
            u32 sp;
            if (unit(s) * (w.forest + savanna) < savanna) sp = 8;          // dryland tree
            else if (rs < w.conifer) sp = 0;                                // conifer
            else sp = unit(s) < 0.45 + 0.4 * w.tropical ? 1u : 2u;          // broadleaf
            const f64 sc = unit(s);
            place(g, sp, sc, 0.4 + std::tan(g.slope) * 1.2, unit(s), unit(s), 0.0);
        });
    } else if (n.level == kNatureLayers[1].level) {
        // Boulders (~10 m grid): scree below cliffs, steep and rocky ground, stony deserts.
        grid(15, [&](f64 u, f64 v, u64& s) {
            Ground g;
            if (!sample(u, v, g)) return;
            const f64 rocky = vnoise(g.dir * (kPlanetRadiusM / 90.0) + 3.0);
            const f64 p = 0.02 + 0.1 * glm::smoothstep(0.35, 0.8, g.slope) + 0.08 * glm::smoothstep(0.6, 0.85, rocky) +
                          0.55 * g.si.talus + (0.06 * g.bw.desert + 0.06 * g.bw.tundra) * (1.0 - g.si.sand);
            if (unit(s) > p * (1.0 - g.si.sand)) return;
            const f64 sc = unit(s) * unit(s); // mostly small, some big
            place(g, 5, sc, heightOf(5, sc) * 0.3 + tiltSink(5, sc, g.slope), std::fmod(g.yaw01 + 0.2 * unit(s), 1.0), g.bw.desert, tiltOf(g.slope, 0.7));
        });
        // Shrubs (~6 m grid): steppe, tundra, savanna, desert margins, forest edges.
        grid(25, [&](f64 u, f64 v, u64& s) {
            Ground g;
            if (!sample(u, v, g) || g.slope > 0.8) return;
            const BiomeWeights& w = g.bw;
            const f64 p = 0.3 * (w.grass * (1.0 - 0.4 * w.tropical) + 0.8 * w.tundra + 0.1 * w.desert + 0.08 * w.forest) *
                          (1.0 - g.si.sand) * (1.0 - g.si.cliff);
            if (unit(s) > p) return;
            place(g, 7, unit(s), 0.08, unit(s), dryness(g), 0.0);
        });
    } else {
        // Ferns (~1.2 m grid) in moist, shaded forest; grass clumps in open ground;
        // mossy stones (~3 m grid) outside deserts.
        grid(64, [&](f64 u, f64 v, u64& s) {
            const f64 accept = unit(s);
            if (accept > 0.45) return;
            Ground g;
            if (!sample(u, v, g) || g.slope > 0.7) return;
            const BiomeWeights& w = g.bw;
            const f64 open = (1.0 - g.si.sand) * (1.0 - g.si.cliff) * glm::smoothstep(0.7, 0.4, g.slope);
            const f64 moist = vnoise(g.dir * (kPlanetRadiusM / 25.0) + 7.0) * 0.6 + vnoise(g.dir * (kPlanetRadiusM / 7.0)) * 0.4;
            const f64 fern = 0.22 * glm::smoothstep(0.4, 0.7, moist) * w.forest * (1.0 - 0.5 * w.conifer) * open;
            const f64 grassPatch = glm::smoothstep(0.25, 0.6, vnoise(g.dir * (kPlanetRadiusM / 14.0) + 2.0));
            const f64 grass = 0.45 * (w.grass + 0.6 * w.tundra + 0.25 * w.forest * (1.0 - moist)) * grassPatch * open;
            if (accept < fern) place(g, 3, unit(s), 0.05, unit(s), unit(s), 0.0);
            else if (accept < fern + grass) place(g, 6, unit(s), 0.03, unit(s), dryness(g), 0.0);
        });
        grid(25, [&](f64 u, f64 v, u64& s) {
            Ground g;
            if (!sample(u, v, g)) return;
            if (unit(s) > (0.025 + 0.07 * glm::smoothstep(0.35, 0.8, g.slope)) * (1.0 - g.bw.desert) * (1.0 - g.si.sand)) return;
            const f64 sc = unit(s);
            place(g, 4, sc, heightOf(4, sc) * 0.35 + tiltSink(4, sc, g.slope), std::fmod(g.yaw01 + 0.3 * unit(s), 1.0), g.bw.desert, tiltOf(g.slope, 0.8));
        });
    }
    return center;
}

NatureTransform natureTransform(const glm::dvec3& p, u32 packed) {
    NatureTransform t;
    t.species = packed & 15u;
    t.scale = (0.5 + f64((packed >> 13) & 511u) / 511.0) * kNatureScaleMul[t.species];
    const f64 rot = f64((packed >> 4) & 511u) / 511.0 * 6.2831853;
    const f64 tilt = f64((packed >> 26) & 63u) / 63.0 * 1.4; // radians, ≤ 80°
    t.origin = p;
    glm::dvec3 Y = glm::normalize(p);
    const glm::dvec3 e = glm::normalize(glm::cross(glm::dvec3(0, 0, 1), Y));
    const glm::dvec3 n = glm::cross(Y, e);
    const glm::dvec3 X = std::cos(rot) * e + std::sin(rot) * n;
    glm::dvec3 Z = glm::cross(X, Y);
    if (tilt > 0.0) {
        const glm::dvec3 Y2 = std::cos(tilt) * Y + std::sin(tilt) * Z;
        Z = std::cos(tilt) * Z - std::sin(tilt) * Y;
        Y = Y2;
    }
    t.X = X, t.Y = Y, t.Z = Z;
    return t;
}

} // namespace phx
