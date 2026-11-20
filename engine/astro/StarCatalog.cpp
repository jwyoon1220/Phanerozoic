#include "engine/astro/StarCatalog.hpp"

#include "engine/astro/Astro.hpp"
#include "engine/core/Log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace phx::astro {

bool loadStarRecords(const std::string& path, std::vector<StarRecord>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    u32 count = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&count), 4);
    if (!f || std::memcmp(magic, kStarFileMagic, 8) != 0 || count > 10'000'000) return false;
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), std::streamsize(count * sizeof(StarRecord)));
    return bool(f);
}

f32 fluxFromMagnitude(f64 vmag) { return f32(std::pow(10.0, -0.4 * (vmag - kSunMagnitude))); }

void colorFromBV(f32 bv, f32 rgb[3]) {
    bv = std::clamp(bv, -0.4f, 2.0f);
    // Ballesteros (2012): B−V → effective temperature.
    const f64 t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    // Planckian locus in CIE xy (Kim et al. cubic spline), then XYZ → linear sRGB.
    const f64 it = 1e3 / t, it2 = it * it, it3 = it2 * it;
    const f64 x = t <= 4000 ? -0.2661239 * it3 - 0.2343589 * it2 + 0.8776956 * it + 0.179910
                            : -3.0258469 * it3 + 2.1070379 * it2 + 0.2226347 * it + 0.240390;
    const f64 x2 = x * x, x3 = x2 * x;
    const f64 y = t <= 2222   ? -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
                  : t <= 4000 ? -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
                              : 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    const f64 X = x / y, Y = 1.0, Z = (1 - x - y) / y;
    f64 r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    f64 g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    f64 b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
    r = std::max(r, 0.0), g = std::max(g, 0.0), b = std::max(b, 0.0);
    const f64 lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    rgb[0] = f32(r / lum), rgb[1] = f32(g / lum), rgb[2] = f32(b / lum);
}

std::vector<GpuStar> buildGpuStars(const std::vector<StarRecord>& records, f64 decimalYear, f32 limitMag) {
    const f32 dt = f32(decimalYear - 2000.0);
    std::vector<GpuStar> out;
    out.reserve(records.size());
    for (const StarRecord& s : records) {
        if (s.vmag > limitMag) continue;
        GpuStar g{};
        f32 len = 0;
        for (int i = 0; i < 3; ++i) {
            g.dir[i] = s.dir[i] + s.dDirDt[i] * dt;
            len += g.dir[i] * g.dir[i];
        }
        len = std::sqrt(len);
        for (f32& c : g.dir) c /= len;
        g.flux = fluxFromMagnitude(s.vmag);
        colorFromBV(s.bv, g.color);
        out.push_back(g);
    }
    // Brightest first: lets the GPU cull by count when the sky is bright.
    std::sort(out.begin(), out.end(), [](const GpuStar& a, const GpuStar& b) { return a.flux > b.flux; });
    return out;
}

} // namespace phx::astro
