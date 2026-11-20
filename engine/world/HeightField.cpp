#include "engine/world/HeightField.hpp"

#include "engine/world/Climate.hpp"

#include "engine/core/Log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace phx {
namespace {
constexpr f64 kDeg = 3.14159265358979323846 / 180.0;

// ---------------------------------------------------------------- Eocene corrections
// Modern topography → early Eocene (~50 Ma), deliberately coarse:
//  * no polar ice: ~+70 m eustatic sea level (higher still in the Eocene, but this keeps
//    today's coastlines recognisable for navigation)
//  * Tibetan plateau and Himalaya not yet uplifted to present heights
//  * Antarctica and Greenland without ice sheets (bedrock roughly 40% of ice surface)
f64 eoceneCorrection(f64 h, f64 lat, f64 lon) {
    auto bump = [](f64 x, f64 lo, f64 hi, f64 edge) {
        return std::clamp((x - lo) / edge, 0.0, 1.0) * std::clamp((hi - x) / edge, 0.0, 1.0);
    };
    if (h > 0) {
        const f64 tibet = bump(lat, 26, 41, 3) * bump(lon, 72, 104, 4);
        if (tibet > 0) h = std::lerp(h, 0.5 * h + 300.0, tibet);
        const f64 ice = std::clamp((-lat - 62.0) / 6.0, 0.0, 1.0) + bump(lat, 60, 84, 3) * bump(lon, -73, -12, 4);
        if (ice > 0) h = std::lerp(h, 0.4 * h - 150.0, std::min(ice, 1.0));
    }
    return h - 70.0;
}

// ---------------------------------------------------------------- noise (double precision)
const u8* permutation() {
    static u8 p[512];
    static bool init = [] {
        u32 s = 0x2545F491u;
        u8 base[256];
        for (int i = 0; i < 256; ++i) base[i] = u8(i);
        for (int i = 255; i > 0; --i) {
            s ^= s << 13, s ^= s >> 17, s ^= s << 5;
            std::swap(base[i], base[s % u32(i + 1)]);
        }
        for (int i = 0; i < 512; ++i) p[i] = base[i & 255];
        return true;
    }();
    (void)init;
    return p;
}

f64 fade(f64 t) { return t * t * t * (t * (t * 6 - 15) + 10); }
f64 grad(int h, f64 x, f64 y, f64 z) {
    const int k = h & 15;
    const f64 u = k < 8 ? x : y, v = k < 4 ? y : (k == 12 || k == 14 ? x : z);
    return ((k & 1) ? -u : u) + ((k & 2) ? -v : v);
}
// Improved Perlin noise, range ≈ [-1, 1].
f64 perlin(f64 x, f64 y, f64 z) {
    const u8* p = permutation();
    const f64 fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    const int X = int(i64(fx) & 255), Y = int(i64(fy) & 255), Z = int(i64(fz) & 255);
    x -= fx, y -= fy, z -= fz;
    const f64 u = fade(x), v = fade(y), w = fade(z);
    const int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z, B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
    auto lerp = [](f64 t, f64 a, f64 b) { return a + t * (b - a); };
    return lerp(w,
                lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                     lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                     lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));
}

f64 catmullRomDeriv(f64 p0, f64 p1, f64 p2, f64 p3, f64 t) {
    return 0.5 * ((-p0 + p2) + 2.0 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t + 3.0 * (-p0 + 3 * p1 - 3 * p2 + p3) * t * t);
}
f64 catmullRom(f64 p0, f64 p1, f64 p2, f64 p3, f64 t) {
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
}
} // namespace

// ---------------------------------------------------------------- frames

glm::dvec3 directionFromLatLon(f64 lat, f64 lon) {
    return {std::cos(lat * kDeg) * std::cos(lon * kDeg), std::cos(lat * kDeg) * std::sin(lon * kDeg), std::sin(lat * kDeg)};
}

void latLonFromDirection(const glm::dvec3& d, f64& lat, f64& lon) {
    lat = std::asin(std::clamp(d.z, -1.0, 1.0)) / kDeg;
    lon = std::atan2(d.y, d.x) / kDeg;
}

glm::dmat3 ecefToEnuMatrix(const glm::dvec3& position) {
    const glm::dvec3 up = glm::normalize(position);
    glm::dvec3 east = glm::cross(glm::dvec3(0, 0, 1), up);
    east = glm::length(east) < 1e-9 ? glm::dvec3(1, 0, 0) : glm::normalize(east);
    const glm::dvec3 north = glm::cross(up, east);
    return glm::transpose(glm::dmat3(east, north, up));
}

// ---------------------------------------------------------------- height field

HeightField::HeightField(const std::string& path, const std::string& regionalDir) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    u32 dims[2];
    f32 hdr[4];
    if (f && f.read(magic, 8) && std::memcmp(magic, "PHXDEM01", 8) == 0 && f.read(reinterpret_cast<char*>(dims), 8) &&
        f.read(reinterpret_cast<char*>(hdr), 16) && dims[0] * u64(dims[1]) < 100'000'000ull) {
        width_ = dims[0], height_ = dims[1];
        lon0_ = hdr[0], lat0_ = hdr[1], dLon_ = hdr[2], dLat_ = hdr[3];
        grid_.resize(std::size_t(width_) * height_);
        if (!f.read(reinterpret_cast<char*>(grid_.data()), std::streamsize(grid_.size() * 4))) grid_.clear();
        // Apply the Eocene corrections once, at grid resolution.
        for (u32 y = 0; y < height_ && !grid_.empty(); ++y)
            for (u32 x = 0; x < width_; ++x) {
                f32& h = grid_[std::size_t(y) * width_ + x];
                h = f32(eoceneCorrection(h, lat0_ + y * dLat_, lon0_ + x * dLon_));
            }
    }
    if (grid_.empty()) log::warn("DEM {} not found — using procedural continents", path);
    else log::info("DEM: {}x{} grid ({}° spacing), Eocene corrections applied", width_, height_, dLon_);

    // Regional 30 m cells: memory-mapped, read lazily by the OS.
    std::error_code ec;
    if (!regionalDir.empty() && std::filesystem::is_directory(regionalDir, ec)) {
        for (const auto& e : std::filesystem::directory_iterator(regionalDir, ec)) {
            if (e.path().extension() != ".phxdem") continue;
            const int fd = ::open(e.path().c_str(), O_RDONLY);
            if (fd < 0) continue;
            struct stat st{};
            ::fstat(fd, &st);
            void* m = ::mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) continue;
            const auto* bytes = static_cast<const char*>(m);
            Cell c;
            if (st.st_size < 32 || std::memcmp(bytes, "PHXD30C1", 8) != 0) {
                ::munmap(m, std::size_t(st.st_size));
                continue;
            }
            std::memcpy(&c.lat, bytes + 8, 4);
            std::memcpy(&c.lon, bytes + 12, 4);
            std::memcpy(&c.size, bytes + 16, 4);
            std::memcpy(&c.mipCount, bytes + 20, 4);
            std::memcpy(&c.scale, bytes + 24, 4);
            std::memcpy(&c.offset, bytes + 28, 4);
            c.mapping = m;
            c.mappingSize = std::size_t(st.st_size);
            std::size_t off = 32, n = c.size;
            bool ok = c.mipCount > 0 && c.mipCount <= 16;
            for (u32 i = 0; ok && i < c.mipCount; ++i, n /= 2) {
                ok = off + n * n * 2 <= c.mappingSize;
                c.mips[i] = reinterpret_cast<const u16*>(bytes + off);
                off += n * n * 2;
            }
            if (!ok) {
                ::munmap(m, c.mappingSize);
                continue;
            }
            ::madvise(m, c.mappingSize, MADV_RANDOM); // tile jobs touch scattered rows
            cells_.push_back(c);
        }
    }
    if (!cells_.empty()) {
        i32 la0 = 90, la1 = -90, lo0 = 180, lo1 = -180;
        for (const Cell& c : cells_) la0 = std::min(la0, c.lat), la1 = std::max(la1, c.lat), lo0 = std::min(lo0, c.lon), lo1 = std::max(lo1, c.lon);
        idxLat0_ = la0, idxLon0_ = lo0, idxH_ = la1 - la0 + 1, idxW_ = lo1 - lo0 + 1;
        cellIndex_.assign(std::size_t(idxW_) * idxH_, -1);
        for (std::size_t i = 0; i < cells_.size(); ++i)
            cellIndex_[std::size_t(cells_[i].lat - la0) * idxW_ + (cells_[i].lon - lo0)] = i32(i);
        std::size_t bytes = 0;
        for (const Cell& c : cells_) bytes += c.mappingSize;
        log::info("Regional DEM: {} cells (30 m, {}° box from {}°N {}°E), {} MiB mapped", cells_.size(),
                  idxW_ * idxH_, la0, lo0, bytes >> 20);
    }
    // Mountain-belt factor from the 1° DEM: high ground and steep large-scale gradients.
    mountain_.resize(1440u * 720u);
    for (u32 y = 0; y < 720; ++y)
        for (u32 x = 0; x < 1440; ++x) {
            const f64 lat = -90.0 + (y + 0.5) / 4.0, lon = -180.0 + (x + 0.5) / 4.0;
            const f64 b = baseHeight(lat, lon);
            const f64 kmLon = std::max(111.32 * std::cos(lat * kDeg), 5.0) * 0.5, kmLat = 111.32 * 0.5;
            const f64 gx = (baseHeight(lat, lon + 0.25) - baseHeight(lat, lon - 0.25)) / kmLon;
            const f64 gy = (baseHeight(std::min(lat + 0.25, 90.0), lon) - baseHeight(std::max(lat - 0.25, -90.0), lon)) / kmLat;
            const f64 g = std::sqrt(gx * gx + gy * gy); // m per km
            auto ss = [](f64 e0, f64 e1, f64 v) { const f64 t = std::clamp((v - e0) / (e1 - e0), 0.0, 1.0); return t * t * (3 - 2 * t); };
            mountain_[std::size_t(y) * 1440 + x] = f32(std::clamp(0.35 * ss(500.0, 3000.0, b) + 0.85 * ss(0.6, 4.0, g), 0.0, 1.0) * ss(-50.0, 150.0, b));
        }
    climate_ = std::make_unique<Climate>(*this);
}

f32 HeightField::mountainFactor(f64 lat, f64 lon) const {
    const f64 fx = (lon + 180.0) * 4.0 - 0.5, fy = (lat + 90.0) * 4.0 - 0.5;
    const i32 x0 = i32(std::floor(fx)), y0 = i32(std::floor(fy));
    const f32 tx = f32(fx - x0), ty = f32(fy - y0);
    auto at = [&](i32 x, i32 y) {
        x = ((x % 1440) + 1440) % 1440;
        y = std::clamp(y, 0, 719);
        return mountain_[std::size_t(y) * 1440 + x];
    };
    return (at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx) * (1 - ty) + (at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx) * ty;
}

HeightField::~HeightField() {
    for (Cell& c : cells_) ::munmap(c.mapping, c.mappingSize);
}

const HeightField::Cell* HeightField::cellAt(i32 lat, i32 lon) const {
    const i32 y = lat - idxLat0_, x = lon - idxLon0_;
    if (cells_.empty() || x < 0 || y < 0 || x >= idxW_ || y >= idxH_) return nullptr;
    const i32 i = cellIndex_[std::size_t(y) * idxW_ + x];
    return i < 0 ? nullptr : &cells_[std::size_t(i)];
}

// Sample (x, y) of mip `mip` in the cell grid; x east, y south from the cell's north
// edge; coordinates outside [0, n) continue into neighbouring cells. NaN if missing.
f64 HeightField::fetch(i32 lat, i32 lon, u32 mip, i64 x, i64 y) const {
    const Cell* c = cellAt(lat, lon);
    if (!c || mip >= c->mipCount) return std::nan("");
    const i64 n = i64(c->size >> mip);
    while (x < 0) x += n, --lon;
    while (x >= n) x -= n, ++lon;
    while (y < 0) y += n, ++lat;
    while (y >= n) y -= n, --lat;
    c = cellAt(lat, lon);
    if (!c || mip >= c->mipCount) return std::nan("");
    return f64(c->mips[mip][y * n + x]) * c->scale + c->offset;
}

f64 HeightField::regionalHeight(f64 lat, f64 lon, f64 res, f64& weight, f64& gradient) const {
    gradient = 0.0;
    weight = 0.0;
    const i32 clat = i32(std::floor(lat)), clon = i32(std::floor(lon));
    const Cell* c = cellAt(clat, clon);
    if (!c) return 0.0;

    // Blend towards the global layer within 0.05° of an edge without a neighbour cell.
    const f64 fx = lon - clon, fy = lat - clat;
    auto ramp = [](f64 d) { const f64 t = std::clamp(d / 0.05, 0.0, 1.0); return t * t * (3 - 2 * t); };
    weight = 1.0;
    if (!cellAt(clat, clon - 1)) weight *= ramp(fx);
    if (!cellAt(clat, clon + 1)) weight *= ramp(1 - fx);
    if (!cellAt(clat - 1, clon)) weight *= ramp(fy);
    if (!cellAt(clat + 1, clon)) weight *= ramp(1 - fy);
    if (weight <= 0.0) return 0.0;

    // Mip choice. CDLOD needs a child tile's even vertices to equal its parent's, so
    // neighbouring levels must sample the same data; switching mips per level opens
    // steps of tens of metres on steep ground. Every tile with spacing up to 8 texels
    // (≈240 m, i.e. everything within several km of the viewer) therefore reads mip 0;
    // only distant coarse tiles use filtered mips, where skirts cover the residual.
    const f64 texel0 = 111320.0 / c->size;
    const f64 ratio = res / (texel0 * 8.0);
    u32 mip = ratio > 1.0 ? u32(std::floor(std::log2(ratio))) + 1u : 0u;
    mip = std::min(mip, c->mipCount - 1);
    const f64 n = f64(c->size >> mip);
    // Copernicus is pixel-is-point: sample 0 sits on the north-west corner.
    const f64 gx = fx * n, gy = (1.0 - fy) * n;
    const i64 x = i64(std::floor(gx)), y = i64(std::floor(gy));
    const f64 tx = gx - f64(x), ty = gy - f64(y);
    f64 rows[4], drows[4];
    for (int j = -1; j <= 2; ++j) {
        f64 p[4];
        for (int i = -1; i <= 2; ++i) {
            p[i + 1] = fetch(clat, clon, mip, x + i, y + j);
            if (std::isnan(p[i + 1])) p[i + 1] = fetch(clat, clon, mip, std::clamp<i64>(x + i, 0, i64(n) - 1),
                                                         std::clamp<i64>(y + j, 0, i64(n) - 1));
        }
        rows[j + 1] = catmullRom(p[0], p[1], p[2], p[3], tx);
        drows[j + 1] = catmullRomDeriv(p[0], p[1], p[2], p[3], tx);
    }
    // Slope of the DEM surface (rise over run, metres per metre) for steepness-driven detail.
    const f64 texelY = 111320.0 / n, texelX = texelY * std::cos(lat * kDeg);
    const f64 dx = catmullRom(drows[0], drows[1], drows[2], drows[3], ty) / texelX;
    const f64 dy = catmullRomDeriv(rows[0], rows[1], rows[2], rows[3], ty) / texelY;
    gradient = std::sqrt(dx * dx + dy * dy);
    // Same eustatic shift as the global layer (modern MSL → Eocene sea level).
    return catmullRom(rows[0], rows[1], rows[2], rows[3], ty) - 70.0;
}

f64 HeightField::sample(i32 x, i32 y) const {
    y = std::clamp(y, 0, i32(height_) - 1);
    x = ((x % i32(width_ - 1)) + i32(width_ - 1)) % i32(width_ - 1); // last column duplicates the first
    return grid_[std::size_t(y) * width_ + u32(x)];
}

f64 HeightField::baseHeight(f64 lat, f64 lon) const {
    if (grid_.empty()) {
        const glm::dvec3 d = directionFromLatLon(lat, lon);
        return 2500.0 * perlin(d.x * 2.1 + 11, d.y * 2.1, d.z * 2.1) + 1200.0 * perlin(d.x * 5, d.y * 5 + 3, d.z * 5) - 600.0;
    }
    const f64 fx = (lon - lon0_) / dLon_, fy = (lat - lat0_) / dLat_;
    const i32 x = i32(std::floor(fx)), y = i32(std::floor(fy));
    const f64 tx = fx - x, ty = fy - y;
    f64 rows[4];
    for (int j = -1; j <= 2; ++j)
        rows[j + 1] = catmullRom(sample(x - 1, y + j), sample(x, y + j), sample(x + 1, y + j), sample(x + 2, y + j), tx);
    return catmullRom(rows[0], rows[1], rows[2], rows[3], ty);
}

f64 HeightField::height(const glm::dvec3& d, f64 resolutionM) const {
    f64 lat, lon;
    latLonFromDirection(d, lat, lon);
    f64 w = 0.0, grad = 0.0;
    const f64 regional = regionalHeight(lat, lon, resolutionM, w, grad);
    f64 h = w < 1.0 ? globalSurface(d, lat, lon, resolutionM).height : 0.0;
    if (w > 0.0) {
        // Sub-30 m relief the DEM cannot resolve.
        const glm::dvec3 q = d * (kPlanetRadiusM / 48.0);
        // Octaves are band-limited by the sampling resolution: a wavelength under ~4
        // vertex spacings would alias into worm-like shading on distant tiles.
        f64 detail = 0, amp = 2.4, f = 1;
        for (int o = 0; o < 5; ++o, f *= 2.1, amp *= 0.45) {
            const f64 wavelength = 48.0 / f;
            const f64 band = resolutionM <= 0.0 ? 1.0 : std::clamp(wavelength / (4.0 * resolutionM) - 1.0, 0.0, 1.0);
            if (band > 0.0) detail += band * amp * perlin(q.x * f + 3.3, q.y * f - 1.7, q.z * f + o);
        }

        // Steep ground (≳35°): granite breaks into blocks; flat ground is untouched.
        // Fade out where the mesh cannot resolve a ledge (it would alias into worms).
        const f64 resolvable = resolutionM <= 0.0 ? 1.0 : std::clamp((2.6 - resolutionM) / 1.2, 0.0, 1.0);
        const f64 steep = std::clamp((grad - 0.7) / 0.7, 0.0, 1.0) * resolvable;
        if (steep > 0.0) {
            // Granite: blocky joint relief (ridged noise). Elevation-keyed terraces were
            // tried and rejected: they follow contour lines and read as a topographic map.
            const glm::dvec3 qb = d * (kPlanetRadiusM / 18.0);
            const f64 blocks = (1.0 - std::abs(perlin(qb.x, qb.y, qb.z))) * 2.0 +
                               (1.0 - std::abs(perlin(qb.x * 2.3 + 5.0, qb.y * 2.3, qb.z * 2.3))) * 0.8 - 1.9;
            detail += steep * blocks * 2.6;
        }
        h = std::lerp(h, regional + detail, w);
    }
    return h;
}

SurfaceInfo HeightField::surface(const glm::dvec3& d, f64 resolutionM) const {
    f64 lat, lon;
    latLonFromDirection(d, lat, lon);
    f64 w = 0.0, grad = 0.0;
    (void)regionalHeight(lat, lon, resolutionM, w, grad);
    SurfaceInfo s = globalSurface(d, lat, lon, resolutionM);
    if (w > 0.0) s.height = height(d, resolutionM);
    return s;
}

namespace {
f64 smooth01(f64 e0, f64 e1, f64 v) {
    const f64 t = std::clamp((v - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3 - 2 * t);
}
} // namespace

// Global procedural terrain on the 1° DEM, shaped by the mountain-belt factor and the
// biome (Climate): eroded ridges in mountains, gentle hills elsewhere, dune seas in
// sandy deserts, and escarpments (cliff bands with scree aprons) — cuesta-like steps
// along isolines of a low-frequency field that is independent of elevation, so they
// form a few long scarps rather than contour-parallel terraces.
SurfaceInfo HeightField::globalSurface(const glm::dvec3& d, f64 lat, f64 lon, f64 res) const {
    SurfaceInfo out;
    const f64 base = baseHeight(lat, lon);
    const f64 land = smooth01(-150.0, 150.0, base);
    const f64 M = climate_ ? f64(mountainFactor(lat, lon)) : 0.0;
    f64 desert = 0.0;
    if (climate_) {
        const ClimateSample c = climate_->sample(lat, lon);
        out.tempC = Climate::temperatureAt(c, std::max(base, 0.0));
        out.precipMm = c.precipMm, out.amplitudeC = c.amplitudeC;
        desert = biomeWeights(out.tempC, c.precipMm, c.amplitudeC).desert;
    }
    out.mountain = f32(M);

    const glm::dvec3 p = d * kPlanetRadiusM; // metres
    // Domain warp (km scale) breaks the regularity of every layer below.
    const glm::dvec3 q = p / 9000.0;
    const glm::dvec3 warp(perlin(q.x + 1.3, q.y, q.z), perlin(q.x, q.y + 7.1, q.z), perlin(q.x, q.y, q.z + 3.9));
    const glm::dvec3 pw = p + warp * 1800.0;

    // 1. Relief: ridged multifractal; amplitude from the mountain belt factor.
    f64 relief = 50.0 + 1500.0 * std::pow(M, 1.2) + 120.0 * land;
    relief *= 1.0 - 0.5 * desert * (1.0 - M); // desert plains are flat
    // Base wavelength 24 km: ranges show ridges and valleys at the 1–10 km scale a hiker sees.
    const glm::dvec3 r = pw / 24000.0;
    f64 sum = 0, amp = 1, freq = 1, weight = 1, norm = 0;
    for (int o = 0; o < 14; ++o) {
        const f64 wavelength = 24000.0 / freq;
        const f64 band = res <= 0.0 ? 1.0 : std::clamp(wavelength / (3.0 * res) - 1.0, 0.0, 1.0);
        f64 n = 1.0 - std::abs(perlin(r.x * freq + o * 17.1, r.y * freq - o * 9.3, r.z * freq + o * 3.7));
        n *= n;
        n *= weight;
        weight = std::clamp(n * 1.6, 0.0, 1.0);
        sum += n * amp * band;
        norm += amp;
        freq *= 2.03;
        amp *= o < 5 ? 0.55 : 0.42;
    }
    const f64 ridged = sum / norm;
    const f64 swell = perlin(r.x * 0.31 + 5.2, r.y * 0.31, r.z * 0.31 - 7.7);
    f64 h = base + land * (relief * (ridged - 0.35) * 2.0 + relief * 0.5 * swell);

    // 2. Escarpments: steps along isolines of a smooth field C (4 km scale).
    const glm::dvec3 qm = p / 30000.0;
    const f64 rocky = std::clamp(0.3 + 0.7 * M + 0.6 * desert, 0.0, 1.0);
    const f64 cliffMask = land * rocky * smooth01(-0.05, 0.35, perlin(qm.x + 4.4, qm.y - 2.2, qm.z + 8.8));
    if (cliffMask > 0.0) {
        auto fieldC = [&](const glm::dvec3& w) {
            const glm::dvec3 qc = w / 4000.0;
            return 0.65 * perlin(qc.x, qc.y + 11.0, qc.z) + 0.25 * perlin(qc.x * 2.3 + 5.0, qc.y * 2.3, qc.z * 2.3);
        };
        const f64 C = fieldC(pw);
        constexpr f64 kSpacing = 0.21; // C units between scarps
        const f64 stepH = 35.0 + 90.0 * M + 40.0 * desert; // scarp height
        // Face width in metres (≥ 12 m, ≥ 1.6 vertex spacings so coarse tiles do not
        // alias the edge into a comb), converted with the field's actual local gradient.
        const glm::dvec3 e1 = glm::normalize(glm::cross(glm::dvec3(0, 0, 1), d)), e2 = glm::cross(d, e1);
        const f64 gC = std::hypot(fieldC(pw + e1 * 20.0) - C, fieldC(pw + e2 * 20.0) - C) / 20.0;
        const f64 faceHalfM = std::max(12.0, 1.6 * res);
        const f64 halfW = std::clamp(faceHalfM * gC / kSpacing, 0.004, 0.2);
        const f64 t = C / kSpacing, k = std::floor(t), fr = t - k;
        const f64 riser = smooth01(0.5 - halfW, 0.5 + halfW, fr);
        // Scree apron at the foot (low side): a bump that rises towards the face and
        // fades out across it, so it is continuous where fr wraps (a discontinuity there
        // is a zero-width step that aliases into a comb at every LOD).
        const f64 apron = smooth01(0.5 - halfW - 0.08, 0.5 - halfW, fr) * (1.0 - riser);
        h += cliffMask * stepH * ((k + riser - t) + 0.35 * apron * apron);
        out.cliff = f32(cliffMask * std::clamp(1.0 - std::abs(fr - 0.5) / (halfW * 1.6), 0.0, 1.0));
        out.talus = f32(cliffMask * apron);
    }

    // 3. Dune seas in flat sandy deserts: transverse dunes, gentle windward / steep lee.
    const glm::dvec3 qe = p / 40000.0;
    const f64 sand = land * desert * (1.0 - M) * (1.0 - M) * smooth01(-0.1, 0.3, perlin(qe.x - 3.3, qe.y + 6.6, qe.z));
    out.sand = f32(sand);
    if (sand > 0.0 && (res <= 0.0 || res < 300.0)) {
        const glm::dvec3 east = glm::normalize(glm::cross(glm::dvec3(0, 0, 1), d));
        const glm::dvec3 north = glm::cross(d, east);
        const f64 along = glm::dot(p, east), across = glm::dot(p, north);
        constexpr f64 kLambda = 600.0;
        // Crest lines bend over kilometres; no short-scale phase noise (it breaks the
        // crests into a comb seen from afar).
        const f64 phase = along / kLambda + 0.9 * perlin(across / 2500.0, along / 4000.0, 1.7) + 0.12 * perlin(across / 1200.0, 3.3, along / 1500.0);
        const f64 f = phase - std::floor(phase);
        const f64 profile = f < 0.88 ? smooth01(0.0, 0.88, f) : 1.0 - smooth01(0.88, 1.0, f); // lee ≈ 30°
        const f64 duneH = 34.0 * (0.6 + 0.4 * perlin(along / 3000.0, across / 3000.0, 4.2));
        const f64 fade = res <= 0.0 ? 1.0 : 1.0 - smooth01(150.0, 300.0, res); // only on km-scale tiles
        h += sand * duneH * profile * fade;
    }

    // 4. Metre-scale roughness, band-limited; smoother on sand, blocky on scarp faces.
    const glm::dvec3 qd = p / 48.0;
    f64 detail = 0, a = 2.0, f = 1;
    for (int o = 0; o < 5; ++o, f *= 2.1, a *= 0.45) {
        const f64 band = res <= 0.0 ? 1.0 : std::clamp(48.0 / f / (4.0 * res) - 1.0, 0.0, 1.0);
        if (band > 0.0) detail += band * a * perlin(qd.x * f + 3.3, qd.y * f - 1.7, qd.z * f + o);
    }
    h += land * detail * (1.0 - 0.7 * sand);
    if (out.cliff > 0.0f && (res <= 0.0 || res < 3.0)) {
        const glm::dvec3 qb = p / 9.0;
        const f64 blocks = (1.0 - std::abs(perlin(qb.x, qb.y, qb.z))) * 2.0 + (1.0 - std::abs(perlin(qb.x * 2.3 + 5, qb.y * 2.3, qb.z * 2.3))) - 2.0;
        h += f64(out.cliff) * blocks * 2.5;
    }
    out.height = h;
    if (climate_) out.tempC = Climate::temperatureAt(climate_->sample(lat, lon), std::max(h, 0.0));
    return out;
}

} // namespace phx
