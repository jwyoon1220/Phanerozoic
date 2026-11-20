// Bakes one Copernicus GLO-30 DEM tile (Cloud-Optimised GeoTIFF, float32, 1°×1°) into
// a mip-mapped, memory-mappable cell file for HeightField's regional layer.
//
// Output (little endian):
//   char[8] "PHXD30C1", i32 latSouth, i32 lonWest, u32 size (level-0 width = height),
//   u32 mipCount, f32 scale, f32 offset, then mip levels 0..n-1 as u16 rows, north row first.
//   height[m] = value * scale + offset   (0.1 m steps, −500…6053 m)
//
// Usage: phx_bake_dem30 <tile.tif> <out.phxdem> <latSouth> <lonWest>
#include <tiffio.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <tile.tif> <out.phxdem> <latSouth> <lonWest>\n", argv[0]);
        return 2;
    }
    TIFF* tif = TIFFOpen(argv[1], "r");
    if (!tif) return 1;
    uint32_t w = 0, h = 0, tw = 0, th = 0;
    uint16_t bps = 0, fmt = 0, spp = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    if (bps != 32 || fmt != SAMPLEFORMAT_IEEEFP || spp != 1 || !TIFFIsTiled(tif)) {
        std::fprintf(stderr, "unsupported TIFF layout (need tiled float32)\n");
        return 1;
    }
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);

    std::vector<float> src(std::size_t(w) * h);
    std::vector<float> tile(std::size_t(tw) * th);
    for (uint32_t ty = 0; ty < h; ty += th)
        for (uint32_t tx = 0; tx < w; tx += tw) {
            if (TIFFReadTile(tif, tile.data(), tx, ty, 0, 0) < 0) {
                std::fprintf(stderr, "tile read failed at %u,%u\n", tx, ty);
                return 1;
            }
            for (uint32_t y = 0; y < th && ty + y < h; ++y)
                for (uint32_t x = 0; x < tw && tx + x < w; ++x) src[std::size_t(ty + y) * w + tx + x] = tile[y * tw + x];
        }
    TIFFClose(tif);

    // Copernicus tiles narrow in longitude above 50°; resample to a square grid so the
    // runtime can treat every cell alike (the corridor is at 20–35°N where w == h).
    const uint32_t n = h;
    std::vector<float> grid(std::size_t(n) * n);
    for (uint32_t y = 0; y < n; ++y)
        for (uint32_t x = 0; x < n; ++x) {
            const uint32_t sx = std::min<uint32_t>(w - 1, uint32_t(uint64_t(x) * w / n));
            float v = src[std::size_t(y) * w + sx];
            if (!std::isfinite(v) || v < -1000.f) v = 0.f; // voids/ocean
            grid[std::size_t(y) * n + x] = v;
        }

    const float scale = 0.1f, offset = -500.f;
    std::vector<std::vector<uint16_t>> mips;
    std::vector<float> level = grid;
    uint32_t size = n;
    float lo = 1e9f, hi = -1e9f;
    for (float v : grid) lo = std::min(lo, v), hi = std::max(hi, v);
    while (true) {
        std::vector<uint16_t> q(level.size());
        for (std::size_t i = 0; i < level.size(); ++i)
            q[i] = uint16_t(std::clamp(std::lround((level[i] - offset) / scale), 0L, 65535L));
        mips.push_back(std::move(q));
        if (size <= 64) break;
        const uint32_t half = size / 2;
        std::vector<float> next(std::size_t(half) * half);
        for (uint32_t y = 0; y < half; ++y)
            for (uint32_t x = 0; x < half; ++x) {
                const float* r0 = &level[std::size_t(2 * y) * size + 2 * x];
                const float* r1 = r0 + size;
                next[std::size_t(y) * half + x] = 0.25f * (r0[0] + r0[1] + r1[0] + r1[1]);
            }
        level = std::move(next);
        size = half;
    }

    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) return 1;
    const int32_t lat = std::atoi(argv[3]), lon = std::atoi(argv[4]);
    const uint32_t mipCount = uint32_t(mips.size());
    std::fwrite("PHXD30C1", 1, 8, out);
    std::fwrite(&lat, 4, 1, out);
    std::fwrite(&lon, 4, 1, out);
    std::fwrite(&n, 4, 1, out);
    std::fwrite(&mipCount, 4, 1, out);
    std::fwrite(&scale, 4, 1, out);
    std::fwrite(&offset, 4, 1, out);
    for (const auto& m : mips) std::fwrite(m.data(), 2, m.size(), out);
    std::fclose(out);
    std::printf("baked %s: %ux%u (src %ux%u), %u mips, %.0f..%.0f m\n", argv[2], n, n, w, h, mipCount, lo, hi);
    return 0;
}
