#pragma once
// Planet surface height, two layers:
//  * global: 1° DEM (PALEOMAP 0 Ma, modern geography) with Eocene corrections and
//    procedural relief, everywhere;
//  * regional: Copernicus GLO-30 cells (30 m, mip-mapped, memory-mapped from disk) over
//    the play corridor, blended into the global layer at the coverage boundary.
// Pure and thread-safe after construction: tile generation jobs call it from any worker;
// the OS pages regional data in on first touch (disk streaming via mmap).
#include "engine/core/Types.hpp"

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace phx {
class Climate;

inline constexpr f64 kPlanetRadiusM = 6371000.0;

// Surface description at a point (global procedural terrain).
struct SurfaceInfo {
    f64 height = 0;     // metres above (Eocene) sea level
    f32 cliff = 0;      // 0..1 on escarpment faces
    f32 talus = 0;      // 0..1 on the scree apron below a face
    f32 sand = 0;       // 0..1 dune sand cover
    f32 mountain = 0;   // 0..1 mountain-belt factor (from the 1° DEM)
    f32 tempC = 15;     // local mean annual temperature
    f32 precipMm = 800;
    f32 amplitudeC = 10;
};

class HeightField {
public:
    // Loads the baked global DEM (without one: procedural continents) and every
    // *.phxdem cell found in `regionalDir` (may be empty / missing).
    HeightField(const std::string& globalDemPath, const std::string& regionalDir);
    ~HeightField();
    HeightField(const HeightField&) = delete;
    HeightField& operator=(const HeightField&) = delete;

    // Height above (Eocene) sea level in metres for a unit direction (ECEF).
    // `resolutionM` is the sample spacing of the caller (selects the DEM mip so coarse
    // terrain tiles do not alias); 0 = finest.
    [[nodiscard]] f64 height(const glm::dvec3& unitDir, f64 resolutionM = 0.0) const;
    // Height plus terrain features and climate (placement, physics, debugging).
    [[nodiscard]] SurfaceInfo surface(const glm::dvec3& unitDir, f64 resolutionM = 0.0) const;
    [[nodiscard]] f32 mountainFactor(f64 latDeg, f64 lonDeg) const;
    [[nodiscard]] u32 regionalCellCount() const { return u32(cells_.size()); }
    [[nodiscard]] f64 baseHeight(f64 latDeg, f64 lonDeg) const; // DEM only, corrected
    [[nodiscard]] bool loaded() const { return !grid_.empty(); }
    [[nodiscard]] const Climate& climate() const { return *climate_; }

private:
    struct Cell {
        i32 lat = 0, lon = 0; // south-west corner, degrees
        u32 size = 0;         // level-0 samples per degree
        u32 mipCount = 0;
        f32 scale = 0.1f, offset = 0.0f;
        const u16* mips[16]{};
        void* mapping = nullptr;
        std::size_t mappingSize = 0;
    };

    [[nodiscard]] f64 sample(i32 x, i32 y) const;
    [[nodiscard]] SurfaceInfo globalSurface(const glm::dvec3& unitDir, f64 lat, f64 lon, f64 resolutionM) const;
    [[nodiscard]] const Cell* cellAt(i32 lat, i32 lon) const;
    // Regional height (Eocene-corrected) and blend weight; weight 0 = outside coverage.
    [[nodiscard]] f64 regionalHeight(f64 lat, f64 lon, f64 resolutionM, f64& weight, f64& gradient) const;
    [[nodiscard]] f64 fetch(i32 lat, i32 lon, u32 mip, i64 x, i64 y) const;

    std::vector<Cell> cells_;
    std::vector<i32> cellIndex_; // dense lookup over the covered lat/lon box, −1 = none
    i32 idxLat0_ = 0, idxLon0_ = 0, idxW_ = 0, idxH_ = 0;

    std::vector<f32> grid_;
    u32 width_ = 0, height_ = 0;
    f64 lon0_ = -180, lat0_ = -90, dLon_ = 1, dLat_ = 1;
    std::unique_ptr<Climate> climate_;
    std::vector<f32> mountain_; // 0.25° grid, row 0 = 90°S
};

// ECEF helpers (spherical planet, z = north pole, x = 0° longitude).
glm::dvec3 directionFromLatLon(f64 latDeg, f64 lonDeg);
void latLonFromDirection(const glm::dvec3& dir, f64& latDeg, f64& lonDeg);
glm::dmat3 ecefToEnuMatrix(const glm::dvec3& position); // rows: east, north, up

} // namespace phx
