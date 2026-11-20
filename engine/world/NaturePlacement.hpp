#pragma once
// Deterministic placement of trees, understorey and rocks (plan M5/M6). Shared by the
// renderer (instances on the GPU) and physics (colliders around the player), so both
// see exactly the same objects. Cells are cube-sphere quadtree nodes of three layers.
#include "engine/core/Types.hpp"
#include "engine/world/TreeMeshes.hpp"

#include <glm/glm.hpp>

#include <vector>

namespace phx {
class HeightField;

struct VegInstance {
    f32 pos[3]; // relative to the cell centre (ECEF axes)
    u32 packed; // species 4 | yaw 9 | scale 9 | tint 4 | tilt 6 bits
};
static_assert(sizeof(VegInstance) == 16);

struct NatureSpeciesDims {
    f32 height = 1, radius = 1; // unscaled asset size (metres)
};
// Per-species multiplier on the packed 0.5-1.5 scale.
inline constexpr f32 kNatureScaleMul[kSpeciesCount] = {1.0f, 1.0f, 1.0f, 1.0f, 0.7f, 1.0f, 1.0f, 1.0f, 1.0f};

struct NatureLayer {
    u32 level;
    f64 radius, step; // residency radius around the viewer, sampling step for cell lookup
};
//   level 14 (~610 m): trees                   level 16 (~150 m): boulders
//   level 16 also shrubs; level 18 (~76 m): ferns, grass clumps, mossy stones
inline constexpr NatureLayer kNatureLayers[3] = {{14, 4200.0, 250.0}, {16, 450.0, 60.0}, {18, 130.0, 25.0}};
inline constexpr u32 kMaxNaturePerCell = 8192;

// Fills `out` with the cell's instances; returns the cell centre (ECEF, on the ground).
glm::dvec3 placeNature(u64 cellKey, const HeightField& heights, const NatureSpeciesDims* dims, std::vector<VegInstance>& out);

// Decoded instance transform (matches veg_common.glsl treeFrame/instanceVertex):
// local mesh point p maps to origin + (X*p.x + Y*p.y + Z*p.z) * scale.
struct NatureTransform {
    u32 species;
    f64 scale;
    glm::dvec3 origin, X, Y, Z;
};
NatureTransform natureTransform(const glm::dvec3& positionEcef, u32 packed);

} // namespace phx
