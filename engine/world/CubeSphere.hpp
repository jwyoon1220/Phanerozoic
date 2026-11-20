#pragma once
// Cube-sphere quadtree addressing shared by terrain tiles and vegetation cells.
#include "engine/core/Types.hpp"
#include "engine/world/HeightField.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

#include <cmath>

namespace phx::cubesphere {

inline constexpr f64 kPi = 3.14159265358979323846;

// Faces: outward normal N, tangent axes U, V with U × V = N.
inline const glm::dvec3 kFaceN[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
inline const glm::dvec3 kFaceU[6] = {{0, 1, 0}, {0, -1, 0}, {-1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}};
inline const glm::dvec3 kFaceV[6] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}};

struct NodeId {
    u32 face, level, x, y;
};

inline u64 makeKey(u32 face, u32 level, u32 x, u32 y) {
    return (u64(face) << 61) | (u64(level) << 56) | (u64(x) << 28) | u64(y);
}
inline NodeId decode(u64 k) {
    return {u32(k >> 61), u32((k >> 56) & 31), u32((k >> 28) & 0xFFFFFFF), u32(k & 0xFFFFFFF)};
}

// Tangent-warped cube → sphere (more uniform cell sizes than a plain normalise). u, v ∈ [-1, 1].
inline glm::dvec3 cubeToSphere(u32 face, f64 u, f64 v) {
    return glm::normalize(kFaceN[face] + std::tan(u * kPi / 4) * kFaceU[face] + std::tan(v * kPi / 4) * kFaceV[face]);
}

// Inverse of cubeToSphere.
inline void sphereToCube(const glm::dvec3& d, u32& face, f64& u, f64& v) {
    const glm::dvec3 a = glm::abs(d);
    if (a.x >= a.y && a.x >= a.z) face = d.x > 0 ? 0 : 1;
    else if (a.y >= a.z) face = d.y > 0 ? 2 : 3;
    else face = d.z > 0 ? 4 : 5;
    const f64 n = glm::dot(d, kFaceN[face]);
    u = std::atan(glm::dot(d, kFaceU[face]) / n) * 4 / kPi;
    v = std::atan(glm::dot(d, kFaceV[face]) / n) * 4 / kPi;
}

// Node containing direction d at `level`.
inline u64 keyAt(const glm::dvec3& d, u32 level) {
    u32 face;
    f64 u, v;
    sphereToCube(d, face, u, v);
    const u32 n = 1u << level;
    const u32 x = u32(std::clamp((u + 1) * 0.5 * n, 0.0, f64(n - 1)));
    const u32 y = u32(std::clamp((v + 1) * 0.5 * n, 0.0, f64(n - 1)));
    return makeKey(face, level, x, y);
}

inline f64 edgeLength(u32 level) { return kPlanetRadiusM * kPi / 2.0 / f64(1u << level); }

} // namespace phx::cubesphere
