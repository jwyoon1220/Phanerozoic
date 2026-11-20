#version 460
#include "common.glsl"
// CDLOD terrain with vertex pulling. Positions are tile-relative floats; the tile
// offset (tile centre − camera) was computed in double on the CPU, so nothing large
// is ever subtracted on the GPU (no jitter at planetary scale).
layout(location = 0) out vec3 vRel;      // camera-relative ECEF position
layout(location = 1) out vec3 vNormal;   // ECEF
layout(location = 2) out float vHeight;  // metres (40 m classes), < 0 = sea
layout(location = 3) flat out float vLevel;
layout(location = 4) out vec4 vHintA;    // sand, cliff, talus, desert
layout(location = 5) out vec4 vHintB;    // forest, conifer, snow, dryness

struct TerrainVertex { vec3 pos; uint packed; uint hintsA; uint hintsB; };
layout(buffer_reference, scalar) readonly buffer VertexPool { TerrainVertex v[]; };
struct TerrainInstance { vec3 offset; uint slot; float morphStart; float morphEnd; float pad0; float pad1; };
layout(buffer_reference, scalar) readonly buffer Instances { TerrainInstance i[]; };

const uint GRID = 65u;
const uint GRID_VERTS = GRID * GRID;
const uint VERTS_PER_TILE = GRID_VERTS + 4u * GRID;

vec3 decodeNormal(uint p) {
    vec2 e = vec2(float(p & 4095u), float((p >> 12) & 4095u)) / 4095.0 * 2.0 - 1.0;
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

void main() {
    TerrainInstance inst = Instances(pc.stars).i[gl_InstanceIndex];
    VertexPool pool = VertexPool(pc.aux);
    uint base = inst.slot * VERTS_PER_TILE;

    uint vi = uint(gl_VertexIndex);
    bool skirt = vi >= GRID_VERTS;
    uint gi = vi;
    if (skirt) { // skirt vertex → the edge vertex it hangs from
        uint s = vi - GRID_VERTS, edge = s / GRID, k = s % GRID;
        gi = edge == 0u ? k : edge == 1u ? (GRID - 1u) * GRID + k : edge == 2u ? k * GRID : k * GRID + GRID - 1u;
    }
    ivec2 ij = ivec2(gi % GRID, gi / GRID);
    TerrainVertex tv = pool.v[base + gi];
    vec3 p = tv.pos;

    // Morph odd vertices onto the parent grid as the node approaches its range end.
    bool oi = (ij.x & 1) == 1, oj = (ij.y & 1) == 1;
    vec3 target = p;
    #define V(x, y) pool.v[base + uint(y) * GRID + uint(x)].pos
    if (oi && !oj) target = 0.5 * (V(ij.x - 1, ij.y) + V(ij.x + 1, ij.y));
    else if (!oi && oj) target = 0.5 * (V(ij.x, ij.y - 1) + V(ij.x, ij.y + 1));
    else if (oi && oj) target = 0.5 * (V(ij.x - 1, ij.y - 1) + V(ij.x + 1, ij.y + 1));
    float dist = length(inst.offset + p);
    dist *= 1.0 + (1.0 - exp(-dist / 20000.0)); // LOD distance warp, mirrors lodDistance() on the CPU
    float morph = clamp((dist - inst.morphStart) / max(inst.morphEnd - inst.morphStart, 1e-3), 0.0, 1.0);
    vec3 pos = mix(p, target, morph);
    if (skirt) pos += pool.v[base + vi].pos - p;

    vRel = inst.offset + pos;
    vNormal = decodeNormal(tv.packed);
    vHintA = unpackUnorm4x8(tv.hintsA);
    vHintB = unpackUnorm4x8(tv.hintsB);
    vHeight = float(tv.packed >> 24) * 40.0 - 4000.0;
    vLevel = inst.pad0 + (skirt ? 100.0 : 0.0);
    gl_Position = FRAME.viewProjEcef * vec4(vRel, 1.0);
}
