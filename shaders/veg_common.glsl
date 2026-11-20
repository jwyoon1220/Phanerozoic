// Vegetation/rocks shared declarations (mirror engine/render/VegetationRenderer.*).
struct TreeVertex { vec3 pos; uint normal; uint uv; uint extra; }; // extra: material 0-7 | sway 8-15 | foliage 16
layout(buffer_reference, scalar) readonly buffer TreeVertices { TreeVertex v[]; };
struct VegInstance { vec3 pos; uint packed; };
layout(buffer_reference, scalar) readonly buffer VegPool { VegInstance v[]; };
struct VegCell { vec3 offset; uint slot; uint count; uint pad0, pad1, pad2; };
layout(buffer_reference, scalar) readonly buffer VegCells { VegCell c[]; };
struct VisibleTree { vec3 rel; uint packed; };
layout(buffer_reference, scalar) buffer VisibleTrees { VisibleTree v[]; };
struct DrawCmd { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };

const uint SPECIES = 9u;
const uint TREE_SPECIES = 3u;
const uint MESH_LODS = 3u;
const uint CAMERA_BUCKETS = SPECIES * MESH_LODS + TREE_SPECIES; // 30
const uint SHADOW_CASTERS = 5u;
const uint BUCKETS = CAMERA_BUCKETS + 3u * SHADOW_CASTERS;     // 45
const uint FLAG_IMPOSTOR = 1u, FLAG_WIND = 2u, FLAG_SHADOW = 4u, FLAG_TILT = 8u;

layout(buffer_reference, scalar) buffer VegIndirect {
    DrawCmd cmds[45];
    uint pad0;
    vec4 species[27];  // per species: [height, radius, impostor extent, flags] [lod end ×4] [scaleMul, -, -, -]
    uint capacity[45];
};

const uint VEG_MAX_PER_CELL = 8192u;
const uint IMPOSTOR_FRAMES = 8u;
const uint TEX_TREE_MATERIAL0 = 32u;
const uint MAX_TREE_MATERIALS = 8u;
uint treeTexture(uint species, uint material, uint map) { return TEX_TREE_MATERIAL0 + (species * MAX_TREE_MATERIALS + material) * 3u + map; }

vec4 speciesA(uint s) { return VegIndirect(pc.aux3).species[s * 3u]; }
vec4 speciesLod(uint s) { return VegIndirect(pc.aux3).species[s * 3u + 1u]; }
float speciesScaleMul(uint s) { return VegIndirect(pc.aux3).species[s * 3u + 2u].x; }
uint speciesFlags(uint s) { return uint(speciesA(s).w); }

// packed: species 4 | yaw 9 | scale 9 | tint 4 | tilt 6
uint vegSpecies(uint p) { return p & 15u; }
float vegRotation(uint p) { return float((p >> 4) & 511u) / 511.0 * 6.2831853; }
float vegScale(uint p) { return (0.5 + float((p >> 13) & 511u) / 511.0) * speciesScaleMul(p & 15u); }
float vegTint(uint p) { return float((p >> 22) & 15u) / 15.0; }
float vegTilt(uint p) { return float((p >> 26) & 63u) / 63.0 * 1.4; } // radians (≤ 80°)

// Instance frame at camera-relative ECEF position: X, Y (local up), Z (all ECEF).
// Rocks are tilted about X so their up follows the slope (Z points downhill by yaw).
void treeFrame(vec3 rel, uint packed, out vec3 X, out vec3 Y, out vec3 Z) {
    float rot = vegRotation(packed);
    Y = normalize(FRAME.cameraEcef.xyz + rel);
    vec3 e = normalize(cross(vec3(0.0, 0.0, 1.0), Y));
    vec3 n = cross(Y, e);
    X = cos(rot) * e + sin(rot) * n;
    Z = cross(X, Y);
    float t = vegTilt(packed);
    if (t > 0.0) {
        vec3 Y2 = cos(t) * Y + sin(t) * Z;
        Z = cos(t) * Z - sin(t) * Y;
        Y = Y2;
    }
}

vec3 decodeTreeNormal(uint p) {
    vec2 e = vec2(float(p & 65535u), float(p >> 16)) / 65535.0 * 2.0 - 1.0;
    vec3 n = vec3(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    if (n.y < 0.0) n.xz = (1.0 - abs(n.zx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.z >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

vec2 hemiOctEncode(vec3 d) {
    vec2 p = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
    return vec2(p.x + p.y, p.x - p.y);
}
vec3 hemiOctDecode(vec2 e) {
    vec2 p = vec2(e.x + e.y, e.x - e.y) * 0.5;
    return normalize(vec3(p.x, 1.0 - abs(p.x) - abs(p.y), p.y));
}
vec3 frameDirection(uint frame) {
    vec2 f = vec2(frame % IMPOSTOR_FRAMES, frame / IMPOSTOR_FRAMES);
    return hemiOctDecode((f + 0.5) / float(IMPOSTOR_FRAMES) * 2.0 - 1.0);
}
uvec2 nearestFrame(vec3 localDir) {
    vec2 e = hemiOctEncode(normalize(vec3(localDir.x, max(localDir.y, 0.0), localDir.z)));
    return uvec2(clamp(floor((e * 0.5 + 0.5) * float(IMPOSTOR_FRAMES)), 0.0, float(IMPOSTOR_FRAMES - 1u)));
}
void frameBasis(vec3 v, out vec3 right, out vec3 upCam) {
    right = abs(v.y) > 0.999 ? vec3(1.0, 0.0, 0.0) : normalize(cross(vec3(0.0, 1.0, 0.0), v));
    upCam = cross(v, right);
}

vec3 seasonalFoliage(vec3 albedo, uint species, float tint) {
    float lum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    float turn = FRAME.world.w * (0.75 + 0.5 * tint);
    float t = clamp(turn * smoothstep(0.0, 0.6, tint + 0.25), 0.0, 1.0);
    if (species == 0u) return mix(albedo, vec3(2.6, 0.95, 0.30) * lum * (0.8 + 0.4 * tint), t);
    if (species == 2u) return mix(albedo, vec3(2.3, 1.75, 0.25) * lum * (0.8 + 0.4 * tint), t);
    // Grass and shrubs: tint = dryness (green when moist → straw in arid climates).
    if (species == 6u || species == 7u) return mix(albedo, vec3(1.9, 1.55, 0.75) * lum, smoothstep(0.15, 0.9, tint) * 0.85);
    return albedo * (0.9 + 0.2 * tint);
}

vec3 windOffset(vec3 rel, float w, float height) {
    vec3 dirEcef = transpose(mat3(FRAME.ecefToEnu)) * FRAME.windDir.xyz;
    float phase = dot(rel, vec3(0.031, 0.027, 0.023));
    float t = FRAME.world.y;
    float s = sin(t * 1.1 + phase) * 0.6 + sin(t * 2.7 + phase * 1.7) * 0.25 + 0.35;
    return dirEcef * (w * w * s * FRAME.world.z * 0.012 * height);
}

// Full world-space transform of a mesh vertex for an instance.
vec3 instanceVertex(VisibleTree inst, TreeVertex tv, out vec3 nEcef) {
    uint species = vegSpecies(inst.packed);
    float scale = vegScale(inst.packed);
    vec3 X, Y, Z;
    treeFrame(inst.rel, inst.packed, X, Y, Z);
    vec3 local = tv.pos * scale;
    vec3 rel = inst.rel + X * local.x + Y * local.y + Z * local.z;
    if ((speciesFlags(species) & FLAG_WIND) != 0u)
        rel += windOffset(inst.rel, float((tv.extra >> 8) & 255u) / 255.0, speciesA(species).x * scale);
    vec3 n = decodeTreeNormal(tv.normal);
    nEcef = X * n.x + Y * n.y + Z * n.z;
    return rel;
}
