#version 460
#include "common.glsl"
#include "veg_common.glsl"
layout(location = 0) out vec3 vRel;
layout(location = 1) out vec3 vNormal;   // ECEF
layout(location = 2) out vec2 vUV;
layout(location = 3) flat out uint vMaterial;  // species << 8 | material | foliage << 16
layout(location = 4) flat out float vTint;

void main() {
    VisibleTree inst = VisibleTrees(pc.stars).v[gl_InstanceIndex];
    TreeVertex tv = TreeVertices(pc.aux).v[gl_VertexIndex];
    vec3 n;
    vRel = instanceVertex(inst, tv, n);
    vNormal = n;
    vUV = unpackHalf2x16(tv.uv);
    vMaterial = (vegSpecies(inst.packed) << 8) | (tv.extra & 255u) | (tv.extra & 0x10000u);
    vTint = vegTint(inst.packed);
    gl_Position = FRAME.viewProjEcef * vec4(vRel, 1.0);
}
