#version 460
#include "common.glsl"
#include "veg_common.glsl"
// Shadow caster (LOD2 mesh) into cascade pc.lightIndex.
layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vTexture; // albedo texture for leaf cut-out, 0 = opaque
void main() {
    VisibleTree inst = VisibleTrees(pc.stars).v[gl_InstanceIndex];
    TreeVertex tv = TreeVertices(pc.aux).v[gl_VertexIndex];
    vec3 n;
    vec3 rel = instanceVertex(inst, tv, n);
    vUV = unpackHalf2x16(tv.uv);
    vTexture = (tv.extra & 0x10000u) != 0u ? treeTexture(vegSpecies(inst.packed), tv.extra & 255u, 0u) : 0u;
    gl_Position = FRAME.csm[pc.lightIndex] * vec4(rel, 1.0);
}
