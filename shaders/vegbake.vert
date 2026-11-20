#version 460
#include "common.glsl"
#include "veg_common.glsl"
// Impostor bake: orthographic view of the full-resolution tree pc.starCount from frame pc.lightIndex.
layout(location = 0) out vec3 vNormal; // tree local
layout(location = 1) out vec3 vPos;    // tree local
layout(location = 2) out vec2 vUV;
layout(location = 3) flat out uint vMaterial;

void main() {
    TreeVertex tv = TreeVertices(pc.aux).v[gl_VertexIndex];
    vec4 sp = speciesA(pc.starCount);
    vec3 v = frameDirection(pc.lightIndex), right, upCam;
    frameBasis(v, right, upCam);
    vec3 p = tv.pos - vec3(0.0, sp.x * 0.5, 0.0);
    float ext = sp.z;
    gl_Position = vec4(dot(p, right) / ext, -dot(p, upCam) / ext, 0.5 - dot(p, v) / (2.0 * ext), 1.0);
    vNormal = decodeTreeNormal(tv.normal);
    vPos = tv.pos;
    vUV = unpackHalf2x16(tv.uv);
    vMaterial = (pc.starCount << 8) | (tv.extra & 255u);
}
