#version 460
#include "common.glsl"
// Performance overlay quads (PerfOverlay): pc.stars = vertex buffer (4 per quad),
// pc.starCount/lightIndex = target size in pixels.
struct OverlayVertex { vec3 pos; uint rgba; };
layout(buffer_reference, scalar) readonly buffer OverlayVertices { OverlayVertex v[]; };
layout(location = 0) out vec4 vColor;
const int kCorner[6] = int[](0, 1, 2, 0, 2, 3);
void main() {
    uint q = uint(gl_VertexIndex) / 6u;
    OverlayVertex v = OverlayVertices(pc.stars).v[q * 4u + uint(kCorner[gl_VertexIndex % 6])];
    vec2 size = vec2(float(pc.starCount), float(pc.lightIndex));
    gl_Position = vec4(v.pos.xy / size * 2.0 - 1.0, 0.0, 1.0);
    vColor = unpackUnorm4x8(v.rgba); // already display-encoded (UNORM swapchain)
}
