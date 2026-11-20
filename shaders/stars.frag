#version 460
#include "common.glsl"
layout(location = 0) in vec2 vOffsetPx;
layout(location = 1) in vec3 vRadiance;
layout(location = 2) in float vSigma;
layout(location = 0) out vec4 outColor;
void main() {
    // Visible only where the scene depth (render resolution, reverse-Z) is still the sky.
    ivec2 p = ivec2(gl_FragCoord.xy * FRAME.display.zw * FRAME.resolution.xy);
    if (texelFetch(gTextures[TEX_DEPTH], min(p, ivec2(FRAME.resolution.xy) - 1), 0).r > 0.0) discard;
    float w = exp(-dot(vOffsetPx, vOffsetPx) / (2.0 * vSigma * vSigma));
    outColor = vec4(min(vRadiance * w, vec3(60000.0)), 0.0);
}
