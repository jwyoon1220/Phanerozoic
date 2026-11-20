#version 460
layout(location = 0) in vec2 vOffsetPx;
layout(location = 1) in vec3 vRadiance;
layout(location = 2) in float vSigma;
layout(location = 0) out vec4 outColor;
void main() {
    float w = exp(-dot(vOffsetPx, vOffsetPx) / (2.0 * vSigma * vSigma));
    outColor = vec4(min(vRadiance * w, vec3(60000.0)), 0.0);
}
