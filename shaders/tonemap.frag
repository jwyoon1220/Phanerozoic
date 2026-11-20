#version 460
#include "common.glsl"
// Pre-exposed HDR → display. Mesopic/scotopic shift (Purkinje) at low adaptation,
// ACES fit, sRGB encode and dithering (night gradients band without it).
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 acesFit(vec3 x) { // Narkowicz 2015
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 srgbEncode(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

void main() {
    vec3 hdr = texture(gTextures[TEX_HDR], vUV).rgb;

    // Rod vision: scotopic luminance (Jensen et al. 2000), blue-shifted and desaturated.
    float adapt = FRAME.tonemap.x; // adaptation luminance, relative to solar units
    float scotopicMix = 1.0 - smoothstep(-8.6, -6.4, log(max(adapt, 1e-12)) / log(10.0));
    if (scotopicMix > 0.0) {
        mat3 rgbToXyz = mat3(0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505);
        vec3 xyz = rgbToXyz * hdr;
        float scot = xyz.y * (1.33 * (1.0 + (xyz.y + xyz.z) / max(xyz.x, 1e-6)) - 1.68);
        vec3 night = max(scot, 0.0) * vec3(0.62, 0.78, 1.05) * 0.9;
        hdr = mix(hdr, night, scotopicMix * 0.75);
    }

    vec3 ldr = srgbEncode(acesFit(hdr));
    float n = hash12(gl_FragCoord.xy + FRAME.tonemap.y * 17.0) + hash12(gl_FragCoord.xy * 1.37 - FRAME.tonemap.y) - 1.0;
    outColor = vec4(ldr + n / 255.0, 1.0);
}
