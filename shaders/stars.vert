#version 460
#include "common.glsl"
#include "atmosphere.glsl"
// One instance per star (or planet): screen-aligned quad carrying a Gaussian PSF.
// Directions are rotated ICRS → ENU with a single matrix; refraction, extinction
// and scintillation are applied here.
layout(location = 0) out vec2 vOffsetPx;
layout(location = 1) out vec3 vRadiance; // pre-exposed, already divided by PSF normalisation
layout(location = 2) out float vSigma;

const vec2 CORNERS[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

float refractionRad(float alt) { // Bennett, alt in radians
    float h = degrees(max(alt, radians(-1.9)));
    return max(radians(1.0 / tan(radians(h + 7.31 / (h + 4.4))) / 60.0), 0.0);
}

void main() {
    GpuStar s = StarRef(pc.stars).stars[gl_InstanceIndex];
    vec3 dir = normalize(mat3(FRAME.icrsToEnu) * s.dir);

    float alt = asin(clamp(dir.z, -1.0, 1.0));
    float viewHeight = viewRadiusKm(FRAME.camera.z);
    vec3 T = transmittanceToSpace(viewHeight, dir.z);
    // Refraction lifts the star (apparent altitude); keep the azimuth.
    float app = alt + refractionRad(alt);
    vec2 hz = normalize(dir.xy + 1e-9);
    dir = vec3(hz * cos(app), sin(app));

    bool hidden = dot(T, vec3(1.0)) < 1e-6 ||
                  dot(dir, FRAME.moonDir.xyz) > cos(FRAME.moonDir.w) || dot(dir, FRAME.sunDir.xyz) > cos(FRAME.sunDir.w);

    // Scintillation grows with airmass; subtle at the zenith.
    float airmass = 1.0 / max(dir.z + 0.15 * pow(max(degrees(app) + 3.885, 0.1), -1.253), 0.03);
    float amp = 0.08 + 0.12 * clamp((airmass - 1.0) / 4.0, 0.0, 1.0);
    float t = FRAME.camera.y * (9.0 + 6.0 * hash12(vec2(gl_InstanceIndex, 1.7)));
    float twinkle = 1.0 + amp * (valueNoise3(vec3(t, gl_InstanceIndex * 0.37, 0.0)) * 2.0 - 1.0);

    float pixelAngle = FRAME.camera.w;
    float pixelSolidAngle = pixelAngle * pixelAngle;
    vec3 radiance = s.color * s.flux * T * twinkle / pixelSolidAngle * FRAME.camera.x; // pre-exposed, all energy in one pixel
    float peak = max(radiance.r, max(radiance.g, radiance.b));
    // PSF: ~0.7 px seeing core, widening for very bright sources (perceived glare).
    float sigma = 0.7 + 0.22 * max(log2(max(peak, 1e-6)) - 1.0, 0.0);
    sigma = min(sigma, 6.0);
    float halfSize = ceil(3.0 * sigma) + 0.5;

    vec4 clip = FRAME.viewProj * vec4(dir, 1.0);
    vec2 corner = CORNERS[gl_VertexIndex];
    vOffsetPx = corner * halfSize;
    clip.xy += corner * halfSize * 2.0 * FRAME.resolution.zw * clip.w;
    vRadiance = radiance / (2.0 * PI * sigma * sigma); // normalise Gaussian over pixels
    vSigma = sigma;
    // Cull: behind the camera, invisible, or below a useful threshold.
    if (hidden || clip.w <= 0.0 || peak < 1e-4) clip = vec4(2.0, 2.0, 2.0, 1.0);
    clip.z = 0.0; // at infinity: depth-tested EQUAL against the cleared far value (terrain occludes)
    gl_Position = clip;
}
