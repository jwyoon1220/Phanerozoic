#version 460
#include "common.glsl"
#include "atmosphere.glsl"
#include "terrain_shadow.glsl"
#include "lighting.glsl"
#include "csm.glsl"
#include "veg_common.glsl"
layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vCentre;
layout(location = 2) flat in uint vPacked;
layout(location = 3) in vec3 vX;
layout(location = 4) in vec3 vY;
layout(location = 5) in vec3 vZ;
layout(location = 0) out vec4 outColor;

void main() {
    uint species = vegSpecies(vPacked);
    vec4 a = texture(gTextures[TEX_IMPOSTOR_ALBEDO0 + species], vUV, FRAME.upscale.x);
    // Stochastic fade beyond ~3.6 km into the canopy-coloured terrain (no hard ring).
    float fade = 1.0 - smoothstep(3500.0, 4100.0, length(vCentre));
    float noise = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    if (a.a < 0.5 || noise > fade) discard;
    vec3 nl = texture(gTextures[TEX_IMPOSTOR_NORMAL0 + species], vUV, FRAME.upscale.x).xyz * 2.0 - 1.0;
    Surface s = makeSurface(vCentre);
    vec3 n = normalize(s.toEnu * normalize(vX * nl.x + vY * nl.y + vZ * nl.z));
    // Foliage pixels are the greener ones in the bake (bark is brown/grey).
    vec3 albedo = a.rgb;
    if (albedo.g > albedo.r * 1.05) albedo = seasonalFoliage(albedo, species, vegTint(vPacked));
    vec3 sunT = sunTransmittance(s);
    float vis = shadowVisibility(vCentre) * csmVisibility(vCentre, normalize(vY), s.dist);
    vec3 E = irradiance(s, n, sunT, vis, 0.55 + 0.45 * dot(n, s.upP));
    vec3 L = albedo / PI * E;
    outColor = vec4(min(applyAerialPerspective(s, L) * FRAME.camera.x, vec3(FP16_SAFE_MAX)), 1.0);
}
