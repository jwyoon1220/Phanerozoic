#version 460
#include "common.glsl"
#include "atmosphere.glsl"
#include "terrain_shadow.glsl"
#include "lighting.glsl"
#include "csm.glsl"
#include "veg_common.glsl"
#include "normalmap.glsl"
// Textured PBR trees (Poly Haven materials): albedo, tangent-space normal, ARM.
layout(location = 0) in vec3 vRel;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) flat in uint vMaterial;
layout(location = 4) flat in float vTint;
layout(location = 0) out vec4 outColor;

void main() {
    uint species = (vMaterial >> 8) & 255u, material = vMaterial & 255u;
    bool foliage = (vMaterial & 0x10000u) != 0u;
    vec3 albedo = texture(gTextures[treeTexture(species, material, 0u)], vUV, FRAME.upscale.x).rgb;
    vec3 mapN = texture(gTextures[treeTexture(species, material, 1u)], vUV, FRAME.upscale.x).xyz * 2.0 - 1.0;
    vec3 arm = texture(gTextures[treeTexture(species, material, 2u)], vUV, FRAME.upscale.x).rgb;
    // Leaf textures are JPGs (no alpha): their black surround shows as dark rims where
    // the leaf UVs overshoot the leaf shape. Cut those texels.
    if (foliage && max(albedo.r, max(albedo.g, albedo.b)) < 0.035) discard;
    if (foliage) albedo = seasonalFoliage(albedo, species, vTint);
    // Rocks take the local rock colour: tint = desert weight → warm sandstone hue,
    // lifted towards the terrain's rock brightness so rocks do not read as black holes.
    if (species == 4u || species == 5u) {
        // Keep the texture's relative detail, re-based on the terrain rock's mean colour
        // (lichen granite ≈ 0.07 in humid lands, sandstone ≈ 0.3/0.2/0.13 in deserts).
        float lum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
        float detailRel = clamp(lum / 0.045, 0.3, 2.5);
        vec3 target = mix(vec3(0.075, 0.068, 0.058), vec3(0.30, 0.20, 0.13), vTint);
        albedo = mix(albedo, target * detailRel, 0.85);
    }

    Surface s = makeSurface(vRel);
    vec3 nGeo = normalize(vNormal);
    if (!gl_FrontFacing) nGeo = -nGeo;
    vec3 nEcef = perturbNormal(nGeo, vRel, vUV, mapN);
    vec3 n = normalize(s.toEnu * nEcef);
    vec3 sun = FRAME.sunDir.xyz;
    vec3 sunT = sunTransmittance(s);
    float vis = shadowVisibility(vRel) * csmVisibility(vRel, nGeo, s.dist);
    float ao = arm.r;
    vec3 E = irradiance(s, n, sunT * ao, vis, (0.55 + 0.45 * dot(n, s.upP)) * ao);
    vec3 L = albedo / PI * E;
    // Specular sun highlight (GGX-ish, roughness from ARM.g): glossy evergreen leaves.
    float rough = max(arm.g, 0.15);
    vec3 h = normalize(sun - s.viewDir);
    float a2 = rough * rough * rough * rough;
    float nh = max(dot(n, h), 0.0);
    float dGGX = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    L += sunT * vis * max(dot(n, sun), 0.0) * dGGX * 0.04 * 0.25;
    if (foliage) L += albedo * sunT * vis * max(-dot(n, sun), 0.0) * 0.15; // light through leaves
    outColor = vec4(min(applyAerialPerspective(s, L) * FRAME.camera.x, vec3(FP16_SAFE_MAX)), 1.0);
}
