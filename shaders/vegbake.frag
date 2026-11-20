#version 460
#include "common.glsl"
#include "veg_common.glsl"
#include "normalmap.glsl"
layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vPos;
layout(location = 2) in vec2 vUV;
layout(location = 3) flat in uint vMaterial;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
void main() {
    uint species = vMaterial >> 8, material = vMaterial & 255u;
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 mapN = texture(gTextures[treeTexture(species, material, 1u)], vUV).xyz * 2.0 - 1.0;
    n = perturbNormal(n, vPos, vUV, mapN);
    vec3 albedo = texture(gTextures[treeTexture(species, material, 0u)], vUV).rgb;
    if (max(albedo.r, max(albedo.g, albedo.b)) < 0.035) discard; // leaf JPG surround
    float ao = texture(gTextures[treeTexture(species, material, 2u)], vUV).r;
    outAlbedo = vec4(albedo * ao, 1.0);
    outNormal = vec4(n * 0.5 + 0.5, 1.0);
}
