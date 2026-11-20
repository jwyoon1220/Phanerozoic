#version 460
#include "common.glsl"
#include "veg_common.glsl"
// Shadow caster: tree impostor quad facing the sun (silhouette as seen from the light).
layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vSpecies;
void main() {
    VisibleTree inst = VisibleTrees(pc.stars).v[gl_InstanceIndex];
    vec2 corner = TreeVertices(pc.aux).v[gl_VertexIndex].pos.xy;
    uint species = vegSpecies(inst.packed);
    float scale = vegScale(inst.packed);
    vec4 sp = speciesA(species);
    vec3 X, Y, Z;
    treeFrame(inst.rel, inst.packed, X, Y, Z);
    vec3 centre = inst.rel + Y * (sp.x * scale * 0.5);
    vec3 sunEcef = transpose(mat3(FRAME.ecefToEnu)) * FRAME.sunDir.xyz;
    uvec2 f = nearestFrame(vec3(dot(sunEcef, X), dot(sunEcef, Y), dot(sunEcef, Z)));
    vec3 v = frameDirection(f.y * IMPOSTOR_FRAMES + f.x), right, upCam;
    frameBasis(v, right, upCam);
    vec3 rW = X * right.x + Y * right.y + Z * right.z;
    vec3 uW = X * upCam.x + Y * upCam.y + Z * upCam.z;
    vec3 pos = centre + (rW * corner.x + uW * corner.y) * (sp.z * scale);
    vUV = (vec2(f) + vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5)) / float(IMPOSTOR_FRAMES);
    vSpecies = species;
    gl_Position = FRAME.csm[pc.lightIndex] * vec4(pos, 1.0);
}
