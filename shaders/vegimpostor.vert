#version 460
#include "common.glsl"
#include "veg_common.glsl"
// Octahedral impostor: a quad perpendicular to the nearest baked view direction.
layout(location = 0) out vec2 vUV;          // atlas UV
layout(location = 1) out vec3 vCentre;      // camera-relative ECEF
layout(location = 2) flat out uint vPacked;
layout(location = 3) out vec3 vX;
layout(location = 4) out vec3 vY;
layout(location = 5) out vec3 vZ;

void main() {
    VisibleTree inst = VisibleTrees(pc.stars).v[gl_InstanceIndex];
    vec2 corner = TreeVertices(pc.aux).v[gl_VertexIndex].pos.xy;
    uint species = vegSpecies(inst.packed);
    float scale = vegScale(inst.packed);
    vec4 sp = speciesA(species);
    vec3 X, Y, Z;
    treeFrame(inst.rel, inst.packed, X, Y, Z);
    float h = sp.x * scale, ext = sp.z * scale;
    vec3 centre = inst.rel + Y * (h * 0.5);
    vec3 toCam = normalize(-centre);
    vec3 local = vec3(dot(toCam, X), max(dot(toCam, Y), 0.0), dot(toCam, Z));
    vec2 e = hemiOctEncode(normalize(local));
    uvec2 f = uvec2(clamp(floor((e * 0.5 + 0.5) * float(IMPOSTOR_FRAMES)), 0.0, float(IMPOSTOR_FRAMES - 1u)));
    uint frame = f.y * IMPOSTOR_FRAMES + f.x;
    vec3 v = frameDirection(frame), right, upCam;
    frameBasis(v, right, upCam);
    vec3 rW = X * right.x + Y * right.y + Z * right.z;
    vec3 uW = X * upCam.x + Y * upCam.y + Z * upCam.z;
    vec3 pos = centre + (rW * corner.x + uW * corner.y) * ext;
    vec2 uvFrame = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    vUV = (vec2(f) + uvFrame) / float(IMPOSTOR_FRAMES);
    vCentre = centre;
    vPacked = inst.packed;
    vX = X, vY = Y, vZ = Z;
    gl_Position = FRAME.viewProjEcef * vec4(pos, 1.0);
}
