#version 460
#include "common.glsl"
layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vTexture;
void main() {
    // Leaf JPGs have no alpha: cut the black surround as in vegmesh.frag.
    if (vTexture != 0u) {
        // Mip 0: coarse mips average leaf and black surround, defeating the threshold.
        vec3 a = textureLod(gTextures[nonuniformEXT(vTexture)], vUV, 0.0).rgb;
        if (max(a.r, max(a.g, a.b)) < 0.035) discard;
    }
}
