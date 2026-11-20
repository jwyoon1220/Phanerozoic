#version 460
#include "common.glsl"
layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vSpecies;
void main() {
    if (texture(gTextures[TEX_IMPOSTOR_ALBEDO0 + vSpecies], vUV).a < 0.5) discard;
}
