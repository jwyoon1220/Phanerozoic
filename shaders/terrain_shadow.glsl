// Terrain shadow height maps (TerrainShadows): sampling helpers.
// Height (m) of the map covering camera-relative ECEF point q; false if outside both.
bool shadowMapHeight(vec3 q, out float h) {
    for (int l = 0; l < 2; ++l) {
        float extent = FRAME.shadowOffset[l].w;
        if (extent <= 0.0) continue;
        vec3 local = mat3(FRAME.shadowEnu[l]) * (q - FRAME.shadowOffset[l].xyz);
        vec2 uv = vec2(local.x / extent + 0.5, 0.5 - local.y / extent);
        if (any(lessThan(uv, vec2(0.002))) || any(greaterThan(uv, vec2(0.998)))) continue;
        h = textureLod(gTextures[l == 0 ? TEX_SHADOW_NEAR : TEX_SHADOW_FAR], uv, 0.0).r;
        return true;
    }
    return false;
}

// Cached sun visibility at q (computed by shadowvis.comp); 1 outside the maps.
float shadowVisibility(vec3 q) {
    if (FRAME.shadowParams.x == 0.0) return 1.0;
    for (int l = 0; l < 2; ++l) {
        float extent = FRAME.shadowOffset[l].w;
        if (extent <= 0.0) continue;
        vec3 local = mat3(FRAME.shadowEnu[l]) * (q - FRAME.shadowOffset[l].xyz);
        vec2 uv = vec2(local.x / extent + 0.5, 0.5 - local.y / extent);
        if (any(lessThan(uv, vec2(0.002))) || any(greaterThan(uv, vec2(0.998)))) continue;
        // Cubic B-spline filter from 4 bilinear taps: hides the texel grid that plain
        // bilinear filtering leaves as straight bands along shadow edges.
        uint tex = l == 0 ? TEX_SHADOW_VIS_NEAR : TEX_SHADOW_VIS_FAR;
        vec2 size = vec2(textureSize(gTextures[tex], 0));
        vec2 t = uv * size - 0.5, f = fract(t), i = floor(t);
        vec2 w0 = (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0;
        vec2 w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0;
        vec2 w2 = (1.0 + 3.0 * f + 3.0 * f * f - 3.0 * f * f * f) / 6.0;
        vec2 w3 = f * f * f / 6.0;
        vec2 g0 = w0 + w1, g1 = w2 + w3;
        vec2 h0 = (i - 1.0 + w1 / g0 + 0.5) / size, h1 = (i + 1.0 + w3 / g1 + 0.5) / size;
        float a = textureLod(gTextures[tex], vec2(h0.x, h0.y), 0.0).r, b = textureLod(gTextures[tex], vec2(h1.x, h0.y), 0.0).r;
        float c = textureLod(gTextures[tex], vec2(h0.x, h1.y), 0.0).r, d = textureLod(gTextures[tex], vec2(h1.x, h1.y), 0.0).r;
        return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
    }
    return 1.0;
}
