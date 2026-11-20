// Cascaded sun shadow lookup (CascadedShadows): 4×4 PCF from four textureGathers.
float csmVisibility(vec3 rel, vec3 nEcef, float dist) {
    if (FRAME.csmSplits.w == 0.0 || dist >= FRAME.csmSplits.z) return 1.0;
    int c = dist < FRAME.csmSplits.x ? 0 : (dist < FRAME.csmSplits.y ? 1 : 2);
    float texel = FRAME.csmTexel[c];
    vec3 p = rel + nEcef * (texel * 1.5); // normal offset against acne
    vec4 clip = FRAME.csm[c] * vec4(p, 1.0);
    vec2 uv = clip.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    float z = clip.z - 0.0004 * float(c + 1);
    float size = FRAME.csmTexel.w;
    vec2 base = (floor(uv * size - 1.5) + 1.0) / size; // gather centres of the 4×4 footprint
    float lit = 0.0;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            vec4 d = textureGather(gTextures[TEX_CASCADE0 + c], base + vec2(i * 2, j * 2) / size, 0);
            lit += dot(step(vec4(z), d), vec4(0.25));
        }
    lit *= 0.25;
    // Fade out the last cascade so the hand-over to the canopy approximation is soft.
    return mix(lit, 1.0, smoothstep(FRAME.csmSplits.z * 0.85, FRAME.csmSplits.z, dist));
}
