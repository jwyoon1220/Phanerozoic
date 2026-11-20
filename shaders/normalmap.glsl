// Fragment-stage only (uses derivatives).
// Normal mapping without stored tangents: cotangent frame from screen-space derivatives
// (Schüler 2013). p, n in any consistent space; returns the perturbed normal there.
vec3 perturbNormal(vec3 n, vec3 p, vec2 uv, vec3 mapN) {
    vec3 dp1 = dFdx(p), dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv), duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, n), dp1perp = cross(n, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-20));
    return normalize(mat3(T * invmax, B * invmax, n) * mapN);
}
