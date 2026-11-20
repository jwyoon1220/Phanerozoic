// Shared opaque-surface lighting (terrain, vegetation): sun/moon/sky irradiance through
// the atmosphere and aerial perspective. Requires common.glsl + atmosphere.glsl.

struct Surface {
    mat3 toEnu;     // camera ECEF → ENU
    vec3 upP;       // local up at the point (camera ENU)
    float rpKm;     // point radius (atmosphere units)
    float rcKm;     // camera radius
    vec3 viewDir;   // camera → point (ENU)
    float dist;     // metres
};

Surface makeSurface(vec3 rel) {
    Surface s;
    s.toEnu = mat3(FRAME.ecefToEnu);
    vec3 worldPos = FRAME.cameraEcef.xyz + rel;
    float rp = length(worldPos);
    s.upP = s.toEnu * (worldPos / rp);
    s.rpKm = clamp(rp / 1000.0, R_BOTTOM + 0.001, R_TOP - 0.01);
    s.rcKm = viewRadiusKm(FRAME.camera.z);
    s.dist = length(rel);
    s.viewDir = s.toEnu * (rel / max(s.dist, 1e-3));
    return s;
}

vec3 transmittanceBetween(float rc, float muC, float rp, float muP) {
    // Ratio of transmittances to space; for downward rays use the reversed path.
    vec3 t = muC >= 0.0 ? transmittanceToSpace(rc, muC) / max(transmittanceToSpace(rp, muP), vec3(1e-6))
                        : transmittanceToSpace(rp, -muP) / max(transmittanceToSpace(rc, -muC), vec3(1e-6));
    return clamp(t, 0.0, 1.0);
}

// Sunlight reaching the point (atmospheric transmittance only; caller applies shadows).
vec3 sunTransmittance(Surface s) { return transmittanceToSpace(s.rpKm, dot(s.upP, FRAME.sunDir.xyz)); }

// Hemispherical sky irradiance (π × mean radiance; zenith + four near-horizon samples).
vec3 skyIrradiance(Surface s) {
    vec3 sun = FRAME.sunDir.xyz, moon = FRAME.moonDir.xyz;
    vec3 hx = normalize(cross(s.upP, vec3(0.0, 0.0, 1.0) + vec3(1e-4, 0.0, 0.0)));
    vec3 hy = cross(s.upP, hx);
    vec3 dirs[5] = vec3[](s.upP, normalize(hx + s.upP * 0.15), normalize(-hx + s.upP * 0.15), normalize(hy + s.upP * 0.15),
                          normalize(-hy + s.upP * 0.15));
    vec3 sky = vec3(0.0);
    for (int k = 0; k < 5; ++k)
        sky += (skyLUT(TEX_SKYVIEW_SUN, dirs[k], sun, s.rcKm) + skyLUT(TEX_SKYVIEW_MOON, dirs[k], moon, s.rcKm) * FRAME.moonParams.x) * 0.2;
    return sky * PI + vec3(1.3e-9 * PI * 2.0); // + natural night-sky background
}

// Total irradiance on a surface with normal n (ENU); sunVis = shadow visibility.
vec3 irradiance(Surface s, vec3 n, vec3 sunT, float sunVis, float skyVis) {
    vec3 moonT = transmittanceToSpace(s.rpKm, dot(s.upP, FRAME.moonDir.xyz));
    vec3 sky = skyIrradiance(s);
    // Light reflected by the surrounding ground onto surfaces facing down/sideways
    // (strong over bright sand and snow, weak over forest).
    vec3 groundE = sunT * max(dot(s.upP, FRAME.sunDir.xyz), 0.0) + sky;
    vec3 bounce = FRAME.ground.rgb * groundE * (0.5 - 0.5 * dot(n, s.upP));
    return sunT * sunVis * max(dot(n, FRAME.sunDir.xyz), 0.0) + moonT * FRAME.moonParams.x * max(dot(n, FRAME.moonDir.xyz), 0.0)
         + (sky + bounce) * skyVis;
}

// Aerial perspective between camera and point. The sky-view LUT holds the in-scatter of
// the whole ray (to the ground sphere or to space); scale it by the fraction of that
// ray's optical depth in front of the point.
vec3 applyAerialPerspective(Surface s, vec3 L) {
    vec3 sun = FRAME.sunDir.xyz, moon = FRAME.moonDir.xyz;
    float muC = s.viewDir.z;
    float muP = dot(s.upP, s.viewDir);
    vec3 T = transmittanceBetween(s.rcKm, muC, s.rpKm, muP);
    vec3 ro = vec3(0.0, 0.0, s.rcKm);
    float tG = raySphere(ro, s.viewDir, R_BOTTOM);
    vec3 Tfull = tG > 0.0 ? transmittanceBetween(s.rcKm, muC, R_BOTTOM + 0.001, dot(normalize(ro + s.viewDir * tG), s.viewDir))
                          : transmittanceToSpace(s.rcKm, muC);
    vec3 inscatterFull = skyLUT(TEX_SKYVIEW_SUN, s.viewDir, sun, s.rcKm) + skyLUT(TEX_SKYVIEW_MOON, s.viewDir, moon, s.rcKm) * FRAME.moonParams.x;
    return L * T + inscatterFull * clamp((1.0 - T) / max(1.0 - Tfull, vec3(1e-4)), 0.0, 1.0);
}
