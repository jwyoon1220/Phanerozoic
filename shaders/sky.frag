#version 460
#include "common.glsl"
#include "atmosphere.glsl"
// Sky background: atmosphere (sun + moon sky-view LUTs), sun & moon disks, airglow
// and the integrated Milky Way. Output is pre-exposed HDR.
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// ICRS → galactic (Hipparcos definition). Written row by row, so the GLSL matrix is
// the transpose: use `v * ICRS_TO_GAL_ROWS` to apply it.
const mat3 ICRS_TO_GAL_ROWS = mat3(
    -0.0548755604, -0.8734370902, -0.4838350155,
     0.4941094279, -0.4448296300,  0.7469822445,
    -0.8676661490, -0.1980763734,  0.4559837762);

// Natural night-sky background ≈ 22 mag/arcsec² → 1.3e-9 of solar irradiance per sr.
const float NIGHT_SKY = 1.3e-9;

// Placeholder Milky Way until the Gaia-derived HDR map (plan §2.6): galactic band
// with central bulge, Great Rift dust lanes and noise structure.
float milkyWay(vec3 icrs) {
    vec3 g = icrs * ICRS_TO_GAL_ROWS;
    float b = asin(clamp(g.z, -1.0, 1.0));
    float l = atan(g.y, g.x); // 0 = galactic centre (Sagittarius)
    float band = exp(-pow(b / radians(7.0), 2.0));
    float along = 0.35 + 0.65 * exp(-pow(l / radians(70.0), 2.0));
    float bulge = 1.6 * exp(-(l * l + b * b) / pow(radians(11.0), 2.0));
    float clouds = 0.55 + 0.9 * fbm3(icrs * 9.0);
    float rift = 1.0 - 0.75 * exp(-pow((b - radians(1.5)) / radians(2.2), 2.0)) * smoothstep(radians(-40.0), radians(-5.0), l) *
                 (1.0 - smoothstep(radians(40.0), radians(70.0), l));
    float dust = mix(1.0, 0.45 + 0.55 * fbm3(icrs * 23.0 + 7.0), exp(-pow(b / radians(4.0), 2.0)));
    return (band * along * clouds + bulge) * rift * dust;
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 wp = FRAME.invViewProj * vec4(ndc, 0.5, 1.0);
    vec3 dir = normalize(wp.xyz / wp.w);

    float viewHeight = viewRadiusKm(FRAME.camera.z);
    vec3 sun = FRAME.sunDir.xyz, moon = FRAME.moonDir.xyz;

    vec3 L;
    vec3 tView;
    float rCam = R_BOTTOM + FRAME.camera.z;
    if (rCam < R_TOP - 0.02) {
        L = skyLUT(TEX_SKYVIEW_SUN, dir, sun, viewHeight) + skyLUT(TEX_SKYVIEW_MOON, dir, moon, viewHeight) * FRAME.moonParams.x;
        tView = transmittanceToSpace(viewHeight, dir.z);
    } else {
        // Above the atmosphere: rays that miss it see black space; rays that enter it
        // are looked up from the entry point (approximation, plan: dedicated space view).
        vec3 ro = vec3(0.0, 0.0, rCam);
        float t = raySphere(ro, dir, R_TOP);
        if (t < 0.0) {
            L = vec3(0.0);
            tView = vec3(1.0);
        } else {
            vec3 entry = ro + dir * t;
            vec3 local = vec3(dir.xy, dot(dir, normalize(entry)));
            L = skyLUT(TEX_SKYVIEW_SUN, local, sun, R_TOP - 0.02) + skyLUT(TEX_SKYVIEW_MOON, local, moon, R_TOP - 0.02) * FRAME.moonParams.x;
            tView = transmittanceToSpace(R_TOP - 0.02, local.z);
        }
    }
    bool ground = dot(tView, vec3(1.0)) == 0.0;

    if (!ground) {
        // Airglow (emission layer at ~90 km: van Rhijn brightening towards the horizon).
        float sinZ2 = 1.0 - dir.z * dir.z;
        float k = R_BOTTOM / (R_BOTTOM + 90.0);
        float vanRhijn = inversesqrt(max(1.0 - k * k * sinZ2, 1e-3));
        vec3 airglow = NIGHT_SKY * 0.6 * vanRhijn * vec3(0.85, 1.0, 0.75);

        vec3 icrs = transpose(mat3(FRAME.icrsToEnu)) * dir;
        vec3 galaxy = NIGHT_SKY * 3.0 * milkyWay(icrs) * vec3(1.0, 0.96, 0.9);
        vec3 background = airglow + galaxy;

        // Sun disk with limb darkening (u = 0.6).
        float sunCos = dot(dir, sun);
        float sunR = FRAME.sunDir.w;
        float ang = acos(clamp(sunCos, -1.0, 1.0));
        if (ang < sunR) {
            float x = ang / sunR;
            float limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - x * x));
            background += vec3(1.0) / (PI * sunR * sunR) * limb / (1.0 - 0.6 / 3.0);
        }

        // Moon: Lommel–Seeliger shading, procedural maria, earthshine. Occludes what is behind.
        float moonR = FRAME.moonDir.w;
        float mAng = acos(clamp(dot(dir, moon), -1.0, 1.0));
        if (mAng < moonR) {
            vec3 right = normalize(cross(moon, vec3(0, 0, 1)));
            vec3 upv = cross(right, moon);
            vec2 q = vec2(dot(dir, right), dot(dir, upv)) / sin(moonR);
            float z = sqrt(max(1.0 - dot(q, q), 0.0));
            vec3 n = q.x * right + q.y * upv - z * moon;
            float mu0 = dot(n, sun), mu = z;
            // Selenographic-ish coordinates are fixed to the disk (libration ignored).
            vec3 local = vec3(q, z);
            float maria = smoothstep(0.45, 0.62, fbm3(local * 2.2 + 3.1));
            float albedo = mix(0.15, 0.075, maria) * (0.85 + 0.3 * fbm3(local * 14.0));
            float lit = mu0 > 0.0 ? 2.0 * mu0 / (mu0 + mu) : 0.0;
            vec3 sunlit = vec3(albedo / PI * lit); // sunlight reaches the Moon unattenuated
            vec3 earthshine = vec3(albedo / PI * FRAME.moonParams.z) * vec3(0.8, 0.9, 1.0);
            // Local adaptation: the eye resolves lunar detail although the disk is ~10^5×
            // brighter than the moonlit sky. Map a fully lit highland to ~1.6 exposed units
            // (never brightening it beyond its physical value, e.g. in daylight).
            float localScale = min(1.0, 1.6 / (0.15 / PI * FRAME.camera.x));
            // Earthshine is boosted after the compression so the ashen light stays visible
            // on a crescent, as it is to the dark-adapted eye.
            background = sunlit * vec3(1.0, 0.97, 0.92) * localScale + earthshine * min(1.0, localScale * 400.0);
        }
        L += background * tView;
    } else {
        // Placeholder ground until terrain (M3): dark soil lit by sun, moon and sky.
        float sunUp = max(sun.z, 0.0), moonUp = max(moon.z, 0.0);
        vec3 tSun = transmittanceToSpace(viewHeight, sun.z);
        vec3 tMoon = transmittanceToSpace(viewHeight, moon.z);
        vec3 E = tSun * sunUp + tMoon * moonUp * FRAME.moonParams.x + NIGHT_SKY * PI * 2.0;
        L += vec3(0.06, 0.05, 0.035) / PI * E * (1.0 - 0.6 * exp(-dir.z * dir.z * 400.0));
    }

    outColor = vec4(min(L * FRAME.camera.x, vec3(FP16_SAFE_MAX)), 1.0);
}
