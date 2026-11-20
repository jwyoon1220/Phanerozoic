#version 460
#include "common.glsl"
#include "atmosphere.glsl"
#include "terrain_shadow.glsl"
#include "lighting.glsl"
#include "csm.glsl"

const uint TEX_GROUND0 = 512u;

// Value noise is lattice-aligned: thresholded, its contours run along the grid axes and
// read as straight seams. Rotate the domain between octaves to break the alignment.
const mat3 NOISE_ROT = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
float organicNoise(vec3 p) {
    float n = valueNoise3(p) * 0.5;
    p = NOISE_ROT * p * 2.03 + 3.7;
    n += valueNoise3(p) * 0.3;
    p = NOISE_ROT * p * 2.01 + 1.3;
    n += valueNoise3(p) * 0.2;
    return n;
}

// Triplanar sampling around world position p (metres, small numbers: cameraMod + rel)
// with whiteout normal blending (Golus 2017). nW: geometric normal (ECEF).
struct Ground { vec3 albedo; vec3 normal; float ao; };
// Explicit gradients: these samples sit in divergent branches (rock/soil masks), where
// implicit derivatives are undefined and pick wrong mips (moiré on distant rock).
Ground sampleGround(uint m, vec3 p, vec3 dpx, vec3 dpy, vec3 nW, float scale, bool full) {
    vec3 w = pow(abs(nW), vec3(4.0));
    // Planes under 3 % weight are skipped (renormalised): usually 1–2 of 3 remain.
    w *= step(vec3(0.03), w / (w.x + w.y + w.z));
    w /= w.x + w.y + w.z;
    uint a = TEX_GROUND0 + m * 3u;
    Ground g;
    g.albedo = vec3(0.0);
    g.ao = 0.0;
    vec3 nsum = vec3(0.0);
    for (int k = 0; k < 3; ++k) {
        if (w[k] == 0.0) continue;
        vec2 uv = (k == 0 ? p.zy : (k == 1 ? p.xz : p.xy)) / scale;
        vec2 gx = (k == 0 ? dpx.zy : (k == 1 ? dpx.xz : dpx.xy)) / scale;
        vec2 gy = (k == 0 ? dpy.zy : (k == 1 ? dpy.xz : dpy.xy)) / scale;
        g.albedo += textureGrad(gTextures[a], uv, gx, gy).rgb * w[k];
        g.ao += textureGrad(gTextures[a + 2u], uv, gx, gy).r * w[k];
        if (full) {
            vec3 t = textureGrad(gTextures[a + 1u], uv, gx, gy).xyz * 2.0 - 1.0;
            // Whiteout blend per plane, swizzled back to world axes.
            if (k == 0) { t = vec3(t.xy + nW.zy, abs(t.z) * nW.x); nsum += t.zyx * w[k]; }
            else if (k == 1) { t = vec3(t.xy + nW.xz, abs(t.z) * nW.y); nsum += t.xzy * w[k]; }
            else { t = vec3(t.xy + nW.xy, abs(t.z) * nW.z); nsum += t * w[k]; }
        }
    }
    g.normal = full ? normalize(nsum) : nW;
    return g;
}
// Weighted accumulation of ground materials (skips negligible weights).
void addGround(inout Ground acc, inout float wsum, uint m, float w, vec3 p, vec3 dpx, vec3 dpy, vec3 nE, float scale) {
    if (w < 0.02) return;
    Ground g = sampleGround(m, p, dpx, dpy, nE, scale, w > 0.25); // minor layers: albedo/AO only
    acc.albedo += g.albedo * w;
    acc.normal += g.normal * w;
    acc.ao += g.ao * w;
    wsum += w;
}
vec3 dryTint(vec3 albedo, float dry) { // green → straw with aridity
    float lum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    return mix(albedo, vec3(1.6, 1.3, 0.75) * lum, dry * 0.8);
}

// Terrain shading: biome materials (per-vertex hints from the CPU height/climate model),
// water with Fresnel sky reflection, shared lighting.
layout(location = 0) in vec3 vRel;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in float vHeight;
layout(location = 3) flat in float vLevel;
layout(location = 4) in vec4 vHintA; // sand, cliff, talus, desert
layout(location = 5) in vec4 vHintB; // forest, conifer, snow, dryness
layout(location = 0) out vec4 outColor;

void main() {
    Surface s = makeSurface(vRel);
    vec3 n = normalize(s.toEnu * normalize(vNormal));
    vec3 sun = FRAME.sunDir.xyz, moon = FRAME.moonDir.xyz;
    float moonE = FRAME.moonParams.x;

    vec3 sunT = sunTransmittance(s);
    float muSun = dot(s.upP, sun);
    float slope = 1.0 - dot(n, s.upP);
    // World-locked coordinates for noise/textures: precise (camera mod 65 km + relative),
    // unlike cameraEcef + rel whose float rounding (0.5 m at Earth radius) makes grid seams.
    vec3 worldPos = FRAME.cameraMod.xyz + vRel;
    // Biome / feature weights (sum to 1 over land): rock (slopes, scarps), snow, sand,
    // then forest floor, stony desert/scree, grass.
    float rockW = max(smoothstep(0.29, 0.42, slope), smoothstep(0.15, 0.55, vHintA.y));
    float snowW = vHintB.z * (1.0 - smoothstep(0.35, 0.5, slope));
    float sandNoise = smoothstep(0.35, 0.7, organicNoise(NOISE_ROT * worldPos / 35.0));
    float sandW = clamp(vHintA.x + vHintA.w * 0.4 * sandNoise, 0.0, 1.0) * (1.0 - rockW);
    float rest = (1.0 - rockW) * (1.0 - snowW) * (1.0 - sandW);
    // Boreal floors are as much moss/grass as needles: part of the conifer forest
    // weight goes to the grass material.
    float forestW = rest * vHintB.x * (1.0 - 0.45 * vHintB.y);
    float stonyW = rest * (1.0 - vHintB.x) * max(vHintA.w, vHintA.z);
    float grassW = max(rest - forestW - stonyW, 0.0);
    rockW *= 1.0 - snowW;
    sandW *= 1.0 - snowW;
    vec3 dpx = dFdx(worldPos), dpy = dFdy(worldPos); // uniform control flow

    // Near field: Poly Haven PBR ground materials (triplanar, world-locked).
    float detail = 1.0 - smoothstep(800.0, 1400.0, s.dist);
    vec3 nEcef = normalize(vNormal);
    vec3 albedoDetail = vec3(0.0);
    float aoDetail = 1.0;
    if (detail > 0.0 && vHeight >= 0.0) {
        vec3 p = worldPos;
        float soilW = smoothstep(0.3, 0.7, organicNoise(NOISE_ROT * worldPos / 24.0));
        Ground g = Ground(vec3(0.0), vec3(0.0), 0.0);
        float wsum = 0.0;
        // Forest floor: broadleaf litter or conifer needles, soil patches between.
        addGround(g, wsum, 0u, forestW * (1.0 - vHintB.y) * (1.0 - soilW * 0.6), p, dpx, dpy, nEcef, 2.0);
        addGround(g, wsum, 8u, forestW * vHintB.y * (1.0 - soilW * 0.6), p, dpx, dpy, nEcef, 2.0);
        addGround(g, wsum, 1u, forestW * soilW * 0.6 + grassW * soilW * 0.25 * vHintB.w, p, dpx, dpy, nEcef, 4.0);
        if (grassW > 0.02) {
            Ground gr = sampleGround(6u, p, dpx, dpy, nEcef, 2.5, true);
            float w = grassW * (1.0 - soilW * 0.25 * vHintB.w);
            g.albedo += dryTint(gr.albedo, vHintB.w) * w;
            g.normal += gr.normal * w;
            g.ao += gr.ao * w;
            wsum += w;
        }
        addGround(g, wsum, 5u, stonyW, p, dpx, dpy, nEcef, 3.0);
        addGround(g, wsum, 4u, sandW, p, dpx, dpy, nEcef, 4.0);
        addGround(g, wsum, 7u, snowW, p, dpx, dpy, nEcef, 4.0);
        if (rockW > 0.02) {
            // Lichen-covered rock on slopes, bare rock on scarps (warmer hue in deserts).
            float cliffW = max(smoothstep(0.45, 0.62, slope), vHintA.y);
            Ground rock = sampleGround(2u, p, dpx, dpy, nEcef, 8.0, true);
            // Second, rotated and larger tiling breaks the repetition seen from afar.
            Ground rock2 = sampleGround(2u, NOISE_ROT * p + 311.0, NOISE_ROT * dpx, NOISE_ROT * dpy, nEcef, 37.0, false);
            float macro = smoothstep(0.25, 0.75, organicNoise(NOISE_ROT * worldPos / 60.0));
            rock.albedo = mix(rock.albedo, rock2.albedo, 0.35 + 0.3 * macro);
            rock.ao = mix(rock.ao, rock2.ao, 0.5);
            if (cliffW > 0.01) {
                Ground cliff = sampleGround(3u, p, dpx, dpy, nEcef, 16.0, true);
                rock.albedo = mix(rock.albedo, cliff.albedo, cliffW);
                rock.normal = normalize(mix(rock.normal, cliff.normal, cliffW));
                rock.ao = mix(rock.ao, cliff.ao, cliffW);
            }
            rock.albedo *= mix(vec3(1.0), vec3(1.45, 1.05, 0.72), vHintA.w);
            g.albedo += rock.albedo * rockW;
            g.normal += rock.normal * rockW;
            g.ao += rock.ao * rockW;
            wsum += rockW;
        }
        g.albedo /= max(wsum, 1e-3);
        g.normal = normalize(g.normal + nEcef * 1e-3);
        g.ao = wsum > 1e-3 ? g.ao / wsum : 1.0;
        albedoDetail = g.albedo;
        aoDetail = mix(1.0, g.ao, detail);
        // Texture normals only where their texels are resolved; beyond, their mip-averaged
        // bumps read as moiré on rock.
        float normalDetail = detail * (1.0 - 0.8 * smoothstep(150.0, 700.0, s.dist));
        n = normalize(mix(n, normalize(s.toEnu * g.normal), normalDetail));
    }

    // Under the forest (where trees stand: not rock, within the 4 km tree range) the
    // canopy shades the ground — dappled sunlight and reduced sky until tree shadow maps.
    float forestMask = forestW * (1.0 - smoothstep(3500.0, 4100.0, s.dist)) * step(2.0, vHeight);
    float dapple = smoothstep(0.3, 0.75, organicNoise(NOISE_ROT * worldPos / 6.0));
    // Real tree shadows (CSM) out to 700 m; the dappled approximation only beyond.
    float beyondCsm = smoothstep(FRAME.csmSplits.z * 0.85, FRAME.csmSplits.z, s.dist);
    float canopySun = mix(1.0, 0.12 + 0.6 * dapple, forestMask * 0.85 * beyondCsm);
    float canopySky = mix(1.0, 0.55, forestMask);

    float sunVis = (muSun > -0.05 && vHeight >= 0.0 && dot(n, sun) > 0.0) ? shadowVisibility(vRel) * csmVisibility(vRel, nEcef, s.dist) : 1.0;
    vec3 E = irradiance(s, n, sunT * canopySun, sunVis, (0.5 + 0.5 * dot(n, s.upP)) * canopySky * aoDetail);

    vec3 L;
    if (vHeight < 0.0) {
        // Sea: deep water absorbs, Fresnel reflects the sky, the sun leaves a glint.
        float depth = clamp(-vHeight / 300.0, 0.0, 1.0);
        vec3 water = mix(vec3(0.02, 0.06, 0.05), vec3(0.004, 0.012, 0.03), depth);
        vec3 nW = s.upP;
        vec3 r = reflect(s.viewDir, nW);
        r.z = abs(r.z);
        float cosV = max(dot(-s.viewDir, nW), 0.0);
        float fresnel = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);
        vec3 skyR = skyLUT(TEX_SKYVIEW_SUN, r, sun, s.rcKm) + skyLUT(TEX_SKYVIEW_MOON, r, moon, s.rcKm) * moonE;
        vec3 h = normalize(sun - s.viewDir);
        float glint = pow(max(dot(h, nW), 0.0), 900.0) * 120.0;
        L = water / PI * E * (1.0 - fresnel) + skyR * fresnel + sunT * glint * fresnel;
    } else {
        // Far field: macro colours. Out where individual trees fade (3.5–4.1 km) the
        // terrain *is* the forest: canopy layer with the seasonal evergreen/deciduous mix.
        vec3 understorey = vec3(0.10, 0.065, 0.025); // leaf-litter mean, shaded by the canopy term above
        float mottle = organicNoise(NOISE_ROT * worldPos / 96.0);
        vec3 evergreen = vec3(0.028, 0.044, 0.018) * (0.85 + 0.3 * mottle); // matches the impostor crowns' mean
        vec3 autumn = mix(vec3(0.20, 0.09, 0.03), vec3(0.27, 0.20, 0.035), mottle);
        vec3 canopy = mix(evergreen, autumn, FRAME.world.w * 0.22 * smoothstep(0.3, 0.8, mottle));
        // From afar the forest floor shows between crowns, as it does in the near field.
        canopy = mix(canopy, understorey * 0.45, 0.35);
        vec3 forest = mix(understorey, canopy, smoothstep(3000.0, 4100.0, s.dist));
        // Mean of the lichen_rock material (linear 0.064/0.049/0.034), varied per outcrop.
        vec3 rock = vec3(0.070, 0.054, 0.038) * (0.8 + 0.5 * organicNoise(NOISE_ROT * worldPos / 40.0));
        rock *= mix(vec3(1.0), vec3(1.45, 1.05, 0.72), vHintA.w);
        vec3 shore = vec3(0.28, 0.25, 0.19);
        // Means of the biome materials (linear), lightly varied.
        vec3 grassFar = dryTint(vec3(0.055, 0.075, 0.026), vHintB.w) * (0.85 + 0.3 * mottle);
        vec3 sandFar = vec3(0.46, 0.33, 0.19) * (0.9 + 0.2 * mottle);
        vec3 stonyFar = vec3(0.22, 0.17, 0.115) * (0.85 + 0.3 * mottle);
        vec3 snowFar = vec3(0.72, 0.75, 0.80);
        vec3 albedo = (forest * forestW + grassFar * grassW + stonyFar * stonyW + sandFar * sandW + snowFar * snowW + rock * rockW) /
                      max(forestW + grassW + stonyW + sandW + snowW + rockW, 1e-3);
        albedo = mix(shore, albedo, smoothstep(0.0, 25.0, vHeight) * 0.6 + 0.4);
        albedo = mix(albedo, albedoDetail, detail);
        L = albedo / PI * E;
    }
    L = applyAerialPerspective(s, L);

    if (pc.lightIndex == 1u) { // debug: LOD level colours (skirts magenta)
        float lv = mod(vLevel, 100.0);
        vec3 c = 0.5 + 0.5 * cos(6.2831 * (lv / 7.0 + vec3(0.0, 0.33, 0.67)));
        if (vLevel >= 100.0) c = vec3(1.0, 0.0, 1.0);
        outColor = vec4(c * (0.35 + 0.65 * max(dot(n, sun), 0.0)), 1.0);
        return;
    }
    outColor = vec4(min(L * FRAME.camera.x, vec3(FP16_SAFE_MAX)), 1.0);
}
