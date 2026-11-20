// Physically based atmosphere after Hillaire, "A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique" (EGSR 2020). Units: km. Illuminance of the
// light source outside the atmosphere = 1 (the sun); radiance results scale with it.

const float R_BOTTOM = 6371.0; // matches kPlanetRadiusM
const float R_TOP = 6471.0;
const vec3 RAYLEIGH_SCATTERING = vec3(5.802, 13.558, 33.1) * 1e-3;
const float RAYLEIGH_H = 8.0;
// Eocene greenhouse climate: a little more aerosol than today's reference (haze
// that also helps hide far-LOD detail, plan §2.7).
const float MIE_SCATTERING = 4.6e-3;
const float MIE_EXTINCTION = 5.1e-3;
const float MIE_H = 1.2;
const float MIE_G = 0.8;
const vec3 OZONE_ABSORPTION = vec3(0.650, 1.881, 0.085) * 1e-3;
const vec3 GROUND_ALBEDO = vec3(0.25);

struct Medium { vec3 scatR; float scatM; vec3 ext; };

Medium sampleMedium(float h) {
    h = max(h, 0.0);
    float dR = exp(-h / RAYLEIGH_H), dM = exp(-h / MIE_H);
    float dO = max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    Medium m;
    m.scatR = RAYLEIGH_SCATTERING * dR;
    m.scatM = MIE_SCATTERING * dM;
    m.ext = m.scatR + vec3(MIE_EXTINCTION * dM) + OZONE_ABSORPTION * dO;
    return m;
}

// Nearest positive intersection with a sphere at the origin, or -1.
float raySphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd), c = dot(ro, ro) - r * r, d = b * b - c;
    if (d < 0.0) return -1.0;
    float s = sqrt(d), t0 = -b - s, t1 = -b + s;
    return t0 > 0.0 ? t0 : (t1 > 0.0 ? t1 : -1.0);
}

float rayleighPhase(float c) { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
float miePhase(float c) { // Cornette-Shanks
    float g = MIE_G, g2 = g * g;
    return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + c * c)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

// Bruneton's transmittance LUT parameterisation.
vec2 transmittanceUV(float r, float mu) {
    float H = sqrt(R_TOP * R_TOP - R_BOTTOM * R_BOTTOM);
    float rho = sqrt(max(r * r - R_BOTTOM * R_BOTTOM, 0.0));
    float d = max(-r * mu + sqrt(max(r * r * (mu * mu - 1.0) + R_TOP * R_TOP, 0.0)), 0.0);
    float dMin = R_TOP - r, dMax = rho + H;
    return vec2((d - dMin) / (dMax - dMin), rho / H);
}
void transmittanceRMu(vec2 uv, out float r, out float mu) {
    float H = sqrt(R_TOP * R_TOP - R_BOTTOM * R_BOTTOM);
    float rho = H * uv.y;
    r = sqrt(rho * rho + R_BOTTOM * R_BOTTOM);
    float dMin = R_TOP - r, dMax = rho + H, d = dMin + uv.x * (dMax - dMin);
    mu = d == 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

// Transmittance from radius r towards direction with zenith cosine mu (to space).
vec3 transmittanceToSpace(float r, float mu) {
    // Below the geometric horizon the ray hits the ground.
    float horizon = -sqrt(max(1.0 - (R_BOTTOM * R_BOTTOM) / (r * r), 0.0));
    if (mu < horizon) return vec3(0.0);
    return texture(gTextures[TEX_TRANSMITTANCE], transmittanceUV(r, mu)).rgb;
}

vec3 multiScattering(float r, float muSun) {
    vec2 uv = vec2(muSun * 0.5 + 0.5, (r - R_BOTTOM) / (R_TOP - R_BOTTOM));
    return texture(gTextures[TEX_MULTISCATTER], clamp(uv, 0.0, 1.0)).rgb;
}

// Sky-view LUT mapping (non-linear in zenith to concentrate texels at the horizon).
vec2 skyViewUV(float viewHeight, float viewZenithCos, float lightViewCos) {
    float vHorizon = sqrt(max(viewHeight * viewHeight - R_BOTTOM * R_BOTTOM, 0.0));
    float beta = acos(clamp(vHorizon / viewHeight, -1.0, 1.0));
    float zenithHorizonAngle = PI - beta;
    float viewZenith = acos(clamp(viewZenithCos, -1.0, 1.0));
    vec2 uv;
    if (viewZenith < zenithHorizonAngle) {
        float c = viewZenith / zenithHorizonAngle;
        uv.y = (1.0 - sqrt(max(1.0 - c, 0.0))) * 0.5;
    } else {
        float c = (viewZenith - zenithHorizonAngle) / beta;
        uv.y = sqrt(clamp(c, 0.0, 1.0)) * 0.5 + 0.5;
    }
    uv.x = sqrt(clamp(-lightViewCos * 0.5 + 0.5, 0.0, 1.0));
    return uv;
}
void skyViewParams(vec2 uv, float viewHeight, out float viewZenithCos, out float lightViewCos) {
    float vHorizon = sqrt(max(viewHeight * viewHeight - R_BOTTOM * R_BOTTOM, 0.0));
    float beta = acos(clamp(vHorizon / viewHeight, -1.0, 1.0));
    float zenithHorizonAngle = PI - beta;
    float viewZenith;
    if (uv.y < 0.5) {
        float c = 1.0 - 2.0 * uv.y;
        c = 1.0 - c * c;
        viewZenith = zenithHorizonAngle * c;
    } else {
        float c = uv.y * 2.0 - 1.0;
        viewZenith = zenithHorizonAngle + beta * c * c;
    }
    viewZenithCos = cos(viewZenith);
    lightViewCos = -(uv.x * uv.x * 2.0 - 1.0);
}

// Camera radius for LUT lookups (the LUT parameterisation needs R_BOTTOM < r < R_TOP).
float viewRadiusKm(float altitudeKm) { return R_BOTTOM + clamp(altitudeKm, 0.005, R_TOP - R_BOTTOM - 0.01); }

// Sky radiance towards `dir` (ENU) from a sky-view LUT built for `lightDir`.
vec3 skyLUT(uint tex, vec3 dir, vec3 lightDir, float viewHeight) {
    vec2 h = dir.xy, l = lightDir.xy;
    float lh = length(h), ll = length(l);
    float lightViewCos = (lh > 1e-5 && ll > 1e-5) ? dot(h / lh, l / ll) : 1.0;
    return texture(gTextures[tex], skyViewUV(viewHeight, dir.z, lightViewCos)).rgb;
}
