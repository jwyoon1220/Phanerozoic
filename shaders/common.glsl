// Shared declarations: bindless set, push constants, per-frame uniforms.
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(set = 0, binding = 0) uniform sampler2D gTextures[];
layout(set = 0, binding = 1, rgba16f) uniform image2D gStorage[];
layout(set = 0, binding = 2, r32f) uniform image2D gStorageR32[];

// Bindless slots (mirror engine/render/SkyRenderer.hpp)
#define TEX_TRANSMITTANCE 0
#define TEX_MULTISCATTER  1
#define TEX_SKYVIEW_SUN   2
#define TEX_SKYVIEW_MOON  3
#define TEX_HDR           4
#define TEX_SHADOW_NEAR   5
#define TEX_SHADOW_FAR    6
#define TEX_SHADOW_VIS_NEAR 7
#define TEX_SHADOW_VIS_FAR  8
#define TEX_DEPTH           9
#define TEX_CASCADE0        20
#define TEX_IMPOSTOR_ALBEDO0 10
#define TEX_IMPOSTOR_NORMAL0 13
// storage slots (gStorage)
#define STOR_SHADOW_VIS_NEAR 4
#define STOR_SHADOW_VIS_FAR  5
#define STOR_MOTION          6

layout(buffer_reference, scalar) readonly buffer FrameRef {
    mat4 viewProj;     // rotation-only camera in ENU (x east, y north, z up)
    mat4 viewProjEcef; // camera-relative ECEF → clip (terrain)
    mat4 ecefToEnu;    // camera's local frame (upper 3x3)
    mat4 invViewProj;
    mat4 icrsToEnu;    // upper-left 3x3
    vec4 sunDir;       // xyz apparent direction, w angular radius
    vec4 moonDir;      // xyz, w angular radius
    vec4 moonParams;   // x illuminance rel. sun, y illuminated fraction, z earthshine factor
    vec4 camera;       // x exposure (pre-exposure), y time [s], z observer altitude [km], w pixel angle [rad]
    vec4 resolution;   // xy size, zw 1/size
    vec4 tonemap;      // x adaptation luminance, y frame counter
    vec4 cameraEcef;   // xyz metres (approximate, float), w planet radius [m]
    mat4 shadowEnu[2]; // terrain shadow height maps: anchor frames
    vec4 shadowOffset[2]; // xyz anchor − camera (ECEF), w extent [m] (0 = invalid)
    vec4 shadowParams; // x enabled, y near texel, z far texel
    vec4 world;        // x day of year, y wind time [s], z wind strength, w autumn factor
    vec4 windDir;      // xyz wind direction (camera ENU), w gust phase
    vec4 cullParams;   // x near plane, y pixels per radian, z/w render size
    vec4 cameraMod;    // xyz camera ECEF mod 65536 m (precise world-locked coordinates), w = 65536
    mat4 csm[3];       // cascaded shadow maps: camera-relative ECEF → light clip
    vec4 csmSplits;    // far distance per cascade, w enabled
    vec4 csmTexel;     // world texel size per cascade, w map size
    vec4 ground;       // rgb mean ground albedo around the camera (bounce light)
    mat4 invViewProjEcef;      // inverse of the jittered viewProjEcef
    mat4 viewProjEcefNoJitter;
    mat4 viewProjEcefPrev;     // previous frame, unjittered, relative to the previous camera
    mat4 viewProjNoJitter;     // ENU rotation only, unjittered (stars after upscaling)
    vec4 camDelta;             // xyz current − previous camera (ECEF m), w history valid
    vec4 upscale;              // x mip bias, y 2^bias, zw jitter (render px)
    vec4 display;              // output width, height, 1/width, 1/height
    vec4 display2;             // x pixel angle at output resolution
};

struct GpuStar { vec3 dir; float flux; vec3 color; float pad; };
layout(buffer_reference, scalar) readonly buffer StarRef { GpuStar stars[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t frame;
    uint64_t stars;
    uint starCount;
    uint lightIndex; // sky-view: 0 sun, 1 moon
    uint64_t aux;    // pass-specific buffer (terrain: vertex pool)
    uint64_t aux2;
    uint64_t aux3;
} pc;

#define FRAME FrameRef(pc.frame)

const float PI = 3.14159265358979;
const float FP16_SAFE_MAX = 60000.0;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float valueNoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm3(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 5; ++i) { s += a * valueNoise3(p); p *= 2.03; a *= 0.5; }
    return s;
}
