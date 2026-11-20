#include "engine/world/Environment.hpp"

#include "engine/astro/StarCatalog.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>

namespace phx {
namespace {
constexpr f64 kRates[] = {0, 1, 10, 60, 300, 1800, 3600, 21600};
constexpr f64 kNightSky = 1.3e-9; // natural night-sky radiance, solar units / sr

// Linear-sRGB tints for planets (luminance-normalised in fill()).
constexpr f32 kPlanetTint[astro::kPlanetCount][3] = {
    {1.0f, 0.95f, 0.88f}, {1.0f, 0.98f, 0.92f}, {1.0f, 0.62f, 0.42f}, {1.0f, 0.92f, 0.80f}, {1.0f, 0.90f, 0.70f}};
} // namespace

f64 systemJulianDateUT() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return 2440587.5 + std::chrono::duration<f64>(now).count() / 86400.0;
}

bool parseIsoUtc(const char* text, f64& jd) {
    int y, mo, d, h = 0, mi = 0;
    double s = 0;
    const int n = std::sscanf(text, "%d-%d-%dT%d:%d:%lf", &y, &mo, &d, &h, &mi, &s);
    if (n < 3) return false;
    jd = astro::julianDate({y, mo, d, h, mi, s});
    return true;
}

Environment::Environment(const EnvironmentConfig& cfg)
    : observer_(cfg.observer), jdUT_(cfg.startJdUT > 0 ? cfg.startJdUT : systemJulianDateUT()), rate_(cfg.timeRate) {
    recompute();
}

void Environment::recompute() { sky_ = astro::computeSky(astro::Instant::fromUT(jdUT_), observer_); }

f64 Environment::targetAdaptation() const {
    // Empirical sky luminance vs. solar altitude (clear sky), in solar units / sr:
    // ~0.05 by day, ~5e-3 at sunset, falling ~0.36 dex per degree through twilight.
    const f64 sunAlt = std::asin(sky_.sun.enu.z) / astro::kDeg;
    const f64 logSun = sunAlt > 0 ? -2.3 + 1.0 * std::clamp(sunAlt / 30.0, 0.0, 1.0) : -2.3 + 0.36 * sunAlt;
    const f64 moonUp = std::max(sky_.moon.enu.z, 0.0);
    const f64 moonSky = sky_.moonIlluminance * 0.08 * std::sqrt(moonUp);
    return std::log10(std::max({std::pow(10.0, logSun), moonSky, 2.0 * kNightSky}));
}

void Environment::advance(f64 realDt) {
    if (!paused_) jdUT_ += realDt * rate_ / 86400.0;
    recompute();
    const f64 target = targetAdaptation();
    if (!adaptationInitialised_) {
        logAdaptation_ = target;
        adaptationInitialised_ = true;
    } else {
        // Eye adaptation, faster when time is accelerated so the view keeps up.
        const f64 tau = 1.5 / std::max(1.0, std::log10(std::max(rate_, 1.0)) + 1.0);
        logAdaptation_ += (target - logAdaptation_) * (1.0 - std::exp(-realDt / tau));
    }
}

bool Environment::handleKey(int key) {
    const f64* begin = std::begin(kRates) + 1, *end = std::end(kRates);
    switch (key) {
    case GLFW_KEY_SPACE: paused_ = !paused_; return true;
    case GLFW_KEY_RIGHT_BRACKET: {
        const f64* it = std::upper_bound(begin, end, rate_);
        rate_ = it == end ? *(end - 1) : *it;
        return true;
    }
    case GLFW_KEY_LEFT_BRACKET: {
        const f64* it = std::lower_bound(begin, end, rate_);
        rate_ = it == begin ? *begin : *(it - 1);
        return true;
    }
    case GLFW_KEY_BACKSPACE: rate_ = 1; return true;
    case GLFW_KEY_N: jdUT_ = systemJulianDateUT(); return true;
    case GLFW_KEY_UP: jdUT_ += 1.0 / 24.0; return true;
    case GLFW_KEY_DOWN: jdUT_ -= 1.0 / 24.0; return true;
    case GLFW_KEY_PAGE_UP: jdUT_ += 1.0; return true;
    case GLFW_KEY_PAGE_DOWN: jdUT_ -= 1.0; return true;
    default: return false;
    }
}

void Environment::fill(FrameSnapshot& snap) const {
    SkySnapshot& s = snap.sky;
    s.icrsToEnu = glm::mat3(sky_.icrsToEnu);
    s.sunDir = glm::vec3(astro::applyRefraction(sky_.sun.enu));
    s.sunRadius = f32(sky_.sun.angularRadius);
    s.moonDir = glm::vec3(astro::applyRefraction(sky_.moon.enu));
    s.moonRadius = f32(sky_.moon.angularRadius);
    s.moonIlluminance = f32(sky_.moonIlluminance);
    s.moonFraction = f32(sky_.moonIlluminatedFraction);
    // Earth phase seen from the Moon ≈ 1 − lunar phase; earthshine falls off steeply past quarter.
    s.earthshine = f32(1.3e-4 * std::pow(1.0 - sky_.moonIlluminatedFraction, 2.0));
    s.observerAltitudeKm = f32(observer_.altitudeM / 1000.0);
    {
        const astro::CalendarDate d = astro::calendarFromJulian(jdUT_);
        const f64 jan1 = astro::julianDate({d.year, 1, 1});
        s.dayOfYear = f32(jdUT_ - jan1);
    }
    const glm::dmat3 enuToIcrs = glm::transpose(sky_.icrsToEnu);
    for (u32 i = 0; i < astro::kPlanetCount; ++i) {
        astro::GpuStar& p = s.planets[i];
        const glm::dvec3 icrs = enuToIcrs * sky_.planets[i].enu;
        for (int k = 0; k < 3; ++k) p.dir[k] = f32(icrs[k]);
        p.flux = astro::fluxFromMagnitude(sky_.planets[i].magnitude);
        const f32* t = kPlanetTint[i];
        const f32 lum = 0.2126f * t[0] + 0.7152f * t[1] + 0.0722f * t[2];
        for (int k = 0; k < 3; ++k) p.color[k] = t[k] / lum;
    }

    // Pre-exposure: middle grey maps the adapted luminance; darker key at night so
    // the scene stays a night scene instead of being normalised to daylight.
    const f64 adapt = std::pow(10.0, logAdaptation_);
    const f64 nightness = std::clamp((-3.0 - logAdaptation_) / 5.0, 0.0, 1.0);
    const f64 key = 0.18 * std::pow(10.0, -0.4 * nightness);
    snap.exposure = f32(key / adapt);
    snap.adaptationLuminance = f32(adapt);
}

std::string Environment::statusLine() const {
    // Local mean solar time: no civil time zones in the Eocene.
    const f64 lmtOffsetDays = observer_.longitude / (2 * astro::kPi);
    const astro::CalendarDate d = astro::calendarFromJulian(jdUT_ + lmtOffsetDays);
    const f64 sunAlt = std::asin(sky_.sun.enu.z) / astro::kDeg;
    const f64 moonAlt = std::asin(sky_.moon.enu.z) / astro::kDeg;
    auto azimuth = [](const glm::dvec3& v) {
        const f64 a = std::atan2(v.x, v.y) / astro::kDeg;
        return a < 0 ? a + 360 : a;
    };
    const std::string rate = paused_ ? "paused" : std::format("x{:g}", rate_);
    return std::format("{:04}-{:02}-{:02} {:02}:{:02} LMT  {}  |  Sun {:+.1f}° az {:.0f}°  Moon {:+.1f}° az {:.0f}° ({:.0f}%)",
                       d.year, d.month, d.day, d.hour, d.minute, rate, sunAlt, azimuth(sky_.sun.enu), moonAlt,
                       azimuth(sky_.moon.enu), sky_.moonIlluminatedFraction * 100);
}

} // namespace phx
