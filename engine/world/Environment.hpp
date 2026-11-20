#pragma once
// Game clock + observer + sky state + eye adaptation. Runs on the supervisor's
// fixed simulation tick and produces the float SkySnapshot for the renderer.
#include "engine/astro/Astro.hpp"
#include "engine/render/FrameSnapshot.hpp"

#include <string>

namespace phx {

struct EnvironmentConfig {
    astro::Observer observer{34.4625 * astro::kDeg, 110.0803 * astro::kDeg, 2016.0}; // Huashan summit (DEM)
    f64 startJdUT = 0.0;  // 0 = system clock now
    f64 timeRate = 1.0;   // game seconds per real second
};

class Environment {
public:
    explicit Environment(const EnvironmentConfig& cfg);

    void advance(f64 realDt);      // advance clock, recompute sky, adapt exposure
    void setObserver(const astro::Observer& obs) { observer_ = obs; }
    bool handleKey(int glfwKey);   // time controls; returns true if consumed
    void fill(FrameSnapshot& snap) const;

    [[nodiscard]] f64 jdUT() const { return jdUT_; }
    [[nodiscard]] f64 decimalYear() const { return astro::decimalYearFromJulian(jdUT_); }
    [[nodiscard]] std::string statusLine() const;
    [[nodiscard]] const astro::SkyState& sky() const { return sky_; }

private:
    void recompute();
    [[nodiscard]] f64 targetAdaptation() const;

    astro::Observer observer_;
    f64 jdUT_;
    f64 rate_;
    bool paused_ = false;
    astro::SkyState sky_{};
    f64 logAdaptation_ = -2.0;  // log10 of adaptation luminance (solar units)
    bool adaptationInitialised_ = false;
};

f64 systemJulianDateUT();
bool parseIsoUtc(const char* text, f64& jdOut); // "YYYY-MM-DD[THH:MM[:SS]]"

} // namespace phx
