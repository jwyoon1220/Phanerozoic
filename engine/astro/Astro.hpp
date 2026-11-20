#pragma once
// Astronomy for the sky (plan §2.6). All angles in radians unless suffixed Deg.
// Accuracy target: < 1 arcminute for stars, sun and planets; a few arcminutes for
// the Moon (truncated ELP series). Algorithms follow Meeus, "Astronomical
// Algorithms" (2nd ed.) and Standish's JPL Keplerian elements.
//
// Frames:
//   ICRS  ~ mean equator & equinox J2000 (frame bias, ~23 mas, ignored)
//   TOD   = true equator & equinox of date
//   ENU   = observer's local frame: x = east, y = north, z = up

#include "engine/core/Types.hpp"

#include <glm/glm.hpp>

namespace phx::astro {

inline constexpr f64 kPi = 3.14159265358979323846;
inline constexpr f64 kDeg = kPi / 180.0;
inline constexpr f64 kArcsec = kDeg / 3600.0;
inline constexpr f64 kJ2000 = 2451545.0;
inline constexpr f64 kAuKm = 149597870.7;
inline constexpr f64 kEarthRadiusKm = 6378.14;
inline constexpr f64 kSunMagnitude = -26.74;

// ------------------------------------------------------------------ time

struct CalendarDate {
    i32 year;
    i32 month;
    i32 day;
    i32 hour = 0;
    i32 minute = 0;
    f64 second = 0.0;
};

f64 julianDate(const CalendarDate& d);       // Gregorian calendar → JD (same time scale as input)
CalendarDate calendarFromJulian(f64 jd);
f64 deltaTSeconds(f64 decimalYear);          // TT − UT (Espenak & Meeus polynomials)
f64 decimalYearFromJulian(f64 jd);

struct Instant {
    f64 jdUT;  // UT1 ≈ UTC
    f64 jdTT;  // terrestrial time
    static Instant fromUT(f64 jdUT);
    [[nodiscard]] f64 centuriesTT() const { return (jdTT - kJ2000) / 36525.0; }
    [[nodiscard]] f64 centuriesUT() const { return (jdUT - kJ2000) / 36525.0; }
};

// ------------------------------------------------------------------ earth orientation

struct Nutation {
    f64 dPsi;  // nutation in longitude
    f64 dEps;  // nutation in obliquity
};
Nutation nutation(f64 T);                      // Meeus ch.22 low-precision (≤0.5")
f64 meanObliquity(f64 T);                      // IAU 1980 (Meeus 22.2)
glm::dmat3 precessionMatrix(f64 T);            // ICRS → mean of date (IAU 1976)
glm::dmat3 nutationMatrix(f64 T);              // mean of date → TOD
f64 greenwichMeanSiderealTime(f64 jdUT);       // radians, [0, 2π)
f64 greenwichApparentSiderealTime(const Instant& t);

struct Observer {
    f64 latitude;     // geodetic, radians (north +)
    f64 longitude;    // radians (east +)
    f64 altitudeM;
};

// TOD equatorial → ENU for the observer at local apparent sidereal time `last`.
glm::dmat3 todToEnu(const Observer& obs, f64 last);

// ------------------------------------------------------------------ bodies

struct BodyState {
    glm::dvec3 enu;          // unit direction, topocentric, before refraction
    f64 distanceKm = 0.0;
    f64 angularRadius = 0.0;
    f64 magnitude = 0.0;     // visual
    f64 phaseAngle = 0.0;    // sun-body-earth angle (radians)
};

struct EclipticPosition {
    f64 lon, lat, distKm; // of date
};

f64 sunApparentLongitude(f64 T);                   // Meeus 25 (includes aberration + nutation)
glm::dvec3 sunEquatorialTOD(f64 T);               // unit vector
EclipticPosition moonEcliptic(f64 T);             // geocentric, mean equinox of date (no nutation)

enum class Planet : u8 { Mercury, Venus, Mars, Jupiter, Saturn, Count };
inline constexpr u32 kPlanetCount = u32(Planet::Count);
const char* planetName(Planet p);

// Geocentric ICRS position in AU (light time ignored, ~arcsec level for outer planets).
glm::dvec3 planetGeocentricICRS(Planet p, f64 T, f64* heliocentricDistAu = nullptr);

// Atmospheric refraction (Bennett), apparent altitude increase for a true altitude.
f64 refraction(f64 trueAltitude);
glm::dvec3 applyRefraction(const glm::dvec3& enu);

// ------------------------------------------------------------------ full sky state

struct SkyState {
    Instant instant;
    glm::dmat3 icrsToEnu;    // stars (apply refraction per star on GPU)
    BodyState sun;
    BodyState moon;
    f64 moonIlluminatedFraction = 0.0;
    f64 moonIlluminance = 0.0;   // relative to the sun's outside the atmosphere (≈2.5e-6 at full)
    BodyState planets[kPlanetCount];
};

SkyState computeSky(const Instant& t, const Observer& obs);

} // namespace phx::astro
