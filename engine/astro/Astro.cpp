#include "engine/astro/Astro.hpp"

#include <algorithm>
#include <cmath>

namespace phx::astro {
namespace {

f64 normDeg(f64 d) {
    d = std::fmod(d, 360.0);
    return d < 0 ? d + 360.0 : d;
}
f64 norm2Pi(f64 r) {
    r = std::fmod(r, 2.0 * kPi);
    return r < 0 ? r + 2.0 * kPi : r;
}
f64 sinD(f64 d) { return std::sin(d * kDeg); }
f64 cosD(f64 d) { return std::cos(d * kDeg); }

glm::dmat3 fromRows(const glm::dvec3& r0, const glm::dvec3& r1, const glm::dvec3& r2) {
    return glm::transpose(glm::dmat3(r0, r1, r2));
}
// Frame rotations (SOFA convention): rotate the coordinate axes by +a.
glm::dmat3 R1(f64 a) {
    const f64 c = std::cos(a), s = std::sin(a);
    return fromRows({1, 0, 0}, {0, c, s}, {0, -s, c});
}
glm::dmat3 R2(f64 a) {
    const f64 c = std::cos(a), s = std::sin(a);
    return fromRows({c, 0, -s}, {0, 1, 0}, {s, 0, c});
}
glm::dmat3 R3(f64 a) {
    const f64 c = std::cos(a), s = std::sin(a);
    return fromRows({c, s, 0}, {-s, c, 0}, {0, 0, 1});
}

glm::dvec3 sphericalToVec(f64 lon, f64 lat) {
    return {std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
}

// Ecliptic of date → TOD equatorial using the true obliquity.
glm::dvec3 eclipticToEquatorial(const glm::dvec3& v, f64 eps) {
    const f64 c = std::cos(eps), s = std::sin(eps);
    return {v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}

} // namespace

// ------------------------------------------------------------------ time

f64 julianDate(const CalendarDate& d) {
    i32 y = d.year, m = d.month;
    if (m <= 2) {
        y -= 1;
        m += 12;
    }
    const i32 a = y / 100;
    const i32 b = 2 - a + a / 4;
    const f64 dayFrac = d.day + (d.hour + (d.minute + d.second / 60.0) / 60.0) / 24.0;
    return std::floor(365.25 * (y + 4716)) + std::floor(30.6001 * (m + 1)) + dayFrac + b - 1524.5;
}

CalendarDate calendarFromJulian(f64 jd) {
    jd += 0.5;
    const f64 z = std::floor(jd);
    const f64 f = jd - z;
    f64 a = z;
    if (z >= 2299161) {
        const f64 alpha = std::floor((z - 1867216.25) / 36524.25);
        a = z + 1 + alpha - std::floor(alpha / 4);
    }
    const f64 b = a + 1524;
    const f64 c = std::floor((b - 122.1) / 365.25);
    const f64 d = std::floor(365.25 * c);
    const f64 e = std::floor((b - d) / 30.6001);
    const f64 dayF = b - d - std::floor(30.6001 * e) + f;
    CalendarDate out{};
    out.day = i32(dayF);
    out.month = i32(e < 14 ? e - 1 : e - 13);
    out.year = i32(out.month > 2 ? c - 4716 : c - 4715);
    f64 secs = (dayF - out.day) * 86400.0;
    out.hour = i32(secs / 3600.0);
    secs -= out.hour * 3600.0;
    out.minute = i32(secs / 60.0);
    out.second = secs - out.minute * 60.0;
    return out;
}

f64 decimalYearFromJulian(f64 jd) { return 2000.0 + (jd - kJ2000) / 365.25; }

f64 deltaTSeconds(f64 y) {
    // Espenak & Meeus (2006) polynomial expressions; adequate for any game date.
    if (y < 1900) {
        const f64 u = (y - 1820) / 100;
        return -20 + 32 * u * u;
    }
    if (y < 1986) {
        const f64 t = y - 1950;
        return 29.07 + 0.407 * t - t * t / 233 + t * t * t / 2547;
    }
    if (y < 2005) {
        const f64 t = y - 2000;
        return 63.86 + 0.3345 * t - 0.060374 * t * t + 0.0017275 * t * t * t + 0.000651814 * t * t * t * t +
               0.00002373599 * t * t * t * t * t;
    }
    if (y < 2050) {
        const f64 t = y - 2000;
        return 62.92 + 0.32217 * t + 0.005589 * t * t;
    }
    if (y < 2150) {
        const f64 u = (y - 1820) / 100;
        return -20 + 32 * u * u - 0.5628 * (2150 - y);
    }
    const f64 u = (y - 1820) / 100;
    return -20 + 32 * u * u;
}

Instant Instant::fromUT(f64 jdUT) {
    return {jdUT, jdUT + deltaTSeconds(decimalYearFromJulian(jdUT)) / 86400.0};
}

// ------------------------------------------------------------------ earth orientation

Nutation nutation(f64 T) {
    const f64 omega = 125.04452 - 1934.136261 * T;
    const f64 L = 280.4665 + 36000.7698 * T;
    const f64 Lp = 218.3165 + 481267.8813 * T;
    const f64 dPsi = -17.20 * sinD(omega) - 1.32 * sinD(2 * L) - 0.23 * sinD(2 * Lp) + 0.21 * sinD(2 * omega);
    const f64 dEps = 9.20 * cosD(omega) + 0.57 * cosD(2 * L) + 0.10 * cosD(2 * Lp) - 0.09 * cosD(2 * omega);
    return {dPsi * kArcsec, dEps * kArcsec};
}

f64 meanObliquity(f64 T) {
    const f64 arcsec = 84381.448 - 46.8150 * T - 0.00059 * T * T + 0.001813 * T * T * T;
    return arcsec * kArcsec;
}

glm::dmat3 precessionMatrix(f64 T) {
    const f64 zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) * kArcsec;
    const f64 z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) * kArcsec;
    const f64 theta = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T * T * T) * kArcsec;
    return R3(-z) * R2(theta) * R3(-zeta);
}

glm::dmat3 nutationMatrix(f64 T) {
    const Nutation n = nutation(T);
    const f64 eps0 = meanObliquity(T);
    return R1(-(eps0 + n.dEps)) * R3(-n.dPsi) * R1(eps0);
}

f64 greenwichMeanSiderealTime(f64 jdUT) {
    const f64 T = (jdUT - kJ2000) / 36525.0;
    const f64 deg = 280.46061837 + 360.98564736629 * (jdUT - kJ2000) + 0.000387933 * T * T - T * T * T / 38710000.0;
    return normDeg(deg) * kDeg;
}

f64 greenwichApparentSiderealTime(const Instant& t) {
    const f64 T = t.centuriesTT();
    const Nutation n = nutation(T);
    return norm2Pi(greenwichMeanSiderealTime(t.jdUT) + n.dPsi * std::cos(meanObliquity(T) + n.dEps));
}

glm::dmat3 todToEnu(const Observer& obs, f64 last) {
    const f64 sl = std::sin(obs.latitude), cl = std::cos(obs.latitude);
    // After R3(LAST): x → local meridian on the equator, y → east, z → celestial pole.
    const glm::dmat3 local = fromRows({0, 1, 0}, {-sl, 0, cl}, {cl, 0, sl});
    return local * R3(last);
}

// ------------------------------------------------------------------ sun

f64 sunApparentLongitude(f64 T) {
    const f64 L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T;
    const f64 M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
    const f64 C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * sinD(M) + (0.019993 - 0.000101 * T) * sinD(2 * M) +
                  0.000289 * sinD(3 * M);
    const f64 omega = 125.04 - 1934.136 * T;
    return normDeg(L0 + C - 0.00569 - 0.00478 * sinD(omega)) * kDeg;
}

namespace {
f64 sunDistanceAu(f64 T) {
    const f64 M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
    const f64 e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T;
    const f64 C = (1.914602 - 0.004817 * T) * sinD(M) + (0.019993 - 0.000101 * T) * sinD(2 * M) + 0.000289 * sinD(3 * M);
    return 1.000001018 * (1 - e * e) / (1 + e * cosD(M + C));
}
} // namespace

glm::dvec3 sunEquatorialTOD(f64 T) {
    const f64 omega = 125.04 - 1934.136 * T;
    const f64 eps = meanObliquity(T) + 0.00256 * kDeg * cosD(omega);
    return eclipticToEquatorial(sphericalToVec(sunApparentLongitude(T), 0.0), eps);
}

// ------------------------------------------------------------------ moon (Meeus ch.47, main terms)

namespace {
struct LunarTerm {
    i8 d, m, mp, f;
    f64 coeff;
};
// Σl (1e-6 deg, sine) and Σr (1e-3 km, cosine), arguments D, M, M', F.
constexpr LunarTerm kLonTerms[] = {
    {0, 0, 1, 0, 6288774},  {2, 0, -1, 0, 1274027}, {2, 0, 0, 0, 658314},   {0, 0, 2, 0, 213618},
    {0, 1, 0, 0, -185116},  {0, 0, 0, 2, -114332},  {2, 0, -2, 0, 58793},   {2, -1, -1, 0, 57066},
    {2, 0, 1, 0, 53322},    {2, -1, 0, 0, 45758},   {0, 1, -1, 0, -40923},  {1, 0, 0, 0, -34720},
    {0, 1, 1, 0, -30383},   {2, 0, 0, -2, 15327},   {0, 0, 1, 2, -12528},   {0, 0, 1, -2, 10980},
    {4, 0, -1, 0, 10675},   {0, 0, 3, 0, 10034},    {4, 0, -2, 0, 8548},    {2, 1, -1, 0, -7888},
    {2, 1, 0, 0, -6766},    {1, 0, -1, 0, -5163},   {1, 1, 0, 0, 4987},     {2, -1, 1, 0, 4036},
};
constexpr LunarTerm kDistTerms[] = {
    {0, 0, 1, 0, -20905355}, {2, 0, -1, 0, -3699111}, {2, 0, 0, 0, -2955968}, {0, 0, 2, 0, -569925},
    {0, 1, 0, 0, 48888},     {0, 0, 0, 2, -3149},     {2, 0, -2, 0, 246158},  {2, -1, -1, 0, -152138},
    {2, 0, 1, 0, -170733},   {2, -1, 0, 0, -204586},  {0, 1, -1, 0, -129620}, {1, 0, 0, 0, 108743},
    {0, 1, 1, 0, 104755},    {2, 0, 0, -2, 10321},    {0, 0, 1, -2, 79661},   {4, 0, -1, 0, -34782},
    {0, 0, 3, 0, -23210},    {4, 0, -2, 0, -21636},   {2, 1, -1, 0, 24208},   {2, 1, 0, 0, 30824},
    {1, 0, -1, 0, -8379},    {1, 1, 0, 0, -16675},    {2, -1, 1, 0, -12831},
};
// Σb (1e-6 deg, sine)
constexpr LunarTerm kLatTerms[] = {
    {0, 0, 0, 1, 5128122}, {0, 0, 1, 1, 280602}, {0, 0, 1, -1, 277693}, {2, 0, 0, -1, 173237},
    {2, 0, -1, 1, 55413},  {2, 0, -1, -1, 46271}, {2, 0, 0, 1, 32573},   {0, 0, 2, 1, 17198},
    {2, 0, 1, -1, 9266},   {0, 0, 2, -1, 8822},   {2, -1, 0, -1, 8216},  {2, 0, -2, -1, 4324},
    {2, 0, 1, 1, 4200},
};
} // namespace

EclipticPosition moonEcliptic(f64 T) {
    const f64 T2 = T * T, T3 = T2 * T, T4 = T3 * T;
    const f64 Lp = normDeg(218.3164477 + 481267.88123421 * T - 0.0015786 * T2 + T3 / 538841 - T4 / 65194000);
    const f64 D = normDeg(297.8501921 + 445267.1114034 * T - 0.0018819 * T2 + T3 / 545868 - T4 / 113065000);
    const f64 M = normDeg(357.5291092 + 35999.0502909 * T - 0.0001536 * T2 + T3 / 24490000);
    const f64 Mp = normDeg(134.9633964 + 477198.8675055 * T + 0.0087414 * T2 + T3 / 69699 - T4 / 14712000);
    const f64 F = normDeg(93.2720950 + 483202.0175233 * T - 0.0036539 * T2 - T3 / 3526000 + T4 / 863310000);
    const f64 A1 = normDeg(119.75 + 131.849 * T);
    const f64 A2 = normDeg(53.09 + 479264.290 * T);
    const f64 A3 = normDeg(313.45 + 481266.484 * T);
    const f64 E = 1 - 0.002516 * T - 0.0000074 * T2;

    auto arg = [&](const LunarTerm& t) { return t.d * D + t.m * M + t.mp * Mp + t.f * F; };
    auto ecc = [&](const LunarTerm& t) { return t.m == 0 ? 1.0 : (t.m == 1 || t.m == -1 ? E : E * E); };

    f64 sl = 0, sr = 0, sb = 0;
    for (const auto& t : kLonTerms) sl += t.coeff * ecc(t) * sinD(arg(t));
    for (const auto& t : kDistTerms) sr += t.coeff * ecc(t) * cosD(arg(t));
    for (const auto& t : kLatTerms) sb += t.coeff * ecc(t) * sinD(arg(t));
    sl += 3958 * sinD(A1) + 1962 * sinD(Lp - F) + 318 * sinD(A2);
    sb += -2235 * sinD(Lp) + 382 * sinD(A3) + 175 * sinD(A1 - F) + 175 * sinD(A1 + F) + 127 * sinD(Lp - Mp) -
          115 * sinD(Lp + Mp);

    return {normDeg(Lp + sl / 1e6) * kDeg, (sb / 1e6) * kDeg, 385000.56 + sr / 1000.0};
}

// ------------------------------------------------------------------ planets (Standish, valid 1800–2050)

namespace {
struct Elements {
    f64 a, e, I, L, wbar, node;          // J2000 values (AU, deg)
    f64 da, de, dI, dL, dwbar, dnode;    // per Julian century
};
constexpr Elements kEmb = {1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
                           0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0};
constexpr Elements kPlanets[kPlanetCount] = {
    {0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593, 0.00000037, 0.00001906,
     -0.00594749, 149472.67411175, 0.16047689, -0.12534081},
    {0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255, 0.00000390, -0.00004107,
     -0.00078890, 58517.81538729, 0.00268329, -0.27769418},
    {1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891, 0.00001847, 0.00007882,
     -0.00813131, 19140.30268499, 0.44441088, -0.29257343},
    {5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909, -0.00011607, -0.00013253,
     -0.00183714, 3034.74612775, 0.21252668, 0.20469106},
    {9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448, -0.00125060, -0.00050991,
     0.00193609, 1222.49362201, -0.41897216, -0.28867794},
};

// Heliocentric ecliptic J2000 position (AU).
glm::dvec3 keplerPosition(const Elements& el, f64 T) {
    const f64 a = el.a + el.da * T, e = el.e + el.de * T;
    const f64 I = (el.I + el.dI * T) * kDeg;
    const f64 L = el.L + el.dL * T, wbar = el.wbar + el.dwbar * T, node = el.node + el.dnode * T;
    const f64 omega = (wbar - node) * kDeg;
    f64 M = std::remainder(L - wbar, 360.0) * kDeg;
    f64 E = M + e * std::sin(M);
    for (int i = 0; i < 8; ++i) E -= (E - e * std::sin(E) - M) / (1 - e * std::cos(E));
    const f64 xp = a * (std::cos(E) - e), yp = a * std::sqrt(1 - e * e) * std::sin(E);
    const f64 co = std::cos(omega), so = std::sin(omega), cn = std::cos(node * kDeg), sn = std::sin(node * kDeg);
    const f64 ci = std::cos(I), si = std::sin(I);
    return {(co * cn - so * sn * ci) * xp + (-so * cn - co * sn * ci) * yp,
            (co * sn + so * cn * ci) * xp + (-so * sn + co * cn * ci) * yp, (so * si) * xp + (co * si) * yp};
}

constexpr f64 kEpsJ2000 = 23.43928 * kDeg;

f64 planetMagnitude(Planet p, f64 r, f64 delta, f64 phaseDeg) {
    const f64 base = 5 * std::log10(r * delta);
    const f64 a = phaseDeg;
    switch (p) { // Mallama & Hilton (2018), simplified
    case Planet::Mercury: return -0.613 + base + 6.3280e-2 * a - 1.6336e-3 * a * a + 3.3644e-5 * a * a * a -
                                 3.4265e-7 * a * a * a * a + 1.6893e-9 * std::pow(a, 5) - 3.0334e-12 * std::pow(a, 6);
    case Planet::Venus: return -4.384 + base - 1.044e-3 * a + 3.687e-4 * a * a - 2.814e-6 * a * a * a +
                               8.938e-9 * a * a * a * a;
    case Planet::Mars: return -1.601 + base + 2.267e-2 * a - 1.302e-4 * a * a;
    case Planet::Jupiter: return -9.395 + base - 3.7e-4 * a + 6.16e-4 * a * a;
    case Planet::Saturn: return -8.914 + base - 0.3; // rough average ring contribution
    default: return 99;
    }
}
} // namespace

const char* planetName(Planet p) {
    static const char* kNames[] = {"Mercury", "Venus", "Mars", "Jupiter", "Saturn"};
    return u32(p) < kPlanetCount ? kNames[u32(p)] : "?";
}

glm::dvec3 planetGeocentricICRS(Planet p, f64 T, f64* helioDist) {
    const glm::dvec3 planet = keplerPosition(kPlanets[u32(p)], T);
    const glm::dvec3 earth = keplerPosition(kEmb, T);
    if (helioDist) *helioDist = glm::length(planet);
    const glm::dvec3 g = planet - earth; // ecliptic J2000
    const f64 c = std::cos(kEpsJ2000), s = std::sin(kEpsJ2000);
    return {g.x, c * g.y - s * g.z, s * g.y + c * g.z};
}

// ------------------------------------------------------------------ refraction

f64 refraction(f64 h) {
    const f64 hd = std::max(h / kDeg, -1.9); // Bennett diverges below about −2°
    const f64 arcmin = 1.0 / std::tan((hd + 7.31 / (hd + 4.4)) * kDeg);
    return std::max(arcmin, 0.0) / 60.0 * kDeg;
}

glm::dvec3 applyRefraction(const glm::dvec3& v) {
    const f64 alt = std::asin(std::clamp(v.z, -1.0, 1.0));
    const f64 az = std::atan2(v.x, v.y);
    const f64 apparent = alt + refraction(alt);
    return {std::cos(apparent) * std::sin(az), std::cos(apparent) * std::cos(az), std::sin(apparent)};
}

// ------------------------------------------------------------------ sky state

SkyState computeSky(const Instant& t, const Observer& obs) {
    SkyState s;
    s.instant = t;
    const f64 T = t.centuriesTT();
    const f64 last = greenwichApparentSiderealTime(t) + obs.longitude;
    const glm::dmat3 tod2enu = todToEnu(obs, last);
    s.icrsToEnu = tod2enu * nutationMatrix(T) * precessionMatrix(T);

    // Observer position in the TOD equatorial frame (km), for topocentric parallax.
    const f64 flattening = 1.0 / 298.257;
    const f64 u = std::atan((1 - flattening) * std::tan(obs.latitude));
    const f64 h = obs.altitudeM / 1000.0;
    const f64 rhoSin = (1 - flattening) * std::sin(u) + h / kEarthRadiusKm * std::sin(obs.latitude);
    const f64 rhoCos = std::cos(u) + h / kEarthRadiusKm * std::cos(obs.latitude);
    const glm::dvec3 observerTod = kEarthRadiusKm * glm::dvec3(rhoCos * std::cos(last), rhoCos * std::sin(last), rhoSin);

    // Sun
    const glm::dvec3 sunTod = sunEquatorialTOD(T);
    const f64 sunDistKm = sunDistanceAu(T) * kAuKm;
    s.sun.distanceKm = sunDistKm;
    s.sun.enu = glm::normalize(tod2enu * (sunTod * sunDistKm - observerTod));
    s.sun.angularRadius = std::asin(696000.0 / sunDistKm);
    s.sun.magnitude = kSunMagnitude;

    // Moon (apparent: add nutation in longitude, true obliquity)
    const EclipticPosition mp = moonEcliptic(T);
    const Nutation n = nutation(T);
    const glm::dvec3 moonGeoTod =
        eclipticToEquatorial(sphericalToVec(mp.lon + n.dPsi, mp.lat), meanObliquity(T) + n.dEps) * mp.distKm;
    const glm::dvec3 moonTopo = moonGeoTod - observerTod;
    s.moon.distanceKm = glm::length(moonTopo);
    s.moon.enu = glm::normalize(tod2enu * moonTopo);
    s.moon.angularRadius = std::asin(1737.4 / s.moon.distanceKm);
    const f64 elongation = std::acos(std::clamp(glm::dot(glm::normalize(moonGeoTod), sunTod), -1.0, 1.0));
    s.moon.phaseAngle = std::atan2(sunDistKm * std::sin(elongation), mp.distKm - sunDistKm * std::cos(elongation));
    s.moon.phaseAngle = std::abs(s.moon.phaseAngle);
    s.moonIlluminatedFraction = (1 + std::cos(s.moon.phaseAngle)) / 2;
    const f64 i = s.moon.phaseAngle / kDeg;
    s.moon.magnitude = -12.73 + 0.026 * i + 4e-9 * i * i * i * i + 5 * std::log10(mp.distKm / 384400.0);
    s.moonIlluminance = std::pow(10.0, -0.4 * (s.moon.magnitude - kSunMagnitude));

    // Planets (phase angle from the sun–planet–earth triangle)
    for (u32 k = 0; k < kPlanetCount; ++k) {
        f64 r = 0;
        const glm::dvec3 g = planetGeocentricICRS(Planet(k), T, &r);
        const f64 delta = glm::length(g);
        const f64 cosPhase = std::clamp((r * r + delta * delta - std::pow(sunDistanceAu(T), 2)) / (2 * r * delta), -1.0, 1.0);
        BodyState& b = s.planets[k];
        b.enu = glm::normalize(s.icrsToEnu * g);
        b.distanceKm = delta * kAuKm;
        b.phaseAngle = std::acos(cosPhase);
        b.magnitude = planetMagnitude(Planet(k), r, delta, b.phaseAngle / kDeg);
    }
    return s;
}

} // namespace phx::astro
