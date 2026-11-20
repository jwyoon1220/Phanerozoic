// Astronomy validation against worked examples in Meeus, "Astronomical Algorithms" 2nd ed.
// Plan M2 completion criterion: angular error < 1' (Moon: a few ').
#include "engine/astro/Astro.hpp"

#include <cmath>
#include <cstdio>

using namespace phx;
using namespace phx::astro;

static int gFailures = 0;

static void check(const char* what, f64 got, f64 want, f64 tol, const char* unit) {
    const f64 err = std::abs(got - want);
    const bool ok = err <= tol;
    std::printf("  %-44s got %.6f want %.6f  err %.3g %s %s\n", what, got, want, err, unit, ok ? "ok" : "FAIL");
    if (!ok) ++gFailures;
}

static f64 hms(f64 h, f64 m, f64 s) { return (h + m / 60 + s / 3600) * 15.0; }
static f64 dms(f64 d, f64 m, f64 s) { return d < 0 || std::signbit(d) ? d - m / 60 - s / 3600 : d + m / 60 + s / 3600; }

static void raDec(const glm::dvec3& v, f64& raDeg, f64& decDeg) {
    raDeg = std::atan2(v.y, v.x) / kDeg;
    if (raDeg < 0) raDeg += 360;
    decDeg = std::asin(v.z / glm::length(v)) / kDeg;
}

int main() {
    std::printf("Julian date / calendar\n");
    check("JD 1957 Oct 4.81 (7.a)", julianDate({1957, 10, 4, 19, 26, 24.0}), 2436116.31, 1e-6, "d");
    check("JD of J2000.0 epoch", julianDate({2000, 1, 1, 12}), 2451545.0, 1e-9, "d");
    {
        const CalendarDate c = calendarFromJulian(2436116.31);
        check("calendar day of JD 2436116.31", c.day + (c.hour + c.minute / 60.0 + c.second / 3600) / 24, 4.81, 1e-6, "d");
    }

    std::printf("Sidereal time / nutation (12.a, 22.a)\n");
    {
        const f64 jd = julianDate({1987, 4, 10});
        check("GMST 1987-04-10 0h UT", greenwichMeanSiderealTime(jd) / kDeg, hms(13, 10, 46.3668), 1e-5, "deg");
        const Nutation n = nutation((jd - kJ2000) / 36525.0);
        check("nutation dPsi", n.dPsi / kArcsec, -3.788, 0.5, "arcsec");
        check("nutation dEps", n.dEps / kArcsec, 9.443, 0.2, "arcsec");
        // 12.a apparent: 13h10m46.1351s; feed TT = UT so the nutation argument matches the book.
        const Instant t{jd, jd};
        check("GAST 1987-04-10 0h UT", greenwichApparentSiderealTime(t) / kDeg, hms(13, 10, 46.1351), 0.5 / 3600.0,
              "deg");
    }

    std::printf("Precession + proper motion, theta Persei (21.b)\n");
    {
        const f64 ra0 = hms(2, 44, 11.986) * kDeg, dec0 = dms(49, 13, 42.48) * kDeg;
        const f64 jd = 2462088.69;
        const f64 years = (jd - kJ2000) / 365.25;
        const f64 ra = ra0 + 0.03425 * 15 * kArcsec * years; // proper motion in RA: s/yr
        const f64 dec = dec0 - 0.0895 * kArcsec * years;
        const glm::dvec3 v{std::cos(dec) * std::cos(ra), std::cos(dec) * std::sin(ra), std::sin(dec)};
        f64 r, d;
        raDec(precessionMatrix((jd - kJ2000) / 36525.0) * v, r, d);
        check("RA  (mean of date)", r, hms(2, 46, 11.331), 1.0 / 3600, "deg");
        check("Dec (mean of date)", d, dms(49, 20, 54.54), 1.0 / 3600, "deg");
    }

    std::printf("Sun 1992 Oct 13.0 TD (25.a)\n");
    {
        const f64 T = (2448908.5 - kJ2000) / 36525.0;
        check("apparent longitude", sunApparentLongitude(T) / kDeg, 199.90895, 0.003, "deg");
        f64 r, d;
        raDec(sunEquatorialTOD(T), r, d);
        check("apparent RA", r, 198.38083, 0.005, "deg");
        check("apparent Dec", d, -7.78507, 0.005, "deg");
    }

    std::printf("Moon 1992 Apr 12.0 TD (47.a) — truncated series\n");
    {
        const f64 T = (2448724.5 - kJ2000) / 36525.0;
        const EclipticPosition m = moonEcliptic(T);
        check("geocentric longitude", m.lon / kDeg, 133.162655, 0.05, "deg");
        check("geocentric latitude", m.lat / kDeg, -3.229126, 0.03, "deg");
        check("distance", m.distKm, 368409.7, 300, "km");
    }

    std::printf("Venus 1992 Dec 20.0 TD (33.a) — Keplerian elements\n");
    {
        const f64 T = (2448976.5 - kJ2000) / 36525.0;
        const glm::dvec3 tod = nutationMatrix(T) * precessionMatrix(T) * planetGeocentricICRS(Planet::Venus, T);
        f64 r, d;
        raDec(tod, r, d);
        check("apparent RA", r, hms(21, 4, 41.454), 2.0 / 60, "deg");
        check("apparent Dec", d, dms(-18, 53, 16.84), 2.0 / 60, "deg");
    }

    std::printf("Horizon frame sanity\n");
    {
        // The celestial pole must sit at altitude = latitude, due north.
        const Observer obs{34.4782 * kDeg, 110.0844 * kDeg, 2154};
        const glm::dmat3 m = todToEnu(obs, 1.234);
        const glm::dvec3 pole = m * glm::dvec3(0, 0, 1);
        check("pole altitude", std::asin(pole.z) / kDeg, 34.4782, 1e-9, "deg");
        check("pole azimuth (east comp.)", pole.x, 0.0, 1e-12, "");
        // A star on the meridian (RA = LAST) culminates due south at alt 90-lat+dec.
        const glm::dvec3 star = m * glm::dvec3(std::cos(1.234), std::sin(1.234), 0); // dec 0
        check("equator star altitude on meridian", std::asin(star.z) / kDeg, 90 - 34.4782, 1e-9, "deg");
        check("refraction at horizon (~34')", refraction(0) / kDeg * 60, 34.5, 0.6, "arcmin");
    }

    if (gFailures) {
        std::fprintf(stderr, "%d astro check(s) failed\n", gFailures);
        return 1;
    }
    std::printf("all astro tests passed\n");
    return 0;
}
