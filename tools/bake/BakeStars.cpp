// Bakes the HYG database (CSV, CC BY-SA 4.0, astronexus/HYG-Database) into stars.bin.
// Usage: phx_bake_stars <hygdata.csv> <out/stars.bin> [limitMag=8.5]
#include "engine/astro/StarCatalog.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace phx;

static std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') cur += '"', ++i;
            else if (c == '"') quoted = false;
            else cur += c;
        } else if (c == '"') quoted = true;
        else if (c == ',') fields.push_back(std::move(cur)), cur.clear();
        else if (c != '\r') cur += c;
    }
    fields.push_back(std::move(cur));
    return fields;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <hygdata.csv> <stars.bin> [limitMag]\n", argv[0]);
        return 2;
    }
    const double limit = argc > 3 ? std::atof(argv[3]) : 8.5;
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::string line;
    std::getline(in, line);
    const auto header = splitCsv(line);
    std::unordered_map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
    for (const char* need : {"id", "rarad", "decrad", "pmrarad", "pmdecrad", "mag", "ci", "dist"})
        if (!col.count(need)) {
            std::fprintf(stderr, "missing column %s\n", need);
            return 1;
        }

    std::vector<astro::StarRecord> stars;
    std::size_t skipped = 0;
    while (std::getline(in, line)) {
        const auto f = splitCsv(line);
        if (f.size() < header.size()) { ++skipped; continue; }
        if (f[col["id"]] == "0") continue; // the Sun
        const double mag = std::atof(f[col["mag"]].c_str());
        if (mag > limit) continue;
        const double ra = std::atof(f[col["rarad"]].c_str()), dec = std::atof(f[col["decrad"]].c_str());
        const double pmra = std::atof(f[col["pmrarad"]].c_str());   // μα·cosδ, rad/yr
        const double pmdec = std::atof(f[col["pmdecrad"]].c_str()); // rad/yr
        const std::string& ci = f[col["ci"]];
        astro::StarRecord r{};
        const double cr = std::cos(ra), sr = std::sin(ra), cd = std::cos(dec), sd = std::sin(dec);
        r.dir[0] = float(cd * cr), r.dir[1] = float(cd * sr), r.dir[2] = float(sd);
        // d(dir)/dt = μα* · ê_α + μδ · ê_δ
        r.dDirDt[0] = float(-sr * pmra - sd * cr * pmdec);
        r.dDirDt[1] = float(cr * pmra - sd * sr * pmdec);
        r.dDirDt[2] = float(cd * pmdec);
        r.vmag = float(mag);
        r.bv = ci.empty() ? 0.6f : float(std::atof(ci.c_str()));
        stars.push_back(r);
    }

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    const auto count = std::uint32_t(stars.size());
    out.write(astro::kStarFileMagic, 8);
    out.write(reinterpret_cast<const char*>(&count), 4);
    out.write(reinterpret_cast<const char*>(stars.data()), std::streamsize(stars.size() * sizeof(astro::StarRecord)));
    std::printf("baked %u stars (V <= %.1f) to %s, %zu malformed rows skipped\n", count, limit, argv[2], skipped);
    return 0;
}
