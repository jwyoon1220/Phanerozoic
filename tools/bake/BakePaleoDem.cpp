// Bakes a Scotese & Wright (2018) PALEOMAP PaleoDEM (netCDF-4/HDF5, 1°×1°, CC BY 4.0)
// into a raw float grid: magic "PHXDEM01", u32 width, u32 height, f32 lon0, lat0, dlon, dlat,
// then width*height f32 elevations (metres, row-major from lat0).
// Usage: phx_bake_paleodem <in.nc> <out.bin> [probeLat probeLon]...
#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <paleodem.nc> <out.bin> [lat lon ...]\n", argv[0]);
        return 2;
    }
    const hid_t file = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 1;
    auto readVec = [&](const char* name, hid_t type, std::vector<hsize_t>& dims, void* out, std::size_t cap) {
        const hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
        if (ds < 0) return false;
        const hid_t sp = H5Dget_space(ds);
        const int nd = H5Sget_simple_extent_ndims(sp);
        dims.resize(std::size_t(nd));
        H5Sget_simple_extent_dims(sp, dims.data(), nullptr);
        std::size_t n = 1;
        for (auto d : dims) n *= std::size_t(d);
        bool ok = n <= cap && H5Dread(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0;
        H5Sclose(sp);
        H5Dclose(ds);
        return ok;
    };
    std::vector<hsize_t> dLat, dLon, dZ;
    std::vector<double> lat(4096), lon(4096);
    if (!readVec("lat", H5T_NATIVE_DOUBLE, dLat, lat.data(), lat.size()) ||
        !readVec("lon", H5T_NATIVE_DOUBLE, dLon, lon.data(), lon.size())) {
        std::fprintf(stderr, "missing lat/lon\n");
        return 1;
    }
    const std::size_t h = dLat[0], w = dLon[0];
    std::vector<float> z(w * h);
    if (!readVec("z", H5T_NATIVE_FLOAT, dZ, z.data(), z.size()) || dZ[0] != h || dZ[1] != w) {
        std::fprintf(stderr, "missing or mismatched z\n");
        return 1;
    }
    H5Fclose(file);

    float zmin = 1e9f, zmax = -1e9f;
    for (float v : z) zmin = std::fmin(zmin, v), zmax = std::fmax(zmax, v);
    const float hdr[4] = {float(lon[0]), float(lat[0]), float(lon[1] - lon[0]), float(lat[1] - lat[0])};
    const unsigned dims[2] = {unsigned(w), unsigned(h)};
    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) return 1;
    std::fwrite("PHXDEM01", 1, 8, out);
    std::fwrite(dims, 4, 2, out);
    std::fwrite(hdr, 4, 4, out);
    std::fwrite(z.data(), 4, z.size(), out);
    std::fclose(out);
    std::printf("baked %zux%zu grid, lon %.1f..%.1f, lat %.1f..%.1f, elevation %.0f..%.0f m\n", w, h, lon[0],
                lon[w - 1], lat[0], lat[h - 1], zmin, zmax);
    for (int i = 3; i + 1 < argc; i += 2) {
        const double qlat = std::atof(argv[i]), qlon = std::atof(argv[i + 1]);
        const std::size_t r = std::size_t(std::lround(qlat - lat[0])), c = std::size_t(std::lround(qlon - lon[0]));
        std::printf("  probe %.2f,%.2f -> %.0f m\n", qlat, qlon, z[r * w + c]);
    }
    return 0;
}
