#include "engine/world/TreeMeshes.hpp"

#include <cstdio>
#include <cstring>

namespace phx {

const char* treeAssetName(u32 species) {
    static const char* kNames[] = {"fir_tree_01", "island_tree_01", "tree_small_02", "fern_02",
                                   "rock_moss_set_01", "boulder_01", "grass_medium_01",
                                   "shrub_02",    "searsia_lucida"};
    return species < 9 ? kNames[species] : "";
}

bool readTreeDims(u32 species, const std::string& dataDir, f32& height, f32& radius) {
    const std::string path = dataDir + "/trees/species" + std::to_string(species) + ".phxtree";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    f32 hr[2];
    const bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "PHXTREE3", 8) == 0 && std::fread(hr, 1, 8, f) == 8;
    std::fclose(f);
    if (ok) height = hr[0], radius = hr[1];
    return ok;
}

bool loadTreeSpecies(u32 species, const std::string& dataDir, const std::string& assetDir, TreeSpeciesData& out) {
    const std::string path = dataDir + "/trees/species" + std::to_string(species) + ".phxtree";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto rd = [&](void* p, std::size_t n) { return std::fread(p, 1, n, f) == n; };
    char magic[8];
    f32 hr[2];
    u32 mc = 0;
    bool ok = rd(magic, 8) && std::memcmp(magic, "PHXTREE3", 8) == 0 && rd(hr, 8) && rd(&mc, 4) && mc <= kMaxTreeMaterials;
    out.height = hr[0], out.radius = hr[1];
    const std::string base = assetDir + "/" + treeAssetName(species) + "/";
    for (u32 i = 0; ok && i < mc; ++i) {
        u32 foliage = 0;
        char paths[3][160];
        ok = rd(&foliage, 4) && rd(paths, sizeof(paths));
        TreeMaterial m;
        m.foliage = foliage != 0;
        m.albedo = paths[0][0] ? base + paths[0] : "";
        m.normal = paths[1][0] ? base + paths[1] : "";
        m.arm = paths[2][0] ? base + paths[2] : "";
        out.materials.push_back(m);
    }
    u32 meshCount = 0;
    ok = ok && rd(&meshCount, 4) && meshCount == kTreeMeshLods + 1;
    for (u32 i = 0; ok && i < meshCount; ++i) {
        TreeMesh& m = i < kTreeMeshLods ? out.lods[i] : out.full;
        u32 vc = 0, ic = 0;
        ok = rd(&vc, 4) && rd(&ic, 4);
        if (!ok) break;
        m.vertices.resize(vc);
        m.indices.resize(ic);
        ok = rd(m.vertices.data(), std::size_t(vc) * sizeof(TreeVertex)) && rd(m.indices.data(), std::size_t(ic) * 4);
    }
    std::fclose(f);
    return ok;
}

} // namespace phx
