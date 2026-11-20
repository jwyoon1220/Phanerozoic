#pragma once
// Tree species baked from Poly Haven assets (tools/bake/BakeTree.cpp, CC0 sources):
//   0 Metasequoia  ← fir_tree_01      (conical conifer; rust autumn tint in the shader)
//   1 Lauraceae    ← island_tree_01   (broadleaf evergreen)
//   2 Ginkgo       ← tree_small_02    (upright broadleaf; golden autumn tint)
// These are stand-ins chosen by silhouette and foliage: no CC0 models of the Eocene
// genera exist. Each species has two near-field LODs and the full-resolution mesh,
// from which octahedral impostors are baked at start-up.
#include "engine/core/Types.hpp"

#include <array>
#include <string>
#include <vector>

namespace phx {

enum class TreeSpecies : u32 { Metasequoia = 0, Laurel = 1, Ginkgo = 2, Count };
inline constexpr u32 kTreeSpeciesCount = u32(TreeSpecies::Count); // species with impostors
// All instanced species: trees, then understorey and rocks.
//   3 fern (fern_02), 4 mossy rock (rock_moss_set_01), 5 boulder (boulder_01),
//   6 grass clumps (grass_medium_01), 7 shrub (shrub_02), 8 dryland tree (searsia_lucida)
// (Cliffs are terrain: escarpments in the height function, not mesh pieces.)
inline constexpr u32 kSpeciesCount = 9;
inline constexpr u32 kTreeMeshLods = 3; // + impostor
inline constexpr u32 kMaxTreeMaterials = 8;

struct TreeVertex {
    f32 pos[3];  // metres, y up, trunk base at origin
    u32 normal;  // octahedral 16:16 (y-up convention)
    u32 uv;      // half2
    u32 extra;   // material 0-7 | sway 8-15 | foliage bit 16
};
static_assert(sizeof(TreeVertex) == 24);

struct TreeMesh {
    std::vector<TreeVertex> vertices;
    std::vector<u32> indices;
};

struct TreeMaterial {
    bool foliage = false;
    std::string albedo, normal, arm; // absolute paths
};

struct TreeSpeciesData {
    f32 height = 0, radius = 0;
    std::vector<TreeMaterial> materials;
    std::array<TreeMesh, kTreeMeshLods> lods;
    TreeMesh full; // impostor bake source
};

// Loads data/trees/species<N>.phxtree; textures resolve under the Poly Haven asset folder.
bool loadTreeSpecies(u32 species, const std::string& dataDir, const std::string& assetDir, TreeSpeciesData& out);
const char* treeAssetName(u32 species);
// Reads only the header (height, crown/footprint radius) of a baked species file.
bool readTreeDims(u32 species, const std::string& dataDir, f32& height, f32& radius);

} // namespace phx
