// Bakes a Poly Haven tree (glTF, film-resolution geometry) into a real-time species file.
//
//   phx_bake_tree <in.gltf> <out.phxtree> <targetHeightM> [--crop xmin xmax] [--lod0 tris] [--lod1 tris]
//
// Output: LOD0 / LOD1 meshes for the near field plus the full mesh, from which the
// runtime bakes octahedral impostors (so the distant forest shows the full-detail asset).
//  * wood (trunk, branches): meshoptimizer simplification to the triangle budget
//  * foliage (modelled leaves): leaves are connected components; a deterministic subset
//    is kept and each kept leaf is enlarged by 1/sqrt(fraction) so crown coverage holds
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <meshoptimizer.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct Vertex {
    glm::vec3 pos, nrm;
    glm::vec2 uv;
    uint32_t material;
};

struct Material {
    std::string name;
    bool foliage = false;
    std::string diff, nor, arm; // texture paths relative to the glTF
};

struct Prim {
    uint32_t material;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

// File format (little endian)
//   "PHXTREE3", f32 height, f32 radius, u32 materialCount,
//   per material: u32 foliage, char[3][160] texture paths (relative to the file's directory)
//   u32 meshCount (4: LOD0, LOD1, LOD2, full), per mesh: u32 vertexCount, u32 indexCount, PackedVertex[], u32[]
struct PackedVertex {
    float pos[3];
    uint32_t normal; // octahedral 16:16, y-up
    uint32_t uv;     // half2
    uint32_t extra;  // material 0-7 | sway 8-15 | foliage bit 16
};
static_assert(sizeof(PackedVertex) == 24);

static uint32_t packNormal(glm::vec3 n) {
    n = glm::normalize(n);
    glm::vec2 p = glm::vec2(n.x, n.z) / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    if (n.y < 0) p = (1.0f - glm::abs(glm::vec2(p.y, p.x))) * glm::vec2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
    return uint32_t(std::lround((p.x * 0.5f + 0.5f) * 65535.f)) | (uint32_t(std::lround((p.y * 0.5f + 0.5f) * 65535.f)) << 16);
}

static uint16_t toHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return uint16_t(sign);
    if (e >= 31) return uint16_t(sign | 0x7C00);
    return uint16_t(sign | (uint32_t(e) << 10) | (m >> 13));
}

static std::string textureOf(const cgltf_texture_view& v) {
    return v.texture && v.texture->image && v.texture->image->uri ? v.texture->image->uri : "";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <in.gltf> <out.phxtree> <heightM> [--crop xmin xmax] [--lod0 n] [--lod1 n]\n", argv[0]);
        return 2;
    }
    const float targetHeight = std::strtof(argv[3], nullptr);
    float cropMin = -1e30f, cropMax = 1e30f;
    size_t lodTarget[3] = {450000, 120000, 20000};
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--crop") && i + 2 < argc) cropMin = std::strtof(argv[++i], nullptr), cropMax = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--lod0") && i + 1 < argc) lodTarget[0] = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--lod1") && i + 1 < argc) lodTarget[1] = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--lod2") && i + 1 < argc) lodTarget[2] = std::strtoul(argv[++i], nullptr, 10);
    }

    cgltf_options opt{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&opt, argv[1], &data) != cgltf_result_success || cgltf_load_buffers(&opt, data, argv[1]) != cgltf_result_success) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }

    std::vector<Material> materials;
    for (size_t i = 0; i < data->materials_count; ++i) {
        const cgltf_material& m = data->materials[i];
        Material out;
        out.name = m.name ? m.name : "";
        std::string lower = out.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        out.foliage = lower.find("lea") != std::string::npos || lower.find("needle") != std::string::npos ||
                      lower.find("twig") != std::string::npos || lower.find("frond") != std::string::npos ||
                      lower.find("grass") != std::string::npos || lower.find("shrub") != std::string::npos ||
                      m.alpha_mode != cgltf_alpha_mode_opaque;
        out.diff = textureOf(m.pbr_metallic_roughness.base_color_texture);
        out.arm = textureOf(m.pbr_metallic_roughness.metallic_roughness_texture);
        out.nor = textureOf(m.normal_texture);
        if (out.nor.empty() && !out.diff.empty()) { // Poly Haven naming: *_diff_* → *_nor_gl_*
            out.nor = out.diff;
            const auto p = out.nor.find("_diff_");
            if (p != std::string::npos) out.nor.replace(p, 6, "_nor_gl_");
        }
        materials.push_back(out);
    }

    // Gather all primitives in world space.
    std::vector<Prim> prims(materials.size());
    for (uint32_t i = 0; i < prims.size(); ++i) prims[i].material = i;
    for (size_t n = 0; n < data->nodes_count; ++n) {
        const cgltf_node& node = data->nodes[n];
        if (!node.mesh) continue;
        float m[16];
        cgltf_node_transform_world(&node, m);
        const glm::mat4 world = glm::make_mat4(m);
        const glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(world)));
        for (size_t p = 0; p < node.mesh->primitives_count; ++p) {
            const cgltf_primitive& prim = node.mesh->primitives[p];
            const cgltf_accessor *pos = nullptr, *nrm = nullptr, *uv = nullptr;
            for (size_t a = 0; a < prim.attributes_count; ++a) {
                const auto& at = prim.attributes[a];
                if (at.type == cgltf_attribute_type_position) pos = at.data;
                if (at.type == cgltf_attribute_type_normal) nrm = at.data;
                if (at.type == cgltf_attribute_type_texcoord && at.index == 0) uv = at.data;
            }
            if (!pos || !prim.indices) continue;
            const uint32_t mat = prim.material ? uint32_t(prim.material - data->materials) : 0;
            Prim& dst = prims[mat];
            const uint32_t base = uint32_t(dst.vertices.size());
            std::vector<uint32_t> remap(pos->count, UINT32_MAX);
            // crop: keep triangles whose centroid x lies in [cropMin, cropMax]
            for (size_t t = 0; t + 2 < prim.indices->count; t += 3) {
                uint32_t idx[3];
                glm::vec3 c(0);
                glm::vec3 wp[3];
                for (int k = 0; k < 3; ++k) {
                    idx[k] = uint32_t(cgltf_accessor_read_index(prim.indices, t + k));
                    float v[3];
                    cgltf_accessor_read_float(pos, idx[k], v, 3);
                    wp[k] = glm::vec3(world * glm::vec4(v[0], v[1], v[2], 1));
                    c += wp[k] / 3.0f;
                }
                if (c.x < cropMin || c.x > cropMax) continue;
                for (int k = 0; k < 3; ++k) {
                    if (remap[idx[k]] == UINT32_MAX) {
                        Vertex vx{};
                        vx.pos = wp[k];
                        float v[3] = {0, 1, 0}, t2[2] = {0, 0};
                        if (nrm) cgltf_accessor_read_float(nrm, idx[k], v, 3);
                        if (uv) cgltf_accessor_read_float(uv, idx[k], t2, 2);
                        vx.nrm = glm::normalize(normalMat * glm::vec3(v[0], v[1], v[2]));
                        vx.uv = glm::vec2(t2[0], t2[1]);
                        vx.material = mat;
                        remap[idx[k]] = uint32_t(dst.vertices.size());
                        dst.vertices.push_back(vx);
                    }
                    dst.indices.push_back(remap[idx[k]]);
                }
            }
            (void)base;
        }
    }
    cgltf_free(data);

    // Normalise: trunk base to origin (x/z centre of the lowest wood vertices), scale to height.
    glm::vec3 lo(1e30f), hi(-1e30f);
    for (auto& p : prims)
        for (auto& v : p.vertices) lo = glm::min(lo, v.pos), hi = glm::max(hi, v.pos);
    glm::vec3 baseCentre(0);
    int baseCount = 0;
    for (auto& p : prims)
        if (!materials[p.material].foliage)
            for (auto& v : p.vertices)
                if (v.pos.y < lo.y + (hi.y - lo.y) * 0.03f) baseCentre += v.pos, ++baseCount;
    if (baseCount) baseCentre /= float(baseCount);
    baseCentre.y = lo.y;
    const float scale = targetHeight / (hi.y - lo.y);
    float radius = 0;
    for (auto& p : prims)
        for (auto& v : p.vertices) {
            v.pos = (v.pos - baseCentre) * scale;
            radius = std::max(radius, std::sqrt(v.pos.x * v.pos.x + v.pos.z * v.pos.z));
        }

    size_t woodTris = 0, leafTris = 0;
    for (auto& p : prims) (materials[p.material].foliage ? leafTris : woodTris) += p.indices.size() / 3;
    std::printf("source: %zu wood + %zu foliage triangles, %.1f m tall, crown radius %.1f m, %zu materials\n", woodTris,
                leafTris, targetHeight, radius, materials.size());

    // Leaf components per foliage primitive (welded positions).
    struct Components {
        std::vector<uint32_t> triComponent;
        uint32_t count = 0;
    };
    std::vector<Components> comps(prims.size());
    for (size_t pi = 0; pi < prims.size(); ++pi) {
        Prim& p = prims[pi];
        if (!materials[p.material].foliage || p.indices.empty()) continue;
        std::vector<uint32_t> weld(p.vertices.size());
        meshopt_generateVertexRemap(weld.data(), p.indices.data(), p.indices.size(), &p.vertices[0].pos.x, p.vertices.size(), sizeof(Vertex));
        std::vector<uint32_t> parent(p.vertices.size());
        std::iota(parent.begin(), parent.end(), 0u);
        auto find = [&](uint32_t x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
        for (size_t t = 0; t < p.indices.size(); t += 3) {
            const uint32_t a = find(weld[p.indices[t]]), b = find(weld[p.indices[t + 1]]), c = find(weld[p.indices[t + 2]]);
            parent[b] = a;
            parent[find(c)] = find(a);
        }
        std::vector<uint32_t> id(p.vertices.size(), UINT32_MAX);
        Components& cc = comps[pi];
        for (size_t t = 0; t < p.indices.size(); t += 3) {
            const uint32_t r = find(weld[p.indices[t]]);
            if (id[r] == UINT32_MAX) id[r] = cc.count++;
            cc.triComponent.push_back(id[r]);
        }
        std::printf("  foliage material '%s': %u leaves\n", materials[p.material].name.c_str(), cc.count);
    }

    auto buildLod = [&](size_t budget) {
        // Split the budget: foliage gets 70 % (or all if there is no wood), wood the rest.
        const double woodShare = leafTris ? 0.3 : 1.0;
        std::vector<PackedVertex> outV;
        std::vector<uint32_t> outI;
        for (size_t pi = 0; pi < prims.size(); ++pi) {
            Prim& p = prims[pi];
            if (p.indices.empty()) continue;
            const bool foliage = materials[p.material].foliage;
            std::vector<uint32_t> idx;
            std::vector<Vertex> verts = p.vertices;
            if (!foliage) {
                const size_t target = std::max<size_t>(3 * 64, size_t(budget * woodShare * (double(p.indices.size() / 3) / std::max<size_t>(woodTris, 1))) * 3);
                idx.resize(p.indices.size());
                float err = 0;
                idx.resize(meshopt_simplify(idx.data(), p.indices.data(), p.indices.size(), &p.vertices[0].pos.x, p.vertices.size(),
                                            sizeof(Vertex), std::min(target, p.indices.size()), 0.05f, 0, &err));
                if (idx.size() > target * 1.3) // topology-locked: fall back to clustering
                    idx.resize(meshopt_simplifySloppy(idx.data(), p.indices.data(), p.indices.size(), &p.vertices[0].pos.x,
                                                      p.vertices.size(), sizeof(Vertex), target, 0.2f, &err));
            } else {
                const Components& cc = comps[pi];
                const double share = (1.0 - woodShare) * double(p.indices.size() / 3) / std::max<size_t>(leafTris, 1);
                const double keep = std::min(1.0, budget * share / double(p.indices.size() / 3));
                const float enlarge = float(std::min(3.0, 1.0 / std::sqrt(keep)));
                // Leaf centroids for enlargement.
                std::vector<glm::vec3> centre(cc.count, glm::vec3(0));
                std::vector<uint32_t> cnt(cc.count, 0);
                for (size_t t = 0; t < p.indices.size(); t += 3)
                    for (int k = 0; k < 3; ++k) centre[cc.triComponent[t / 3]] += p.vertices[p.indices[t + k]].pos, ++cnt[cc.triComponent[t / 3]];
                for (uint32_t c = 0; c < cc.count; ++c) centre[c] /= float(std::max(cnt[c], 1u));
                std::vector<char> kept(cc.count);
                for (uint32_t c = 0; c < cc.count; ++c) kept[c] = double((c * 2654435761u) & 0xFFFFFF) / 16777216.0 < keep;
                std::vector<char> moved(verts.size(), 0);
                for (size_t t = 0; t < p.indices.size(); t += 3) {
                    const uint32_t c = cc.triComponent[t / 3];
                    if (!kept[c]) continue;
                    for (int k = 0; k < 3; ++k) {
                        const uint32_t v = p.indices[t + k];
                        if (!moved[v]) verts[v].pos = centre[c] + (verts[v].pos - centre[c]) * enlarge, moved[v] = 1;
                        idx.push_back(v);
                    }
                }
            }
            // Compact and append.
            std::vector<uint32_t> remap(verts.size());
            const size_t unique = meshopt_generateVertexRemap(remap.data(), idx.data(), idx.size(), verts.data(), verts.size(), sizeof(Vertex));
            std::vector<Vertex> cv(unique);
            meshopt_remapVertexBuffer(cv.data(), verts.data(), verts.size(), sizeof(Vertex), remap.data());
            meshopt_remapIndexBuffer(idx.data(), idx.data(), idx.size(), remap.data());
            meshopt_optimizeVertexCache(idx.data(), idx.data(), idx.size(), unique);
            const uint32_t base = uint32_t(outV.size());
            for (const Vertex& v : cv) {
                PackedVertex pv{};
                pv.pos[0] = v.pos.x, pv.pos[1] = v.pos.y, pv.pos[2] = v.pos.z;
                pv.normal = packNormal(v.nrm);
                pv.uv = uint32_t(toHalf(v.uv.x)) | (uint32_t(toHalf(v.uv.y)) << 16);
                const float h = std::clamp(v.pos.y / targetHeight, 0.0f, 1.0f);
                const float out = std::clamp(std::sqrt(v.pos.x * v.pos.x + v.pos.z * v.pos.z) / std::max(radius, 0.1f), 0.0f, 1.0f);
                const float sway = std::clamp(std::pow(h, 1.5f) * (foliage ? 1.0f : 0.6f) + out * 0.3f * h, 0.0f, 1.0f);
                pv.extra = v.material | (uint32_t(sway * 255.f) << 8) | (foliage ? (1u << 16) : 0u);
                outV.push_back(pv);
            }
            for (uint32_t i : idx) outI.push_back(base + i);
        }
        return std::make_pair(std::move(outV), std::move(outI));
    };

    auto lod0 = buildLod(lodTarget[0]);
    auto lod1 = buildLod(lodTarget[1]);
    auto lod2 = buildLod(lodTarget[2]);
    auto full = buildLod(SIZE_MAX / 4);
    std::printf("LOD0 %zu, LOD1 %zu, LOD2 %zu, full %zu tris\n", lod0.second.size() / 3, lod1.second.size() / 3,
                lod2.second.size() / 3, full.second.size() / 3);

    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) return 1;
    std::fwrite("PHXTREE3", 1, 8, out);
    const float hr[2] = {targetHeight, radius};
    std::fwrite(hr, 4, 2, out);
    const uint32_t mc = uint32_t(materials.size());
    std::fwrite(&mc, 4, 1, out);
    for (const Material& m : materials) {
        const uint32_t fol = m.foliage;
        std::fwrite(&fol, 4, 1, out);
        for (const std::string* s : {&m.diff, &m.nor, &m.arm}) {
            char buf[160] = {};
            std::snprintf(buf, sizeof(buf), "%s", s->c_str());
            std::fwrite(buf, 1, sizeof(buf), out);
        }
    }
    const uint32_t meshCount = 4;
    std::fwrite(&meshCount, 4, 1, out);
    for (auto* m : {&lod0, &lod1, &lod2, &full}) {
        const uint32_t vc = uint32_t(m->first.size()), ic = uint32_t(m->second.size());
        std::fwrite(&vc, 4, 1, out);
        std::fwrite(&ic, 4, 1, out);
        std::fwrite(m->first.data(), sizeof(PackedVertex), vc, out);
        std::fwrite(m->second.data(), 4, ic, out);
    }
    std::fclose(out);
    return 0;
}
