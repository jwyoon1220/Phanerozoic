#include "engine/game/PhysicsWorld.hpp"

#include "engine/core/Log.hpp"
#include "engine/world/CubeSphere.hpp"
#include "engine/world/HeightField.hpp"
#include "engine/world/TreeMeshes.hpp"

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <unordered_set>

#ifndef PHX_DATA_DIR
#define PHX_DATA_DIR "data"
#endif
#ifndef PHX_ASSET_DIR
#define PHX_ASSET_DIR "data/raw/polyhaven"
#endif

namespace phx {
namespace {
using namespace JPH::literals;

void joltTrace(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    log::info("jolt: {}", buf);
}
#ifdef JPH_ENABLE_ASSERTS
bool joltAssert(const char* expr, const char* msg, const char* file, JPH::uint line) {
    log::error("jolt assert {} ({}) at {}:{}", expr, msg ? msg : "", file, line);
    return false; // don't break
}
#endif

// One-time Jolt runtime setup (allocator, factory, type registry).
void initJolt() {
    static std::once_flag once;
    std::call_once(once, [] {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = joltTrace;
        JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = joltAssert;)
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    });
}

namespace bp {
constexpr JPH::BroadPhaseLayer kStatic(0), kMoving(1);
}

class BroadPhaseLayers final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return 2; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer l) const override {
        return l == physlayers::kStatic ? bp::kStatic : bp::kMoving;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer l) const override { return l == bp::kStatic ? "static" : "moving"; }
#endif
};
class ObjectVsBroadPhase final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::BroadPhaseLayer b) const override {
        return a == physlayers::kMoving || b == bp::kMoving;
    }
};
class ObjectPairs final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        return a == physlayers::kMoving || b == physlayers::kMoving;
    }
};

// Tree trunk radius per species at scale 1 (metres): Metasequoia, Lauraceae, Ginkgo.
constexpr f32 kTrunkRadius[3] = {0.28f, 0.32f, 0.2f};
} // namespace

struct PhysicsWorld::Filters {
    BroadPhaseLayers layers;
    ObjectVsBroadPhase objectVsBroadPhase;
    ObjectPairs pairs;
};

PhysicsWorld::PhysicsWorld(const HeightField& heights, jobs::JobSystem& jobs) : heights_(heights), jobs_(jobs) {
    initJolt();
    temp_ = std::make_unique<JPH::TempAllocatorImpl>(8 * 1024 * 1024);
    filters_ = std::make_unique<Filters>();
    system_ = std::make_unique<JPH::PhysicsSystem>();
    // Two patches coexist briefly (the next one is created while the current is live).
    system_->Init(32768, 0, 32768, 4096, filters_->layers, filters_->objectVsBroadPhase, filters_->pairs);

    for (u32 sp = 0; sp < kSpeciesCount; ++sp)
        if (!readTreeDims(sp, PHX_DATA_DIR, dims_[sp].height, dims_[sp].radius))
            PHX_FATAL("Species {} missing (build target phx_tree_data)", sp);
    // Rocks collide with their LOD2 mesh (the rendered LOD0 differs by centimetres).
    for (u32 sp : {4u, 5u}) {
        TreeSpeciesData data;
        if (!loadTreeSpecies(sp, PHX_DATA_DIR, PHX_ASSET_DIR, data)) PHX_FATAL("Species {} missing", sp);
        const TreeMesh& m = data.lods[2];
        JPH::VertexList verts;
        verts.reserve(m.vertices.size());
        for (const TreeVertex& v : m.vertices) verts.push_back(JPH::Float3(v.pos[0], v.pos[1], v.pos[2]));
        JPH::IndexedTriangleList tris;
        for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3)
            tris.push_back(JPH::IndexedTriangle(m.indices[i], m.indices[i + 1], m.indices[i + 2]));
        JPH::MeshShapeSettings settings(std::move(verts), std::move(tris));
        auto result = settings.Create();
        if (result.HasError()) PHX_FATAL("Rock collider {}: {}", sp, result.GetError().c_str());
        rockShapes_[sp] = result.Get();
    }
}

PhysicsWorld::~PhysicsWorld() {
    if (pendingInFlight_) jobs_.wait(pendingCounter_);
    JPH::BodyInterface& bi = system_->GetBodyInterfaceNoLock();
    for (Patch* p : {current_.get(), pending_.get()}) {
        if (!p || p->bodies.empty()) continue;
        if (p == current_.get()) bi.RemoveBodies(p->bodies.data(), int(p->bodies.size()));
        bi.DestroyBodies(p->bodies.data(), int(p->bodies.size()));
    }
}

// ---- frames

JPH::Vec3 PhysicsWorld::localIn(const Frame& f, const glm::dvec3& ecef) {
    const glm::dvec3 enu = f.enu * (ecef - f.origin);
    return JPH::Vec3(f32(enu.x), f32(enu.z), f32(-enu.y));
}
JPH::Vec3 PhysicsWorld::dirIn(const Frame& f, const glm::dvec3& d) {
    const glm::dvec3 enu = f.enu * d;
    return JPH::Vec3(f32(enu.x), f32(enu.z), f32(-enu.y));
}
JPH::Vec3 PhysicsWorld::toLocal(const glm::dvec3& ecef) const { return localIn(current_->frame, ecef); }
JPH::Vec3 PhysicsWorld::dirToLocal(const glm::dvec3& d) const { return dirIn(current_->frame, d); }
glm::dvec3 PhysicsWorld::dirToEcef(JPH::Vec3Arg d) const {
    return glm::transpose(current_->frame.enu) * glm::dvec3(d.GetX(), -f64(d.GetZ()), d.GetY());
}
glm::dvec3 PhysicsWorld::toEcef(JPH::Vec3Arg l) const { return current_->frame.origin + dirToEcef(l); }

// ---- patch management (supervisor)

bool PhysicsWorld::update(const glm::dvec3& focus, bool waitUntilReady) {
    auto groundUnder = [&](const glm::dvec3& p) {
        const glm::dvec3 dir = glm::normalize(p);
        return dir * (kPlanetRadiusM + std::max(heights_.height(dir, 1.2), 0.0));
    };
    if (!pendingInFlight_) {
        bool needed = !current_;
        if (current_) {
            const glm::dvec3 enu = current_->frame.enu * (focus - current_->frame.origin);
            needed = std::hypot(enu.x, enu.y) > kRebaseDistance;
        }
        if (needed) {
            pending_ = std::make_unique<Patch>();
            pending_->frame.origin = groundUnder(focus);
            pending_->frame.enu = ecefToEnuMatrix(pending_->frame.origin);
            pendingInFlight_ = true;
            Patch* p = pending_.get();
            jobs_.submit({.priority = current_ ? jobs::Priority::Normal : jobs::Priority::Critical, .signal = &pendingCounter_},
                         [this, p] { build(*p); });
        }
    }
    if (pendingInFlight_ && waitUntilReady) jobs_.wait(pendingCounter_);
    if (!pendingInFlight_ || !pendingCounter_.done()) return false;

    // Swap: old bodies out, new bodies in (one batch each).
    JPH::BodyInterface& bi = system_->GetBodyInterface();
    if (current_ && !current_->bodies.empty()) {
        bi.RemoveBodies(current_->bodies.data(), int(current_->bodies.size()));
        bi.DestroyBodies(current_->bodies.data(), int(current_->bodies.size()));
    }
    if (!pending_->bodies.empty()) {
        const JPH::BodyInterface::AddState state = bi.AddBodiesPrepare(pending_->bodies.data(), int(pending_->bodies.size()));
        bi.AddBodiesFinalize(pending_->bodies.data(), int(pending_->bodies.size()), state, JPH::EActivation::DontActivate);
    }
    system_->OptimizeBroadPhase();
    current_ = std::move(pending_);
    pendingInFlight_ = false;
    log::info("physics patch: {} colliders, built in {:.1f} ms on a worker", current_->bodies.size(), current_->buildMs);
    return true;
}

std::shared_ptr<const PhysicsWorld::CellData> PhysicsWorld::cell(u64 key) {
    {
        std::lock_guard lock(cacheMutex_);
        if (auto it = cellCache_.find(key); it != cellCache_.end()) return it->second;
    }
    auto data = std::make_shared<CellData>();
    data->center = placeNature(key, heights_, dims_, data->instances);
    std::lock_guard lock(cacheMutex_);
    return cellCache_.emplace(key, std::move(data)).first->second;
}

// ---- patch build (worker job)

void PhysicsWorld::build(Patch& p) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const Frame& f = p.frame;
    const glm::dvec3 E = glm::dvec3(f.enu[0][0], f.enu[1][0], f.enu[2][0]);
    const glm::dvec3 N = glm::dvec3(f.enu[0][1], f.enu[1][1], f.enu[2][1]);
    const glm::dvec3 U = glm::dvec3(f.enu[0][2], f.enu[1][2], f.enu[2][2]);
    JPH::BodyInterface& bi = system_->GetBodyInterface();

    // Heightfield: sample (i, j) at local x = i - c, z = j - c  (east, south).
    constexpr f32 c = f32(kSamples - 1) * 0.5f;
    std::vector<float> samples(std::size_t(kSamples) * kSamples);
    for (u32 j = 0; j < kSamples; ++j)
        for (u32 i = 0; i < kSamples; ++i) {
            const glm::dvec3 q = f.origin + E * f64(f32(i) - c) - N * f64(f32(j) - c);
            const glm::dvec3 dir = glm::normalize(q);
            const glm::dvec3 surface = dir * (kPlanetRadiusM + std::max(heights_.height(dir, 1.2), 0.0));
            samples[std::size_t(j) * kSamples + i] = f32(glm::dot(surface - f.origin, U));
        }
    JPH::HeightFieldShapeSettings hs(samples.data(), JPH::Vec3(-c, 0, -c), JPH::Vec3::sReplicate(1.0f), kSamples);
    hs.mBlockSize = 4;
    if (auto r = hs.Create(); r.IsValid()) {
        JPH::BodyCreationSettings bcs(r.Get(), JPH::RVec3::sZero(), JPH::Quat::sIdentity(), JPH::EMotionType::Static,
                                      physlayers::kStatic);
        if (JPH::Body* b = bi.CreateBody(bcs)) p.bodies.push_back(b->GetID());
    } else {
        log::error("heightfield collider: {}", r.GetError().c_str());
    }

    // Instance colliders from the nature cells overlapping the patch.
    std::unordered_set<u64> keys;
    for (const NatureLayer& L : kNatureLayers) {
        const f64 step = std::min(30.0, cubesphere::edgeLength(L.level) * 0.5);
        for (f64 y = -kColliderRadius; y <= kColliderRadius + 0.1; y += step)
            for (f64 x = -kColliderRadius; x <= kColliderRadius + 0.1; x += step)
                keys.insert(cubesphere::keyAt(glm::normalize(f.origin + E * x + N * y), L.level));
    }
    JPH::Ref<JPH::Shape> trunk[3];
    for (const u64 key : keys) {
        const auto data = cell(key);
        for (const VegInstance& inst : data->instances) {
            const glm::dvec3 pos = data->center + glm::dvec3(inst.pos[0], inst.pos[1], inst.pos[2]);
            const JPH::Vec3 local = localIn(f, pos);
            if (std::abs(local.GetX()) > kColliderRadius || std::abs(local.GetZ()) > kColliderRadius) continue;
            const NatureTransform t = natureTransform(pos, inst.packed);
            if (t.species == 3 || t.species == 6 || t.species == 7) continue; // ferns, grass, shrubs: walk through
            JPH::RefConst<JPH::Shape> shape;
            JPH::Quat rot = JPH::Quat::sIdentity();
            JPH::Vec3 at = local;
            if (t.species < 3 || t.species == 8) {
                const f32 r = (t.species == 8 ? 0.12f : kTrunkRadius[t.species]) * f32(t.scale);
                const f32 halfH = std::min(3.0f, dims_[t.species].height * f32(t.scale) * 0.25f);
                shape = new JPH::CapsuleShape(halfH, r);
                at = local + dirIn(f, t.Y) * (halfH + r - 0.3f);
                rot = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), dirIn(f, t.Y).Normalized());
            } else {
                shape = new JPH::ScaledShape(rockShapes_[t.species], JPH::Vec3::sReplicate(f32(t.scale)));
                const JPH::Mat44 m(JPH::Vec4(dirIn(f, t.X), 0), JPH::Vec4(dirIn(f, t.Y), 0), JPH::Vec4(dirIn(f, t.Z), 0),
                                   JPH::Vec4(0, 0, 0, 1));
                rot = m.GetQuaternion().Normalized();
            }
            JPH::BodyCreationSettings bcs(shape, at, rot, JPH::EMotionType::Static, physlayers::kStatic);
            if (JPH::Body* b = bi.CreateBody(bcs)) p.bodies.push_back(b->GetID());
        }
    }
    {   // keep the cell cache bounded: drop cells no longer near the player
        std::lock_guard lock(cacheMutex_);
        if (cellCache_.size() > 48)
            for (auto it = cellCache_.begin(); it != cellCache_.end();)
                it = keys.count(it->first) ? std::next(it) : cellCache_.erase(it);
    }
    p.buildMs = std::chrono::duration<f64, std::milli>(Clock::now() - t0).count();
}

} // namespace phx
