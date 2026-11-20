#pragma once
// Collision around the player (plan §2.5, M6), on Jolt Physics.
//  * Frame: a local tangent frame anchored on the ground near the player, Jolt-style
//    Y up: x = east, y = up, z = south. Earth curvature over the patch is < 1 mm.
//  * Patch: 128 × 128 height samples at 1 m from the same HeightField as the renderer,
//    plus static colliders for the same instances the renderer draws (NaturePlacement):
//    capsule trunks for trees, LOD2 triangle meshes for rocks. Ferns do not collide.
//  * Built in a background worker job; the supervisor swaps it in atomically (old
//    bodies out, new in, frame re-anchored) when the player moves 24 m from the anchor.
#include "engine/core/Types.hpp"
#include "engine/jobs/JobSystem.hpp"
#include "engine/world/NaturePlacement.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <glm/glm.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace JPH {
class PhysicsSystem;
class TempAllocator;
} // namespace JPH

namespace phx {
class HeightField;

namespace physlayers {
inline constexpr JPH::ObjectLayer kStatic = 0;
inline constexpr JPH::ObjectLayer kMoving = 1;
} // namespace physlayers

class PhysicsWorld {
public:
    static constexpr u32 kSamples = 128;   // heightfield samples per side (1 m apart)
    static constexpr f64 kRebaseDistance = 24.0;
    static constexpr f64 kColliderRadius = 62.0;

    PhysicsWorld(const HeightField& heights, jobs::JobSystem& jobs);
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // Supervisor: keeps a patch around `focus` (feet, ECEF). Swaps in a finished patch;
    // returns true when the local frame changed this call (callers re-anchor).
    bool update(const glm::dvec3& focusEcef, bool waitUntilReady = false);
    [[nodiscard]] bool ready() const { return current_ != nullptr; }

    [[nodiscard]] JPH::Vec3 toLocal(const glm::dvec3& ecef) const;
    [[nodiscard]] glm::dvec3 toEcef(JPH::Vec3Arg local) const;
    [[nodiscard]] JPH::Vec3 dirToLocal(const glm::dvec3& d) const;
    [[nodiscard]] glm::dvec3 dirToEcef(JPH::Vec3Arg d) const;

    [[nodiscard]] JPH::PhysicsSystem& system() { return *system_; }
    [[nodiscard]] JPH::TempAllocator& tempAllocator() { return *temp_; }
    [[nodiscard]] u32 colliderCount() const { return current_ ? u32(current_->bodies.size()) : 0; }
    [[nodiscard]] f64 lastBuildMs() const { return current_ ? current_->buildMs : 0.0; }

private:
    struct Frame {
        glm::dvec3 origin{0.0}; // ECEF, on the ground
        glm::dmat3 enu{1.0};    // rows: east, north, up
    };
    struct Patch {
        Frame frame;
        std::vector<JPH::BodyID> bodies; // created, not yet added
        f64 buildMs = 0;
    };
    struct CellData {
        glm::dvec3 center;
        std::vector<VegInstance> instances;
    };

    void build(Patch& p);
    std::shared_ptr<const CellData> cell(u64 key);
    static JPH::Vec3 localIn(const Frame& f, const glm::dvec3& ecef);
    static JPH::Vec3 dirIn(const Frame& f, const glm::dvec3& d);

    const HeightField& heights_;
    jobs::JobSystem& jobs_;
    NatureSpeciesDims dims_[kSpeciesCount];
    JPH::RefConst<JPH::Shape> rockShapes_[kSpeciesCount]; // unit-scale LOD2 meshes (rocks only)

    std::unique_ptr<JPH::TempAllocator> temp_;
    struct Filters;
    std::unique_ptr<Filters> filters_;
    std::unique_ptr<JPH::PhysicsSystem> system_;

    std::unique_ptr<Patch> current_, pending_;
    std::atomic<bool> pendingDone_{false};
    bool pendingInFlight_ = false;
    jobs::Counter pendingCounter_;

    std::mutex cacheMutex_;
    std::unordered_map<u64, std::shared_ptr<const CellData>> cellCache_;
};

} // namespace phx
