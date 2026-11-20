#include "engine/game/Player.hpp"

#include "engine/game/PhysicsWorld.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include <algorithm>
#include <cmath>

namespace phx {
namespace {
constexpr f64 kPi = 3.14159265358979323846;
constexpr f32 kRadius = 0.3f, kHalfCylinder = 0.6f; // 1.8 m capsule
constexpr f32 kMaxWalkSlope = 50.0f * f32(kPi) / 180.0f;
constexpr f32 kScrambleSlope = 30.0f * f32(kPi) / 180.0f;
constexpr f32 kClimbSlope = 45.0f * f32(kPi) / 180.0f; // plan: 45°+ is climbed (hold Space)
constexpr f64 kMassKg = 75.0;

// Tobler's hiking function: walking speed (m/s) for a grade dh/dx.
f64 tobler(f64 grade) { return 6.0 * std::exp(-3.5 * std::abs(grade + 0.05)) / 3.6; }
} // namespace

const char* locomotionName(Locomotion l) {
    switch (l) {
    case Locomotion::Walk: return "walk";
    case Locomotion::Scramble: return "scramble";
    case Locomotion::Slide: return "slide";
    case Locomotion::Climb: return "climb";
    case Locomotion::Air: return "air";
    }
    return "?";
}

f64 airTemperatureC(f64 latitudeDeg, f64 altitudeM, f64 dayOfYear, f64 sunAltitudeSin) {
    // Early Eocene: warm, weak equator-pole gradient (≈ 21 °C mean at 34° vs ≈ 13 °C today).
    const f64 meanSeaLevel = 29.0 - 0.0045 * latitudeDeg * latitudeDeg;
    const f64 hemisphere = latitudeDeg >= 0 ? 1.0 : -1.0;
    const f64 annual = 6.0 * hemisphere * std::cos(2.0 * kPi * (dayOfYear - 200.0) / 365.25) *
                       std::min(1.0, std::abs(latitudeDeg) / 40.0);
    const f64 diurnal = 5.0 * std::clamp(sunAltitudeSin, -0.25, 1.0) - 1.0;
    return meanSeaLevel - 0.0065 * altitudeM + annual + diurnal;
}

Player::Player(PhysicsWorld& world, const glm::dvec3& feetEcef) : world_(world), feet_(feetEcef) {
    world_.update(feet_, true);
    JPH::CharacterVirtualSettings s;
    s.mShape = JPH::RotatedTranslatedShapeSettings(JPH::Vec3(0, kHalfCylinder + kRadius, 0), JPH::Quat::sIdentity(),
                                                   new JPH::CapsuleShape(kHalfCylinder, kRadius))
                   .Create()
                   .Get();
    s.mMaxSlopeAngle = kMaxWalkSlope;
    s.mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -kRadius); // contacts below the lower sphere centre support
    s.mMass = f32(kMassKg);
    s.mMaxStrength = 200.0f;
    s.mCharacterPadding = 0.02f;
    s.mPredictiveContactDistance = 0.1f;
    s.mEnhancedInternalEdgeRemoval = true; // no snagging on heightfield triangle edges
    character_.reset(new JPH::CharacterVirtual(&s, world_.toLocal(feet_), JPH::Quat::sIdentity(), &world_.system()));
}

Player::~Player() = default;

glm::dvec3 Player::eyeEcef() const {
    const glm::dvec3 up = glm::normalize(feet_);
    // A little head bob while walking (1.5 cm), none when standing.
    const f64 bob = mode_ == Locomotion::Walk || mode_ == Locomotion::Scramble ? 0.015 * std::sin(bob_) * std::min(speed_, 1.5) : 0.0;
    return feet_ + up * (kEyeHeight + bob);
}

void Player::reanchor() { character_->SetPosition(world_.toLocal(feet_)); }

void Player::teleport(const glm::dvec3& feetEcef) {
    feet_ = feetEcef;
    world_.update(feet_, true);
    reanchor();
    character_->SetLinearVelocity(JPH::Vec3::sZero());
    wasSupported_ = true;
}

void Player::step(f64 dt, const PlayerControls& in, f64 airTempC) {
    using JPH::Vec3;
    if (world_.update(feet_)) reanchor();
    const f32 fdt = f32(dt);
    const Vec3 up = Vec3::sAxisY();
    const f32 sy = std::sin(in.yaw), cy = std::cos(in.yaw);
    const Vec3 fwd(sy, 0, -cy), right(cy, 0, sy); // local frame: x east, z south
    Vec3 wish = fwd * in.forward + right * in.strafe;
    if (wish.LengthSq() > 1.0f) wish = wish.Normalized();

    character_->UpdateGroundVelocity();
    const auto ground = character_->GetGroundState();
    const Vec3 n = character_->GetGroundNormal();

    // Steepest touching surface (rock face to climb).
    Vec3 wall = Vec3::sZero();
    f32 wallAngle = 0;
    for (const JPH::CharacterVirtual::Contact& c : character_->GetActiveContacts()) {
        if (!c.mHadCollision && c.mDistance > 0.06f) continue;
        const f32 a = std::acos(std::clamp(c.mSurfaceNormal.GetY(), -1.0f, 1.0f));
        if (a > kClimbSlope && a > wallAngle) wallAngle = a, wall = c.mSurfaceNormal;
    }
    const bool canClimb = wallAngle > 0 && vitals_.stamina > 1.0 && vitals_.health > 0;
    const bool pushingIntoWall = canClimb && wish.Dot(-wall) > 0.3f;

    Vec3 v = character_->GetLinearVelocity();
    // Grounded only when not moving away from the ground (a jump leaves OnGround a few ticks late).
    const bool onGround = ground == JPH::CharacterBase::EGroundState::OnGround &&
                          (v - character_->GetGroundVelocity()).Dot(up) < 0.5f;
    const bool climb = in.jumpHeld && canClimb && (mode_ == Locomotion::Climb || !onGround || pushingIntoWall);
    Vec3 gravity(0, -9.81f, 0);
    const bool jumpEdge = in.jumpHeld && !jumpWasHeld_;
    jumpWasHeld_ = in.jumpHeld;

    if (climb) {
        mode_ = Locomotion::Climb;
        const Vec3 w = wall.Normalized();
        const Vec3 along = (up - w * w.Dot(up)).NormalizedOr(up);
        const Vec3 side = (right - w * w.Dot(right)).NormalizedOr(right);
        v = along * (in.forward * 0.35f) + side * (in.strafe * 0.3f) - w * 0.4f; // press into the rock
        gravity = Vec3::sZero();
    } else {
        if (mode_ == Locomotion::Climb && in.jumpHeld && in.forward > 0) v = up * 2.2f + fwd * 1.2f; // mantle over the top
        const Vec3 horizontal = v - up * v.Dot(up);
        if (onGround) {
            const f32 slope = std::acos(std::clamp(n.GetY(), -1.0f, 1.0f));
            const Vec3 t = wish - n * n.Dot(wish);
            const f64 run = std::hypot(f64(t.GetX()), f64(t.GetZ()));
            const f64 grade = run > 1e-4 ? f64(t.GetY()) / run : 0.0;
            f64 target = tobler(grade) * 1.05;
            mode_ = slope > kScrambleSlope ? Locomotion::Scramble : Locomotion::Walk;
            if (mode_ == Locomotion::Scramble) target *= 0.8; // hands help; Tobler already slows steep grades
            else if (in.sprint && vitals_.stamina > 5.0) target *= 2.3;
            target = std::max(target, 0.2);
            target *= 0.4 + 0.6 * std::clamp(vitals_.stamina / 25.0, 0.0, 1.0); // exhausted: slow down
            // Move along the ground plane; Tobler's speed is horizontal (map) speed.
            const f32 tlen = t.Length();
            const Vec3 desired = tlen > 1e-4f ? t * f32(target / std::max(run, 0.2)) : Vec3::sZero();
            const Vec3 cur = v - n * v.Dot(n); // current velocity along the ground
            v = cur + (desired - cur) * std::min(1.0f, fdt * 10.0f) + character_->GetGroundVelocity();
            if (jumpEdge && vitals_.stamina > 8.0) {
                v += up * 4.0f;
                vitals_.stamina -= 8.0;
            }
        } else if (ground == JPH::CharacterBase::EGroundState::OnSteepGround) {
            mode_ = Locomotion::Slide; // too steep to stand: slide, slight steering
            v += (wish * 0.6f - horizontal * 0.0f) * fdt;
        } else {
            mode_ = Locomotion::Air;
            v += wish * (1.5f * fdt); // a little air control
        }
        if (!onGround || v.Dot(up) > 0.5f) v += gravity * fdt;
    }

    const f32 vyBefore = v.GetY();
    character_->SetLinearVelocity(v);
    JPH::CharacterVirtual::ExtendedUpdateSettings us;
    us.mStickToFloorStepDown = climb ? Vec3::sZero() : Vec3(0, -0.5f, 0);
    us.mWalkStairsStepUp = Vec3(0, 0.4f, 0);
    const JPH::Vec3 before = character_->GetPosition();
    character_->ExtendedUpdate(fdt, gravity, us, world_.system().GetDefaultBroadPhaseLayerFilter(physlayers::kMoving),
                               world_.system().GetDefaultLayerFilter(physlayers::kMoving), JPH::BodyFilter(),
                               JPH::ShapeFilter(), world_.tempAllocator());
    const JPH::Vec3 after = character_->GetPosition();
    feet_ = world_.toEcef(after);

    // Landing impact.
    const bool supported = character_->IsSupported();
    if (!supported) fallSpeed_ = std::max(fallSpeed_, f64(-vyBefore));
    if (supported && !wasSupported_) {
        if (fallSpeed_ > 7.0) vitals_.health -= (fallSpeed_ - 7.0) * (fallSpeed_ - 7.0) * 2.5;
        fallSpeed_ = 0;
    }
    if (climb) fallSpeed_ = 0;
    wasSupported_ = supported || climb;

    const JPH::Vec3 d = after - before;
    speed_ = std::hypot(f64(d.GetX()), f64(d.GetZ())) / dt;
    verticalRate_ += (f64(d.GetY()) / dt - verticalRate_) * std::min(1.0, dt * 2.0);
    groundNormal_ = world_.dirToEcef(wallAngle > 0 ? wall : character_->GetGroundNormal());
    slopeDeg_ = supported ? std::acos(std::clamp(f64(character_->GetGroundNormal().GetY()), -1.0, 1.0)) * 180.0 / kPi : 0.0;
    bob_ = std::fmod(bob_ + speed_ * dt / 0.75 * kPi, 2 * kPi);
    updateVitals(dt, airTempC, verticalRate_);
}

void Player::updateVitals(f64 dt, f64 airT, f64 vz) {
    Vitals& v = vitals_;
    // Metabolic power (W): resting + horizontal transport + lifting the body (25 %
    // efficient) + a fraction for braking downhill + static effort while climbing.
    const f64 costPerMetre = speed_ > 2.2 ? 4.2 : 3.0; // J/kg/m, running vs walking
    f64 P = 95.0 + costPerMetre * kMassKg * speed_;
    P += std::max(vz, 0.0) * kMassKg * 9.81 / 0.25 + std::max(-vz, 0.0) * kMassKg * 9.81 * 0.1;
    if (mode_ == Locomotion::Climb) P += 250.0;
    v.powerW += (P - v.powerW) * std::min(1.0, dt / 2.0);

    // Anaerobic reserve: a fit hiker sustains ~750 W metabolic; above that the reserve
    // drains, below it recovers in proportion to the margin. Climbing costs grip.
    constexpr f64 kSustainable = 750.0;
    if (mode_ == Locomotion::Climb) {
        v.stamina -= (2.5 + 8.0 * speed_) * dt;
    } else if (v.powerW > kSustainable) {
        v.stamina -= (v.powerW - kSustainable) / 40.0 * dt;
    } else {
        const f64 fuel = std::min(1.0, v.waterL / 0.5) * std::min(1.0, v.energyKcal / 300.0);
        v.stamina += 8.0 * fuel * std::clamp((kSustainable - v.powerW) / 350.0, 0.1, 1.0) * dt;
    }
    v.stamina = std::clamp(v.stamina, 0.0, 100.0);

    v.energyKcal = std::max(0.0, v.energyKcal - v.powerW * dt / 4184.0);
    const f64 sweatLph = std::max(0.0, v.powerW - 120.0) / 1000.0 * (0.5 + 0.05 * std::max(airT - 15.0, 0.0));
    v.waterL = std::max(0.0, v.waterL - (0.05 + sweatLph) * dt / 3600.0);

    const f64 eq = std::clamp(36.9 + 0.0025 * (v.powerW - 100.0) + 0.03 * (airT - 22.0), 34.0, 41.0);
    v.coreTempC += (eq - v.coreTempC) * std::min(1.0, dt / 300.0);

    f64 dmg = 0;
    if (v.waterL < 0.25) dmg += 0.02;
    if (v.energyKcal < 100) dmg += 0.01;
    if (v.coreTempC > 39.5) dmg += 0.2 * (v.coreTempC - 39.5);
    if (v.coreTempC < 35.5) dmg += 0.2 * (35.5 - v.coreTempC);
    if (dmg > 0) v.health -= dmg * dt;
    else if (v.stamina > 50) v.health += 0.02 * dt;
    v.health = std::clamp(v.health, 0.0, 100.0);
}

} // namespace phx
