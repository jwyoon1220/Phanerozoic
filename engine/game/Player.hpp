#pragma once
// Player on foot (plan §1.3, M6): Jolt CharacterVirtual in the PhysicsWorld frame.
//  * Walking speed follows Tobler's hiking function of the grade along the heading;
//    sprinting (Shift) costs stamina; 30–50° ground is a slow scramble; steeper
//    ground slides unless climbed.
//  * Climbing: hold Space against rock steeper than 50°: move along the face with
//    WASD while stamina lasts; topping out mantles onto the ledge.
//  * Physiology: metabolic power from actual movement drives stamina, energy
//    (kcal), hydration (sweat) and core temperature against the air temperature
//    (altitude lapse, season, sun). Hard landings injure.
#include "engine/core/Types.hpp"

#include <glm/glm.hpp>

#include <memory>

namespace JPH {
class CharacterVirtual;
}

namespace phx {
class PhysicsWorld;

enum class Locomotion : u8 { Walk, Scramble, Slide, Climb, Air };
const char* locomotionName(Locomotion l);

struct PlayerControls {
    f32 forward = 0, strafe = 0; // -1..1
    f32 yaw = 0;                 // view azimuth (radians, 0 = north)
    bool sprint = false;
    bool jumpHeld = false;       // jump / climb grip
};

struct Vitals {
    f64 health = 100;    // %
    f64 stamina = 100;   // %
    f64 energyKcal = 2000;
    f64 waterL = 2.0;
    f64 coreTempC = 37.0;
    f64 powerW = 100;    // smoothed metabolic power
    static constexpr f64 kEnergyMax = 2400, kWaterMax = 2.5;
};

class Player {
public:
    static constexpr f64 kEyeHeight = 1.65;

    Player(PhysicsWorld& world, const glm::dvec3& feetEcef);
    ~Player();

    void step(f64 dt, const PlayerControls& in, f64 airTempC);
    void teleport(const glm::dvec3& feetEcef);

    [[nodiscard]] glm::dvec3 feetEcef() const { return feet_; }
    [[nodiscard]] glm::dvec3 eyeEcef() const;
    [[nodiscard]] const Vitals& vitals() const { return vitals_; }
    [[nodiscard]] Locomotion mode() const { return mode_; }
    [[nodiscard]] f64 groundSpeed() const { return speed_; }   // m/s, horizontal
    [[nodiscard]] f64 slopeDeg() const { return slopeDeg_; }   // ground under the feet
    [[nodiscard]] f64 bobPhase() const { return bob_; }
    [[nodiscard]] glm::dvec3 groundNormalEcef() const { return groundNormal_; } // last support / steepest contact

private:
    void reanchor();
    void updateVitals(f64 dt, f64 airTempC, f64 verticalRate);

    PhysicsWorld& world_;
    std::unique_ptr<JPH::CharacterVirtual> character_;
    glm::dvec3 feet_;
    Vitals vitals_;
    Locomotion mode_ = Locomotion::Walk;
    f64 speed_ = 0, slopeDeg_ = 0, bob_ = 0, verticalRate_ = 0;
    bool wasSupported_ = true;
    bool jumpWasHeld_ = false;
    f64 fallSpeed_ = 0;
    glm::dvec3 groundNormal_{0.0};
};

// Near-surface air temperature (°C): Eocene greenhouse baseline at the latitude, lapse
// rate with altitude, annual cycle and a diurnal swing following the sun.
f64 airTemperatureC(f64 latitudeDeg, f64 altitudeM, f64 dayOfYear, f64 sunAltitudeSin);

} // namespace phx
