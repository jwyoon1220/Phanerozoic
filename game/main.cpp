#include "engine/Engine.hpp"

int main(int argc, char** argv) {
    phx::EngineConfig cfg;
    cfg.title = "Phanerozoic — Eocene";
#ifdef NDEBUG
    cfg.validation = false;
#endif
    return phx::runEngine(phx::engineConfigFromEnvironment(cfg));
}
