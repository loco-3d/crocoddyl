
#include "crocoddyl/core/actuation-base-squash.hpp"

namespace crocoddyl {
    ActuationModelSquashingAbstract::ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state, const std::size_t& nu) : ActuationModelAbstract(state, nu) {}

    ActuationModelSquashingAbstract::~ActuationModelSquashingAbstract() {}
}