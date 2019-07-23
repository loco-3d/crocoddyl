#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

ActivationModelAbstract::ActivationModelAbstract(const unsigned int& nr) : nr_(nr) {}

ActivationModelAbstract::~ActivationModelAbstract() {}

unsigned int ActivationModelAbstract::get_nr() const { return nr_; }

}  // namespace crocoddyl