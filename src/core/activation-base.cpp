#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

ActivationModelAbstract::ActivationModelAbstract(const unsigned int& nr) : nr_(nr) {}

ActivationModelAbstract::~ActivationModelAbstract() {}

boost::shared_ptr<ActivationDataAbstract> ActivationModelAbstract::createData() {
  return boost::make_shared<ActivationDataAbstract>(this);
}

unsigned int ActivationModelAbstract::get_nr() const { return nr_; }

void ActivationModelAbstract::set_nr(const unsigned int& nr) { nr_ = nr; }

}  // namespace crocoddyl