#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/squashing-base.hpp"

namespace crocoddyl {

SquashingModelAbstract::SquashingModelAbstract(boost::shared_ptr<ActuationModelAbstract> actuation, const std::size_t& ns) : ns_(ns), actuation_(actuation) {
  if (ns_ == 0) {
    throw_pretty("Invalid argument: "
                << "ns cannot be zero");
  }
}

SquashingModelAbstract::~SquashingModelAbstract() {}

boost::shared_ptr<SquashingDataAbstract> SquashingModelAbstract::createData() {
  return boost::make_shared<SquashingDataAbstract>(this);
}

const std::size_t& SquashingModelAbstract::get_ns() const { return ns_;}

const boost::shared_ptr<ActuationModelAbstract>& SquashingModelAbstract::get_actuation() const {
  return actuation_; }

} //namespace crocoddyl