///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"

namespace crocoddyl {

ActivationModelQuadraticBarrier::ActivationModelQuadraticBarrier(const ActivationBounds& bounds)
    : ActivationModelAbstract(bounds.lb.size()), bounds_(bounds) {}

ActivationModelQuadraticBarrier::~ActivationModelQuadraticBarrier() {}

void ActivationModelQuadraticBarrier::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& r) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: " << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  boost::shared_ptr<ActivationDataQuadraticBarrier> d =
      boost::static_pointer_cast<ActivationDataQuadraticBarrier>(data);

  d->rlb_min_ = (r - bounds_.lb).array().min(0.);
  d->rub_max_ = (r - bounds_.ub).array().max(0.);
  data->a_value = 0.5 * d->rlb_min_.matrix().squaredNorm() + 0.5 * d->rub_max_.matrix().squaredNorm();
}

void ActivationModelQuadraticBarrier::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                               const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: " << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  if (recalc) {
    calc(data, r);
  }

  boost::shared_ptr<ActivationDataQuadraticBarrier> d =
      boost::static_pointer_cast<ActivationDataQuadraticBarrier>(data);
  data->Ar = (d->rlb_min_ + d->rub_max_).matrix();
  data->Arr.diagonal() = (((r - bounds_.lb).array() <= 0.) + ((r - bounds_.ub).array() >= 0.)).matrix().cast<double>();
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelQuadraticBarrier::createData() {
  return boost::make_shared<ActivationDataQuadraticBarrier>(this);
}

const ActivationBounds& ActivationModelQuadraticBarrier::get_bounds() const { return bounds_; }

void ActivationModelQuadraticBarrier::set_bounds(const ActivationBounds& bounds) { bounds_ = bounds; }

}  // namespace crocoddyl
