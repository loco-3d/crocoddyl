///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/inequality.hpp"
#include <iostream>
namespace crocoddyl {

ActivationModelInequality::ActivationModelInequality(const ActivationBounds& bounds)
    : ActivationModelAbstract((unsigned int)bounds.lb.size()), bounds_(bounds) {}

ActivationModelInequality::~ActivationModelInequality() {}

void ActivationModelInequality::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(r.size() == nr_ && "r has wrong dimension");
  boost::shared_ptr<ActivationDataInequality> d = boost::static_pointer_cast<ActivationDataInequality>(data);

  d->rlb_min_ = (r - bounds_.lb).array().min(0.);
  d->rub_max_ = (r - bounds_.ub).array().max(0.);
  data->a_value = 0.5 * d->rlb_min_.matrix().squaredNorm() + 0.5 * d->rub_max_.matrix().squaredNorm();
}

void ActivationModelInequality::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                         const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(r.size() == nr_ && "r has wrong dimension");
  if (recalc) {
    calc(data, r);
  }

  boost::shared_ptr<ActivationDataInequality> d = boost::static_pointer_cast<ActivationDataInequality>(data);
  data->Ar = (d->rlb_min_ + d->rub_max_).matrix();
  data->Arr.diagonal() = (((r - bounds_.lb).array() <= 0.) + ((r - bounds_.ub).array() >= 0.)).matrix().cast<double>();
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelInequality::createData() {
  return boost::make_shared<ActivationDataInequality>(this);
}

const ActivationBounds& ActivationModelInequality::get_bounds() const { return bounds_; }

void ActivationModelInequality::set_bounds(const ActivationBounds& bounds) { bounds_ = bounds; }

}  // namespace crocoddyl
