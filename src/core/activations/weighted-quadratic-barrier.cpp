///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include <iostream>
namespace crocoddyl {

ActivationModelWeightedQuadraticBarrier::ActivationModelWeightedQuadraticBarrier(const ActivationBounds& bounds,
                                                                                 const Eigen::VectorXd& weights)
    : ActivationModelAbstract(bounds.lb.size()), bounds_(bounds), weights_(weights) {}

ActivationModelWeightedQuadraticBarrier::~ActivationModelWeightedQuadraticBarrier() {}

void ActivationModelWeightedQuadraticBarrier::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                                   const Eigen::Ref<const Eigen::VectorXd>& r) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  boost::shared_ptr<ActivationDataQuadraticBarrier> d =
      boost::static_pointer_cast<ActivationDataQuadraticBarrier>(data);

  d->rlb_min_ = (r - bounds_.lb).array().min(0.);
  d->rub_max_ = (r - bounds_.ub).array().max(0.);
  d->rlb_min_.array() *= weights_.array();
  d->rub_max_.array() *= weights_.array();
  data->a_value = 0.5 * d->rlb_min_.matrix().squaredNorm() + 0.5 * d->rub_max_.matrix().squaredNorm();
}

void ActivationModelWeightedQuadraticBarrier::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                                       const Eigen::Ref<const Eigen::VectorXd>& r,
                                                       const bool& recalc) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  if (recalc) {
    calc(data, r);
  }

  boost::shared_ptr<ActivationDataQuadraticBarrier> d =
      boost::static_pointer_cast<ActivationDataQuadraticBarrier>(data);
  data->Ar = (d->rlb_min_ + d->rub_max_).matrix();
  data->Ar.array() *= weights_.array();
  data->Arr.diagonal() = (((r - bounds_.lb).array() <= 0.) + ((r - bounds_.ub).array() >= 0.)).matrix().cast<double>();
  data->Arr.diagonal().array() *= weights_.array();
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelWeightedQuadraticBarrier::createData() {
  return boost::make_shared<ActivationDataQuadraticBarrier>(this);
}

const ActivationBounds& ActivationModelWeightedQuadraticBarrier::get_bounds() const { return bounds_; }

void ActivationModelWeightedQuadraticBarrier::set_bounds(const ActivationBounds& bounds) { bounds_ = bounds; }

const Eigen::VectorXd& ActivationModelWeightedQuadraticBarrier::get_weights() const { return weights_; }

void ActivationModelWeightedQuadraticBarrier::set_weights(const Eigen::VectorXd& weights) {
  if (weights.size() != weights_.size()) {
    throw_pretty("Invalid argument: "
                 << "weight vector has wrong dimension (it should be " + std::to_string(weights_.size()) + ")");
  }

  weights_ = weights;
}

}  // namespace crocoddyl
