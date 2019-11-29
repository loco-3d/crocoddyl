///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/weighted-quadratic.hpp"

namespace crocoddyl {

ActivationModelWeightedQuad::ActivationModelWeightedQuad(const Eigen::VectorXd& weights)
    : ActivationModelAbstract(weights.size()), weights_(weights) {}

ActivationModelWeightedQuad::~ActivationModelWeightedQuad() {}

void ActivationModelWeightedQuad::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                       const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(static_cast<std::size_t>(r.size()) == nr_ && "r has wrong dimension");
  boost::shared_ptr<ActivationDataWeightedQuad> d = boost::static_pointer_cast<ActivationDataWeightedQuad>(data);

  d->Wr = weights_.cwiseProduct(r);
  data->a_value = 0.5 * r.transpose() * d->Wr;
}

void ActivationModelWeightedQuad::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(static_cast<std::size_t>(r.size()) == nr_ && "r has wrong dimension");
  if (recalc) {
    calc(data, r);
  }

  boost::shared_ptr<ActivationDataWeightedQuad> d = boost::static_pointer_cast<ActivationDataWeightedQuad>(data);
  data->Ar = d->Wr;
  // The Hessian has constant values which were set in createData.
  assert(data->Arr == Arr_ && "Arr has wrong value");
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelWeightedQuad::createData() {
  boost::shared_ptr<ActivationDataWeightedQuad> data = boost::make_shared<ActivationDataWeightedQuad>(this);
  data->Arr.diagonal() = weights_;

#ifndef NDEBUG
  Arr_ = data->Arr;
#endif

  return data;
}

const Eigen::VectorXd& ActivationModelWeightedQuad::get_weights() const { return weights_; }

void ActivationModelWeightedQuad::set_weights(const Eigen::VectorXd& weights) { weights_ = weights; }

}  // namespace crocoddyl
