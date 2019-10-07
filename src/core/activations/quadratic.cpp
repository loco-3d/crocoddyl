///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

ActivationModelQuad::ActivationModelQuad(const std::size_t& nr) : ActivationModelAbstract(nr) {}

ActivationModelQuad::~ActivationModelQuad() {}

void ActivationModelQuad::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(r.size() == nr_ && "r has wrong dimension");
  data->a_value = 0.5 * r.transpose() * r;
}

void ActivationModelQuad::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(r.size() == nr_ && "r has wrong dimension");
  if (recalc) {
    calc(data, r);
  }
  data->Ar = r;
  // The Hessian has constant values which were set in createData.
  assert(data->Arr == Eigen::MatrixXd::Identity(nr_, nr_) && "Arr has wrong value");
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelQuad::createData() {
  boost::shared_ptr<ActivationDataAbstract> data = boost::make_shared<ActivationDataAbstract>(this);
  data->Arr.diagonal().fill(1.);
  return data;
}

}  // namespace crocoddyl
