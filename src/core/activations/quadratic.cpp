///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

ActivationModelQuad::ActivationModelQuad(const std::size_t& nr) : ActivationModelAbstract(nr) {}

ActivationModelQuad::~ActivationModelQuad() {}

void ActivationModelQuad::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& r) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  data->a_value = 0.5 * r.transpose() * r;
}

void ActivationModelQuad::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& r) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }

  data->Ar = r;
  // The Hessian has constant values which were set in createData.
  assert_pretty(data->Arr == Eigen::MatrixXd::Identity(nr_, nr_), "Arr has wrong value");
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelQuad::createData() {
  boost::shared_ptr<ActivationDataAbstract> data = boost::make_shared<ActivationDataAbstract>(this);
  data->Arr.diagonal().fill(1.);
  return data;
}

}  // namespace crocoddyl
