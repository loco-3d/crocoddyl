///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/full.hpp"

namespace crocoddyl {

ActuationModelFull::ActuationModelFull(boost::shared_ptr<StateMultibody> state)
    : ActuationModelAbstract(state, state->get_nv()) {
  pinocchio::JointModelFreeFlyer ff_joint;
  if (state->get_pinocchio().joints[1].shortname() == ff_joint.shortname()) {
    throw std::invalid_argument("the first joint cannot be free-flyer");
  }
}

ActuationModelFull::~ActuationModelFull() {}

void ActuationModelFull::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw std::invalid_argument("u has wrong dimension (it should be " + to_string(nu_) + ")");
  }
  data->tau = u;
}

void ActuationModelFull::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                  const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // The derivatives has constant values which were set in createData.
  assert(data->dtau_dx == Eigen::MatrixXd::Zero(state_->get_nv(), state_->get_ndx()) && "dtau_dx has wrong value");
  assert(data->dtau_du == Eigen::MatrixXd::Identity(state_->get_nv(), nu_) && "dtau_du has wrong value");
}

boost::shared_ptr<ActuationDataAbstract> ActuationModelFull::createData() {
  boost::shared_ptr<ActuationDataAbstract> data = boost::make_shared<ActuationDataAbstract>(this);
  data->dtau_du.diagonal().fill(1);
  return data;
}

}  // namespace crocoddyl
