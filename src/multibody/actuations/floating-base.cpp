///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"

namespace crocoddyl {

ActuationModelFloatingBase::ActuationModelFloatingBase(boost::shared_ptr<StateMultibody> state)
    : ActuationModelAbstract(state, state->get_nv() - 6) {
  pinocchio::JointModelFreeFlyer ff_joint;
  if (state->get_pinocchio().joints[1].shortname() != ff_joint.shortname()) {
    throw_pretty("Invalid argument: " << "the first joint has to be free-flyer");
  }
}

ActuationModelFloatingBase::~ActuationModelFloatingBase() {}

void ActuationModelFloatingBase::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                      const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: " << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  data->tau.tail(nu_) = u;
}

void ActuationModelFloatingBase::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // The derivatives has constant values which were set in createData.
#ifndef NDEBUG
  assert_pretty(data->dtau_dx == Eigen::MatrixXd::Zero(state_->get_nv(), state_->get_ndx()) , "dtau_dx has wrong value");
  assert_pretty(data->dtau_du == dtau_du_ , "dtau_du has wrong value");
#endif
}

boost::shared_ptr<ActuationDataAbstract> ActuationModelFloatingBase::createData() {
  boost::shared_ptr<ActuationDataAbstract> data = boost::make_shared<ActuationDataAbstract>(this);
  data->dtau_du.diagonal(-6).fill(1.);

#ifndef NDEBUG
  dtau_du_ = data->dtau_du;
#endif

  return data;
}

}  // namespace crocoddyl
