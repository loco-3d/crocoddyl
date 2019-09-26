///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/floating-base.hpp"

namespace crocoddyl {

ActuationModelFloatingBase::ActuationModelFloatingBase(StateMultibody& state)
    : ActuationModelAbstract(state, state.get_nv() - 6) {
  pinocchio::JointModelFreeFlyer ff_joint;
  assert(state.get_pinocchio().joints[1].shortname() == ff_joint.shortname() &&
         "The first joint has to be free-flyer");
  if (state.get_pinocchio().joints[1].shortname() != ff_joint.shortname()) {
    std::cout << "Warning: the first joint has to be a free-flyer" << std::endl;
  }
}

ActuationModelFloatingBase::~ActuationModelFloatingBase() {}

void ActuationModelFloatingBase::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                      const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(u.size() == nu_ && "u has wrong dimension");
  data->tau.tail(nu_) = u;
}

void ActuationModelFloatingBase::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // The derivatives has constant values which were set in createData.
  assert(data->dtau_dx == Eigen::MatrixXd::Zero(state_.get_nv(), state_.get_ndx()) && "dtau_dx has wrong value");
  assert(data->dtau_du == dtau_du_ && "dtau_du has wrong value");
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
