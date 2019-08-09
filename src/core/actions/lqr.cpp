///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/lqr.hpp"

namespace crocoddyl {

ActionModelLQR::ActionModelLQR(const unsigned int& nx, const unsigned int& nu, bool drift_free)
    : ActionModelAbstract(*new StateVector(nx), nu, 0), drift_free_(drift_free) {
  // TODO substitute by random (vectors) and random-orthogonal (matrices)
  Fx_ = Eigen::MatrixXd::Identity(nx, nx);
  Fu_ = Eigen::MatrixXd::Identity(nx, nu);
  f0_ = Eigen::VectorXd::Ones(nx);
  Lxx_ = Eigen::MatrixXd::Identity(nx, nx);
  Lxu_ = Eigen::MatrixXd::Identity(nx, nu);
  Luu_ = Eigen::MatrixXd::Identity(nu, nu);
  lx_ = Eigen::VectorXd::Ones(nx);
  lu_ = Eigen::VectorXd::Ones(nu);
}

ActionModelLQR::~ActionModelLQR() {
  // delete state_; //TODO @Carlos this breaks the test_actions c++ unit-test
}

void ActionModelLQR::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "ActionModelLQR::calc: x has wrong dimension");
  assert(u.size() == nu_ && "ActionModelLQR::calc: u has wrong dimension");

  if (drift_free_) {
    data->xnext = Fx_ * x + Fu_ * u;
  } else {
    data->xnext = Fx_ * x + Fu_ * u + f0_;
  }
  data->cost = 0.5 * x.dot(Lxx_ * x) + 0.5 * u.dot(Luu_ * u) + x.dot(Lxu_ * u) + lx_.dot(x) + lu_.dot(u);
}

void ActionModelLQR::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  assert(x.size() == state_.get_nx() && "ActionModelLQR::calcDiff: x has wrong dimension");
  assert(u.size() == nu_ && "ActionModelLQR::calcDiff: u has wrong dimension");

  if (recalc) {
    calc(data, x, u);
  }
  data->Lx = lx_ + Lxx_ * x + Lxu_ * u;
  data->Lu = lu_ + Lxu_.transpose() * x + Luu_ * u;
  data->Fx = Fx_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

boost::shared_ptr<ActionDataAbstract> ActionModelLQR::createData() { return boost::make_shared<ActionDataLQR>(this); }

}  // namespace crocoddyl
