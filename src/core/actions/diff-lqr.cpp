///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/diff-lqr.hpp"

namespace crocoddyl {

DifferentialActionModelLQR::DifferentialActionModelLQR(const unsigned int& nq, const unsigned int& nu, bool drift_free)
    : DifferentialActionModelAbstract(*new StateVector(2 * nq), nu), drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal (matrices)
  Fq_ = Eigen::MatrixXd::Identity(state_.get_nq(), state_.get_nq());
  Fv_ = Eigen::MatrixXd::Identity(state_.get_nv(), state_.get_nv());
  Fu_ = Eigen::MatrixXd::Identity(state_.get_nq(), nu_);
  f0_ = Eigen::VectorXd::Ones(state_.get_nv());
  Lxx_ = Eigen::MatrixXd::Identity(state_.get_nx(), state_.get_nx());
  Lxu_ = Eigen::MatrixXd::Identity(state_.get_nx(), nu_);
  Luu_ = Eigen::MatrixXd::Identity(nu_, nu_);
  lx_ = Eigen::VectorXd::Ones(state_.get_nx());
  lu_ = Eigen::VectorXd::Ones(nu_);
}

DifferentialActionModelLQR::~DifferentialActionModelLQR() {}

void DifferentialActionModelLQR::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                      const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "DifferentialActionModelLQR::calc: x has wrong dimension");
  assert(u.size() == nu_ && "DifferentialActionModelLQR::calc: u has wrong dimension");

  const Eigen::VectorXd& q = x.head(state_.get_nq());
  const Eigen::VectorXd& v = x.tail(state_.get_nv());
  if (drift_free_) {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u;
  } else {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u + f0_;
  }
  data->cost = 0.5 * x.dot(Lxx_ * x) + 0.5 * u.dot(Luu_ * u) + x.dot(Lxu_ * u) + lx_.dot(x) + lu_.dot(u);
}

void DifferentialActionModelLQR::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == state_.get_nx() && "DifferentialActionModelLQR::calcDiff: x has wrong dimension");
  assert(u.size() == nu_ && "DifferentialActionModelLQR::calcDiff: u has wrong dimension");

  if (recalc) {
    calc(data, x, u);
  }
  data->Lx = lx_ + Lxx_ * x + Lxu_ * u;
  data->Lu = lu_ + Lxu_.transpose() * x + Luu_ * u;
  data->Fx.leftCols(state_.get_nq()) = Fq_;
  data->Fx.rightCols(state_.get_nv()) = Fv_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelLQR::createData() {
  return boost::make_shared<DifferentialActionDataLQR>(this);
}

}  // namespace crocoddyl
