///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/diff-lqr.hpp"

namespace crocoddyl {

DifferentialActionModelLQR::DifferentialActionModelLQR(const std::size_t& nq, const std::size_t& nu, bool drift_free)
    : DifferentialActionModelAbstract(internal_state_, nu), internal_state_(2 * nq), drift_free_(drift_free) {
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
  assert(static_cast<std::size_t>(x.size()) == state_.get_nx() && "x has wrong dimension");
  assert(static_cast<std::size_t>(u.size()) == nu_ && "u has wrong dimension");

  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_.get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(state_.get_nv());

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
  assert(static_cast<std::size_t>(x.size()) == state_.get_nx() && "x has wrong dimension");
  assert(static_cast<std::size_t>(u.size()) == nu_ && "u has wrong dimension");

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
