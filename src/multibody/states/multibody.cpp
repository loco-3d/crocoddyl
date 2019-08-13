///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl {

StateMultibody::StateMultibody(pinocchio::Model& model)
    : StateAbstract(model.nq + model.nv, 2 * model.nv),
      pinocchio_(model),
      x0_(Eigen::VectorXd::Zero(model.nq + model.nv)),
      dx_(Eigen::VectorXd::Zero(2 * model.nv)),
      q0_(Eigen::VectorXd::Zero(model.nq)),
      dq0_(Eigen::VectorXd::Zero(model.nv)),
      q1_(Eigen::VectorXd::Zero(model.nq)),
      dq1_(Eigen::VectorXd::Zero(model.nv)),
      Ji_(Eigen::MatrixXd::Zero(model.nv, model.nv)),
      Jd_(Eigen::MatrixXd::Zero(model.nv, model.nv)) {
  x0_.head(nq_) = pinocchio::neutral(pinocchio_);
}

StateMultibody::~StateMultibody() {}

Eigen::VectorXd StateMultibody::zero() { return x0_; }

Eigen::VectorXd StateMultibody::rand() {
  Eigen::VectorXd xrand = Eigen::VectorXd::Random(nx_);
  xrand.head(nq_) = pinocchio::randomConfiguration(pinocchio_);
  return xrand;
}

void StateMultibody::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                          Eigen::Ref<Eigen::VectorXd> dxout) {
  assert(x0.size() == nx_ && "StateMultibody::diff: x0 has wrong dimension");
  assert(x1.size() == nx_ && "StateMultibody::diff: x1 has wrong dimension");
  assert(dxout.size() == ndx_ && "StateMultibody::diff: output must be pre-allocated");

  q0_ = x0.head(nq_);
  q1_ = x1.head(nq_);
  dq0_ = x0.tail(nv_);
  dq1_ = x1.tail(nv_);
  pinocchio::difference(pinocchio_, q0_, q1_, dxout.head(nv_));
  dxout.tail(nv_) = dq1_ - dq0_;
}

void StateMultibody::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                               Eigen::Ref<Eigen::VectorXd> xout) {
  assert(x.size() == nx_ && "StateMultibody::diff: x has wrong dimension");
  assert(dx.size() == ndx_ && "StateMultibody::diff: dx has wrong dimension");
  assert(xout.size() == nx_ && "StateMultibody::diff: output must be pre-allocated");

  q0_ = x.head(nq_);
  dq0_ = dx.head(nv_);
  pinocchio::integrate(pinocchio_, q0_, dq0_, q1_);
  xout.head(nq_) = q1_;

  dq0_ = x.tail(nv_);
  dq1_ = dx.tail(nv_);
  xout.tail(nv_) = dq0_ + dq1_;
}

void StateMultibody::Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                           Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                           Jcomponent firstsecond) {
  assert(x0.size() == nx_ && "StateMultibody::Jdiff: x0 has wrong dimension");
  assert(x1.size() == nx_ && "StateMultibody::Jdiff: x1 has wrong dimension");
  assert(is_a_Jcomponent(firstsecond) && ("StateMultibody::Jdiff: firstsecond "
                                          "must be one of the Jcomponent {both, first, second}"));

  if (firstsecond == first) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateMultibody::Jdiff: Jfirst must be of the good size");

    diff(x1, x0, dx_);
    q1_ = x1.head(nq_);
    dq1_ = dx_.head(nv_);
    pinocchio::dIntegrate(pinocchio_, q1_, dq1_, Ji_, pinocchio::ARG1);
    updateJdiff(Ji_, false);

    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = Jd_;
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = -Eigen::VectorXd::Ones(nv_);
  } else if (firstsecond == second) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    diff(x0, x1, dx_);
    q0_ = x0.head(nq_);
    dq0_ = dx_.head(nv_);
    pinocchio::dIntegrate(pinocchio_, q0_, dq0_, Ji_, pinocchio::ARG1);
    updateJdiff(Ji_);

    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Jd_;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else {  // computing both
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateMultibody::Jdiff: Jfirst must be of the good size");
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    // Computing Jfirst
    diff(x1, x0, dx_);
    q1_ = x1.head(nq_);
    dq1_ = dx_.head(nv_);
    pinocchio::dIntegrate(pinocchio_, q1_, dq1_, Ji_, pinocchio::ARG1);
    updateJdiff(Ji_, false);

    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = Jd_;
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = -Eigen::VectorXd::Ones(nv_);

    // Computing Jsecond
    diff(x0, x1, dx_);
    const Eigen::VectorXd& q0 = x0.head(nq_);
    const Eigen::VectorXd& dq0 = dx_.head(nv_);
    pinocchio::dIntegrate(pinocchio_, q0, dq0, Ji_, pinocchio::ARG1);
    updateJdiff(Ji_);

    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Jd_;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  }
}

void StateMultibody::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                                const Eigen::Ref<const Eigen::VectorXd>& dx, Eigen::Ref<Eigen::MatrixXd> Jfirst,
                                Eigen::Ref<Eigen::MatrixXd> Jsecond, Jcomponent firstsecond) {
  assert(x.size() == nx_ && "StateMultibody::Jintegrate: x has wrong dimension");
  assert(dx.size() == ndx_ && "StateMultibody::Jintegrate: dx has wrong dimension");
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("StateMultibody::Jintegrate: firstsecond must be one of the Jcomponent "
          "{both, first, second}"));

  q0_ = x.head(nq_);
  dq0_ = dx.head(nv_);
  if (firstsecond == first) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ &&
           "StateMultibody::Jintegrate: Jfirst must be of the good size");

    pinocchio::dIntegrate(pinocchio_, q0_, dq0_, Ji_, pinocchio::ARG0);
    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = Ji_;
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else if (firstsecond == second) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    pinocchio::dIntegrate(pinocchio_, q0_, dq0_, Ji_, pinocchio::ARG1);
    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Ji_;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else {  // computing both
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateMultibody::Jdiff: Jfirst must be of the good size");
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    // Computing Jfirst
    pinocchio::dIntegrate(pinocchio_, q0_, dq0_, Ji_, pinocchio::ARG0);
    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = Ji_;
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);

    // Computing Jsecond
    pinocchio::dIntegrate(pinocchio_, q0_, dq0_, Ji_, pinocchio::ARG1);
    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Ji_;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  }
}

pinocchio::Model& StateMultibody::get_pinocchio() const { return pinocchio_; }

void StateMultibody::updateJdiff(const Eigen::Ref<const Eigen::MatrixXd>& Jdq, bool positive) {
  if (positive) {
    Jd_.diagonal() = Jdq.diagonal();
    Jd_.block<3, 3>(0, 0) = Jdq.block<3, 3>(0, 0).transpose();
    Jd_.block<3, 3>(0, 3) = Jdq.block<3, 3>(0, 3).transpose();
    Jd_.block<3, 3>(3, 3) = Jdq.block<3, 3>(3, 3).transpose();
  } else {
    Jd_.diagonal() = -Jdq.diagonal();
    Jd_.block<3, 3>(0, 0) = -Jdq.block<3, 3>(0, 0).transpose();
    Jd_.block<3, 3>(0, 3) = -Jdq.block<3, 3>(0, 3).transpose();
    Jd_.block<3, 3>(3, 3) = -Jdq.block<3, 3>(3, 3).transpose();
  }
}

}  // namespace crocoddyl
