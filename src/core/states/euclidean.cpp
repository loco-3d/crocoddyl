///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

StateVector::StateVector(const unsigned int& nx) : StateAbstract(nx, nx) {}

StateVector::~StateVector() {}

Eigen::VectorXd StateVector::zero() { return Eigen::VectorXd::Zero(nx_); }

Eigen::VectorXd StateVector::rand() { return Eigen::VectorXd::Random(nx_); }

void StateVector::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                       Eigen::Ref<Eigen::VectorXd> dxout) {
  assert(x0.size() == nx_ && "StateVector::diff: x0 has wrong dimension");
  assert(x1.size() == nx_ && "StateVector::diff: x1 has wrong dimension");
  assert(dxout.size() == ndx_ && "StateVector::diff: output must be pre-allocated");
  dxout = x1 - x0;
}

void StateVector::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                            Eigen::Ref<Eigen::VectorXd> xout) {
  assert(x.size() == nx_ && "StateVector::integrate: x has wrong dimension");
  assert(dx.size() == ndx_ && "StateVector::integrate: dx has wrong dimension");
  assert(xout.size() == nx_ && "StateVector::integrate: output must be pre-allocated");
  xout = x + dx;
}

void StateVector::Jdiff(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                        Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                        Jcomponent firstsecond) {
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("StateVector::Jdiff: firstsecond must be one of the Jcomponent "
          "{both, first, second }"));
  if (firstsecond == first || firstsecond == both) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateVector::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, -1.);
  }
  if (firstsecond == second || firstsecond == both) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ && "StateVector::Jdiff: Jfirst must be of the good size");
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

void StateVector::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                             Jcomponent firstsecond) {
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("StateVector::Jdiff: firstsecond must be one of the Jcomponent "
          "{both, first, second }"));
  if (firstsecond == first || firstsecond == both) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateVector::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
  if (firstsecond == second || firstsecond == both) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ && "StateVector::Jdiff: Jfirst must be of the good size");
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

}  // namespace crocoddyl
