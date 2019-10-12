///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

StateVector::StateVector(const std::size_t& nx) : StateAbstract(nx, nx) {}

StateVector::~StateVector() {}

Eigen::VectorXd StateVector::zero() { return Eigen::VectorXd::Zero(nx_); }

Eigen::VectorXd StateVector::rand() { return Eigen::VectorXd::Random(nx_); }

void StateVector::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                       Eigen::Ref<Eigen::VectorXd> dxout) {
  assert(static_cast<std::size_t>(x0.size()) == nx_ && "x0 has wrong dimension");
  assert(static_cast<std::size_t>(x1.size()) == nx_ && "x1 has wrong dimension");
  assert(static_cast<std::size_t>(dxout.size()) == ndx_ && "output must be pre-allocated");
  dxout = x1 - x0;
}

void StateVector::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                            Eigen::Ref<Eigen::VectorXd> xout) {
  assert(static_cast<std::size_t>(x.size()) == nx_ && "x has wrong dimension");
  assert(static_cast<std::size_t>(dx.size()) == ndx_ && "dx has wrong dimension");
  assert(static_cast<std::size_t>(xout.size()) == nx_ && "Output must be pre-allocated");
  xout = x + dx;
}

void StateVector::Jdiff(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                        Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                        Jcomponent firstsecond) {
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (firstsecond == first || firstsecond == both) {
    assert(static_cast<std::size_t>(Jfirst.rows()) == ndx_ && static_cast<std::size_t>(Jfirst.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, -1.);
  }
  if (firstsecond == second || firstsecond == both) {
    assert(static_cast<std::size_t>(Jsecond.rows()) == ndx_ && static_cast<std::size_t>(Jsecond.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

void StateVector::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                             Jcomponent firstsecond) {
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (firstsecond == first || firstsecond == both) {
    assert(static_cast<std::size_t>(Jfirst.rows()) == ndx_ && static_cast<std::size_t>(Jfirst.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
  if (firstsecond == second || firstsecond == both) {
    assert(static_cast<std::size_t>(Jsecond.rows()) == ndx_ && static_cast<std::size_t>(Jsecond.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

}  // namespace crocoddyl
