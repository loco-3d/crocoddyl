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

Eigen::VectorXd StateVector::zero() const { return Eigen::VectorXd::Zero(nx_); }

Eigen::VectorXd StateVector::rand() const { return Eigen::VectorXd::Random(nx_); }

void StateVector::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                       Eigen::Ref<Eigen::VectorXd> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw CrocoddylException("x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw CrocoddylException("x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw CrocoddylException("dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  dxout = x1 - x0;
}

void StateVector::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                            Eigen::Ref<Eigen::VectorXd> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw CrocoddylException("x has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw CrocoddylException("dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw CrocoddylException("xout has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  xout = x + dx;
}

void StateVector::Jdiff(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                        Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                        Jcomponent firstsecond) const {
  assert(is_a_Jcomponent(firstsecond) && ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw CrocoddylException("Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                               std::to_string(ndx_) + ")");
    }
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, -1.);
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw CrocoddylException("Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                               std::to_string(ndx_) + ")");
    }
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

void StateVector::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                             Jcomponent firstsecond) const {
  assert(is_a_Jcomponent(firstsecond) && ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw CrocoddylException("Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                               std::to_string(ndx_) + ")");
    }
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw CrocoddylException("Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                               std::to_string(ndx_) + ")");
    }
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

}  // namespace crocoddyl
