///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_STATES_STATE_EUCLIDEAN_HPP_
#define CROCODDYL_CORE_STATES_STATE_EUCLIDEAN_HPP_

#include <crocoddyl/core/state-base.hpp>

namespace crocoddyl {

class StateVector : public StateAbstract {
public:
  StateVector(int nx) : StateAbstract(nx, nx) { }
  Eigen::VectorXd zero() override {
    return Eigen::VectorXd::Zero(nx);
  }
  Eigen::VectorXd rand() override {
    return Eigen::VectorXd::Random(nx);
  }
  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0,
            const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout) override {
    dxout = x1 - x0;
  }
  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                 const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> xout) override {
    xout = x + dx;
  }
  void Jdiff(const Eigen::Ref<const Eigen::VectorXd>&,
             const Eigen::Ref<const Eigen::VectorXd>&,
             Eigen::Ref<Eigen::MatrixXd> Jfirst,
             Eigen::Ref<Eigen::MatrixXd> Jsecond,
             Jcomponent firstsecond = Jcomponent::both) override {
    switch (firstsecond) {
    case first: {
      Jfirst = Eigen::MatrixXd::Zero(ndx, ndx);
      Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx, -1.);
      break;
    } case second: {
      Jsecond.setIdentity(ndx, ndx);
      break;
    } case both: {
      Jfirst = Eigen::MatrixXd::Zero(ndx, ndx);
      Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx, -1.);
      Jsecond.setIdentity(ndx, ndx);
      break;
    } default: {
      Jfirst = Eigen::MatrixXd::Zero(ndx, ndx);
      Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx, -1.);
      Jsecond.setIdentity(ndx, ndx);
    }}
  }
  void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&,
                  const Eigen::Ref<const Eigen::VectorXd>&,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst,
                  Eigen::Ref<Eigen::MatrixXd> Jsecond,
                  Jcomponent firstsecond = Jcomponent::both) override {
    switch (firstsecond) {
    case first: {
      Jfirst.setIdentity(ndx, ndx);
      break;
    } case second: {
      Jsecond.setIdentity(ndx, ndx);
      break;
    } case both: {
      Jfirst.setIdentity(ndx, ndx);
      Jsecond.setIdentity(ndx, ndx);
      break;
    } default: {
      Jfirst.setIdentity(ndx, ndx);
      Jfirst *= -1.;
      Jsecond.setIdentity(ndx, ndx);
    }}
  }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_STATES_STATE_EUCLIDEAN_HPP_
