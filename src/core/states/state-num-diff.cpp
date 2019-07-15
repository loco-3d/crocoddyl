///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "crocoddyl/core/states/state-num-diff.hpp"

namespace crocoddyl {

StateNumDiff::StateNumDiff(StateAbstract& state) : StateAbstract(state.get_nx(), state.get_ndx()), state_(state) {
  disturbance_ = 1e-6;
  // disturbance vector
  dx_.resize(ndx_);
  dx_.setZero();
  // State around which to compute the finite integrale jacobians
  x0_.resize(nx_);
  x0_.setZero();
  // State difference around which to compute the finite difference jacobians
  dx0_.resize(ndx_);
  dx0_.setZero();
  // temporary variable needed
  tmp_x_.resize(nx_);
  tmp_x_.setZero();
}

StateNumDiff::~StateNumDiff() {}

Eigen::VectorXd StateNumDiff::zero() { return state_.zero(); }

Eigen::VectorXd StateNumDiff::rand() { return state_.rand(); }

void StateNumDiff::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                        Eigen::Ref<Eigen::VectorXd> dxout) {
  state_.diff(x0, x1, dxout);
}

void StateNumDiff::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                             Eigen::Ref<Eigen::VectorXd> xout) {
  state_.integrate(x, dx, xout);
}

void StateNumDiff::Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x1, const Eigen::Ref<const Eigen::VectorXd>& x2,
                         Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                         Jcomponent firstsecond) {
  assert((firstsecond == Jcomponent::first || firstsecond == Jcomponent::second || firstsecond == Jcomponent::both) &&
         ("StateNumDiff::Jdiff: firstsecond must be one of the Jcomponent "
          "{both, first, second }"));
  dx_.setZero();
  diff(x1, x2, dx0_);
  if (firstsecond == Jcomponent::first || firstsecond == Jcomponent::both) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x = int(x_1, d_x)
      integrate(x1, dx_, tmp_x_);
      // Jfirst[:,k] = diff(tmp_x, x2) = diff(int(x_1 + d_x), x2)
      diff(tmp_x_, x2, Jfirst.col(i));
      // Jfirst[:,k] = Jfirst[:,k] - tmp_dx_, or
      // Jfirst[:,k] = Jfirst[:,k] - diff(x1, x2)
      Jfirst.col(i) -= dx0_;
      dx_(i) = 0.0;
    }
    Jfirst /= disturbance_;
  }
  if (firstsecond == Jcomponent::second || firstsecond == Jcomponent::both) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");

    Jsecond.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x = int(x_2 + d_x)
      integrate(x2, dx_, tmp_x_);
      // Jsecond[:,k] = diff(x1, tmp_x) = diff(x1, int(x_2 + d_x))
      diff(x1, tmp_x_, Jsecond.col(i));
      // Jsecond[:,k] = J[:,k] - tmp_dx_
      // Jsecond[:,k] = Jsecond[:,k] - diff(x1, x2)
      Jsecond.col(i) -= dx0_;
      dx_(i) = 0.0;
    }
    Jsecond /= disturbance_;
  }
}

void StateNumDiff::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                              Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                              Jcomponent firstsecond) {
  assert((firstsecond == Jcomponent::first || firstsecond == Jcomponent::second || firstsecond == Jcomponent::both) &&
         ("StateNumDiff::Jdiff: firstsecond must be one of the Jcomponent "
          "{both, first, second }"));
  dx_.setZero();
  // x0_ = integrate(x, dx)
  integrate(x, dx, x0_);

  if (firstsecond == Jcomponent::first || firstsecond == Jcomponent::both) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x_ = integrate(x, dx_) = integrate(x, disturbance_vector)
      integrate(x, dx_, tmp_x_);
      // tmp_x_ = integrate(tmp_x_, dx) = integrate(integrate(x, dx_), dx)
      integrate(tmp_x_, dx, tmp_x_);
      // Jfirst[:,i] = diff(x0_, tmp_x_)
      // Jfirst[:,i] = diff( integrate(x, dx), integrate(integrate(x, dx_), dx))
      diff(x0_, tmp_x_, Jfirst.col(i));
      dx_(i) = 0.0;
    }
    Jfirst /= disturbance_;
  }
  if (firstsecond == Jcomponent::second || firstsecond == Jcomponent::both) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");
    Jsecond.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x_ = integrate(x, dx + dx_) = integrate(x, dx + disturbance_vector)
      integrate(x, dx + dx_, tmp_x_);
      // Jsecond[:,i] = diff(x0_, tmp_x_)
      // Jsecond[:,i] = diff( integrate(x, dx), integrate(x, dx_ + dx) )
      diff(x0_, tmp_x_, Jsecond.col(i));
      dx_(i) = 0.0;
    }
    Jsecond /= disturbance_;
  }
}

}  // namespace crocoddyl
