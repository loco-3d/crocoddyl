///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/state.hpp"

namespace crocoddyl {

StateNumDiff::StateNumDiff(boost::shared_ptr<StateAbstract> state)
    : StateAbstract(state->get_nx(), state->get_ndx()), state_(state), disturbance_(1e-6) {}

StateNumDiff::~StateNumDiff() {}

Eigen::VectorXd StateNumDiff::zero() const { return state_->zero(); }

Eigen::VectorXd StateNumDiff::rand() const { return state_->rand(); }

void StateNumDiff::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                        Eigen::Ref<Eigen::VectorXd> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw std::invalid_argument("x0 has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw std::invalid_argument("x1 has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw std::invalid_argument("dxout has wrong dimension (it should be " + to_string(ndx_) + ")");
  }
  state_->diff(x0, x1, dxout);
}

void StateNumDiff::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                             Eigen::Ref<Eigen::VectorXd> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw std::invalid_argument("x has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw std::invalid_argument("dx has wrong dimension (it should be " + to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw std::invalid_argument("xout has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  state_->integrate(x, dx, xout);
}

void StateNumDiff::Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                         Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                         Jcomponent firstsecond) const {
  assert(is_a_Jcomponent(firstsecond) && ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw std::invalid_argument("x0 has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw std::invalid_argument("x1 has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  Eigen::VectorXd tmp_x_ = Eigen::VectorXd::Zero(nx_);
  Eigen::VectorXd dx_ = Eigen::VectorXd::Zero(ndx_);
  Eigen::VectorXd dx0_ = Eigen::VectorXd::Zero(ndx_);

  dx_.setZero();
  diff(x0, x1, dx0_);
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw std::invalid_argument("Jfirst has wrong dimension (it should be " + to_string(ndx_) + "," +
                                  to_string(ndx_) + ")");
    }
    Jfirst.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x = int(x0, dx)
      integrate(x0, dx_, tmp_x_);
      // Jfirst[:,k] = diff(tmp_x, x1) = diff(int(x0 + dx), x1)
      diff(tmp_x_, x1, Jfirst.col(i));
      // Jfirst[:,k] = Jfirst[:,k] - tmp_dx_, or
      // Jfirst[:,k] = Jfirst[:,k] - diff(x0, x1)
      Jfirst.col(i) -= dx0_;
      dx_(i) = 0.0;
    }
    Jfirst /= disturbance_;
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw std::invalid_argument("Jsecond has wrong dimension (it should be " + to_string(ndx_) + "," +
                                  to_string(ndx_) + ")");
    }

    Jsecond.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x = int(x1 + dx)
      integrate(x1, dx_, tmp_x_);
      // Jsecond[:,k] = diff(x0, tmp_x) = diff(x0, int(x1 + dx))
      diff(x0, tmp_x_, Jsecond.col(i));
      // Jsecond[:,k] = J[:,k] - tmp_dx_
      // Jsecond[:,k] = Jsecond[:,k] - diff(x0, x1)
      Jsecond.col(i) -= dx0_;
      dx_(i) = 0.0;
    }
    Jsecond /= disturbance_;
  }
}

void StateNumDiff::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                              Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                              Jcomponent firstsecond) const {
  assert(is_a_Jcomponent(firstsecond) && ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw std::invalid_argument("x has wrong dimension (it should be " + to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw std::invalid_argument("dx has wrong dimension (it should be " + to_string(ndx_) + ")");
  }
  Eigen::VectorXd tmp_x_ = Eigen::VectorXd::Zero(nx_);
  Eigen::VectorXd dx_ = Eigen::VectorXd::Zero(ndx_);
  Eigen::VectorXd x0_ = Eigen::VectorXd::Zero(nx_);

  // x0_ = integrate(x, dx)
  integrate(x, dx, x0_);

  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw std::invalid_argument("Jfirst has wrong dimension (it should be " + to_string(ndx_) + "," +
                                  to_string(ndx_) + ")");
    }
    Jfirst.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
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
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw std::invalid_argument("Jsecond has wrong dimension (it should be " + to_string(ndx_) + "," +
                                  to_string(ndx_) + ")");
    }
    Jsecond.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
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

const double& StateNumDiff::get_disturbance() const { return disturbance_; }

void StateNumDiff::set_disturbance(const double& disturbance) {
  if (disturbance < 0.) {
    throw std::invalid_argument("Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
