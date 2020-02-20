///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
StateNumDiffTpl<Scalar>::StateNumDiffTpl(boost::shared_ptr<Base> state)
    : Base(state->get_nx(), state->get_ndx()), state_(state), disturbance_(1e-6) {}

template <typename Scalar>
StateNumDiffTpl<Scalar>::~StateNumDiffTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateNumDiffTpl<Scalar>::zero() const {
  return state_->zero();
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateNumDiffTpl<Scalar>::rand() const {
  return state_->rand();
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                   Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  state_->diff(x0, x1, dxout);
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                        Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "xout has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  state_->integrate(x, dx, xout);
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                    Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                    Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  VectorXs tmp_x_ = VectorXs::Zero(nx_);
  VectorXs dx_ = VectorXs::Zero(ndx_);
  VectorXs dx0_ = VectorXs::Zero(ndx_);

  dx_.setZero();
  diff(x0, x1, dx0_);
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
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
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
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

template <typename Scalar>
void StateNumDiffTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                         Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                         Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  VectorXs tmp_x_ = VectorXs::Zero(nx_);
  VectorXs dx_ = VectorXs::Zero(ndx_);
  VectorXs x0_ = VectorXs::Zero(nx_);

  // x0_ = integrate(x, dx)
  integrate(x, dx, x0_);

  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
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
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
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

template <typename Scalar>
const Scalar& StateNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::set_disturbance(const Scalar& disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
