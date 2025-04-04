///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          New York University, Max Planck Gesellschaft,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
StateNumDiffTpl<Scalar>::StateNumDiffTpl(std::shared_ptr<Base> state)
    : Base(state->get_nx(), state->get_ndx()),
      state_(state),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

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
void StateNumDiffTpl<Scalar>::diff(const Eigen::Ref<const VectorXs>& x0,
                                   const Eigen::Ref<const VectorXs>& x1,
                                   Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x1 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw_pretty(
        "Invalid argument: " << "dxout has wrong dimension (it should be " +
                                    std::to_string(ndx_) + ")");
  }
  state_->diff(x0, x1, dxout);
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs>& x,
                                        const Eigen::Ref<const VectorXs>& dx,
                                        Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty(
        "Invalid argument: " << "dx has wrong dimension (it should be " +
                                    std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "xout has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  state_->integrate(x, dx, xout);
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs>& x0,
                                    const Eigen::Ref<const VectorXs>& x1,
                                    Eigen::Ref<MatrixXs> Jfirst,
                                    Eigen::Ref<MatrixXs> Jsecond,
                                    Jcomponent firstsecond) const {
  assert_pretty(
      is_a_Jcomponent(firstsecond),
      ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x1 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  VectorXs tmp_x_ = VectorXs::Zero(nx_);
  VectorXs dx_ = VectorXs::Zero(ndx_);
  VectorXs dx0_ = VectorXs::Zero(ndx_);

  dx_.setZero();
  diff(x0, x1, dx0_);
  if (firstsecond == first || firstsecond == both) {
    const Scalar x0h_jac = e_jac_ * std::max(Scalar(1.), x0.norm());
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jfirst has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    Jfirst.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
      dx_(i) = x0h_jac;
      integrate(x0, dx_, tmp_x_);
      diff(tmp_x_, x1, Jfirst.col(i));
      Jfirst.col(i) -= dx0_;
      dx_(i) = Scalar(0.);
    }
    Jfirst /= x0h_jac;
  }
  if (firstsecond == second || firstsecond == both) {
    const Scalar x1h_jac = e_jac_ * std::max(Scalar(1.), x1.norm());
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jsecond has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }

    Jsecond.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
      dx_(i) = x1h_jac;
      integrate(x1, dx_, tmp_x_);
      diff(x0, tmp_x_, Jsecond.col(i));
      Jsecond.col(i) -= dx0_;
      dx_(i) = Scalar(0.);
    }
    Jsecond /= x1h_jac;
  }
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs>& x,
                                         const Eigen::Ref<const VectorXs>& dx,
                                         Eigen::Ref<MatrixXs> Jfirst,
                                         Eigen::Ref<MatrixXs> Jsecond,
                                         const Jcomponent firstsecond,
                                         const AssignmentOp) const {
  assert_pretty(
      is_a_Jcomponent(firstsecond),
      ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty(
        "Invalid argument: " << "dx has wrong dimension (it should be " +
                                    std::to_string(ndx_) + ")");
  }
  VectorXs tmp_x_ = VectorXs::Zero(nx_);
  VectorXs dx_ = VectorXs::Zero(ndx_);
  VectorXs x0_ = VectorXs::Zero(nx_);

  // x0_ = integrate(x, dx)
  integrate(x, dx, x0_);

  if (firstsecond == first || firstsecond == both) {
    const Scalar xh_jac = e_jac_ * std::max(Scalar(1.), x.norm());
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jfirst has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    Jfirst.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
      dx_(i) = xh_jac;
      integrate(x, dx_, tmp_x_);
      integrate(tmp_x_, dx, tmp_x_);
      diff(x0_, tmp_x_, Jfirst.col(i));
      dx_(i) = Scalar(0.);
    }
    Jfirst /= xh_jac;
  }
  if (firstsecond == second || firstsecond == both) {
    const Scalar dxh_jac = e_jac_ * std::max(Scalar(1.), dx.norm());
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jsecond has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    Jsecond.setZero();
    for (std::size_t i = 0; i < ndx_; ++i) {
      dx_(i) = dxh_jac;
      integrate(x, dx + dx_, tmp_x_);
      diff(x0_, tmp_x_, Jsecond.col(i));
      dx_(i) = Scalar(0.);
    }
    Jsecond /= dxh_jac;
  }
}

template <typename Scalar>
template <typename NewScalar>
StateNumDiffTpl<NewScalar> StateNumDiffTpl<Scalar>::cast() const {
  typedef StateNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(state_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::JintegrateTransport(
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&,
    Eigen::Ref<MatrixXs>, const Jcomponent) const {}

template <typename Scalar>
const Scalar StateNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::set_disturbance(Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

template <typename Scalar>
void StateNumDiffTpl<Scalar>::print(std::ostream& os) const {
  os << "StateNumDiff {state=" << *state_ << "}";
}

}  // namespace crocoddyl
