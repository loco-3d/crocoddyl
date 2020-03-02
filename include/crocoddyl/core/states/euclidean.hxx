///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
StateVectorTpl<Scalar>::StateVectorTpl(const std::size_t& nx) : StateAbstractTpl<Scalar>(nx, nx) {}

template <typename Scalar>
StateVectorTpl<Scalar>::~StateVectorTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateVectorTpl<Scalar>::zero() const {
  return MathBase::VectorXs::Zero(nx_);
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateVectorTpl<Scalar>::rand() const {
  return MathBase::VectorXs::Random(nx_);
}

template <typename Scalar>
void StateVectorTpl<Scalar>::diff(const Eigen::Ref<const typename MathBase::VectorXs>& x0,
                                  const Eigen::Ref<const typename MathBase::VectorXs>& x1,
                                  Eigen::Ref<typename MathBase::VectorXs> dxout) const {
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
  dxout = x1 - x0;
}

template <typename Scalar>
void StateVectorTpl<Scalar>::integrate(const Eigen::Ref<const typename MathBase::VectorXs>& x,
                                       const Eigen::Ref<const typename MathBase::VectorXs>& dx,
                                       Eigen::Ref<typename MathBase::VectorXs> xout) const {
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
  xout = x + dx;
}

template <typename Scalar>
void StateVectorTpl<Scalar>::Jdiff(const Eigen::Ref<const typename MathBase::VectorXs>&,
                                   const Eigen::Ref<const typename MathBase::VectorXs>&,
                                   Eigen::Ref<typename MathBase::MatrixXs> Jfirst,
                                   Eigen::Ref<typename MathBase::MatrixXs> Jsecond, Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    Jfirst.setZero();
    Jfirst.diagonal() = MathBase::VectorXs::Constant(ndx_, -1.);
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    Jsecond.setZero();
    Jsecond.diagonal() = MathBase::VectorXs::Constant(ndx_, 1.);
  }
}

template <typename Scalar>
void StateVectorTpl<Scalar>::Jintegrate(const Eigen::Ref<const typename MathBase::VectorXs>&,
                                        const Eigen::Ref<const typename MathBase::VectorXs>&,
                                        Eigen::Ref<typename MathBase::MatrixXs> Jfirst,
                                        Eigen::Ref<typename MathBase::MatrixXs> Jsecond,
                                        Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    Jfirst.setZero();
    Jfirst.diagonal() = MathBase::VectorXs::Constant(ndx_, 1.);
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    Jsecond.setZero();
    Jsecond.diagonal() = MathBase::VectorXs::Constant(ndx_, 1.);
  }
}

}  // namespace crocoddyl
