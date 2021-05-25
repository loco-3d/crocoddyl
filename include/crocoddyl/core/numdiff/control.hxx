///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ControlNumDiffTpl<Scalar>::ControlNumDiffTpl(boost::shared_ptr<Base> control)
    : Base(control->get_nu(), control->get_np()), control_(control), disturbance_(1e-6) {}

template <typename Scalar>
ControlNumDiffTpl<Scalar>::~ControlNumDiffTpl() {}

template <typename Scalar>
void ControlNumDiffTpl<Scalar>::resize(const std::size_t nu)
{
  control_->resize(nu);
  nu_ = nu;
  np_ = control_->get_np();
}

template <typename Scalar>
void ControlNumDiffTpl<Scalar>::value(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const
{
  control_->value(t, p, u_out);
}

template <typename Scalar>
void ControlNumDiffTpl<Scalar>::dValue(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const
{
  VectorXs tmp_p = VectorXs::Zero(np_);
  VectorXs u = VectorXs::Zero(nu_);
  if (static_cast<std::size_t>(J_out.rows()) != nu_ || static_cast<std::size_t>(J_out.cols()) != np_) {
    throw_pretty("Invalid argument: "
                  << "Jout has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(np_) +
                        ")");
  }
  value(t, p, u);
  J_out.setZero();
  for (std::size_t i = 0; i < np_; ++i) {
    tmp_p = p;
    tmp_p(i) += disturbance_;
    value(t, tmp_p, J_out.col(i));
    J_out.col(i) -= u;
  }
  J_out /= disturbance_;
}

template <typename Scalar>
void ControlNumDiffTpl<Scalar>::multiplyByDValue(double t, const Eigen::Ref<const VectorXs>& p, 
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const
{
  MatrixXs J(nu_, np_);
  dValue(t, p, J);
  out.noalias() = A * J;
}

template <typename Scalar>
void ControlNumDiffTpl<Scalar>::multiplyDValueTransposeBy(double t, const Eigen::Ref<const VectorXs>& p, 
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const
{
  MatrixXs J(nu_, np_);
  dValue(t, p, J);
  out.noalias() = J.transpose() * A;
}

template <typename Scalar>
const Scalar ControlNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ControlNumDiffTpl<Scalar>::set_disturbance(Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
