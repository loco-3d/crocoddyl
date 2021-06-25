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
ControlParametrizationModelNumDiffTpl<Scalar>::ControlParametrizationModelNumDiffTpl(boost::shared_ptr<Base> control)
    : Base(control->get_nu(), control->get_np()), control_(control), disturbance_(1e-6) {
  data_ = control_->createData();
}

template <typename Scalar>
ControlParametrizationModelNumDiffTpl<Scalar>::~ControlParametrizationModelNumDiffTpl() {}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data,
                                     double t, const Eigen::Ref<const VectorXs>& p) const
{
  control_->calc(data, t, p);
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data,
                                       double t, const Eigen::Ref<const VectorXs>& u) const
{
  control_->params(data, t, u);
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, 
    const Eigen::Ref<const VectorXs>& u_ub, Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const
{
  control_->convert_bounds(u_lb, u_ub, p_lb, p_ub);
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data,
                                         double t, const Eigen::Ref<const VectorXs>& p) const
{
  VectorXs tmp_p = VectorXs::Zero(np_);
  VectorXs u0 = VectorXs::Zero(nu_);
  calc(data, t, p);
  u0 = data->u;
  data->J.setZero();
  for (std::size_t i = 0; i < np_; ++i) {
    tmp_p = p;
    tmp_p(i) += disturbance_;
    calc(data, t, tmp_p);
    data->J.col(i) = data->u - u0;
  }
  data->J /= disturbance_;
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p, 
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const
{
  MatrixXs J(nu_, np_);
  calcDiff(data_, t, p);
  out.noalias() = A * data_->J;
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::multiplyJacobianTransposeBy(double t, const Eigen::Ref<const VectorXs>& p, 
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const
{
  MatrixXs J(nu_, np_);
  calcDiff(data_, t, p);
  out.noalias() = data_->J.transpose() * A;
}

template <typename Scalar>
const Scalar ControlParametrizationModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::set_disturbance(Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
