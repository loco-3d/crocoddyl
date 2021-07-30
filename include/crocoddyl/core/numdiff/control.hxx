///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include <iostream>

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelNumDiffTpl<Scalar>::ControlParametrizationModelNumDiffTpl(boost::shared_ptr<Base> control)
    : Base(control->get_nw(), control->get_np()), control_(control), disturbance_(1e-6) {
  data0_ = control_->createData();
  dataCalcDiff_ = control_->createData(); 
  dataNumDiff_ = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
ControlParametrizationModelNumDiffTpl<Scalar>::~ControlParametrizationModelNumDiffTpl() {}

template <typename Scalar>
boost::shared_ptr<ControlParametrizationDataAbstractTpl<Scalar> > ControlParametrizationModelNumDiffTpl<Scalar>::createData() {
  return control_->createData();
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::calc(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>& p) const {
  control_->calc(data, t, p);
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::params(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>& u) const {
  control_->params(data, t, u);
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::convertBounds(const Eigen::Ref<const VectorXs>& u_lb,
                                                                   const Eigen::Ref<const VectorXs>& u_ub,
                                                                   Eigen::Ref<VectorXs> p_lb,
                                                                   Eigen::Ref<VectorXs> p_ub) const {
  control_->convertBounds(u_lb, u_ub, p_lb, p_ub);
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>& p) const {
  data->J.setZero();
  for (std::size_t i = 0; i < np_; ++i) {
    dataNumDiff_->dp = p;
    dataNumDiff_->dp(i) += disturbance_;
    calc(dataCalcDiff_, t, dataNumDiff_->dp);
    data->J.col(i) = dataCalcDiff_->w - data->w;
  }
  data->J /= disturbance_;
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p,
                                                                       const Eigen::Ref<const MatrixXs>& A,
                                                                       Eigen::Ref<MatrixXs> out) const {
  MatrixXs J(nw_, np_);
  calc(data0_, t, p);
  calcDiff(data0_, t, p);
  out.noalias() = A * data0_->J;
}

template <typename Scalar>
void ControlParametrizationModelNumDiffTpl<Scalar>::multiplyJacobianTransposeBy(double t,
                                                                                const Eigen::Ref<const VectorXs>& p,
                                                                                const Eigen::Ref<const MatrixXs>& A,
                                                                                Eigen::Ref<MatrixXs> out) const {
  MatrixXs J(nw_, np_);
  calc(data0_, t, p);
  calcDiff(data0_, t, p);
  out.noalias() = data0_->J.transpose() * A;
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
