///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include <boost/make_shared.hpp>
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelAbstractTpl<Scalar>::ControlParametrizationModelAbstractTpl(const std::size_t nu,
                                                                                       const std::size_t np)
    : nu_(nu), np_(np) {}

template <typename Scalar>
ControlParametrizationModelAbstractTpl<Scalar>::~ControlParametrizationModelAbstractTpl() {}

template <typename Scalar>
boost::shared_ptr<ControlParametrizationDataAbstractTpl<Scalar> >
ControlParametrizationModelAbstractTpl<Scalar>::createData() {
  return boost::allocate_shared<ControlParametrizationDataAbstract>(
      Eigen::aligned_allocator<ControlParametrizationDataAbstract>(), this);
}

template <typename Scalar>
bool ControlParametrizationModelAbstractTpl<Scalar>::checkData(
    const boost::shared_ptr<ControlParametrizationDataAbstract>&) {
  return false;
}
// template <typename Scalar>
// typename MathBaseTpl<Scalar>::VectorXs ControlParametrizationModelAbstractTpl<Scalar>::calc_u(double t, const
// Eigen::Ref<const VectorXs>& p) const{
//   VectorXs u(nu_);
//   calc(t, p, u);
//   return u;
// }

// template <typename Scalar>
// typename MathBaseTpl<Scalar>::VectorXs ControlParametrizationModelAbstractTpl<Scalar>::params_p(double t, const
// Eigen::Ref<const VectorXs>& u) const{
//   VectorXs p(np_);
//   value_inv(t, u, p);
//   return p;
// }

// template <typename Scalar>
// typename MathBaseTpl<Scalar>::MatrixXs ControlParametrizationModelAbstractTpl<Scalar>::dValue_J(double t, const
// Eigen::Ref<const VectorXs>& p) const{
//   MatrixXs J(nu_, np_);
//   dValue(t, p, J);
//   return J;
// }

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs ControlParametrizationModelAbstractTpl<Scalar>::multiplyByJacobian_J(
    double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A) const {
  MatrixXs AJ(A.rows(), np_);
  multiplyByJacobian(t, p, A, AJ);
  return AJ;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs ControlParametrizationModelAbstractTpl<Scalar>::multiplyJacobianTransposeBy_J(
    double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A) const {
  MatrixXs JTA(np_, A.cols());
  multiplyJacobianTransposeBy(t, p, A, JTA);
  return JTA;
}

template <typename Scalar>
std::size_t ControlParametrizationModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ControlParametrizationModelAbstractTpl<Scalar>::get_np() const {
  return np_;
}

}  // namespace crocoddyl
