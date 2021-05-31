///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

template <typename Scalar>
ControlAbstractTpl<Scalar>::ControlAbstractTpl(const std::size_t nu, const std::size_t np)
    : nu_(nu),
      np_(np)
{}

template <typename Scalar>
ControlAbstractTpl<Scalar>::ControlAbstractTpl()
    : nu_(0),
      np_(0)
      {}

template <typename Scalar>
ControlAbstractTpl<Scalar>::~ControlAbstractTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ControlAbstractTpl<Scalar>::value_u(double t, const Eigen::Ref<const VectorXs>& p) const{
  VectorXs u(nu_);
  value(t, p, u);
  return u;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ControlAbstractTpl<Scalar>::value_inv_p(double t, const Eigen::Ref<const VectorXs>& u) const{
  VectorXs p(np_);
  value_inv(t, u, p);
  return p;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs ControlAbstractTpl<Scalar>::dValue_J(double t, const Eigen::Ref<const VectorXs>& p) const{
  MatrixXs J(nu_, np_);
  dValue(t, p, J);
  return J;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs ControlAbstractTpl<Scalar>::multiplyByDValue_J(double t, const Eigen::Ref<const VectorXs>& p, 
      const Eigen::Ref<const MatrixXs>& A) const{
  MatrixXs AJ(A.rows(), np_);
  multiplyByDValue(t, p, A, AJ);
  return AJ;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs ControlAbstractTpl<Scalar>::multiplyDValueTransposeBy_J(double t, const Eigen::Ref<const VectorXs>& p, 
      const Eigen::Ref<const MatrixXs>& A) const{
  MatrixXs JTA(np_, A.cols());
  multiplyDValueTransposeBy(t, p, A, JTA);
  return JTA;
}

template <typename Scalar>
std::size_t ControlAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ControlAbstractTpl<Scalar>::get_np() const {
  return np_;
}

}  // namespace crocoddyl
