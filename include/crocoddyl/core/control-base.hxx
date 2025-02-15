///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/make_shared.hpp>

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelAbstractTpl<
    Scalar>::ControlParametrizationModelAbstractTpl(const std::size_t nw,
                                                    const std::size_t nu)
    : nw_(nw), nu_(nu) {}

template <typename Scalar>
ControlParametrizationModelAbstractTpl<
    Scalar>::~ControlParametrizationModelAbstractTpl() {}

template <typename Scalar>
std::shared_ptr<ControlParametrizationDataAbstractTpl<Scalar> >
ControlParametrizationModelAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<ControlParametrizationDataAbstract>(
      Eigen::aligned_allocator<ControlParametrizationDataAbstract>(), this);
}

template <typename Scalar>
bool ControlParametrizationModelAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<ControlParametrizationDataAbstract>&) {
  return false;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs
ControlParametrizationModelAbstractTpl<Scalar>::multiplyByJacobian_J(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Eigen::Ref<const MatrixXs>& A, const AssignmentOp op) const {
  MatrixXs AJ(A.rows(), nu_);
  multiplyByJacobian(data, A, AJ, op);
  return AJ;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs
ControlParametrizationModelAbstractTpl<Scalar>::multiplyJacobianTransposeBy_J(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Eigen::Ref<const MatrixXs>& A, const AssignmentOp op) const {
  MatrixXs JTA(nu_, A.cols());
  multiplyJacobianTransposeBy(data, A, JTA, op);
  return JTA;
}

template <typename Scalar>
std::size_t ControlParametrizationModelAbstractTpl<Scalar>::get_nw() const {
  return nw_;
}

template <typename Scalar>
std::size_t ControlParametrizationModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

}  // namespace crocoddyl
