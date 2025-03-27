///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelAbstractTpl<
    Scalar>::ControlParametrizationModelAbstractTpl(const std::size_t nw,
                                                    const std::size_t nu)
    : nw_(nw), nu_(nu) {}

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
std::ostream& operator<<(
    std::ostream& os,
    const ControlParametrizationModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

template <typename Scalar>
void ControlParametrizationModelAbstractTpl<Scalar>::print(
    std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
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
