///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActivationModelAbstract> activation,
    std::shared_ptr<ResidualModelAbstract> residual)
    : state_(state),
      activation_(activation),
      residual_(residual),
      nu_(residual->get_nu()),
      unone_(VectorXs::Zero(residual->get_nu())) {
  if (activation_->get_nr() != residual_->get_nr()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(residual_->get_nr()));
  }
}

template <typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActivationModelAbstract> activation, const std::size_t nu)
    : state_(state),
      activation_(activation),
      residual_(std::make_shared<ResidualModelAbstractTpl<Scalar>>(
          state, activation->get_nr(), nu)),
      nu_(nu),
      unone_(VectorXs::Zero(nu)) {}

template <typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActivationModelAbstract> activation)
    : state_(state),
      activation_(activation),
      residual_(std::make_shared<ResidualModelAbstractTpl<Scalar>>(
          state, activation->get_nr())),
      nu_(state->get_nv()),
      unone_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ResidualModelAbstract> residual)
    : state_(state),
      activation_(
          std::make_shared<ActivationModelQuadTpl<Scalar>>(residual->get_nr())),
      residual_(residual),
      nu_(residual->get_nu()),
      unone_(VectorXs::Zero(residual->get_nu())) {}

template <typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr,
    const std::size_t nu)
    : state_(state),
      activation_(std::make_shared<ActivationModelQuadTpl<Scalar>>(nr)),
      residual_(std::make_shared<ResidualModelAbstract>(state, nr, nu)),
      nu_(nu),
      unone_(VectorXs::Zero(nu)) {}

template <typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr)
    : state_(state),
      activation_(std::make_shared<ActivationModelQuadTpl<Scalar>>(nr)),
      residual_(std::make_shared<ResidualModelAbstract>(state, nr)),
      nu_(state->get_nv()),
      unone_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
void CostModelAbstractTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void CostModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
std::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelAbstractTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<CostDataAbstract>(
      Eigen::aligned_allocator<CostDataAbstract>(), this, data);
}

template <typename Scalar>
void CostModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar>>&
CostModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const std::shared_ptr<ActivationModelAbstractTpl<Scalar>>&
CostModelAbstractTpl<Scalar>::get_activation() const {
  return activation_;
}

template <typename Scalar>
const std::shared_ptr<ResidualModelAbstractTpl<Scalar>>&
CostModelAbstractTpl<Scalar>::get_residual() const {
  return residual_;
}

template <typename Scalar>
std::size_t CostModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
template <class ReferenceType>
void CostModelAbstractTpl<Scalar>::set_reference(ReferenceType ref) {
  set_referenceImpl(typeid(ref), &ref);
}

template <typename Scalar>
void CostModelAbstractTpl<Scalar>::set_referenceImpl(const std::type_info&,
                                                     const void*) {
  throw_pretty("It has not been implemented the set_referenceImpl() function");
}

template <typename Scalar>
template <class ReferenceType>
ReferenceType CostModelAbstractTpl<Scalar>::get_reference() {
#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has
                             // been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  ReferenceType ref;
  get_referenceImpl(typeid(ref), &ref);
  return ref;
#pragma GCC diagnostic pop
}

template <typename Scalar>
void CostModelAbstractTpl<Scalar>::get_referenceImpl(const std::type_info&,
                                                     void*) {
  throw_pretty("It has not been implemented the set_referenceImpl() function");
}

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const CostModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
