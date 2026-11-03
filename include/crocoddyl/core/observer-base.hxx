///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ObserverModelAbstractTpl<Scalar>::ObserverModelAbstractTpl(
    std::shared_ptr<StateAbstractTpl<Scalar> > state, const std::size_t ntau,
    const std::size_t nu, const std::size_t nr, const std::size_t ng,
    const std::size_t nh, const std::size_t ng_T, const std::size_t nh_T,
    const std::size_t np)
    : Base(state, nu, nr, ng, nh, ng_T, nh_T, np),
      ntau_(ntau),
      tau_meas_(VectorXs::Zero(ntau)) {}

template <typename Scalar>
void ObserverModelAbstractTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  if (static_cast<std::size_t>(tau_meas.size()) != ntau_) {
    throw_pretty(
        "Invalid argument: " << "tau_meas has wrong dimension (it should be " +
                                    std::to_string(ntau_) + ")");
  }
  tau_meas_ = tau_meas;
}

template <typename Scalar>
void ObserverModelAbstractTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  Base::set_params(data, params);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ObserverModelAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<ObserverDataAbstract>(
      Eigen::aligned_allocator<ObserverDataAbstract>(), this);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ObserverModelAbstractTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>&) {
  return createData();
}

template <typename Scalar>
bool ObserverModelAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  return std::dynamic_pointer_cast<ObserverDataAbstract>(data) != nullptr;
}

template <typename Scalar>
std::size_t ObserverModelAbstractTpl<Scalar>::get_ntau() const {
  return ntau_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ObserverModelAbstractTpl<Scalar>::get_tau_meas() const {
  return tau_meas_;
}

}  // namespace crocoddyl
