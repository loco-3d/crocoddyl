///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ParamsAbstractTpl<Scalar>::ParamsAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t np)
    : np_(np),
      state_(state),
      lb_(VectorXs::Constant(np, -std::numeric_limits<Scalar>::max())),
      ub_(VectorXs::Constant(np, std::numeric_limits<Scalar>::max())) {}

template <typename Scalar>
void ParamsAbstractTpl<Scalar>::update(
    const std::shared_ptr<ParamsDataAbstract>&,
    const Eigen::Ref<const VectorXs>&) {}

template <typename Scalar>
std::shared_ptr<typename ParamsAbstractTpl<Scalar>::ParamsDataAbstract>
ParamsAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<ParamsDataAbstract>(
      Eigen::aligned_allocator<ParamsDataAbstract>(), np_, 0);
}

template <typename Scalar>
bool ParamsAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  return data != nullptr && data->np == np_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ParamsAbstractTpl<Scalar>::zero() const {
  return VectorXs::Zero(np_);
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ParamsAbstractTpl<Scalar>::rand() const {
  return Scalar(0.5) *
         (vector_random_cast<Scalar, Eigen::Dynamic, Eigen::ColMajor,
                             Eigen::Dynamic>(static_cast<Eigen::Index>(np_))
              .array() +
          Scalar(1.))
             .matrix();
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar> >&
ParamsAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ParamsAbstractTpl<Scalar>::get_np() const {
  return np_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ParamsAbstractTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ParamsAbstractTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
void ParamsAbstractTpl<Scalar>::set_lb(const VectorXs& lb) {
  if (static_cast<std::size_t>(lb.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "lb has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  lb_ = lb;
}

template <typename Scalar>
void ParamsAbstractTpl<Scalar>::set_ub(const VectorXs& ub) {
  if (static_cast<std::size_t>(ub.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "ub has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  ub_ = ub;
}

template <typename Scalar>
void ParamsAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
std::shared_ptr<ParamsModelBase> ParamsAbstractTpl<Scalar>::cloneAsDouble()
    const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this parameter "
      "model");
}

template <typename Scalar>
std::shared_ptr<ParamsModelBase> ParamsAbstractTpl<Scalar>::cloneAsFloat()
    const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this parameter "
      "model");
}

#ifdef CROCODDYL_WITH_CODEGEN
template <typename Scalar>
std::shared_ptr<ParamsModelBase> ParamsAbstractTpl<Scalar>::cloneAsADDouble()
    const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this parameter "
      "model");
}

template <typename Scalar>
std::shared_ptr<ParamsModelBase> ParamsAbstractTpl<Scalar>::cloneAsADFloat()
    const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this parameter "
      "model");
}
#endif

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ParamsAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

template <typename Scalar>
ActionModelParamsAbstractTpl<Scalar>::ActionModelParamsAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t np)
    : Base(state, np) {}

template <typename Scalar>
typename ActionModelParamsAbstractTpl<Scalar>::MatrixXs
ActionModelParamsAbstractTpl<Scalar>::computeParamSensitivity_x(
    const std::shared_ptr<ActionDataAbstract>& data,
    const std::shared_ptr<ParamsDataAbstract>& params,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  MatrixXs dx_dp(this->state_->get_ndx(), this->np_);
  computeParamSensitivity(data, params, dx_dp, x, u);
  return dx_dp;
}

template <typename Scalar>
std::shared_ptr<
    typename ActionModelParamsAbstractTpl<Scalar>::ParamsDataAbstract>
ActionModelParamsAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<ActionModelParamsDataAbstractTpl<Scalar> >(
      Eigen::aligned_allocator<ActionModelParamsDataAbstractTpl<Scalar> >(),
      this->np_);
}

template <typename Scalar>
DynamicsParamsAbstractTpl<Scalar>::DynamicsParamsAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t np)
    : Base(state, np) {}

template <typename Scalar>
typename DynamicsParamsAbstractTpl<Scalar>::MatrixXs
DynamicsParamsAbstractTpl<Scalar>::computeJointTorqueRegressor_x(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const std::shared_ptr<ParamsDataAbstract>& params,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  MatrixXs dtau_dp(this->state_->get_nv(), this->np_);
  computeJointTorqueRegressor(data, params, dtau_dp, x, u);
  return dtau_dp;
}

template <typename Scalar>
std::shared_ptr<typename DynamicsParamsAbstractTpl<Scalar>::ParamsDataAbstract>
DynamicsParamsAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<DynamicsParamsDataAbstractTpl<Scalar> >(
      Eigen::aligned_allocator<DynamicsParamsDataAbstractTpl<Scalar> >(),
      this->np_);
}

}  // namespace crocoddyl
