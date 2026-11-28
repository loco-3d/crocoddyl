///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActuationMultibodyParamsTpl<Scalar>::ActuationMultibodyParamsTpl(
    std::shared_ptr<ActuationModelMultibody> actuation)
    : Base(actuation == nullptr ? std::shared_ptr<StateAbstract>()
                                : std::static_pointer_cast<StateAbstract>(
                                      actuation->get_state()),
           actuation == nullptr ? 0 : actuation->get_np()),
      actuation_(actuation) {
  if (actuation_ == nullptr) {
    throw_pretty("Invalid argument: actuation is null");
  }
}

template <typename Scalar>
std::shared_ptr<
    typename ActuationMultibodyParamsTpl<Scalar>::ActuationMultibodyParamsData>
ActuationMultibodyParamsTpl<Scalar>::castData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  std::shared_ptr<ActuationMultibodyParamsData> d =
      std::dynamic_pointer_cast<ActuationMultibodyParamsData>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data is not an ActuationMultibodyParamsData");
  }
  return d;
}

template <typename Scalar>
void ActuationMultibodyParamsTpl<Scalar>::update(
    const std::shared_ptr<ParamsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(this->np_) + ")");
  }
  const std::shared_ptr<ActuationMultibodyParamsData> d = castData(data);
  d->p = p;
  actuation_->update_p(p, d->gamma);
  actuation_->updateParametrizationDerivative(d->dgamma_dp);
}

template <typename Scalar>
void ActuationMultibodyParamsTpl<Scalar>::computeJointTorqueRegressor(
    const std::shared_ptr<DynamicsDataAbstract>&,
    const std::shared_ptr<ParamsDataAbstract>& params,
    Eigen::Ref<MatrixXs> dtau_dp, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  castData(params);
  actuation_->computeJointTorqueRegressor(dtau_dp, x, u);
}

template <typename Scalar>
std::shared_ptr<
    typename ActuationMultibodyParamsTpl<Scalar>::ParamsDataAbstract>
ActuationMultibodyParamsTpl<Scalar>::createData() {
  return std::allocate_shared<ActuationMultibodyParamsData>(
      Eigen::aligned_allocator<ActuationMultibodyParamsData>(), this);
}

template <typename Scalar>
bool ActuationMultibodyParamsTpl<Scalar>::checkData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  return std::dynamic_pointer_cast<ActuationMultibodyParamsData>(data) !=
             nullptr &&
         Base::checkData(data);
}

template <typename Scalar>
const std::shared_ptr<
    typename ActuationMultibodyParamsTpl<Scalar>::ActuationModelMultibody>&
ActuationMultibodyParamsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
template <typename NewScalar>
ActuationMultibodyParamsTpl<NewScalar>
ActuationMultibodyParamsTpl<Scalar>::cast() const {
  typedef ActuationMultibodyParamsTpl<NewScalar> ReturnType;
  typedef ActuationModelMultibodyTpl<NewScalar> ActuationModelMultibodyNew;
  ReturnType ret(std::make_shared<ActuationModelMultibodyNew>(
      actuation_->template cast<NewScalar>()));
  ret.set_lb(this->lb_.template cast<NewScalar>());
  ret.set_ub(this->ub_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void ActuationMultibodyParamsTpl<Scalar>::print(std::ostream& os) const {
  os << "ActuationMultibodyParams {np=" << this->np_ << "}";
}

}  // namespace crocoddyl
