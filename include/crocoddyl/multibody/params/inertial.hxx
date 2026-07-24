///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
MultibodyInertialParamsTpl<Scalar>::MultibodyInertialParamsTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<InertialParametrizationAbstract> parametrization)
    : MultibodyInertialParamsTpl(state, parametrization,
                                 createDefaultBodyNames(state)) {}

template <typename Scalar>
std::vector<std::string>
MultibodyInertialParamsTpl<Scalar>::createDefaultBodyNames(
    const std::shared_ptr<StateMultibody>& state) {
  if (state == nullptr) {
    return std::vector<std::string>();
  }
  const std::vector<std::string>& names = state->get_pinocchio()->names;
  return std::vector<std::string>(names.begin() + 1, names.end());
}

template <typename Scalar>
MultibodyInertialParamsTpl<Scalar>::MultibodyInertialParamsTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<InertialParametrizationAbstract> parametrization,
    const std::vector<std::string>& body_names)
    : Base(state, body_names.size() * kParametersPerBody),
      state_multibody_(state),
      parametrization_(parametrization),
      joint_ids_(),
      layout_version_(0) {
  if (state_multibody_ == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
  if (parametrization_ == nullptr) {
    throw_pretty("Invalid argument: parametrization is null");
  }
  joint_ids_ = resolveBodyIds(body_names);
  this->np_ = joint_ids_.size() * kParametersPerBody;
}

template <typename Scalar>
std::vector<pinocchio::JointIndex>
MultibodyInertialParamsTpl<Scalar>::resolveBodyIds(
    const std::vector<std::string>& body_names) const {
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state_multibody_->get_pinocchio();
  std::vector<pinocchio::JointIndex> ids;
  ids.reserve(body_names.size());
  for (std::vector<std::string>::const_iterator it = body_names.begin();
       it != body_names.end(); ++it) {
    pinocchio::JointIndex id = 0;
    if (pin_model->existJointName(*it)) {
      id = pin_model->getJointId(*it);
    } else if (pin_model->existFrame(*it)) {
      id = pin_model->frames[pin_model->getFrameId(*it)].parentJoint;
    } else {
      throw_pretty("Invalid argument: body '" << *it << "' does not exist");
    }
    if (id == 0) {
      throw_pretty("Invalid argument: body '" << *it
                                              << "' resolves to the universe");
    }
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
      throw_pretty("Invalid argument: body '" << *it << "' is duplicated");
    }
    ids.push_back(id);
  }
  return ids;
}

template <typename Scalar>
std::shared_ptr<
    typename MultibodyInertialParamsTpl<Scalar>::MultibodyInertialParamsData>
MultibodyInertialParamsTpl<Scalar>::castData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  std::shared_ptr<MultibodyInertialParamsData> d =
      std::dynamic_pointer_cast<MultibodyInertialParamsData>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data is not a MultibodyInertialParamsData");
  }
  return d;
}

template <typename Scalar>
void MultibodyInertialParamsTpl<Scalar>::update(
    const std::shared_ptr<ParamsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(this->np_) + ")");
  }
  const std::shared_ptr<MultibodyInertialParamsData> d = castData(data);
  if (!checkData(data)) {
    throw_pretty(
        "Invalid argument: inertial parameter data layout is stale; recreate "
        "data after changing body status");
  }
  pinocchio::ModelTpl<Scalar>& pin_model = *state_multibody_->get_pinocchio();
  d->p = p;
  d->dtau_dp.setZero();
  for (std::size_t j = 0; j < joint_ids_.size(); ++j) {
    const Eigen::Index offset =
        static_cast<Eigen::Index>(j * kParametersPerBody);
    parametrization_->fromParametrization(
        d->parametrization, d->psi[j], p.segment(offset, kParametersPerBody));
    pin_model.inertias[joint_ids_[j]] =
        pinocchio::InertiaTpl<Scalar>::FromDynamicParameters(d->psi[j]);
    parametrization_->updateParametrizationDerivative(
        d->parametrization, d->dpsi_dp[j],
        p.segment(offset, kParametersPerBody), d->psi[j]);
  }
}

template <typename Scalar>
void MultibodyInertialParamsTpl<Scalar>::computeJointTorqueRegressor(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const std::shared_ptr<ParamsDataAbstract>& params,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  (void)u;
  if (data == nullptr) {
    throw_pretty("Invalid argument: dynamics data is null");
  }
  const std::shared_ptr<MultibodyInertialParamsData> d = castData(params);
  if (!checkData(params)) {
    throw_pretty(
        "Invalid argument: inertial parameter data layout is stale; recreate "
        "data after changing body status");
  }
  if (data->shared == nullptr) {
    throw_pretty("Invalid argument: dynamics data has no shared collector");
  }
  DataCollectorMultibody* const multibody =
      dynamic_cast<DataCollectorMultibody*>(data->shared);
  if (multibody == nullptr) {
    throw_pretty(
        "Invalid argument: dynamics data does not contain a multibody "
        "collector");
  }
  if (multibody->pinocchio == nullptr) {
    throw_pretty(
        "Invalid argument: multibody collector has null Pinocchio data");
  }
  const std::size_t nv = state_multibody_->get_nv();
  if (static_cast<std::size_t>(data->vdot.size()) != nv) {
    throw_pretty(
        "Invalid argument: dynamics data does not expose a "
        "continuous-time acceleration vector");
  }

  const std::size_t nq = state_multibody_->get_nq();
  const Eigen::Ref<const VectorXs> q = x.head(nq);
  const Eigen::Ref<const VectorXs> v = x.tail(nv);

  pinocchio::computeJointTorqueRegressor(*state_multibody_->get_pinocchio(),
                                         *multibody->pinocchio, q, v,
                                         data->vdot);
  d->dtau_dp.setZero();
  const MatrixXs& dtau_dpsi = multibody->pinocchio->jointTorqueRegressor;
  for (std::size_t j = 0; j < joint_ids_.size(); ++j) {
    const Eigen::Index input_offset =
        static_cast<Eigen::Index>((joint_ids_[j] - 1) * kParametersPerBody);
    const Eigen::Index output_offset =
        static_cast<Eigen::Index>(j * kParametersPerBody);
    d->dtau_dp.middleCols(output_offset, kParametersPerBody).noalias() =
        dtau_dpsi.middleCols(input_offset, kParametersPerBody) * d->dpsi_dp[j];
  }
}

template <typename Scalar>
std::shared_ptr<typename MultibodyInertialParamsTpl<Scalar>::ParamsDataAbstract>
MultibodyInertialParamsTpl<Scalar>::createData() {
  return std::allocate_shared<MultibodyInertialParamsData>(
      Eigen::aligned_allocator<MultibodyInertialParamsData>(), this);
}

template <typename Scalar>
bool MultibodyInertialParamsTpl<Scalar>::checkData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  const std::shared_ptr<MultibodyInertialParamsData> d =
      std::dynamic_pointer_cast<MultibodyInertialParamsData>(data);
  if (d == nullptr || d->layout_version_ != layout_version_ ||
      d->psi.size() != joint_ids_.size() ||
      d->dpsi_dp.size() != joint_ids_.size() ||
      !parametrization_->checkData(d->parametrization) ||
      !Base::checkData(data)) {
    return false;
  }
  return true;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
MultibodyInertialParamsTpl<Scalar>::zero() const {
  VectorXs p_zero = VectorXs::Zero(this->np_);
  const pinocchio::ModelTpl<Scalar>& pin_model =
      *state_multibody_->get_pinocchio();
  for (std::size_t j = 0; j < joint_ids_.size(); ++j) {
    const Eigen::Index offset =
        static_cast<Eigen::Index>(j * kParametersPerBody);
    parametrization_->toParametrization(
        p_zero.segment(offset, kParametersPerBody),
        pin_model.inertias[joint_ids_[j]].toDynamicParameters());
  }
  return p_zero;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
MultibodyInertialParamsTpl<Scalar>::rand() const {
  return Scalar(0.5) * (vector_random_cast<Scalar, Eigen::Dynamic,
                                           Eigen::ColMajor, Eigen::Dynamic>(
                            static_cast<Eigen::Index>(this->np_))
                            .array() +
                        Scalar(1.))
                           .matrix();
}

template <typename Scalar>
void MultibodyInertialParamsTpl<Scalar>::changeBodyStatus(
    const std::string& body_name, const bool active) {
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state_multibody_->get_pinocchio();
  pinocchio::JointIndex joint_id = 0;
  if (pin_model->existJointName(body_name)) {
    joint_id = pin_model->getJointId(body_name);
  } else if (pin_model->existFrame(body_name)) {
    joint_id = pin_model->frames[pin_model->getFrameId(body_name)].parentJoint;
  } else {
    std::cerr << "Warning: we couldn't change the status of body '" << body_name
              << "', it doesn't exist." << std::endl;
    return;
  }
  if (joint_id == 0) {
    std::cerr << "Warning: we couldn't change the status of body '" << body_name
              << "', it resolves to the universe." << std::endl;
    return;
  }

  typename std::vector<pinocchio::JointIndex>::iterator it =
      std::find(joint_ids_.begin(), joint_ids_.end(), joint_id);
  if (active) {
    if (it != joint_ids_.end()) {
      return;
    }
    joint_ids_.push_back(joint_id);
    this->np_ += kParametersPerBody;
    this->lb_.conservativeResize(static_cast<Eigen::Index>(this->np_));
    this->ub_.conservativeResize(static_cast<Eigen::Index>(this->np_));
    this->lb_.tail(kParametersPerBody)
        .setConstant(-std::numeric_limits<Scalar>::max());
    this->ub_.tail(kParametersPerBody)
        .setConstant(std::numeric_limits<Scalar>::max());
  } else {
    if (it == joint_ids_.end()) {
      return;
    }
    const std::size_t block =
        static_cast<std::size_t>(std::distance(joint_ids_.begin(), it));
    const Eigen::Index offset =
        static_cast<Eigen::Index>(block * kParametersPerBody);
    const Eigen::Index old_np = static_cast<Eigen::Index>(this->np_);
    const Eigen::Index new_np =
        old_np - static_cast<Eigen::Index>(kParametersPerBody);
    VectorXs lb(new_np);
    VectorXs ub(new_np);
    if (offset != 0) {
      lb.head(offset) = this->lb_.head(offset);
      ub.head(offset) = this->ub_.head(offset);
    }
    const Eigen::Index trailing =
        old_np - offset - static_cast<Eigen::Index>(kParametersPerBody);
    if (trailing != 0) {
      lb.tail(trailing) = this->lb_.tail(trailing);
      ub.tail(trailing) = this->ub_.tail(trailing);
    }
    joint_ids_.erase(it);
    this->np_ = static_cast<std::size_t>(new_np);
    this->lb_.swap(lb);
    this->ub_.swap(ub);
  }
  ++layout_version_;
}

template <typename Scalar>
const std::shared_ptr<typename MultibodyInertialParamsTpl<
    Scalar>::InertialParametrizationAbstract>&
MultibodyInertialParamsTpl<Scalar>::get_parametrization() const {
  return parametrization_;
}

template <typename Scalar>
const std::vector<pinocchio::JointIndex>&
MultibodyInertialParamsTpl<Scalar>::get_joint_ids() const {
  return joint_ids_;
}

template <typename Scalar>
std::vector<std::string> MultibodyInertialParamsTpl<Scalar>::get_body_names()
    const {
  std::vector<std::string> body_names;
  body_names.reserve(joint_ids_.size());
  const std::vector<std::string>& names =
      state_multibody_->get_pinocchio()->names;
  for (std::vector<pinocchio::JointIndex>::const_iterator it =
           joint_ids_.begin();
       it != joint_ids_.end(); ++it) {
    body_names.push_back(names[*it]);
  }
  return body_names;
}

template <typename Scalar>
template <typename NewScalar>
MultibodyInertialParamsTpl<NewScalar> MultibodyInertialParamsTpl<Scalar>::cast()
    const {
  typedef MultibodyInertialParamsTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateMultibodyNew;
  ReturnType ret(std::make_shared<StateMultibodyNew>(
                     state_multibody_->template cast<NewScalar>()),
                 parametrization_->template cast<NewScalar>(),
                 get_body_names());
  typename MathBaseTpl<NewScalar>::VectorXs lb =
      this->lb_.template cast<NewScalar>();
  typename MathBaseTpl<NewScalar>::VectorXs ub =
      this->ub_.template cast<NewScalar>();
  for (Eigen::Index i = 0; i < this->lb_.size(); ++i) {
    if (this->lb_[i] == -std::numeric_limits<Scalar>::max()) {
      lb[i] = -std::numeric_limits<NewScalar>::max();
    }
    if (this->ub_[i] == std::numeric_limits<Scalar>::max()) {
      ub[i] = std::numeric_limits<NewScalar>::max();
    }
  }
  ret.set_lb(lb);
  ret.set_ub(ub);
  return ret;
}

template <typename Scalar>
void MultibodyInertialParamsTpl<Scalar>::print(std::ostream& os) const {
  os << "MultibodyInertialParams {nbodies=" << joint_ids_.size()
     << ", np=" << this->np_ << ", parametrization=" << *parametrization_
     << "}";
}

}  // namespace crocoddyl
