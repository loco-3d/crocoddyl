///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {
namespace internal {

template <typename Scalar>
class JointDynamicsModelPassiveTpl
    : public JointDynamicsModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(JointDynamicsModelBase, JointDynamicsModelPassiveTpl)

  typedef JointDynamicsModelAbstractTpl<Scalar> Base;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef typename Base::VectorXs VectorXs;

  JointDynamicsModelPassiveTpl(const pinocchio::JointIndex id,
                               const std::size_t nq, const std::size_t nv)
      : Base(id, nq, nv, 0) {}

  virtual void calc(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& q,
                    const Eigen::Ref<const VectorXs>& v,
                    const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(q.size()) != this->nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(this->nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != this->nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(this->nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != this->nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(this->nu_) + ")");
    }
    data->tau.setZero();
    data->friction.setZero();
  }

  virtual void calcDiff(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& q,
                        const Eigen::Ref<const VectorXs>& v,
                        const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(q.size()) != this->nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(this->nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != this->nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(this->nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != this->nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(this->nu_) + ")");
    }
    data->dtau_dq.setZero();
    data->dtau_dv.setZero();
    data->dtau_du.setZero();
  }

  virtual void commands(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& q,
                        const Eigen::Ref<const VectorXs>& v,
                        const Eigen::Ref<const VectorXs>& tau) override {
    if (static_cast<std::size_t>(q.size()) != this->nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(this->nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != this->nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(this->nv_) + ")");
    }
    if (static_cast<std::size_t>(tau.size()) != this->nv_) {
      throw_pretty(
          "Invalid argument: " << "tau has wrong dimension (it should be " +
                                      std::to_string(this->nv_) + ")");
    }
    data->u.setZero();
    data->friction.setZero();
  }

  virtual void print(std::ostream& os) const override {
    os << "JointDynamicsModelPassive {id=" << this->id_ << ", nq=" << this->nq_
       << ", nv=" << this->nv_ << "}";
  }

  template <typename NewScalar>
  JointDynamicsModelPassiveTpl<NewScalar> cast() const {
    typedef JointDynamicsModelPassiveTpl<NewScalar> ReturnType;
    ReturnType ret(this->id_, this->nq_, this->nv_);
    return ret;
  }
};

}  // namespace internal

template <typename Scalar>
ActuationModelMultibodyTpl<Scalar>::ActuationModelMultibodyTpl(
    std::shared_ptr<StateMultibody> state)
    : ActuationModelMultibodyTpl(
          state, std::vector<std::shared_ptr<JointDynamicsModelAbstract> >()) {}

template <typename Scalar>
ActuationModelMultibodyTpl<Scalar>::ActuationModelMultibodyTpl(
    std::shared_ptr<StateMultibody> state,
    const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints)
    : ActuationModelMultibodyTpl(
          state, joints.empty() ? createDefaultIdentityModels(state) : joints,
          ConstructorTag()) {}

template <typename Scalar>
ActuationModelMultibodyTpl<Scalar>::ActuationModelMultibodyTpl(
    std::shared_ptr<StateMultibody> state,
    const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints,
    ConstructorTag)
    : Base(state, computeNu(state, joints)), np_(0) {
  initialize(joints);
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ActuationModelMultibodyTpl<Scalar>::JointDynamicsModelAbstract> >
ActuationModelMultibodyTpl<Scalar>::createDefaultIdentityModels(
    const std::shared_ptr<StateMultibody>& state) {
  if (state == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state->get_pinocchio();
  const pinocchio::JointIndex first_actuated_joint =
      pin_model->existJointName("root_joint")
          ? pin_model->getJointId("root_joint") + 1
          : 1;
  std::vector<std::shared_ptr<JointDynamicsModelAbstract> > joints;
  joints.reserve(static_cast<std::size_t>(pin_model->njoints));
  for (pinocchio::JointIndex jid = first_actuated_joint;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    joints.push_back(std::make_shared<JointDynamicsModelIdentityTpl<Scalar> >(
        jid, static_cast<std::size_t>(joint.nq()),
        static_cast<std::size_t>(joint.nv())));
  }
  return joints;
}

template <typename Scalar>
std::size_t ActuationModelMultibodyTpl<Scalar>::computeNu(
    const std::shared_ptr<StateMultibody>& state,
    const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints) {
  if (state == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state->get_pinocchio();
  std::vector<bool> seen(static_cast<std::size_t>(pin_model->njoints), false);
  std::vector<std::size_t> joint_nu(
      static_cast<std::size_t>(pin_model->njoints), 0);
  std::size_t nu = 0;
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (joints[i] == nullptr) {
      throw_pretty("Invalid argument: joint model at index " << i
                                                             << " is null");
    }
    const pinocchio::JointIndex jid = joints[i]->get_id();
    if (jid <= 0 ||
        jid >= static_cast<pinocchio::JointIndex>(pin_model->njoints)) {
      throw_pretty("Invalid argument: joint id " << jid << " is out of bounds");
    }
    const auto& joint = pin_model->joints[jid];
    const std::size_t joint_nq = static_cast<std::size_t>(joint.nq());
    const std::size_t joint_nv = static_cast<std::size_t>(joint.nv());
    if (joints[i]->get_nq() != joint_nq) {
      throw_pretty(
          "Invalid argument: joint model nq is inconsistent with the "
          "Pinocchio model");
    }
    if (joints[i]->get_nv() != joint_nv) {
      throw_pretty(
          "Invalid argument: joint model nv is inconsistent with the "
          "Pinocchio model");
    }
    if (!seen[static_cast<std::size_t>(jid)]) {
      seen[static_cast<std::size_t>(jid)] = true;
      joint_nu[static_cast<std::size_t>(jid)] = joints[i]->get_nu();
      nu += joints[i]->get_nu();
    } else if (joint_nu[static_cast<std::size_t>(jid)] != joints[i]->get_nu()) {
      throw_pretty("Invalid argument: all joint models attached to joint "
                   << jid << " must share the same control dimension");
    }
  }
  return nu;
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::initialize(
    const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints) {
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  joints_.clear();
  joints_.resize(static_cast<std::size_t>(pin_model->njoints));
  joint_control_dims_.assign(static_cast<std::size_t>(pin_model->njoints), 0);
  joint_u_offset_.assign(static_cast<std::size_t>(pin_model->njoints), 0);
  np_ = 0;

  for (std::size_t i = 0; i < joints.size(); ++i) {
    joints_[static_cast<std::size_t>(joints[i]->get_id())].push_back(joints[i]);
    np_ += joints[i]->get_np();
  }

  std::size_t u_offset = 0;
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joint_models =
        joints_[static_cast<std::size_t>(jid)];
    if (joint_models.empty()) {
      joint_models.push_back(
          std::make_shared<internal::JointDynamicsModelPassiveTpl<Scalar> >(
              jid, static_cast<std::size_t>(joint.nq()),
              static_cast<std::size_t>(joint.nv())));
    }
    const std::size_t joint_nu = joint_models.front()->get_nu();
    for (std::size_t i = 0; i < joint_models.size(); ++i) {
      if (joint_models[i]->get_nq() != static_cast<std::size_t>(joint.nq())) {
        throw_pretty(
            "Invalid argument: joint model nq is inconsistent with "
            "the Pinocchio model");
      }
      if (joint_models[i]->get_nv() != static_cast<std::size_t>(joint.nv())) {
        throw_pretty(
            "Invalid argument: joint model nv is inconsistent with "
            "the Pinocchio model");
      }
      if (joint_models[i]->get_nu() != joint_nu) {
        throw_pretty("Invalid argument: all joint models attached to joint "
                     << jid << " must share the same control dimension");
      }
    }
    joint_control_dims_[static_cast<std::size_t>(jid)] = joint_nu;
    joint_u_offset_[static_cast<std::size_t>(jid)] = u_offset;
    u_offset += joint_nu;
  }
  updateBounds();
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::updateBounds() {
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  u_lb_.setConstant(-std::numeric_limits<Scalar>::infinity());
  u_ub_.setConstant(std::numeric_limits<Scalar>::infinity());
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const std::size_t joint_nu =
        joint_control_dims_[static_cast<std::size_t>(jid)];
    if (joint_nu == 0) {
      continue;
    }
    bool identity_only = true;
    const JointDynamicsModelThrusterTpl<Scalar>* thruster_model = NULL;
    const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >&
        joint_models = joints_[static_cast<std::size_t>(jid)];
    for (std::size_t i = 0; i < joint_models.size(); ++i) {
      if (dynamic_cast<JointDynamicsModelIdentityTpl<Scalar>*>(
              joint_models[i].get()) == NULL &&
          dynamic_cast<JointDynamicsModelFrictionTpl<Scalar>*>(
              joint_models[i].get()) == NULL) {
        const JointDynamicsModelThrusterTpl<Scalar>* candidate =
            dynamic_cast<JointDynamicsModelThrusterTpl<Scalar>*>(
                joint_models[i].get());
        if (candidate != NULL) {
          thruster_model = candidate;
          identity_only = false;
          continue;
        }
        thruster_model = NULL;
        identity_only = false;
        break;
      }
    }
    const std::size_t u_offset = joint_u_offset_[static_cast<std::size_t>(jid)];
    if (thruster_model != NULL) {
      const std::vector<ThrusterTpl<Scalar> >& thrusters =
          thruster_model->get_thrusters();
      for (std::size_t i = 0; i < joint_nu; ++i) {
        u_lb_[u_offset + i] = thrusters[i].min_thrust;
        u_ub_[u_offset + i] = thrusters[i].max_thrust;
      }
      continue;
    }
    const auto& joint = pin_model->joints[jid];
    if (identity_only && joint_nu == static_cast<std::size_t>(joint.nv())) {
      const std::size_t v_offset = static_cast<std::size_t>(joint.idx_v());
      u_lb_.segment(u_offset, joint_nu) =
          Scalar(-1.) * pin_model->effortLimit.segment(v_offset, joint_nu);
      u_ub_.segment(u_offset, joint_nu) =
          pin_model->effortLimit.segment(v_offset, joint_nu);
    }
  }
}

template <typename Scalar>
std::shared_ptr<
    typename ActuationModelMultibodyTpl<Scalar>::ActuationDataMultibody>
ActuationModelMultibodyTpl<Scalar>::castData(
    const std::shared_ptr<ActuationDataAbstract>& data) const {
  std::shared_ptr<ActuationDataMultibody> d =
      std::dynamic_pointer_cast<ActuationDataMultibody>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data is not an ActuationDataMultibody");
  }
  return d;
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::calc(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  std::shared_ptr<ActuationDataMultibody> d = castData(data);
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  const std::size_t nq = state_->get_nq();
  d->u = u;
  d->tau.setZero();
  d->friction.setZero();
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    const std::size_t joint_q = static_cast<std::size_t>(joint.idx_q());
    const std::size_t joint_v = static_cast<std::size_t>(joint.idx_v());
    const std::size_t joint_nq = static_cast<std::size_t>(joint.nq());
    const std::size_t joint_nv = static_cast<std::size_t>(joint.nv());
    const std::size_t joint_nu =
        joint_control_dims_[static_cast<std::size_t>(jid)];
    const std::size_t joint_u = joint_u_offset_[static_cast<std::size_t>(jid)];
    for (std::size_t i = 0; i < joints_[static_cast<std::size_t>(jid)].size();
         ++i) {
      joints_[static_cast<std::size_t>(jid)][i]->calc(
          d->joint[static_cast<std::size_t>(jid)][i],
          x.segment(joint_q, joint_nq), x.segment(nq + joint_v, joint_nv),
          u.segment(joint_u, joint_nu));
      d->tau.segment(joint_v, joint_nv) +=
          d->joint[static_cast<std::size_t>(jid)][i]->tau;
      d->friction.segment(joint_v, joint_nv) +=
          d->joint[static_cast<std::size_t>(jid)][i]->friction;
    }
  }
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  std::shared_ptr<ActuationDataMultibody> d = castData(data);
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  d->dtau_dx.setZero();
  d->dtau_du.setZero();
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    const std::size_t joint_q = static_cast<std::size_t>(joint.idx_q());
    const std::size_t joint_v = static_cast<std::size_t>(joint.idx_v());
    const std::size_t joint_nq = static_cast<std::size_t>(joint.nq());
    const std::size_t joint_nv = static_cast<std::size_t>(joint.nv());
    const std::size_t joint_nu =
        joint_control_dims_[static_cast<std::size_t>(jid)];
    const std::size_t joint_u = joint_u_offset_[static_cast<std::size_t>(jid)];
    for (std::size_t i = 0; i < joints_[static_cast<std::size_t>(jid)].size();
         ++i) {
      joints_[static_cast<std::size_t>(jid)][i]->calcDiff(
          d->joint[static_cast<std::size_t>(jid)][i],
          x.segment(joint_q, joint_nq), x.segment(nq + joint_v, joint_nv),
          u.segment(joint_u, joint_nu));
      d->dtau_dx.block(joint_v, joint_v, joint_nv, joint_nv) +=
          d->joint[static_cast<std::size_t>(jid)][i]->dtau_dq;
      d->dtau_dx.block(joint_v, nv + joint_v, joint_nv, joint_nv) +=
          d->joint[static_cast<std::size_t>(jid)][i]->dtau_dv;
      if (joint_nu != 0) {
        d->dtau_du.block(joint_v, joint_u, joint_nv, joint_nu) +=
            d->joint[static_cast<std::size_t>(jid)][i]->dtau_du;
      }
    }
  }
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::commands(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& tau) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(tau.size()) != state_->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "tau has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + ")");
  }
  std::shared_ptr<ActuationDataMultibody> d = castData(data);
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  const std::size_t nq = state_->get_nq();
  d->u.setZero();
  d->friction.setZero();
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    const std::size_t joint_q = static_cast<std::size_t>(joint.idx_q());
    const std::size_t joint_v = static_cast<std::size_t>(joint.idx_v());
    const std::size_t joint_nq = static_cast<std::size_t>(joint.nq());
    const std::size_t joint_nv = static_cast<std::size_t>(joint.nv());
    const std::size_t joint_nu =
        joint_control_dims_[static_cast<std::size_t>(jid)];
    const std::size_t joint_u = joint_u_offset_[static_cast<std::size_t>(jid)];
    for (std::size_t i = 0; i < joints_[static_cast<std::size_t>(jid)].size();
         ++i) {
      joints_[static_cast<std::size_t>(jid)][i]->commands(
          d->joint[static_cast<std::size_t>(jid)][i],
          x.segment(joint_q, joint_nq), x.segment(nq + joint_v, joint_nv),
          tau.segment(joint_v, joint_nv));
      if (joint_nu != 0) {
        d->u.segment(joint_u, joint_nu) +=
            d->joint[static_cast<std::size_t>(jid)][i]->u;
      }
      d->friction.segment(joint_v, joint_nv) +=
          d->joint[static_cast<std::size_t>(jid)][i]->friction;
    }
  }
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::updateTorqueTransform(
    const std::shared_ptr<ActuationDataMultibody>& data) const {
  data->Mtau.setZero();
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    const std::size_t joint_v = static_cast<std::size_t>(joint.idx_v());
    const std::size_t joint_nv = static_cast<std::size_t>(joint.nv());
    const std::size_t joint_nu =
        joint_control_dims_[static_cast<std::size_t>(jid)];
    if (joint_nu == 0) {
      continue;
    }
    const std::size_t joint_u = joint_u_offset_[static_cast<std::size_t>(jid)];
    for (std::size_t i = 0; i < joints_[static_cast<std::size_t>(jid)].size();
         ++i) {
      data->Mtau.block(joint_u, joint_v, joint_nu, joint_nv) +=
          data->joint[static_cast<std::size_t>(jid)][i]->Mtau;
    }
  }
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::torqueTransform(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  std::shared_ptr<ActuationDataMultibody> d = castData(data);
  updateTorqueTransform(d);
}

template <typename Scalar>
std::shared_ptr<
    typename ActuationModelMultibodyTpl<Scalar>::ActuationDataAbstract>
ActuationModelMultibodyTpl<Scalar>::createData() {
  std::shared_ptr<ActuationDataMultibody> data =
      std::allocate_shared<ActuationDataMultibody>(
          Eigen::aligned_allocator<ActuationDataMultibody>(), this);
  data->joint.resize(joints_.size());
  for (std::size_t jid = 0; jid < joints_.size(); ++jid) {
    data->joint[jid].reserve(joints_[jid].size());
    for (std::size_t i = 0; i < joints_[jid].size(); ++i) {
      data->joint[jid].push_back(joints_[jid][i]->createData());
    }
  }
  updateTorqueTransform(data);
  std::fill(data->tau_set.begin(), data->tau_set.end(), false);
  if (nu_ != 0) {
    data->dtau_du = pseudoInverse(data->Mtau);
    const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
        std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
    for (pinocchio::JointIndex jid = 1;
         jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
      const auto& joint = pin_model->joints[jid];
      for (std::size_t row = 0; row < static_cast<std::size_t>(joint.nv());
           ++row) {
        for (std::size_t i = 0; i < data->joint[jid].size(); ++i) {
          if (!data->joint[jid][i]
                   ->dtau_du.row(static_cast<Eigen::Index>(row))
                   .isZero()) {
            data->tau_set[static_cast<std::size_t>(joint.idx_v()) + row] = true;
            break;
          }
        }
      }
    }
  }
  return data;
}

template <typename Scalar>
const std::vector<
    std::vector<std::shared_ptr<typename ActuationModelMultibodyTpl<
        Scalar>::JointDynamicsModelAbstract> > >&
ActuationModelMultibodyTpl<Scalar>::get_joints() const {
  return joints_;
}

template <typename Scalar>
std::size_t ActuationModelMultibodyTpl<Scalar>::get_np() const {
  return np_;
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::update_p(
    const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> gamma) {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(gamma.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "gamma has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  std::size_t p_offset = 0;
  for (std::size_t jid = 1; jid < joints_.size(); ++jid) {
    for (std::size_t i = 0; i < joints_[jid].size(); ++i) {
      const std::size_t joint_np = joints_[jid][i]->get_np();
      if (joint_np == 0) {
        continue;
      }
      joints_[jid][i]->set_parameters(p.segment(p_offset, joint_np));
      gamma.segment(p_offset, joint_np) = joints_[jid][i]->get_parameters();
      p_offset += joint_np;
    }
  }
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::updateParametrizationDerivative(
    Eigen::Ref<MatrixXs> dgamma_dp) const {
  if (static_cast<std::size_t>(dgamma_dp.rows()) != np_ ||
      static_cast<std::size_t>(dgamma_dp.cols()) != np_) {
    throw_pretty("Invalid argument: dgamma_dp has wrong dimensions");
  }
  dgamma_dp.setZero();
  std::size_t p_offset = 0;
  for (std::size_t jid = 1; jid < joints_.size(); ++jid) {
    for (std::size_t i = 0; i < joints_[jid].size(); ++i) {
      const std::size_t joint_np = joints_[jid][i]->get_np();
      if (joint_np == 0) {
        continue;
      }
      joints_[jid][i]->updateParametrizationDerivative(
          dgamma_dp.block(p_offset, p_offset, joint_np, joint_np));
      p_offset += joint_np;
    }
  }
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::computeJointTorqueRegressor(
    Eigen::Ref<MatrixXs> dtau_dp, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(dtau_dp.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(dtau_dp.cols()) != np_) {
    throw_pretty("Invalid argument: dtau_dp has wrong dimensions");
  }
  dtau_dp.setZero();
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      std::static_pointer_cast<StateMultibody>(state_)->get_pinocchio();
  const std::size_t nq = state_->get_nq();
  std::size_t p_offset = 0;
  for (pinocchio::JointIndex jid = 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    const std::size_t joint_q = static_cast<std::size_t>(joint.idx_q());
    const std::size_t joint_v = static_cast<std::size_t>(joint.idx_v());
    const std::size_t joint_nq = static_cast<std::size_t>(joint.nq());
    const std::size_t joint_nv = static_cast<std::size_t>(joint.nv());
    const std::size_t joint_nu =
        joint_control_dims_[static_cast<std::size_t>(jid)];
    const std::size_t joint_u = joint_u_offset_[static_cast<std::size_t>(jid)];
    for (std::size_t i = 0; i < joints_[static_cast<std::size_t>(jid)].size();
         ++i) {
      const std::size_t joint_np =
          joints_[static_cast<std::size_t>(jid)][i]->get_np();
      if (joint_np == 0) {
        continue;
      }
      joints_[static_cast<std::size_t>(jid)][i]->computeJointTorqueRegressor(
          dtau_dp.block(joint_v, p_offset, joint_nv, joint_np),
          x.segment(joint_q, joint_nq), x.segment(nq + joint_v, joint_nv),
          u.segment(joint_u, joint_nu));
      p_offset += joint_np;
    }
  }
}

template <typename Scalar>
template <typename NewScalar>
ActuationModelMultibodyTpl<NewScalar> ActuationModelMultibodyTpl<Scalar>::cast()
    const {
  typedef ActuationModelMultibodyTpl<NewScalar> ReturnType;
  typedef JointDynamicsModelAbstractTpl<NewScalar>
      JointDynamicsModelAbstractNew;
  typedef StateMultibodyTpl<NewScalar> StateMultibodyNew;
  std::vector<std::shared_ptr<JointDynamicsModelAbstractNew> > joints;
  for (std::size_t jid = 1; jid < joints_.size(); ++jid) {
    for (std::size_t i = 0; i < joints_[jid].size(); ++i) {
      if (dynamic_cast<internal::JointDynamicsModelPassiveTpl<Scalar>*>(
              joints_[jid][i].get()) != NULL) {
        continue;
      }
      const JointDynamicsModelIdentityTpl<Scalar>* identity =
          dynamic_cast<JointDynamicsModelIdentityTpl<Scalar>*>(
              joints_[jid][i].get());
      if (identity != NULL) {
        joints.push_back(
            std::make_shared<JointDynamicsModelIdentityTpl<NewScalar> >(
                identity->get_id(), identity->get_nq(), identity->get_nv()));
        continue;
      }
      const JointDynamicsModelFrictionTpl<Scalar>* friction =
          dynamic_cast<JointDynamicsModelFrictionTpl<Scalar>*>(
              joints_[jid][i].get());
      if (friction != NULL) {
        joints.push_back(
            std::make_shared<JointDynamicsModelFrictionTpl<NewScalar> >(
                friction->get_id(), friction->get_nq(),
                friction->get_parametrization().template cast<NewScalar>(),
                friction->get_friction_type(), friction->get_passive(),
                friction->get_fixed_smoothing_parametrization()
                    .template cast<NewScalar>()));
        continue;
      }
      const JointDynamicsModelThrusterTpl<Scalar>* thruster =
          dynamic_cast<JointDynamicsModelThrusterTpl<Scalar>*>(
              joints_[jid][i].get());
      if (thruster != NULL) {
        joints.push_back(
            std::make_shared<JointDynamicsModelThrusterTpl<NewScalar> >(
                vector_cast<NewScalar>(thruster->get_thrusters())));
        continue;
      }
      throw_pretty(
          "Invalid argument: cast only supports the built-in joint dynamics "
          "models in this migration step");
    }
  }
  ReturnType ret(std::static_pointer_cast<StateMultibodyNew>(
                     state_->template cast<NewScalar>()),
                 joints);
  return ret;
}

template <typename Scalar>
void ActuationModelMultibodyTpl<Scalar>::print(std::ostream& os) const {
  os << "ActuationModelMultibody {nu=" << nu_ << ", nv=" << state_->get_nv()
     << ", np=" << np_ << "}";
}

}  // namespace crocoddyl
