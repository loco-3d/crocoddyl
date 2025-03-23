///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActionModelImpulseFwdDynamicsTpl<Scalar>::ActionModelImpulseFwdDynamicsTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<ImpulseModelMultiple> impulses,
    std::shared_ptr<CostModelSum> costs, const Scalar r_coeff,
    const Scalar JMinvJt_damping, const bool enable_force)
    : Base(state, 0, costs->get_nr(), 0, 0),
      impulses_(impulses),
      costs_(costs),
      constraints_(nullptr),
      pinocchio_(state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      r_coeff_(r_coeff),
      JMinvJt_damping_(JMinvJt_damping),
      enable_force_(enable_force),
      gravity_(state->get_pinocchio()->gravity) {
  init();
}

template <typename Scalar>
ActionModelImpulseFwdDynamicsTpl<Scalar>::ActionModelImpulseFwdDynamicsTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<ImpulseModelMultiple> impulses,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints, const Scalar r_coeff,
    const Scalar JMinvJt_damping, const bool enable_force)
    : Base(state, 0, costs->get_nr(), constraints->get_ng(),
           constraints->get_nh(), constraints->get_ng_T(),
           constraints->get_nh_T()),
      impulses_(impulses),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      r_coeff_(r_coeff),
      JMinvJt_damping_(JMinvJt_damping),
      enable_force_(enable_force),
      gravity_(state->get_pinocchio()->gravity) {
  init();
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::init() {
  if (r_coeff_ < Scalar(0.)) {
    r_coeff_ = Scalar(0.);
    throw_pretty("Invalid argument: "
                 << "The restitution coefficient has to be positive, set to 0");
  }
  if (JMinvJt_damping_ < Scalar(0.)) {
    JMinvJt_damping_ = Scalar(0.);
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive, set to 0");
  }
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());

  // Computing impulse dynamics and forces
  initCalc(d, x);

  // Computing the cost and constraints
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x, u);
  }
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());

  // Computing impulse dynamics and forces
  initCalc(d, x);

  // Computing the cost and constraints
  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d, false);
    constraints_->calc(d->constraints, x);
  }
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());

  // Computing derivatives of impulse dynamics and forces
  initCalcDiff(d, x);

  // Computing derivatives of cost and constraints
  costs_->calcDiff(d->costs, x, u);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());

  // Computing derivatives of impulse dynamics and forces
  initCalcDiff(d, x);

  // Computing derivatives of cost and constraints
  costs_->calcDiff(d->costs, x);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x);
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ActionModelImpulseFwdDynamicsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
ActionModelImpulseFwdDynamicsTpl<NewScalar>
ActionModelImpulseFwdDynamicsTpl<Scalar>::cast() const {
  typedef ActionModelImpulseFwdDynamicsTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ImpulseModelMultipleTpl<NewScalar> ImpulseType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  if (constraints_) {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        std::make_shared<ImpulseType>(impulses_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        std::make_shared<ConstraintType>(
            constraints_->template cast<NewScalar>()),
        scalar_cast<NewScalar>(r_coeff_),
        scalar_cast<NewScalar>(JMinvJt_damping_), enable_force_);
    return ret;
  } else {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        std::make_shared<ImpulseType>(impulses_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        scalar_cast<NewScalar>(r_coeff_),
        scalar_cast<NewScalar>(JMinvJt_damping_), enable_force_);
    return ret;
  }
}

template <typename Scalar>
bool ActionModelImpulseFwdDynamicsTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>&, Eigen::Ref<VectorXs>,
    const Eigen::Ref<const VectorXs>&, const std::size_t, const Scalar) {
  // do nothing
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::initCalc(
    Data* data, const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = impulses_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  // Computing the forward dynamics with the holonomic constraints defined by
  // the contact model
  pinocchio::computeAllTerms(*pinocchio_, data->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, data->pinocchio);

  if (!with_armature_) {
    data->pinocchio.M.diagonal() += armature_;
  }
  impulses_->calc(data->multibody.impulses, x);

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXs> Jc_lu(data->multibody.impulses->Jc.topRows(nc));

  if (Jc_lu.rank() < data->multibody.impulses->Jc.topRows(nc).rows() &&
      JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty(
        "It is needed a damping factor since the contact Jacobian is not "
        "full-rank");
  }
#endif

  pinocchio::impulseDynamics(*pinocchio_, data->pinocchio, v,
                             data->multibody.impulses->Jc.topRows(nc), r_coeff_,
                             JMinvJt_damping_);
  data->xnext.head(nq) = q;
  data->xnext.tail(nv) = data->pinocchio.dq_after;
  impulses_->updateVelocity(data->multibody.impulses, data->pinocchio.dq_after);
  impulses_->updateForce(data->multibody.impulses, data->pinocchio.impulse_c);
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::initCalcDiff(
    Data* data, const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t nv = state_->get_nv();
  const std::size_t nc = impulses_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  // Computing the dynamics derivatives
  // We resize the Kinv matrix because Eigen cannot call block operations
  // recursively: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=408. Therefore,
  // it is not possible to pass d->Kinv.topLeftCorner(nv + nc, nv + nc)
  data->Kinv.resize(nv + nc, nv + nc);
  pinocchio::computeRNEADerivatives(*pinocchio_, data->pinocchio, q,
                                    data->vnone, data->pinocchio.dq_after - v,
                                    data->multibody.impulses->fext);
  pinocchio::computeGeneralizedGravityDerivatives(*pinocchio_, data->pinocchio,
                                                  q, data->dgrav_dq);
  pinocchio::getKKTContactDynamicMatrixInverse(
      *pinocchio_, data->pinocchio, data->multibody.impulses->Jc.topRows(nc),
      data->Kinv);

  pinocchio::computeForwardKinematicsDerivatives(
      *pinocchio_, data->pinocchio, q, data->pinocchio.dq_after, data->vnone);
  impulses_->calcDiff(data->multibody.impulses, x);
  impulses_->updateRneaDiff(data->multibody.impulses, data->pinocchio);

  Eigen::Block<MatrixXs> a_partial_dtau = data->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<MatrixXs> a_partial_da = data->Kinv.topRightCorner(nv, nc);
  Eigen::Block<MatrixXs> f_partial_dtau = data->Kinv.bottomLeftCorner(nc, nv);
  Eigen::Block<MatrixXs> f_partial_da = data->Kinv.bottomRightCorner(nc, nc);

  data->pinocchio.dtau_dq -= data->dgrav_dq;
  data->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      data->pinocchio.M.transpose()
          .template triangularView<Eigen::StrictlyLower>();
  data->Fx.topLeftCorner(nv, nv).setIdentity();
  data->Fx.topRightCorner(nv, nv).setZero();
  data->Fx.bottomLeftCorner(nv, nv).noalias() =
      -a_partial_dtau * data->pinocchio.dtau_dq;
  data->Fx.bottomLeftCorner(nv, nv).noalias() -=
      a_partial_da * data->multibody.impulses->dv0_dq.topRows(nc);
  data->Fx.bottomRightCorner(nv, nv).noalias() =
      a_partial_dtau * data->pinocchio.M;

  // Computing the cost derivatives
  if (enable_force_) {
    data->df_dx.topLeftCorner(nc, nv).noalias() =
        f_partial_dtau * data->pinocchio.dtau_dq;
    data->df_dx.topLeftCorner(nc, nv).noalias() +=
        f_partial_da * data->multibody.impulses->dv0_dq.topRows(nc);
    data->df_dx.topRightCorner(nc, nv).noalias() =
        f_partial_da * data->multibody.impulses->Jc.topRows(nc);
    impulses_->updateVelocityDiff(data->multibody.impulses,
                                  data->Fx.bottomRows(nv));
    impulses_->updateForceDiff(data->multibody.impulses,
                               data->df_dx.topRows(nc));
  }
}

template <typename Scalar>
std::size_t ActionModelImpulseFwdDynamicsTpl<Scalar>::get_ng() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return Base::get_ng();
  }
}

template <typename Scalar>
std::size_t ActionModelImpulseFwdDynamicsTpl<Scalar>::get_nh() const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return Base::get_nh();
  }
}

template <typename Scalar>
std::size_t ActionModelImpulseFwdDynamicsTpl<Scalar>::get_ng_T() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng_T();
  } else {
    return Base::get_ng_T();
  }
}

template <typename Scalar>
std::size_t ActionModelImpulseFwdDynamicsTpl<Scalar>::get_nh_T() const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh_T();
  } else {
    return Base::get_nh_T();
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelImpulseFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nc=" << impulses_->get_nc() << "}";
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
const std::shared_ptr<ImpulseModelMultipleTpl<Scalar> >&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_impulses() const {
  return impulses_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
const Scalar
ActionModelImpulseFwdDynamicsTpl<Scalar>::get_restitution_coefficient() const {
  return r_coeff_;
}

template <typename Scalar>
const Scalar ActionModelImpulseFwdDynamicsTpl<Scalar>::get_damping_factor()
    const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_armature(
    const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }
  armature_ = armature;
  with_armature_ = false;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_restitution_coefficient(
    const Scalar r_coeff) {
  if (r_coeff < 0.) {
    throw_pretty("Invalid argument: "
                 << "The restitution coefficient has to be positive");
  }
  r_coeff_ = r_coeff;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_damping_factor(
    const Scalar damping) {
  if (damping < 0.) {
    throw_pretty(
        "Invalid argument: " << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
