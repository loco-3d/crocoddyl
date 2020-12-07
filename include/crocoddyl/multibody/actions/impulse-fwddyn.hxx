///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActionModelImpulseFwdDynamicsTpl<Scalar>::ActionModelImpulseFwdDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ImpulseModelMultiple> impulses,
    boost::shared_ptr<CostModelSum> costs, Scalar r_coeff, Scalar JMinvJt_damping,
    bool enable_force)
    : Base(state, 0, costs->get_nr()),
      impulses_(impulses),
      costs_(costs),
      pinocchio_(*state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      r_coeff_(r_coeff),
      JMinvJt_damping_(JMinvJt_damping),
      enable_force_(enable_force),
      gravity_(state->get_pinocchio()->gravity) {
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
ActionModelImpulseFwdDynamicsTpl<Scalar>::~ActionModelImpulseFwdDynamicsTpl() {}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                                    const Eigen::Ref<const VectorXs>& x,
                                                    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  std::size_t nq = state_->get_nq();
  std::size_t nv = state_->get_nv();
  std::size_t ni = impulses_->get_ni();
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);
  pinocchio::computeCentroidalMomentum(pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  impulses_->calc(d->multibody.impulses, x);

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXs> Jc_lu(d->multibody.impulses->Jc.topRows(ni));

  if (Jc_lu.rank() < d->multibody.impulses->Jc.topRows(ni).rows() && JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty("It is needed a damping factor since the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::impulseDynamics(pinocchio_, d->pinocchio, v, d->multibody.impulses->Jc.topRows(ni), r_coeff_,
                             JMinvJt_damping_);
  d->xnext.head(nq) = q;
  d->xnext.tail(nv) = d->pinocchio.dq_after;
  impulses_->updateVelocity(d->multibody.impulses, d->pinocchio.dq_after);
  impulses_->updateForce(d->multibody.impulses, d->pinocchio.impulse_c);

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                                        const Eigen::Ref<const VectorXs>& x,
                                                        const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  std::size_t nv = state_->get_nv();
  std::size_t ni = impulses_->get_ni();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  Data* d = static_cast<Data*>(data.get());

  // Computing the dynamics derivatives
  // We resize the Kinv matrix because Eigen cannot call block operations recursively:
  // https://eigen.tuxfamily.org/bz/show_bug.cgi?id=408.
  // Therefore, it is not possible to pass d->Kinv.topLeftCorner(nv + ni, nv + ni)
  d->Kinv.resize(nv + ni, nv + ni);
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, d->vnone, d->pinocchio.dq_after - v,
                                    d->multibody.impulses->fext);
  pinocchio::computeGeneralizedGravityDerivatives(pinocchio_, d->pinocchio, q, d->dgrav_dq);
  pinocchio::getKKTContactDynamicMatrixInverse(pinocchio_, d->pinocchio, d->multibody.impulses->Jc.topRows(ni),
                                               d->Kinv);

  pinocchio::computeForwardKinematicsDerivatives(pinocchio_, d->pinocchio, q, d->pinocchio.dq_after, d->vnone);
  impulses_->calcDiff(d->multibody.impulses, x);

  Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, ni);
  Eigen::Block<MatrixXs> f_partial_dtau = d->Kinv.bottomLeftCorner(ni, nv);
  Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(ni, ni);

  d->pinocchio.dtau_dq -= d->dgrav_dq;
  d->Fx.topLeftCorner(nv, nv).setIdentity();
  d->Fx.topRightCorner(nv, nv).setZero();
  d->Fx.bottomLeftCorner(nv, nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.bottomLeftCorner(nv, nv).noalias() -= a_partial_da * d->multibody.impulses->dv0_dq.topRows(ni);
  d->Fx.bottomRightCorner(nv, nv).noalias() = a_partial_dtau * d->pinocchio.M.template selfadjointView<Eigen::Upper>();

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.topLeftCorner(ni, nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.topLeftCorner(ni, nv).noalias() += f_partial_da * d->multibody.impulses->dv0_dq.topRows(ni);
    d->df_dx.topRightCorner(ni, nv).noalias() = f_partial_da * d->multibody.impulses->Jc.topRows(ni);
    impulses_->updateVelocityDiff(d->multibody.impulses, d->Fx.bottomRows(nv));
    impulses_->updateForceDiff(d->multibody.impulses, d->df_dx.topRows(ni));
  }
  costs_->calcDiff(d->costs, x, u);
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > ActionModelImpulseFwdDynamicsTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool ActionModelImpulseFwdDynamicsTpl<Scalar>::checkData(const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& ActionModelImpulseFwdDynamicsTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ImpulseModelMultipleTpl<Scalar> >& ActionModelImpulseFwdDynamicsTpl<Scalar>::get_impulses()
    const {
  return impulses_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& ActionModelImpulseFwdDynamicsTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelImpulseFwdDynamicsTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
Scalar ActionModelImpulseFwdDynamicsTpl<Scalar>::get_restitution_coefficient() const {
  return r_coeff_;
}

template <typename Scalar>
Scalar ActionModelImpulseFwdDynamicsTpl<Scalar>::get_damping_factor() const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  armature_ = armature;
  with_armature_ = false;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_restitution_coefficient(Scalar r_coeff) {
  if (r_coeff < 0.) {
    throw_pretty("Invalid argument: "
                 << "The restitution coefficient has to be positive");
  }
  r_coeff_ = r_coeff;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_damping_factor(Scalar damping) {
  if (damping < 0.) {
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
