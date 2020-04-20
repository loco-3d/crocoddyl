///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActionModelImpulseFwdDynamicsTpl<Scalar>::ActionModelImpulseFwdDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ImpulseModelMultiple> impulses,
    boost::shared_ptr<CostModelSum> costs, const Scalar& r_coeff, const Scalar& JMinvJt_damping,
    const bool& enable_force)
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
  if (r_coeff_ < 0.) {
    r_coeff_ = 0.;
    throw_pretty("Invalid argument: "
                 << "The restitution coefficient has to be positive, set to 0");
  }
  if (JMinvJt_damping_ < 0.) {
    JMinvJt_damping_ = 0.;
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

  const std::size_t& nq = state_->get_nq();
  const std::size_t& nv = state_->get_nv();
  ActionDataImpulseFwdDynamicsTpl<Scalar>* d = static_cast<ActionDataImpulseFwdDynamicsTpl<Scalar>*>(data.get());
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
  Eigen::FullPivLU<MatrixXs> Jc_lu(d->multibody.impulses->Jc);

  if (Jc_lu.rank() < d->multibody.impulses->Jc.rows() && JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty("It is needed a damping factor since the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::impulseDynamics(pinocchio_, d->pinocchio, v, d->multibody.impulses->Jc, r_coeff_, JMinvJt_damping_);
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

  const std::size_t& nv = state_->get_nv();
  const std::size_t& ni = impulses_->get_ni();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  ActionDataImpulseFwdDynamicsTpl<Scalar>* d = static_cast<ActionDataImpulseFwdDynamicsTpl<Scalar>*>(data.get());

  // Computing the dynamics derivatives
  pinocchio_.gravity.setZero();
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, d->vnone, d->pinocchio.dq_after - v,
                                    d->multibody.impulses->fext);
  pinocchio_.gravity = gravity_;
  pinocchio::getKKTContactDynamicMatrixInverse(pinocchio_, d->pinocchio, d->multibody.impulses->Jc, d->Kinv);

  pinocchio::computeForwardKinematicsDerivatives(pinocchio_, d->pinocchio, q, d->pinocchio.dq_after, d->vnone);
  impulses_->calcDiff(d->multibody.impulses, x);

  Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, ni);
  Eigen::Block<MatrixXs> f_partial_dtau = d->Kinv.bottomLeftCorner(ni, nv);
  Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(ni, ni);

  d->Fx.topLeftCorner(nv, nv).setIdentity();
  d->Fx.topRightCorner(nv, nv).setZero();
  d->Fx.bottomLeftCorner(nv, nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.bottomLeftCorner(nv, nv).noalias() -= a_partial_da * d->multibody.impulses->dv0_dq;
  d->Fx.bottomRightCorner(nv, nv).noalias() = a_partial_dtau * d->pinocchio.M.template selfadjointView<Eigen::Upper>();

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dq.noalias() = f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dq.noalias() += f_partial_da * d->multibody.impulses->dv0_dq;
    impulses_->updateVelocityDiff(d->multibody.impulses, d->Fx.bottomRows(nv));
    impulses_->updateForceDiff(d->multibody.impulses, d->df_dq);
  }
  costs_->calcDiff(d->costs, x, u);
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > ActionModelImpulseFwdDynamicsTpl<Scalar>::createData() {
  return boost::make_shared<ActionDataImpulseFwdDynamicsTpl<Scalar> >(this);
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
const Scalar& ActionModelImpulseFwdDynamicsTpl<Scalar>::get_restitution_coefficient() const {
  return r_coeff_;
}

template <typename Scalar>
const Scalar& ActionModelImpulseFwdDynamicsTpl<Scalar>::get_damping_factor() const {
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
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_restitution_coefficient(const Scalar& r_coeff) {
  if (r_coeff < 0.) {
    throw_pretty("Invalid argument: "
                 << "The restitution coefficient has to be positive");
  }
  r_coeff_ = r_coeff;
}

template <typename Scalar>
void ActionModelImpulseFwdDynamicsTpl<Scalar>::set_damping_factor(const Scalar& damping) {
  if (damping < 0.) {
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
