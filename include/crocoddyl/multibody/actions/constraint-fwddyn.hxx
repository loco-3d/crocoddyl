///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#if PINOCCHIO_VERSION_AT_LEAST(2, 9, 0)

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::DifferentialActionModelConstraintFwdDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel) & contacts,
    boost::shared_ptr<CostModelSum> costs, const ProximalSettings& settings)
    : Base(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      pinocchio_(*state->get_pinocchio().get()),
      settings_(settings) {
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }

  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
}

template <typename Scalar>
DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::~DifferentialActionModelConstraintFwdDynamicsTpl() {}

template <typename Scalar>
void DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  DifferentialActionDataConstraintFwdDynamicsTpl<Scalar>* d =
      static_cast<DifferentialActionDataConstraintFwdDynamicsTpl<Scalar>*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model

  actuation_->calc(d->multibody.actuation, x, u);
  // TODO: Add mu
  d->xout = pinocchio::constraintDynamics(pinocchio_, d->pinocchio, q, v, d->multibody.actuation->tau, contacts_,
                                          d->multibody.contacts);  //, mu_contacts);
  pinocchio::jacobianCenterOfMass(pinocchio_, d->pinocchio, false);
  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

template <typename Scalar>
void DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t& nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  DifferentialActionDataConstraintFwdDynamicsTpl<Scalar>* d =
      static_cast<DifferentialActionDataConstraintFwdDynamicsTpl<Scalar>*>(data.get());

  // Computing the dynamics derivatives
  actuation_->calcDiff(d->multibody.actuation, x, u);
  pinocchio::computeConstraintDynamicsDerivatives(
      pinocchio_, d->pinocchio, contacts_, d->multibody.contacts, d->Fx.leftCols(nv), d->Fx.rightCols(nv),
      d->pinocchio.ddq_dtau, d->pinocchio.dlambda_dq, d->pinocchio.dlambda_dv, d->pinocchio.dlambda_dtau);

  d->Fu.noalias() = d->pinocchio.ddq_dtau * d->multibody.actuation->dtau_du;
  costs_->calcDiff(d->costs, x, u);
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::createData() {
  boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > data =
      boost::make_shared<DifferentialActionDataConstraintFwdDynamics>(this);
  DifferentialActionDataConstraintFwdDynamics* d =
      static_cast<DifferentialActionDataConstraintFwdDynamics*>(data.get());
  pinocchio::initConstraintDynamics(pinocchio_, d->pinocchio, contacts_);

  return data;
}

template <typename Scalar>
bool DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<DifferentialActionDataConstraintFwdDynamics> d =
    boost::dynamic_pointer_cast<DifferentialActionDataConstraintFwdDynamics>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}
  
  
/*
template <typename Scalar>
void DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::quasiStatic(const
boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                                                                     const std::size_t& maxiter, const Scalar& tol) {
if (static_cast<std::size_t>(u.size()) != nu_) {
  throw_pretty("Invalid argument: "
               << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
}
if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
  throw_pretty("Invalid argument: "
               << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
}
// Static casting the data
DifferentialActionDataConstraintFwdDynamicsTpl<Scalar>* d =
    static_cast<DifferentialActionDataConstraintFwdDynamicsTpl<Scalar>*>(data.get());
const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

d->multibody.actuation->tau = pinocchio::rnea(pinocchio_, d->pinocchio, q, v, VectorXs::Zero(state_->get_nv()));
actuation_->get_actuated(d->multibody.actuation, u);
}
*/

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const std::vector<pinocchio::RigidConstraintModelTpl<Scalar, 0>,
                  Eigen::aligned_allocator<pinocchio::RigidConstraintModelTpl<Scalar, 0> > >&
DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::get_armature()
    const {
  return pinocchio_.armature;
}

template <typename Scalar>
void DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  pinocchio_.armature = armature;
}

template <typename Scalar>
const pinocchio::ProximalSettingsTpl<Scalar>& DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::get_settings()
    const {
  return settings_;
}

template <typename Scalar>
void DifferentialActionModelConstraintFwdDynamicsTpl<Scalar>::set_settings(const ProximalSettings& settings) {
  settings_ = settings;
}

}  // namespace crocoddyl

#endif  // PINOCCHIO_VERSION_AT_LEAST(2,9,0)
