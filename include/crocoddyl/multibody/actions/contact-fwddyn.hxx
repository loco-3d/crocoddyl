///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::
    DifferentialActionModelContactFwdDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs, const Scalar JMinvJt_damping,
        const bool enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr(), 0, 0),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(nullptr),
      pinocchio_(state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force) {
  init();
}

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::
    DifferentialActionModelContactFwdDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs,
        std::shared_ptr<ConstraintModelManager> constraints,
        const Scalar JMinvJt_damping, const bool enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr(), constraints->get_ng(),
           constraints->get_nh(), constraints->get_ng_T(),
           constraints->get_nh_T()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force) {
  init();
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::init() {
  if (JMinvJt_damping_ < Scalar(0.)) {
    JMinvJt_damping_ = Scalar(0.);
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive, set to 0");
  }
  if (contacts_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Contacts doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }

  Base::set_u_lb(Scalar(-1.) * pinocchio_->effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_->effortLimit.tail(nu_));
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
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

  const std::size_t nc = contacts_->get_nc();
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by
  // the contact model
  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_->calc(d->multibody.actuation, x, u);
  contacts_->calc(d->multibody.contacts, x);

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXs> Jc_lu(d->multibody.contacts->Jc.topRows(nc));

  if (Jc_lu.rank() < d->multibody.contacts->Jc.topRows(nc).rows() &&
      JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty(
        "A damping factor is needed as the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::forwardDynamics(
      *pinocchio_, d->pinocchio, d->multibody.actuation->tau,
      d->multibody.contacts->Jc.topRows(nc), d->multibody.contacts->a0.head(nc),
      JMinvJt_damping_);
  d->xout = d->pinocchio.ddq;
  contacts_->updateAcceleration(d->multibody.contacts, d->pinocchio.ddq);
  contacts_->updateForce(d->multibody.contacts, d->pinocchio.lambda_c);
  d->multibody.joint->a = d->pinocchio.ddq;
  d->multibody.joint->tau = u;
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d, false);
    constraints_->calc(d->constraints, x);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
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

  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  Data* d = static_cast<Data*>(data.get());

  // Computing the dynamics derivatives
  // We resize the Kinv matrix because Eigen cannot call block operations
  // recursively: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=408. Therefore,
  // it is not possible to pass d->Kinv.topLeftCorner(nv + nc, nv + nc)
  d->Kinv.resize(nv + nc, nv + nc);
  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, v, d->xout,
                                    d->multibody.contacts->fext);
  contacts_->updateRneaDiff(d->multibody.contacts, d->pinocchio);
  pinocchio::getKKTContactDynamicMatrixInverse(
      *pinocchio_, d->pinocchio, d->multibody.contacts->Jc.topRows(nc),
      d->Kinv);

  actuation_->calcDiff(d->multibody.actuation, x, u);
  contacts_->calcDiff(d->multibody.contacts, x);

  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  const Eigen::Block<MatrixXs> f_partial_dtau =
      d->Kinv.bottomLeftCorner(nc, nv);
  const Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -= a_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
  d->Fx.noalias() += a_partial_dtau * d->multibody.actuation->dtau_dx;
  d->Fu.noalias() = a_partial_dtau * d->multibody.actuation->dtau_du;
  d->multibody.joint->da_dx = d->Fx;
  d->multibody.joint->da_du = d->Fu;

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.topLeftCorner(nc, nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.topRightCorner(nc, nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dv;
    d->df_dx.topRows(nc).noalias() +=
        f_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
    d->df_dx.topRows(nc).noalias() -=
        f_partial_dtau * d->multibody.actuation->dtau_dx;
    d->df_du.topRows(nc).noalias() =
        -f_partial_dtau * d->multibody.actuation->dtau_du;
    contacts_->updateAccelerationDiff(d->multibody.contacts,
                                      d->Fx.bottomRows(nv));
    contacts_->updateForceDiff(d->multibody.contacts, d->df_dx.topRows(nc),
                               d->df_du.topRows(nc));
  }
  costs_->calcDiff(d->costs, x, u);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  costs_->calcDiff(d->costs, x);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x);
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelContactFwdDynamicsTpl<NewScalar>
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::cast() const {
  typedef DifferentialActionModelContactFwdDynamicsTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ContactModelMultipleTpl<NewScalar> ContactType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  if (constraints_) {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        std::make_shared<ConstraintType>(
            constraints_->template cast<NewScalar>()),
        scalar_cast<NewScalar>(JMinvJt_damping_), enable_force_);
    return ret;
  } else {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        scalar_cast<NewScalar>(JMinvJt_damping_), enable_force_);
    return ret;
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x, std::size_t,
    Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  // Static casting the data
  DifferentialActionDataContactFwdDynamicsTpl<Scalar>* d =
      static_cast<DifferentialActionDataContactFwdDynamicsTpl<Scalar>*>(
          data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();

  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.tail(nv));
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio, q);
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv),
                  d->tmp_xstatic.tail(nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic);

  // Allocates memory
  d->tmp_Jstatic.conservativeResize(nv, nu_ + nc);
  d->tmp_Jstatic.leftCols(nu_) = d->multibody.actuation->dtau_du;
  d->tmp_Jstatic.rightCols(nc) =
      d->multibody.contacts->Jc.topRows(nc).transpose();
  u.noalias() = (pseudoInverse(d->tmp_Jstatic) * d->pinocchio.tau).head(nu_);
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
bool DifferentialActionModelContactFwdDynamicsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelContactFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
     << ", nc=" << contacts_->get_nc() << "}";
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_ng()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return Base::get_ng();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_nh()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return Base::get_nh();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_ng_T()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng_T();
  } else {
    return Base::get_ng_T();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_nh_T()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh_T();
  } else {
    return Base::get_nh_T();
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const std::shared_ptr<ContactModelMultipleTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
const Scalar DifferentialActionModelContactFwdDynamicsTpl<
    Scalar>::get_damping_factor() const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::set_armature(
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
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::set_damping_factor(
    const Scalar damping) {
  if (damping < 0.) {
    throw_pretty(
        "Invalid argument: " << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
