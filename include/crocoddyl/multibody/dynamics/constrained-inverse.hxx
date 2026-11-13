///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/dynamics/dissipative.hpp"
#include "crocoddyl/multibody/dynamics/parameter-cast.hpp"

namespace crocoddyl {

template <typename Scalar>
DynamicsModelConstrainedInverseTpl<Scalar>::DynamicsModelConstrainedInverseTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<ActuationModelAbstract> actuation,
    std::shared_ptr<ImplicitConstraintModelMultiple> implicit_constraints,
    const std::size_t np, const DynamicsType dyn_type)
    : Base(state, dyn_type, np,
           (state == nullptr ? 0 : state->get_nv()) +
               (implicit_constraints == nullptr
                    ? 0
                    : implicit_constraints->get_nc()),
           0,
           dyn_type == DynamicsType::ContinuousControl
               ? (state != nullptr && actuation != nullptr &&
                          state->get_nv() > actuation->get_nu()
                      ? state->get_nv() - actuation->get_nu()
                      : 0) +
                     (implicit_constraints == nullptr
                          ? 0
                          : implicit_constraints->get_nc())
               : (state == nullptr ? 0 : state->get_nv()) +
                     (implicit_constraints == nullptr
                          ? 0
                          : implicit_constraints->get_nc())),
      actuation_(actuation),
      implicit_constraints_(implicit_constraints),
      params_(nullptr),
      pinocchio_(state == nullptr ? nullptr : state->get_pinocchio().get()) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
  if (actuation_ == nullptr) {
    throw_pretty("Invalid argument: actuation is null");
  }
  if (implicit_constraints_ == nullptr) {
    throw_pretty("Invalid argument: implicit_constraints is null");
  }
  if (implicit_constraints_->get_state()->get_nx() != state_->get_nx() ||
      implicit_constraints_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty(
        "Invalid argument: implicit_constraints has an incompatible state");
  }
  if (implicit_constraints_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "implicit_constraints doesn't have the same control "
                    "dimension as constrained inverse dynamics (it should be " +
                        std::to_string(nu_) + ")");
  }
  if (dyn_type_ == DynamicsType::DiscreteTime) {
    throw_pretty(
        "Invalid argument: constrained inverse dynamics is continuous");
  }
  tau_meas_.resize(actuation_->get_nu());
  tau_meas_.setZero();
  p_lb_ = VectorXs::Constant(np_, -std::numeric_limits<Scalar>::infinity());
  p_ub_ = VectorXs::Constant(np_, std::numeric_limits<Scalar>::infinity());
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
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

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = implicit_constraints_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a =
      u.head(nv);
  const bool compute_all_constraints =
      implicit_constraints_->getComputeAllConstraints();
  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(implicit_constraints_);

  data->vdot = a;
  data->h.setZero();
  data->g.setZero();
  pinocchio::forwardKinematics(*pinocchio_, d->pinocchio, q, v, a);
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio, q);
  implicit_constraints_->calc(d->multibody.constraints, x);
  if (nc != 0) {
    implicit_constraints_->updateForce(d->multibody.constraints, u.tail(nc));
  }
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, v, a,
                  d->multibody.constraints->fext);
  pinocchio::updateGlobalPlacements(*pinocchio_, d->pinocchio);
  pinocchio::centerOfMass(*pinocchio_, d->pinocchio, q, v, a);
  actuation_->commands(d->multibody.actuation, x, d->pinocchio.tau);
  d->multibody.joint->a = a;
  d->multibody.joint->tau = d->multibody.actuation->u;
  actuation_->calc(d->multibody.actuation, x, d->multibody.joint->tau);

  if (dyn_type_ == DynamicsType::ContinuousControl) {
    const std::size_t nh_act =
        (nv > actuation_->get_nu() ? nv - actuation_->get_nu() : 0);
    std::size_t nrow = 0;
    for (std::size_t k = 0;
         k < static_cast<std::size_t>(d->multibody.actuation->tau_set.size()) &&
         nrow < nh_act;
         ++k) {
      if (!d->multibody.actuation->tau_set[k]) {
        data->h(nrow) = d->pinocchio.tau(k);
        ++nrow;
      }
    }
    if (nc != 0) {
      data->h.tail(nc) = d->multibody.constraints->a0.head(nc);
    }
  } else {
    actuation_->calc(d->multibody.actuation, x, tau_meas_);
    data->h.head(nv) = d->pinocchio.tau - d->multibody.actuation->tau;
    if (nc != 0) {
      data->h.tail(nc) = d->multibody.constraints->a0.head(nc);
    }
  }
  internal::updateDissipativePowerFromActuation(d->multibody.actuation, v,
                                                data->dissipative_P);

  constraint_mode.restore();
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, x);
  }
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
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
  implicit_constraints_->calc(d->multibody.constraints, x);
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::calcDiff(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff_xu(data, x);
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
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
  const std::size_t nc = implicit_constraints_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a =
      u.head(nv);
  const bool compute_all_constraints =
      implicit_constraints_->getComputeAllConstraints();
  Data* d = static_cast<Data*>(data.get());
  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(implicit_constraints_);
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, x);
  }

  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, v, a,
                                    d->multibody.constraints->fext);
  implicit_constraints_->updateRneaDiff(d->multibody.constraints, d->pinocchio);
  d->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      d->pinocchio.M.template triangularView<Eigen::StrictlyUpper>()
          .transpose();
  pinocchio::jacobianCenterOfMass(*pinocchio_, d->pinocchio, false);
  actuation_->calcDiff(d->multibody.actuation, x, d->multibody.joint->tau);
  internal::updateDissipativePowerFromActuation(
      d->multibody.actuation, v, data->dissipative_P, &data->dP_dv);
  data->dP_dp.setZero();
  actuation_->torqueTransform(d->multibody.actuation, x,
                              d->multibody.actuation->u);
  d->multibody.joint->dtau_dx.leftCols(nv).noalias() =
      d->multibody.actuation->Mtau * d->pinocchio.dtau_dq;
  d->multibody.joint->dtau_dx.rightCols(nv).noalias() =
      d->multibody.actuation->Mtau * d->pinocchio.dtau_dv;
  d->multibody.joint->dtau_du.leftCols(nv).noalias() =
      d->multibody.actuation->Mtau * d->pinocchio.M;
  if (nc != 0) {
    d->multibody.joint->dtau_du.rightCols(nc).noalias() =
        -d->multibody.actuation->Mtau *
        d->multibody.constraints->Jc.topRows(nc).transpose();
  } else {
    d->multibody.joint->dtau_du.rightCols(0).setZero();
  }
  implicit_constraints_->calcDiff(d->multibody.constraints, x);

  d->dtau_dx.leftCols(nv) = d->pinocchio.dtau_dq;
  d->dtau_dx.rightCols(nv) = d->pinocchio.dtau_dv;

  if (dyn_type_ == DynamicsType::ContinuousControl) {
    const std::size_t nh_act =
        (nv > actuation_->get_nu() ? nv - actuation_->get_nu() : 0);
    d->dtau_dx -= d->multibody.actuation->dtau_dx;
    data->Hx.setZero();
    data->Hu.setZero();
    std::size_t nrow = 0;
    for (std::size_t k = 0;
         k < static_cast<std::size_t>(d->multibody.actuation->tau_set.size()) &&
         nrow < nh_act;
         ++k) {
      if (!d->multibody.actuation->tau_set[k]) {
        data->Hx.row(nrow) = d->dtau_dx.row(k);
        data->Hu.row(nrow).head(nv) = d->pinocchio.M.row(k);
        if (nc != 0) {
          data->Hu.row(nrow).tail(nc) =
              -d->multibody.constraints->Jc.topRows(nc).transpose().row(k);
        }
        ++nrow;
      }
    }
    if (nc != 0) {
      data->Hx.bottomRows(nc) = d->multibody.constraints->da0_dx.topRows(nc);
      data->Hu.bottomRows(nc).leftCols(nv) =
          d->multibody.constraints->Jc.topRows(nc);
    }
  } else {
    d->dtau_dx -= d->multibody.actuation->dtau_dx;
    data->Hx.topRows(nv) = d->dtau_dx;
    data->Hu.topLeftCorner(nv, nv) = d->pinocchio.M;
    if (nc != 0) {
      data->Hu.topRightCorner(nv, nc) =
          -d->multibody.constraints->Jc.topRows(nc).transpose();
      data->Hx.bottomRows(nc) = d->multibody.constraints->da0_dx.topRows(nc);
      data->Hu.bottomRows(nc).leftCols(nv) =
          d->multibody.constraints->Jc.topRows(nc);
    }
  }
  d->multibody.joint->da_dx = data->Fx;
  d->multibody.joint->da_du = data->Fu;

  constraint_mode.restore();
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, x);
    implicit_constraints_->calcDiff(d->multibody.constraints, x);
  }
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  implicit_constraints_->calcDiff(d->multibody.constraints, x);
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::calcDiff_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  (void)u;
  data->Fp.setZero();
  data->Hp.setZero();
  data->Gp.setZero();
  data->dP_dp.setZero();
  if (np_ == 0) {
    return;
  }
  if (params_ == nullptr) {
    throw_pretty(
        "Invalid call: constrained inverse dynamics parameters are not set");
  }

  Data* d = static_cast<Data*>(data.get());
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: constrained inverse dynamics data has no "
        "parameter-manager payload");
  }

  d->parameter_regressor->vdot = d->vdot;
  const VectorXs& regressor_u = dyn_type_ == DynamicsType::ContinuousControl
                                    ? d->multibody.actuation->u
                                    : tau_meas_;
  const std::size_t np_action = params_->get_np_action();
  const std::size_t np_dynamics = params_->get_np_dynamics();
  auto dtau_dp = d->parameter_regressor->Fp.middleCols(np_action, np_dynamics);
  params_->calcDiff_dynamics(d->params, d->parameter_regressor, dtau_dp, x,
                             regressor_u);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  internal::updateDissipativePowerParams(params_, dtau_dp, v, data->dP_dp);
  if (np_dynamics == 0) {
    return;
  }

  const std::size_t nv = state_->get_nv();
  const std::size_t nc = implicit_constraints_->get_nc();
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    const std::size_t nh_act =
        (nv > actuation_->get_nu() ? nv - actuation_->get_nu() : 0);
    std::size_t nrow = 0;
    for (std::size_t k = 0;
         k < static_cast<std::size_t>(d->multibody.actuation->tau_set.size()) &&
         nrow < nh_act;
         ++k) {
      if (!d->multibody.actuation->tau_set[k]) {
        data->Hp.row(nrow).middleCols(np_action, np_dynamics) = dtau_dp.row(k);
        ++nrow;
      }
    }
    (void)nc;
  } else {
    data->Hp.topRows(nv).middleCols(np_action, np_dynamics) = dtau_dp;
  }
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelConstrainedInverseTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelConstrainedInverseTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
template <typename NewScalar>
DynamicsModelConstrainedInverseTpl<NewScalar>
DynamicsModelConstrainedInverseTpl<Scalar>::cast() const {
  typedef DynamicsModelConstrainedInverseTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ImplicitConstraintModelMultipleTpl<NewScalar> ConstraintType;
  const std::shared_ptr<StateType> state =
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>());
  const std::shared_ptr<ActuationModelAbstractTpl<NewScalar> > actuation =
      actuation_->template cast<NewScalar>();
  ReturnType ret(state, actuation,
                 std::make_shared<ConstraintType>(
                     implicit_constraints_->template cast<NewScalar>()),
                 np_, dyn_type_);
  if (tau_meas_.size() != 0) {
    ret.update_tau(tau_meas_.template cast<NewScalar>());
  }
  if (params_ != nullptr) {
    ret.params_ = internal::castDynamicsParameters<Scalar, NewScalar>(
        params_, state, actuation);
    ret.np_ = ret.params_->get_np();
    ret.p_lb_ = p_lb_.template cast<NewScalar>();
    ret.p_ub_ = p_ub_.template cast<NewScalar>();
  }
  return ret;
}

template <typename Scalar>
bool DynamicsModelConstrainedInverseTpl<Scalar>::checkData(
    const std::shared_ptr<DynamicsDataAbstract>& data) {
  return std::dynamic_pointer_cast<Data>(data) != NULL;
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DynamicsDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t, Scalar) {
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

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t na = actuation_->get_nu();
  const std::size_t nc = implicit_constraints_->get_nc();
  const bool compute_all_constraints =
      implicit_constraints_->getComputeAllConstraints();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);

  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(implicit_constraints_);
  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.tail(nv));
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio, q);
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv),
                  d->tmp_xstatic.tail(nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic,
                   d->tmp_rstatic.head(na));
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic,
                       d->tmp_rstatic.head(na));
  implicit_constraints_->calc(d->multibody.constraints, d->tmp_xstatic);

  d->tmp_Jstatic.leftCols(na) = d->multibody.actuation->dtau_du;
  if (nc != 0) {
    d->tmp_Jstatic.middleCols(na, nc) =
        d->multibody.constraints->Jc.topRows(nc).transpose();
  }
  d->tmp_rstatic.head(na + nc).noalias() =
      pseudoInverse(d->tmp_Jstatic.leftCols(na + nc).eval()) * d->pinocchio.tau;
  if (nc != 0) {
    u.segment(nv, nc) = d->tmp_rstatic.segment(na, nc);
  }
  d->pinocchio.tau.setZero();

  constraint_mode.restore();
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, d->tmp_xstatic);
  }
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  if (static_cast<std::size_t>(tau_meas.size()) != actuation_->get_nu()) {
    throw_pretty(
        "Invalid argument: " << "tau_meas has wrong dimension (it should be " +
                                    std::to_string(actuation_->get_nu()) + ")");
  }
  tau_meas_ = tau_meas;
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::set_params(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
  Data* d = static_cast<Data*>(data.get());
  params_ = params;
  np_ = params_->get_np();
  p_lb_ = VectorXs::Constant(np_, -std::numeric_limits<Scalar>::infinity());
  p_ub_ = VectorXs::Constant(np_, std::numeric_limits<Scalar>::infinity());
  d->resize(this);
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::update_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty(
        "Invalid call: constrained inverse dynamics parameters are not set");
  }
  Data* d = static_cast<Data*>(data.get());
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: constrained inverse dynamics data has no "
        "parameter-manager payload");
  }
  params_->update(d->params, p);
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DynamicsModelConstrainedInverseTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const std::shared_ptr<ImplicitConstraintModelMultipleTpl<Scalar> >&
DynamicsModelConstrainedInverseTpl<Scalar>::get_constraints() const {
  return implicit_constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
DynamicsModelConstrainedInverseTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DynamicsModelConstrainedInverseTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
void DynamicsModelConstrainedInverseTpl<Scalar>::print(std::ostream& os) const {
  os << "DynamicsModelConstrainedInverse {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << ", nh=" << nh_
     << ", np=" << np_ << ", nc=" << implicit_constraints_->get_nc()
     << ", nc_total=" << implicit_constraints_->get_nc_total() << "}";
}

}  // namespace crocoddyl
