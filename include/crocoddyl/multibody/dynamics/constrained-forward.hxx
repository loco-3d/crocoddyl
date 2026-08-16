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
DynamicsModelConstrainedForwardTpl<Scalar>::DynamicsModelConstrainedForwardTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<ActuationModelAbstract> actuation,
    std::shared_ptr<ImplicitConstraintModelMultiple> implicit_constraints,
    const std::size_t np, const DynamicsType dyn_type)
    : Base(state, dyn_type, np,
           dyn_type == DynamicsType::ContinuousControl && actuation != nullptr
               ? actuation->get_nu()
               : 0,
           0, 0),
      actuation_(actuation),
      implicit_constraints_(implicit_constraints),
      params_(nullptr),
      pinocchio_(state == nullptr ? nullptr : state->get_pinocchio().get()),
      without_armature_(true),
      armature_(VectorXs::Zero(state == nullptr ? 0 : state->get_nv())) {
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
  if (implicit_constraints_->get_nu() != actuation_->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "implicit_constraints doesn't have the same control "
                    "dimension as actuation (it should be " +
                        std::to_string(actuation_->get_nu()) + ")");
  }
  if (dyn_type_ == DynamicsType::DiscreteTime) {
    throw_pretty(
        "Invalid argument: constrained forward dynamics is continuous");
  }
  tau_meas_.resize(actuation_->get_nu());
  tau_meas_.setZero();
  p_lb_ = VectorXs::Constant(np_, -std::numeric_limits<Scalar>::infinity());
  p_ub_ = VectorXs::Constant(np_, std::numeric_limits<Scalar>::infinity());
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::calc(
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
  const std::size_t nc = implicit_constraints_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  const bool compute_all_constraints =
      implicit_constraints_->getComputeAllConstraints();
  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(implicit_constraints_);

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  if (!without_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    actuation_->calc(d->multibody.actuation, x, u);
  } else {
    actuation_->calc(d->multibody.actuation, x, tau_meas_);
  }
  implicit_constraints_->calc(d->multibody.constraints, x);
  pinocchio::forwardDynamics(*pinocchio_, d->pinocchio,
                             d->multibody.actuation->tau,
                             d->multibody.constraints->Jc.topRows(nc),
                             d->multibody.constraints->a0.head(nc), Scalar(0.));
  d->vdot = d->pinocchio.ddq;
  implicit_constraints_->updateAcceleration(d->multibody.constraints,
                                            d->pinocchio.ddq);
  implicit_constraints_->updateForce(d->multibody.constraints,
                                     d->pinocchio.lambda_c.head(nc));
  d->multibody.joint->a = d->vdot;
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    d->multibody.joint->tau = u;
  } else {
    d->multibody.joint->tau = tau_meas_;
  }
  internal::updateDissipativePowerFromActuation(d->multibody.actuation, v,
                                                data->dissipative_P);

  constraint_mode.restore();
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, x);
  }
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::calc(
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
void DynamicsModelConstrainedForwardTpl<Scalar>::calcDiff(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff_xu(data, x);
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::calcDiff_xu(
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
void DynamicsModelConstrainedForwardTpl<Scalar>::calcDiff_xu(
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
  const bool compute_all_constraints =
      implicit_constraints_->getComputeAllConstraints();
  Data* d = static_cast<Data*>(data.get());
  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(implicit_constraints_);
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, x);
  }

  d->Kinv.resize(nv + nc, nv + nc);
  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, v, d->vdot,
                                    d->multibody.constraints->fext);
  implicit_constraints_->updateRneaDiff(d->multibody.constraints, d->pinocchio);
  pinocchio::getKKTContactDynamicMatrixInverse(
      *pinocchio_, d->pinocchio, d->multibody.constraints->Jc.topRows(nc),
      d->Kinv);
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    actuation_->calcDiff(d->multibody.actuation, x, u);
  } else {
    actuation_->calcDiff(d->multibody.actuation, x, tau_meas_);
  }
  internal::updateDissipativePowerFromActuation(
      d->multibody.actuation, v, data->dissipative_P, &data->dP_dv);
  implicit_constraints_->calcDiff(d->multibody.constraints, x);

  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  const Eigen::Block<MatrixXs> f_partial_dtau =
      d->Kinv.bottomLeftCorner(nc, nv);
  const Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -=
      a_partial_da * d->multibody.constraints->da0_dx.topRows(nc);
  d->Fx.noalias() += a_partial_dtau * d->multibody.actuation->dtau_dx;
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    d->Fu.noalias() = a_partial_dtau * d->multibody.actuation->dtau_du;
  } else {
    d->Fu.setZero();
  }

  d->df_dx.topRows(nc).leftCols(nv).noalias() =
      f_partial_dtau * d->pinocchio.dtau_dq;
  d->df_dx.topRows(nc).rightCols(nv).noalias() =
      f_partial_dtau * d->pinocchio.dtau_dv;
  d->df_dx.topRows(nc).noalias() +=
      f_partial_da * d->multibody.constraints->da0_dx.topRows(nc);
  d->df_dx.topRows(nc).noalias() -=
      f_partial_dtau * d->multibody.actuation->dtau_dx;
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    d->df_du.topRows(nc).noalias() =
        -f_partial_dtau * d->multibody.actuation->dtau_du;
  } else {
    d->df_du.topRows(nc).setZero();
  }
  implicit_constraints_->updateAccelerationDiff(d->multibody.constraints,
                                                d->Fx);
  if (nc > 0u) {
    implicit_constraints_->updateForceDiff(
        d->multibody.constraints, d->df_dx.topRows(nc), d->df_du.topRows(nc));
  }
  d->multibody.joint->da_dx = d->Fx;
  if (dyn_type_ == DynamicsType::ContinuousControl) {
    d->multibody.joint->da_du = d->Fu;
  } else {
    d->multibody.joint->da_du.setZero();
  }

  constraint_mode.restore();
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, x);
    implicit_constraints_->calcDiff(d->multibody.constraints, x);
  }
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::calcDiff_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  data->Fp.setZero();
  if (np_ == 0) {
    return;
  }
  if (params_ == nullptr) {
    throw_pretty(
        "Invalid call: constrained forward dynamics parameters are not set");
  }

  Data* d = static_cast<Data*>(data.get());
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: constrained forward dynamics data has no "
        "parameter-manager payload");
  }

  d->parameter_regressor->vdot = d->vdot;
  const std::size_t np_action = params_->get_np_action();
  const std::size_t np_dynamics = params_->get_np_dynamics();
  auto dtau_dp = d->parameter_regressor->Fp.middleCols(np_action, np_dynamics);
  if (dyn_type_ == DynamicsType::ContinuousEstimation) {
    params_->calcDiff_dynamics(d->params, d->parameter_regressor, dtau_dp, x,
                               tau_meas_);
  } else {
    params_->calcDiff_dynamics(d->params, d->parameter_regressor, dtau_dp, x,
                               u);
  }

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  internal::updateDissipativePowerParams(params_, dtau_dp, v, data->dP_dp);
  if (np_dynamics == 0) {
    return;
  }

  const std::size_t nv = state_->get_nv();
  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  data->Fp.middleCols(np_action, np_dynamics).noalias() =
      -a_partial_dtau * dtau_dp;
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelConstrainedForwardTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelConstrainedForwardTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
template <typename NewScalar>
DynamicsModelConstrainedForwardTpl<NewScalar>
DynamicsModelConstrainedForwardTpl<Scalar>::cast() const {
  typedef DynamicsModelConstrainedForwardTpl<NewScalar> ReturnType;
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
  if (!without_armature_) {
    ret.set_armature(armature_.template cast<NewScalar>());
  }
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
bool DynamicsModelConstrainedForwardTpl<Scalar>::checkData(
    const std::shared_ptr<DynamicsDataAbstract>& data) {
  return std::dynamic_pointer_cast<Data>(data) != NULL;
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::quasiStatic(
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
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = implicit_constraints_->get_nc();
  const bool compute_all_constraints =
      implicit_constraints_->getComputeAllConstraints();
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
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);
  implicit_constraints_->calc(d->multibody.constraints, d->tmp_xstatic);

  d->tmp_Jstatic.leftCols(nu_) = d->multibody.actuation->dtau_du;
  if (nc != 0) {
    d->tmp_Jstatic.middleCols(nu_, nc) =
        d->multibody.constraints->Jc.topRows(nc).transpose();
  }
  const MatrixXs tmp_Jstatic = d->tmp_Jstatic.leftCols(nu_ + nc);
  u.noalias() = (pseudoInverse(tmp_Jstatic) * d->pinocchio.tau).head(nu_);
  d->pinocchio.tau.setZero();

  constraint_mode.restore();
  if (compute_all_constraints) {
    implicit_constraints_->calc(d->multibody.constraints, d->tmp_xstatic);
  }
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  if (static_cast<std::size_t>(tau_meas.size()) != actuation_->get_nu()) {
    throw_pretty(
        "Invalid argument: " << "tau_meas has wrong dimension (it should be " +
                                    std::to_string(actuation_->get_nu()) + ")");
  }
  tau_meas_ = tau_meas;
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::set_params(
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
void DynamicsModelConstrainedForwardTpl<Scalar>::update_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty(
        "Invalid call: constrained forward dynamics parameters are not set");
  }
  Data* d = static_cast<Data*>(data.get());
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: constrained forward dynamics data has no "
        "parameter-manager payload");
  }
  params_->update(d->params, p);
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DynamicsModelConstrainedForwardTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const std::shared_ptr<ImplicitConstraintModelMultipleTpl<Scalar> >&
DynamicsModelConstrainedForwardTpl<Scalar>::get_constraints() const {
  return implicit_constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
DynamicsModelConstrainedForwardTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DynamicsModelConstrainedForwardTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DynamicsModelConstrainedForwardTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::set_armature(
    const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }
  armature_ = armature;
  without_armature_ = false;
}

template <typename Scalar>
void DynamicsModelConstrainedForwardTpl<Scalar>::print(std::ostream& os) const {
  os << "DynamicsModelConstrainedForward {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << ", np=" << np_
     << ", nc=" << implicit_constraints_->get_nc()
     << ", nc_total=" << implicit_constraints_->get_nc_total() << "}";
}

}  // namespace crocoddyl
