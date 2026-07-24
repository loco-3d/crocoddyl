///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "crocoddyl/multibody/dynamics/parameter-cast.hpp"

namespace crocoddyl {

template <typename Scalar>
DynamicsModelImpulseForwardTpl<Scalar>::DynamicsModelImpulseForwardTpl(
    std::shared_ptr<StateMultibody> state,
    std::shared_ptr<ImplicitConstraintModelMultiple> constraints,
    const std::size_t np, const Scalar r_coeff, const Scalar JMinvJt_damping)
    : Base(state, DynamicsType::DiscreteTime, np, 0, 0, 0),
      constraints_(constraints),
      params_(nullptr),
      pinocchio_(state == nullptr ? nullptr : state->get_pinocchio().get()),
      r_coeff_(r_coeff),
      JMinvJt_damping_(JMinvJt_damping) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
  if (constraints_ == nullptr) {
    throw_pretty("Invalid argument: constraints is null");
  }
  if (constraints_->get_state()->get_nx() != state_->get_nx() ||
      constraints_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: constraints has an incompatible state");
  }
  if (constraints_->get_nu() != 0) {
    throw_pretty(
        "Invalid argument: impulse forward dynamics expects "
        "constraints with nu = 0");
  }
  if (JMinvJt_damping_ < Scalar(0.)) {
    throw_pretty("Invalid argument: JMinvJt_damping has to be nonnegative");
  }
  p_lb_ = VectorXs::Constant(np_, -std::numeric_limits<Scalar>::infinity());
  p_ub_ = VectorXs::Constant(np_, std::numeric_limits<Scalar>::infinity());
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  constraints_->calc(d->multibody.constraints, x);

  data->vdot.setZero();
  data->h.setZero();
  data->g.setZero();
  data->dissipative_P.setZero();
  data->dP_dv.setZero();
  data->dP_dp.setZero();
  d->joint->a.setZero();
  d->joint->tau.setZero();
  d->joint->dtau_dx.setZero();
  d->joint->dtau_du.setZero();
  d->joint->da_dx.setZero();
  d->joint->da_du.setZero();
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = constraints_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const bool compute_all_constraints = constraints_->getComputeAllConstraints();
  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(constraints_);

  data->h.setZero();
  data->g.setZero();
  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  constraints_->calc(d->multibody.constraints, x);
  pinocchio::impulseDynamics(*pinocchio_, d->pinocchio, v,
                             d->multibody.constraints->Jc.topRows(nc), r_coeff_,
                             JMinvJt_damping_);
  data->vdot.head(nq) = q;
  data->vdot.tail(nv) = d->pinocchio.dq_after;
  data->dissipative_P.setZero();
  data->dP_dv.setZero();
  data->dP_dp.setZero();
  d->joint->tau.setZero();
  d->joint->a = d->pinocchio.dq_after - v;
  constraints_->updateVelocity(d->multibody.constraints, d->pinocchio.dq_after);
  constraints_->updateForce(d->multibody.constraints, d->pinocchio.impulse_c);

  constraint_mode.restore();
  if (compute_all_constraints) {
    constraints_->calc(d->multibody.constraints, x);
  }
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::calcDiff(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff_xu(data, x);
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  constraints_->calcDiff(d->multibody.constraints, x);

  data->Fx.setZero();
  data->Fu.setZero();
  data->Fp.setZero();
  data->Hx.setZero();
  data->Hu.setZero();
  data->Hp.setZero();
  data->Gx.setZero();
  data->Gu.setZero();
  data->Gp.setZero();
  data->dP_dv.setZero();
  data->dP_dp.setZero();
  d->df_dx.setZero();
  d->df_du.setZero();
  d->joint->dtau_dx.setZero();
  d->joint->dtau_du.setZero();
  d->joint->da_dx.setZero();
  d->joint->da_du.setZero();
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = constraints_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const bool compute_all_constraints = constraints_->getComputeAllConstraints();
  internal::ActiveConstraintModeGuardTpl<ImplicitConstraintModelMultiple>
      constraint_mode(constraints_);
  if (compute_all_constraints) {
    constraints_->calc(d->multibody.constraints, x);
  }

  d->Kinv.conservativeResize(nv + nc, nv + nc);

  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, d->vnone,
                                    d->pinocchio.dq_after - v,
                                    d->multibody.constraints->fext);
  pinocchio::computeGeneralizedGravityDerivatives(*pinocchio_, d->pinocchio, q,
                                                  d->dgrav_dq);
  d->pinocchio.dtau_dq -= d->dgrav_dq;
  d->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      d->pinocchio.M.transpose()
          .template triangularView<Eigen::StrictlyLower>();

  pinocchio::getKKTContactDynamicMatrixInverse(
      *pinocchio_, d->pinocchio, d->multibody.constraints->Jc.topRows(nc),
      d->Kinv);
  pinocchio::computeForwardKinematicsDerivatives(
      *pinocchio_, d->pinocchio, q, d->pinocchio.dq_after, d->vnone);
  constraints_->calcDiff(d->multibody.constraints, x);
  constraints_->updateRneaDiff(d->multibody.constraints, d->pinocchio);

  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  const Eigen::Block<MatrixXs> f_partial_dtau =
      d->Kinv.bottomLeftCorner(nc, nv);
  const Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  data->Gx.setZero();
  data->Gu.setZero();
  data->Hx.setZero();
  data->Hu.setZero();
  data->Fp.setZero();
  data->Hp.setZero();
  data->Gp.setZero();
  data->dP_dv.setZero();
  data->dP_dp.setZero();

  data->Fx.topLeftCorner(nv, nv).setIdentity();
  data->Fx.topRightCorner(nv, nv).setZero();
  data->Fx.bottomLeftCorner(nv, nv).noalias() =
      -a_partial_dtau * d->pinocchio.dtau_dq;
  if (nc != 0) {
    data->Fx.bottomLeftCorner(nv, nv).noalias() -=
        a_partial_da * d->multibody.constraints->dv0_dq.topRows(nc);
  }
  data->Fx.bottomRightCorner(nv, nv).noalias() =
      a_partial_dtau * d->pinocchio.M;
  data->Fu.setZero();

  d->joint->dtau_dx.setZero();
  d->joint->dtau_du.setZero();
  d->joint->da_dx.leftCols(nv) = data->Fx.bottomLeftCorner(nv, nv);
  d->joint->da_dx.rightCols(nv) = data->Fx.bottomRightCorner(nv, nv);
  d->joint->da_dx.rightCols(nv).diagonal().array() -= Scalar(1.);
  d->joint->da_du.setZero();

  if (nc != 0) {
    d->df_dx.topRows(nc).leftCols(nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.topRows(nc).leftCols(nv).noalias() +=
        f_partial_da * d->multibody.constraints->dv0_dq.topRows(nc);
    d->df_dx.topRows(nc).rightCols(nv).noalias() =
        f_partial_da * d->multibody.constraints->Jc.topRows(nc);
    d->df_du.topRows(nc).setZero();
    constraints_->updateVelocityDiff(d->multibody.constraints,
                                     data->Fx.bottomRows(nv));
    constraints_->updateForceDiff(d->multibody.constraints,
                                  d->df_dx.topRows(nc), d->df_du.topRows(nc));
  } else {
    d->df_dx.setZero();
    d->df_du.setZero();
  }

  constraint_mode.restore();
  if (compute_all_constraints) {
    constraints_->calc(d->multibody.constraints, x);
    constraints_->calcDiff(d->multibody.constraints, x);
  }
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::calcDiff_p(
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
        "Invalid call: impulse forward dynamics parameters are not set");
  }

  Data* d = static_cast<Data*>(data.get());
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: impulse forward dynamics data has no "
        "parameter-manager payload");
  }

  const std::size_t np_action = params_->get_np_action();
  const std::size_t np_dynamics = params_->get_np_dynamics();
  if (np_dynamics == 0) {
    return;
  }

  const std::size_t nv = state_->get_nv();
  d->tmp_xparams = x;
  d->tmp_xparams.tail(nv).setZero();
  const VectorXs& regressor_u = d->parameter_regressor->u;

  d->parameter_regressor->vdot = d->joint->a;
  params_->calcDiff_dynamics(d->params, d->parameter_regressor, d->tmp_xparams,
                             regressor_u);
  d->tmp_dtau_dp.leftCols(np_dynamics) = d->params->params->dtau_dp;

  d->parameter_regressor->vdot.setZero();
  params_->calcDiff_dynamics(d->params, d->parameter_regressor, d->tmp_xparams,
                             regressor_u);
  d->tmp_dtau_dp.leftCols(np_dynamics) -= d->params->params->dtau_dp;

  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  data->Fp.bottomRows(nv).middleCols(np_action, np_dynamics).noalias() =
      -a_partial_dtau * d->tmp_dtau_dp.leftCols(np_dynamics);
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelImpulseForwardTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelImpulseForwardTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
template <typename NewScalar>
DynamicsModelImpulseForwardTpl<NewScalar>
DynamicsModelImpulseForwardTpl<Scalar>::cast() const {
  typedef DynamicsModelImpulseForwardTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ImplicitConstraintModelMultipleTpl<NewScalar> ConstraintType;
  const std::shared_ptr<StateType> state =
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>());
  ReturnType ret(state,
                 std::make_shared<ConstraintType>(
                     constraints_->template cast<NewScalar>()),
                 np_, scalar_cast<NewScalar>(r_coeff_),
                 scalar_cast<NewScalar>(JMinvJt_damping_));
  if (params_ != nullptr) {
    ret.params_ = internal::castDynamicsParameters<Scalar, NewScalar>(
        params_, state,
        std::shared_ptr<ActuationModelAbstractTpl<NewScalar> >());
    ret.np_ = ret.params_->get_np();
    ret.p_lb_ = p_lb_.template cast<NewScalar>();
    ret.p_ub_ = p_ub_.template cast<NewScalar>();
  }
  return ret;
}

template <typename Scalar>
bool DynamicsModelImpulseForwardTpl<Scalar>::checkData(
    const std::shared_ptr<DynamicsDataAbstract>& data) {
  return std::dynamic_pointer_cast<Data>(data) != nullptr;
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::set_params(
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
void DynamicsModelImpulseForwardTpl<Scalar>::update_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty(
        "Invalid call: impulse forward dynamics parameters are not set");
  }

  Data* d = static_cast<Data*>(data.get());
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: impulse forward dynamics data has no "
        "parameter-manager payload");
  }
  params_->update(d->params, p);
}

template <typename Scalar>
const std::shared_ptr<ImplicitConstraintModelMultipleTpl<Scalar> >&
DynamicsModelImpulseForwardTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
DynamicsModelImpulseForwardTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DynamicsModelImpulseForwardTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
Scalar DynamicsModelImpulseForwardTpl<Scalar>::get_r_coeff() const {
  return r_coeff_;
}

template <typename Scalar>
Scalar DynamicsModelImpulseForwardTpl<Scalar>::get_damping_factor() const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void DynamicsModelImpulseForwardTpl<Scalar>::print(std::ostream& os) const {
  os << "DynamicsModelImpulseForward {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx()
     << ", nc_total=" << constraints_->get_nc_total() << ", np=" << np_
     << ", r_coeff=" << r_coeff_ << ", JMinvJt_damping=" << JMinvJt_damping_
     << "}";
}

}  // namespace crocoddyl
