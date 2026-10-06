///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <limits>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

#include "crocoddyl/core/integ-observer-base.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/constrained-inverse.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedObserverModelAbstractTpl<Scalar>::IntegratedObserverModelAbstractTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints, const Scalar time_step)
    : Base(dynamics != nullptr ? dynamics->get_state() : nullptr, 0,
           dynamics != nullptr
               ? dynamics->get_state()->get_ndx() + dynamics->get_nu()
               : 0,
           /*nr=*/0,
           dynamics != nullptr
               ? dynamics->get_ng() +
                     (constraints != nullptr ? constraints->get_ng() : 0)
               : 0,
           dynamics != nullptr
               ? dynamics->get_nh() +
                     (constraints != nullptr ? constraints->get_nh() : 0)
               : 0,
           constraints != nullptr ? constraints->get_ng_T() : 0,
           constraints != nullptr ? constraints->get_nh_T() : 0,
           dynamics != nullptr ? dynamics->get_np() : 0),
      dynamics_(dynamics),
      costs_(costs),
      constraints_(constraints),
      params_(nullptr),
      state_multibody_(nullptr),
      dynamics_forward_(nullptr),
      dynamics_inverse_(nullptr),
      dynamics_constraints_(nullptr),
      time_step_(time_step),
      time_step2_(time_step * time_step),
      u_zero_(dynamics != nullptr ? VectorXs::Zero(dynamics->get_nu())
                                  : VectorXs()) {
  if (dynamics_ == nullptr) {
    throw_pretty("Invalid argument: dynamics is null");
  }
  if (costs_ == nullptr) {
    throw_pretty("Invalid argument: costs is null");
  }
  if (time_step_ < Scalar(0.)) {
    throw_pretty("Invalid argument: time_step has to be positive");
  }
  if (dynamics_->get_dyn_type() == DynamicsType::DiscreteTime) {
    throw_pretty(
        "Invalid argument: this is not a time-continuous dynamics "
        "(time-discrete dynamics should be incorporated using a "
        "discretized observer model)");
  }

  state_multibody_ =
      std::dynamic_pointer_cast<StateMultibody>(dynamics_->get_state());
  if (state_multibody_ == nullptr) {
    throw_pretty(
        "Invalid argument: observer integration currently requires "
        "a multibody state");
  }

  dynamics_forward_ =
      std::dynamic_pointer_cast<DynamicsModelConstrainedForward>(dynamics_);
  dynamics_inverse_ =
      std::dynamic_pointer_cast<DynamicsModelConstrainedInverse>(dynamics_);
  if (dynamics_forward_ == nullptr && dynamics_inverse_ == nullptr) {
    throw_pretty(
        "Invalid argument: integrated observer base currently "
        "supports constrained forward and inverse dynamics only");
  }
  dynamics_constraints_ = dynamics_forward_ != nullptr
                              ? dynamics_forward_->get_constraints()
                              : dynamics_inverse_->get_constraints();
  if (dynamics_constraints_ == nullptr) {
    throw_pretty("Invalid argument: dynamics constraints are null");
  }

  const std::size_t ntau = dynamics_forward_ != nullptr
                               ? dynamics_forward_->get_actuation()->get_nu()
                               : dynamics_inverse_->get_actuation()->get_nu();
  Base::ntau_ = ntau;
  Base::tau_meas_ = VectorXs::Zero(ntau);

  if (costs_->get_state()->get_nx() != state_->get_nx() ||
      costs_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: costs has an incompatible state");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: costs doesn't have the observer control "
        "dimension (it should be " +
        std::to_string(nu_) + ")");
  }
  if (constraints_ != nullptr) {
    if (constraints_->get_state()->get_nx() != state_->get_nx() ||
        constraints_->get_state()->get_ndx() != state_->get_ndx()) {
      throw_pretty("Invalid argument: constraints has an incompatible state");
    }
    if (constraints_->get_nu() != nu_) {
      throw_pretty(
          "Invalid argument: constraints doesn't have the observer control "
          "dimension (it should be " +
          std::to_string(nu_) + ")");
    }
  }

  Base::set_u_lb(
      VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity()));
  Base::set_u_ub(
      VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity()));
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  g_lb_live_.resize(dynamics_->get_ng() + ng_c);
  g_ub_live_.resize(dynamics_->get_ng() + ng_c);
  refresh_constraint_bounds();
  Base::set_g_lb(g_lb_live_);
  Base::set_g_ub(g_ub_live_);
}

template <typename Scalar>
void IntegratedObserverModelAbstractTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  params_ = params;
  np_ = params_->get_np();
  if (constraints_ != nullptr && constraints_->get_np() != 0 &&
      constraints_->get_np() != np_) {
    throw_pretty(
        "Invalid argument: constraints parameter dimension does not match "
        "integrated observer parameter dimension");
  }
  d->resize(this);
  if (dynamics_forward_ != nullptr) {
    dynamics_forward_->set_params(d->dynamics, params);
  } else {
    dynamics_inverse_->set_params(d->dynamics, params);
  }
  d->costs = costs_->createData(d->dynamics->shared);
  if (constraints_ != nullptr) {
    d->constraints = constraints_->createData(d->dynamics->shared);
    d->constraints_terminal = constraints_->createData(d->dynamics->shared);
    d->constraints_terminal->resize(constraints_.get(), false);
  } else {
    d->constraints.reset();
    d->constraints_terminal.reset();
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void IntegratedObserverModelAbstractTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty("Invalid call: integrated observer parameters are not set");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  if (dynamics_forward_ != nullptr) {
    dynamics_forward_->update_p(d->dynamics, p);
  } else {
    dynamics_inverse_->update_p(d->dynamics, p);
  }
}

template <typename Scalar>
void IntegratedObserverModelAbstractTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  Base::update_tau(tau_meas);
  dynamics_->update_tau(tau_meas_);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedObserverModelAbstractTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedObserverModelAbstractTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
bool IntegratedObserverModelAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  return std::dynamic_pointer_cast<Data>(data) != nullptr;
}

template <typename Scalar>
std::size_t IntegratedObserverModelAbstractTpl<Scalar>::get_ng() const {
  return dynamics_->get_ng() +
         (constraints_ != nullptr ? constraints_->get_ng() : 0);
}

template <typename Scalar>
std::size_t IntegratedObserverModelAbstractTpl<Scalar>::get_nh() const {
  return dynamics_->get_nh() +
         (constraints_ != nullptr ? constraints_->get_nh() : 0);
}

template <typename Scalar>
std::size_t IntegratedObserverModelAbstractTpl<Scalar>::get_ng_T() const {
  return constraints_ != nullptr ? constraints_->get_ng_T() : 0;
}

template <typename Scalar>
std::size_t IntegratedObserverModelAbstractTpl<Scalar>::get_nh_T() const {
  return constraints_ != nullptr ? constraints_->get_nh_T() : 0;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
IntegratedObserverModelAbstractTpl<Scalar>::get_g_lb() const {
  refresh_constraint_bounds();
  return g_lb_live_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
IntegratedObserverModelAbstractTpl<Scalar>::get_g_ub() const {
  refresh_constraint_bounds();
  return g_ub_live_;
}

template <typename Scalar>
void IntegratedObserverModelAbstractTpl<Scalar>::refresh_constraint_bounds()
    const {
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t ng_max = std::max(ng_d + ng_c, ng_T);
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  g_lb_live_.setConstant(ng_max, -inf);
  g_ub_live_.setConstant(ng_max, inf);
  if (ng_d != 0) {
    g_ub_live_.head(ng_d).setZero();
  }
  if (constraints_ != nullptr && ng_c != 0) {
    g_lb_live_.segment(ng_d, ng_c) = constraints_->get_lb().head(ng_c);
    g_ub_live_.segment(ng_d, ng_c) = constraints_->get_ub().head(ng_c);
  }
}

template <typename Scalar>
const std::shared_ptr<DynamicsModelAbstractTpl<Scalar> >&
IntegratedObserverModelAbstractTpl<Scalar>::get_dynamics() const {
  return dynamics_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
IntegratedObserverModelAbstractTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
IntegratedObserverModelAbstractTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
IntegratedObserverModelAbstractTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
IntegratedObserverModelAbstractTpl<Scalar>::get_multibody_state() const {
  return state_multibody_;
}

template <typename Scalar>
const std::shared_ptr<ImplicitConstraintModelMultipleTpl<Scalar> >&
IntegratedObserverModelAbstractTpl<Scalar>::get_dynamics_constraints() const {
  return dynamics_constraints_;
}

template <typename Scalar>
const Scalar IntegratedObserverModelAbstractTpl<Scalar>::get_dt() const {
  return time_step_;
}

template <typename Scalar>
void IntegratedObserverModelAbstractTpl<Scalar>::set_dt(const Scalar dt) {
  if (dt < Scalar(0.)) {
    throw_pretty("Invalid argument: dt has to be positive");
  }
  time_step_ = dt;
  time_step2_ = dt * dt;
}

template <typename Scalar>
void IntegratedObserverModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedObserverModelAbstract {dt=" << time_step_ << ", nu=" << nu_
     << ", ntau=" << Base::get_ntau() << ", np=" << np_ << "}";
}

template <typename Scalar>
std::shared_ptr<typename IntegratedObserverModelAbstractTpl<Scalar>::Data>
IntegratedObserverModelAbstractTpl<Scalar>::cast_data(
    const std::shared_ptr<ActionDataAbstract>& data) const {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data is not an integrated observer data");
  }
  return d;
}

template <typename Scalar>
std::shared_ptr<typename IntegratedObserverModelAbstractTpl<
    Scalar>::DynamicsDataConstrainedForward>
IntegratedObserverModelAbstractTpl<Scalar>::get_forward_data(
    const std::shared_ptr<Data>& data) const {
  const std::shared_ptr<DynamicsDataConstrainedForward> d =
      std::dynamic_pointer_cast<DynamicsDataConstrainedForward>(data->dynamics);
  if (d == nullptr) {
    throw_pretty("Invalid argument: dynamics data is not constrained forward");
  }
  return d;
}

template <typename Scalar>
std::shared_ptr<typename IntegratedObserverModelAbstractTpl<
    Scalar>::DynamicsDataConstrainedInverse>
IntegratedObserverModelAbstractTpl<Scalar>::get_inverse_data(
    const std::shared_ptr<Data>& data) const {
  const std::shared_ptr<DynamicsDataConstrainedInverse> d =
      std::dynamic_pointer_cast<DynamicsDataConstrainedInverse>(data->dynamics);
  if (d == nullptr) {
    throw_pretty("Invalid argument: dynamics data is not constrained inverse");
  }
  return d;
}

template <typename Scalar>
std::size_t
IntegratedObserverModelAbstractTpl<Scalar>::get_active_constraint_dim() const {
  return dynamics_constraints_ != nullptr ? dynamics_constraints_->get_nc() : 0;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs&
IntegratedObserverModelAbstractTpl<Scalar>::compute_noise_projector(
    const std::shared_ptr<Data>& data,
    const Eigen::Ref<const VectorXs>& x) const {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = get_active_constraint_dim();

  data->noise_projector.setIdentity();
  data->Nc.setIdentity();
  if (nc == 0u) {
    return data->noise_projector;
  }

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  pinocchio::DataTpl<Scalar>* pin_data = nullptr;
  const MatrixXs* Jc = nullptr;
  const bool is_forward_dynamics = dynamics_forward_ != nullptr;
  if (is_forward_dynamics) {
    const std::shared_ptr<DynamicsDataConstrainedForward> dyn_data =
        get_forward_data(data);
    pin_data = &dyn_data->pinocchio;
    Jc = &dyn_data->multibody.constraints->Jc;
  } else {
    const std::shared_ptr<DynamicsDataConstrainedInverse> dyn_data =
        get_inverse_data(data);
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        v = x.tail(nv);
    pinocchio::computeAllTerms(*state_multibody_->get_pinocchio(),
                               data->pinocchioW, q, v);
    pin_data = &data->pinocchioW;
    Jc = &dyn_data->multibody.constraints->Jc;
  }

  data->M = pin_data->M;
  data->Kinv.setZero();
  if (is_forward_dynamics) {
    pinocchio::getKKTContactDynamicMatrixInverse(
        *state_multibody_->get_pinocchio(), *pin_data, Jc->topRows(nc),
        data->Kinv.topLeftCorner(nv + nc, nv + nc));
  } else {
    pinocchio::computeKKTContactDynamicMatrixInverse(
        *state_multibody_->get_pinocchio(), *pin_data, q, Jc->topRows(nc),
        data->Kinv.topLeftCorner(nv + nc, nv + nc), Scalar(0.));
  }
  data->Kinv.block(nv, 0, nc, nv + nc) *= Scalar(-1);

  data->Nc.noalias() = data->Kinv.topLeftCorner(nv, nv) * data->M;
  data->noise_projector.setZero();
  data->noise_projector.topLeftCorner(nv, nv) = data->Nc;
  data->noise_projector.bottomRightCorner(nv, nv) = data->Nc;
  return data->noise_projector;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs&
IntegratedObserverModelAbstractTpl<Scalar>::compute_projected_noise(
    const std::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& w) const {
  if (static_cast<std::size_t>(w.size()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(state_->get_ndx()) + ")");
  }
  data->projected_noise.noalias() = compute_noise_projector(data, x) * w;
  return data->projected_noise;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs&
IntegratedObserverModelAbstractTpl<Scalar>::
    compute_projected_noise_block_jacobian(
        const std::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
        const Eigen::Ref<const VectorXs>& w_block,
        const Eigen::Ref<const VectorXs>& projected_block) const {
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = get_active_constraint_dim();
  if (static_cast<std::size_t>(w_block.size()) != nv ||
      static_cast<std::size_t>(projected_block.size()) != nv) {
    throw_pretty(
        "Invalid argument: projected-noise blocks have wrong "
        "dimension");
  }

  data->projected_noise_block_dx.setZero();
  if (nc == 0u || w_block.isZero()) {
    return data->projected_noise_block_dx;
  }

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  data->projector_x.head(nq) = q;
  data->projector_x.tail(nv) = projected_block;
  data->projector_rhs.setZero();
  data->projector_rhs.head(nv).noalias() = data->M * w_block;
  data->projector_sol.setZero();
  data->projector_sol.head(nv + nc).noalias() =
      data->Kinv.topLeftCorner(nv + nc, nv + nc) *
      data->projector_rhs.head(nv + nc);
  data->projector_force.setZero();
  data->projector_force.head(nc) = data->projector_sol.segment(nv, nc);

  pinocchio::computeAllTerms(*state_multibody_->get_pinocchio(),
                             data->pinocchioW, q, projected_block);
  dynamics_constraints_->calc(data->constraintsW, data->projector_x);
  dynamics_constraints_->updateForce(data->constraintsW,
                                     data->projector_force.head(nc));
  pinocchio::computeRNEADerivatives(*state_multibody_->get_pinocchio(),
                                    data->pinocchioW, q, data->projector_zero,
                                    projected_block - w_block,
                                    data->constraintsW->fext);
  pinocchio::computeGeneralizedGravityDerivatives(
      *state_multibody_->get_pinocchio(), data->pinocchioW, q, data->dgrav_dq);
  pinocchio::computeForwardKinematicsDerivatives(
      *state_multibody_->get_pinocchio(), data->pinocchioW, q, projected_block,
      data->projector_zero);
  dynamics_constraints_->calcDiff(data->constraintsW, data->projector_x);
  dynamics_constraints_->updateRneaDiff(data->constraintsW, data->pinocchioW);

  data->projected_noise_block_dx.noalias() =
      -data->Kinv.topLeftCorner(nv, nv) *
      (data->pinocchioW.dtau_dq - data->dgrav_dq);
  data->projected_noise_block_dx.noalias() -=
      data->Kinv.block(0, nv, nv, nc) * data->constraintsW->dv0_dq.topRows(nc);
  return data->projected_noise_block_dx;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs&
IntegratedObserverModelAbstractTpl<Scalar>::compute_projected_noise_jacobian(
    const std::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& w) const {
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = get_active_constraint_dim();
  if (static_cast<std::size_t>(w.size()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(state_->get_ndx()) + ")");
  }

  data->dprojected_noise_dx.setZero();
  compute_projected_noise(data, x, w);
  if (nc == 0u) {
    return data->dprojected_noise_dx;
  }

  data->dprojected_noise_dx.topLeftCorner(nv, nv) =
      compute_projected_noise_block_jacobian(data, x, w.head(nv),
                                             data->projected_noise.head(nv));
  data->dprojected_noise_dx.block(nv, 0, nv, nv) =
      compute_projected_noise_block_jacobian(
          data, x, w.segment(nv, nv), data->projected_noise.segment(nv, nv));
  compute_projected_noise(data, x, w);
  return data->dprojected_noise_dx;
}

template <typename Scalar>
template <template <typename _TmpScalar> class Model>
IntegratedObserverDataAbstractTpl<Scalar>::IntegratedObserverDataAbstractTpl(
    Model<Scalar>* const model,
    const std::shared_ptr<ParameterDataManager>& params_data)
    : Base(model),
      dynamics(params_data != nullptr
                   ? model->get_dynamics()->createData(params_data)
                   : model->get_dynamics()->createData()),
      costs(nullptr),
      constraints(nullptr),
      constraints_terminal(nullptr),
      dx(model->get_state()->get_ndx()),
      projected_noise(model->get_state()->get_ndx()),
      noise_projector(model->get_state()->get_ndx(),
                      model->get_state()->get_ndx()),
      dprojected_noise_dx(model->get_state()->get_ndx(),
                          model->get_state()->get_ndx()),
      Nc(model->get_state()->get_nv(), model->get_state()->get_nv()),
      projected_noise_block_dx(model->get_state()->get_nv(),
                               model->get_state()->get_nv()),
      M(model->get_state()->get_nv(), model->get_state()->get_nv()),
      Kinv(model->get_state()->get_nv() +
               (model->get_dynamics_constraints() != nullptr
                    ? model->get_dynamics_constraints()->get_nc_total()
                    : 0),
           model->get_state()->get_nv() +
               (model->get_dynamics_constraints() != nullptr
                    ? model->get_dynamics_constraints()->get_nc_total()
                    : 0)),
      dgrav_dq(model->get_state()->get_nv(), model->get_state()->get_nv()),
      pinocchioW(*model->get_multibody_state()->get_pinocchio()),
      constraintsW(
          model->get_dynamics_constraints() != nullptr
              ? model->get_dynamics_constraints()->createData(&pinocchioW)
              : nullptr),
      projector_x(model->get_state()->get_nx()),
      projector_zero(model->get_state()->get_nv()),
      projector_rhs(model->get_state()->get_nv() +
                    (model->get_dynamics_constraints() != nullptr
                         ? model->get_dynamics_constraints()->get_nc_total()
                         : 0)),
      projector_sol(model->get_state()->get_nv() +
                    (model->get_dynamics_constraints() != nullptr
                         ? model->get_dynamics_constraints()->get_nc_total()
                         : 0)),
      projector_force(model->get_dynamics_constraints() != nullptr
                          ? model->get_dynamics_constraints()->get_nc_total()
                          : 0) {
  costs = model->get_costs()->createData(dynamics->shared);
  if (model->get_constraints() != nullptr) {
    constraints = model->get_constraints()->createData(dynamics->shared);
    constraints_terminal =
        model->get_constraints()->createData(dynamics->shared);
    constraints_terminal->resize(model->get_constraints().get(), false);
  }
  setZero();
}

}  // namespace crocoddyl
