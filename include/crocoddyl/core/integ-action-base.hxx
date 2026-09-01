///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <limits>

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/controls/poly-zero.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/integ-action-base.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::IntegratedActionModelAbstractTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model->get_state(), control->get_nu(), model->get_nr(),
           model->get_ng(), model->get_nh()),
      differential_(model),
      dynamics_(nullptr),
      costs_(nullptr),
      constraints_(nullptr),
      control_(control),
      params_(nullptr),
      integrator_time_(std::make_shared<IntegratorTime>(
          time_step < Scalar(0.) ? Scalar(1e-3) : time_step, false)),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual) {
  init();
}

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::IntegratedActionModelAbstractTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model->get_state(), model->get_nu(), model->get_nr(),
           model->get_ng(), model->get_nh()),
      differential_(model),
      dynamics_(nullptr),
      costs_(nullptr),
      constraints_(nullptr),
      control_(
          new ControlParametrizationModelPolyZeroTpl<Scalar>(model->get_nu())),
      params_(nullptr),
      integrator_time_(std::make_shared<IntegratorTime>(
          time_step < Scalar(0.) ? Scalar(1e-3) : time_step, false)),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual) {
  init();
}

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::IntegratedActionModelAbstractTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    std::shared_ptr<IntegratorTime> integrator_time)
    : Base(dynamics != nullptr ? dynamics->get_state() : nullptr,
           control != nullptr ? control->get_nu()
                              : (dynamics != nullptr ? dynamics->get_nu() : 0),
           costs != nullptr ? costs->get_nr() : 0,
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
      differential_(nullptr),
      dynamics_(dynamics),
      costs_(costs),
      constraints_(constraints),
      control_(control != nullptr
                   ? control
                   : std::shared_ptr<ControlParametrizationModelAbstract>(
                         new ControlParametrizationModelPolyZeroTpl<Scalar>(
                             dynamics != nullptr ? dynamics->get_nu() : 0))),
      params_(nullptr),
      integrator_time_(integrator_time != nullptr
                           ? integrator_time
                           : std::make_shared<IntegratorTime>()),
      time_step_(integrator_time_ != nullptr ? integrator_time_->get_time_step()
                                             : Scalar(0.)),
      with_cost_residual_(true) {
  init();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::init() {
  if (integrator_time_ == nullptr) {
    throw_pretty("Invalid argument: integrator_time is null");
  }
  if (control_ == nullptr) {
    throw_pretty("Invalid argument: control is null");
  }
  if (time_step_ < Scalar(0.)) {
    std::cerr << "Warning: dt should be positive, set to 1e-3" << std::endl;
  }

  refresh_integrator_time();

  if (dynamics_ == nullptr) {
    if (differential_ == nullptr) {
      throw_pretty("Invalid argument: differential backend is null");
    }
    if (control_->get_nw() != differential_->get_nu()) {
      throw_pretty("Invalid argument: "
                   << "control.nw (" + std::to_string(control_->get_nw()) +
                          ") is not equal to model.nu (" +
                          std::to_string(differential_->get_nu()) + ")");
    }

    VectorXs u_lb(nu_), u_ub(nu_);
    control_->convertBounds(differential_->get_u_lb(),
                            differential_->get_u_ub(), u_lb, u_ub);
    Base::set_u_lb(u_lb);
    Base::set_u_ub(u_ub);
    return;
  }

  if (costs_ == nullptr) {
    throw_pretty("Invalid argument: costs is null");
  }
  if (dynamics_->get_dyn_type() == DynamicsType::DiscreteTime) {
    throw_pretty(
        "Invalid argument: continuous integrators cannot be built "
        "from DiscreteTime dynamics");
  }
  if (control_->get_nw() != dynamics_->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "control.nw (" + std::to_string(control_->get_nw()) +
                        ") is not equal to dynamics.nu (" +
                        std::to_string(dynamics_->get_nu()) + ")");
  }
  if (costs_->get_state()->get_nx() != dynamics_->get_state()->get_nx() ||
      costs_->get_state()->get_ndx() != dynamics_->get_state()->get_ndx()) {
    throw_pretty("Invalid argument: costs has an incompatible state");
  }
  if (costs_->get_nu() != dynamics_->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "costs doesn't have the same control dimension as dynamics "
                    "(it should be " +
                        std::to_string(dynamics_->get_nu()) + ")");
  }
  if (constraints_ != nullptr) {
    if (constraints_->get_state()->get_nx() !=
            dynamics_->get_state()->get_nx() ||
        constraints_->get_state()->get_ndx() !=
            dynamics_->get_state()->get_ndx()) {
      throw_pretty("Invalid argument: constraints has an incompatible state");
    }
    if (constraints_->get_nu() != dynamics_->get_nu()) {
      throw_pretty(
          "Invalid argument: "
          << "constraints doesn't have the same control dimension as dynamics "
             "(it should be " +
                 std::to_string(dynamics_->get_nu()) + ")");
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
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelAbstractTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelAbstractTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>&) {
  const std::size_t nw =
      dynamics_ != nullptr ? dynamics_->get_nu() : differential_->get_nu();
  if (control_->get_nu() > nw) {
    std::cerr << "Warning: It is useless to use an Euler integrator with a "
                 "control parametrization larger than PolyZero"
              << std::endl;
  }
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_ng() const {
  return dynamics_ != nullptr
             ? dynamics_->get_ng() +
                   (constraints_ != nullptr ? constraints_->get_ng() : 0)
             : differential_->get_ng();
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_nh() const {
  return dynamics_ != nullptr
             ? dynamics_->get_nh() +
                   (constraints_ != nullptr ? constraints_->get_nh() : 0)
             : differential_->get_nh();
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_ng_T() const {
  return dynamics_ != nullptr
             ? (constraints_ != nullptr ? constraints_->get_ng_T() : 0)
             : differential_->get_ng_T();
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_nh_T() const {
  return dynamics_ != nullptr
             ? (constraints_ != nullptr ? constraints_->get_nh_T() : 0)
             : differential_->get_nh_T();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
IntegratedActionModelAbstractTpl<Scalar>::get_g_lb() const {
  if (dynamics_ != nullptr) {
    refresh_constraint_bounds();
    return g_lb_live_;
  }
  return differential_->get_g_lb();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
IntegratedActionModelAbstractTpl<Scalar>::get_g_ub() const {
  if (dynamics_ != nullptr) {
    refresh_constraint_bounds();
    return g_ub_live_;
  }
  return differential_->get_g_ub();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::refresh_constraint_bounds()
    const {
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t ng_max = std::max(ng_c + ng_d, ng_T);
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
const std::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_differential() const {
  return differential_;
}

template <typename Scalar>
const std::shared_ptr<DynamicsModelAbstractTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_dynamics() const {
  return dynamics_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const std::shared_ptr<ControlParametrizationModelAbstractTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_control() const {
  return control_;
}

template <typename Scalar>
const std::shared_ptr<IntegratorTimeTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_integrator_time() const {
  return integrator_time_;
}

template <typename Scalar>
const Scalar IntegratedActionModelAbstractTpl<Scalar>::get_dt() const {
  return integrator_time_->get_time_step();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::set_dt(const Scalar dt) {
  if (dt < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "dt has positive value");
  }
  integrator_time_->set_time_step(dt);
  refresh_integrator_time();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::refresh_integrator_time() {
  time_step_ = integrator_time_->get_time_step();
  time_step2_ = integrator_time_->get_time_step2();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::set_differential(
    std::shared_ptr<DifferentialActionModelAbstract> model) {
  if (control_->get_nw() != model->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "control.nw (" + std::to_string(control_->get_nw()) +
                        ") is not equal to model.nu (" +
                        std::to_string(model->get_nu()) + ")");
  }

  nr_ = model->get_nr();
  state_ = model->get_state();
  differential_ = model;
  dynamics_.reset();
  costs_.reset();
  constraints_.reset();

  VectorXs p_lb(nu_), p_ub(nu_);
  control_->convertBounds(differential_->get_u_lb(), differential_->get_u_ub(),
                          p_lb, p_ub);
  Base::set_u_lb(p_lb);
  Base::set_u_ub(p_ub);
}

}  // namespace crocoddyl
