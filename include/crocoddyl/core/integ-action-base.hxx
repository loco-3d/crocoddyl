///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/controls/poly-zero.hpp"
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
      control_(control),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual) {
  if (control->get_nw() != model->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "control.nw (" + std::to_string(control->get_nw()) +
                        ") is not equal to model.nu (" +
                        std::to_string(model->get_nu()) + ")");
  }
  init();
}

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::IntegratedActionModelAbstractTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model->get_state(), model->get_nu(), model->get_nr(),
           model->get_ng(), model->get_nh()),
      differential_(model),
      control_(
          new ControlParametrizationModelPolyZeroTpl<Scalar>(model->get_nu())),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual) {
  init();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::init() {
  time_step2_ = time_step_ * time_step_;
  VectorXs u_lb(nu_), u_ub(nu_);
  control_->convertBounds(differential_->get_u_lb(), differential_->get_u_ub(),
                          u_lb, u_ub);
  Base::set_u_lb(u_lb);
  Base::set_u_ub(u_ub);
  if (time_step_ < Scalar(0.)) {
    time_step_ = Scalar(1e-3);
    time_step2_ = time_step_ * time_step_;
    std::cerr << "Warning: dt should be positive, set to 1e-3" << std::endl;
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelAbstractTpl<Scalar>::createData() {
  if (control_->get_nu() > differential_->get_nu())
    std::cerr << "Warning: It is useless to use an Euler integrator with a "
                 "control parametrization larger than PolyZero"
              << std::endl;
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_ng() const {
  return differential_->get_ng();
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_nh() const {
  return differential_->get_nh();
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_ng_T() const {
  return differential_->get_ng_T();
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_nh_T() const {
  return differential_->get_nh_T();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
IntegratedActionModelAbstractTpl<Scalar>::get_g_lb() const {
  return differential_->get_g_lb();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
IntegratedActionModelAbstractTpl<Scalar>::get_g_ub() const {
  return differential_->get_g_ub();
}

template <typename Scalar>
const std::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_differential() const {
  return differential_;
}

template <typename Scalar>
const std::shared_ptr<ControlParametrizationModelAbstractTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_control() const {
  return control_;
}

template <typename Scalar>
const Scalar IntegratedActionModelAbstractTpl<Scalar>::get_dt() const {
  return time_step_;
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::set_dt(const Scalar dt) {
  if (dt < 0.) {
    throw_pretty("Invalid argument: " << "dt has positive value");
  }
  time_step_ = dt;
  time_step2_ = dt * dt;
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

  VectorXs p_lb(nu_), p_ub(nu_);
  control_->convertBounds(differential_->get_u_lb(), differential_->get_u_ub(),
                          p_lb, p_ub);
  Base::set_u_lb(p_lb);
  Base::set_u_ub(p_ub);
}

}  // namespace crocoddyl
