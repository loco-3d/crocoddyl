///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Oxford,
//                     University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/integr-action-base.hpp"
#include "crocoddyl/core/controls/poly-zero.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::IntegratedActionModelAbstractTpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, const Scalar time_step, const bool with_cost_residual)
    : Base(model->get_state(), model->get_nu(), model->get_nr()),
      control_(new ControlPolyZero(model->get_nu())),
      differential_(model),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual)
{
  init();
}

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::IntegratedActionModelAbstractTpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, boost::shared_ptr<ControlAbstract> control, 
    const Scalar time_step, const bool with_cost_residual)
    : Base(model->get_state(), control->get_np(), model->get_nr()),
      control_(control),
      differential_(model),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual)
{
  init();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::init()
{
  time_step2_ = time_step_ * time_step_;
  enable_integration_ = true;
  VectorXs p_lb(control_->get_np()), p_ub(control_->get_np());
  control_->convert_bounds(differential_->get_u_lb(), differential_->get_u_ub(), p_lb, p_ub);
  Base::set_u_lb(p_lb);
  Base::set_u_ub(p_ub);
  if (time_step_ < Scalar(0.)) {
    time_step_ = Scalar(1e-3);
    time_step2_ = time_step_ * time_step_;
    std::cerr << "Warning: dt should be positive, set to 1e-3" << std::endl;
  }
  if (time_step_ == Scalar(0.)) {
    enable_integration_ = false;
  }
}

template <typename Scalar>
IntegratedActionModelAbstractTpl<Scalar>::~IntegratedActionModelAbstractTpl() {}

template <typename Scalar>
const boost::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
IntegratedActionModelAbstractTpl<Scalar>::get_differential() const {
  return differential_;
}

template <typename Scalar>
const Scalar IntegratedActionModelAbstractTpl<Scalar>::get_dt() const {
  return time_step_;
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::set_dt(const Scalar dt) {
  if (dt < 0.) {
    throw_pretty("Invalid argument: "
                 << "dt has positive value");
  }
  time_step_ = dt;
  time_step2_ = dt * dt;
}

template <typename Scalar>
std::size_t IntegratedActionModelAbstractTpl<Scalar>::get_nu_diff() const {
  return differential_->get_nu();
}

template <typename Scalar>
void IntegratedActionModelAbstractTpl<Scalar>::set_differential(
    boost::shared_ptr<DifferentialActionModelAbstract> model) {
  const std::size_t nu_diff = model->get_nu();
  if (control_->get_nu() != nu_diff) {
    control_->resize(nu_diff);
    nu_ = control_->get_np();
    unone_ = VectorXs::Zero(nu_);
  }
  nr_ = model->get_nr();
  state_ = model->get_state();
  differential_ = model;
  
  VectorXs p_lb(nu_), p_ub(nu_);
  control_->convert_bounds(differential_->get_u_lb(), differential_->get_u_ub(), p_lb, p_ub);
  Base::set_u_lb(p_lb);
  Base::set_u_ub(p_ub);
}

}  // namespace crocoddyl
