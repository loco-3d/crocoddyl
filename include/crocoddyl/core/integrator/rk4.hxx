///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/integrator/rk4.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelRK4Tpl<Scalar>::IntegratedActionModelRK4Tpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, const Scalar& time_step, const bool& with_cost_residual)
    : Base(model->get_state(), model->get_nu(), model->get_nr()),
      differential_(model),
      time_step_(time_step),
      with_cost_residual_(with_cost_residual),
      enable_integration_(true) {
  Base::set_u_lb(differential_->get_u_lb());
  Base::set_u_ub(differential_->get_u_ub());
  if (time_step_ < Scalar(0.)) {
    time_step_ = Scalar(1e-3);
    std::cerr << "Warning: dt should be positive, set to 1e-3" << std::endl;
  }
  if (time_step == 0.) {
    enable_integration_ = false;
  }
  rk4_c_ = {0.0, 0.5, 0.5, 1.0};
}

template <typename Scalar>
IntegratedActionModelRK4Tpl<Scalar>::~IntegratedActionModelRK4Tpl() {}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x,
                                               const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t& nv = differential_->get_state()->get_nv();

  // Static casting the data
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

  // Computing the acceleration and cost
  d->y[0] = x;
  differential_->calc(d->differential[0], d->y[0], u);
  d->ki[0] << y[0].tail(nv), d->differential[0].xout;
  for (std::size_t i = 1; i < 4; ++i) {
    d->dx[i].noalias() = time_step_ * rk4_c_[i] * d->ki[i - 1];
    differential_->get_state()->integrate(x, d->dx[i], d->y[i]);
    differential_->calc(d->differential[i], d->y[i], u);
    d->ki[i] << y[i].tail(nv), d->differential[i].xout;
    d->integral[i] = d->differential[i].cost;
  }

  // Computing the next state (discrete time)
  if (enable_integration_) {
    d->dx_rk4 = (d->ki[0] + 2 * d->ki[1] + 2 * d->ki[2] + d->ki[3]) * time_step_ / 6;
    differential_->get_state()->integrate(x, d->dx, d->xnext);
    d->cost = (d->integral[0] + 2 * d->integral[1] + 2 * d->integral[2] + d->integral[3]) * time_step_ / 6;
  } else {
    d->dx.setZero();
    d->xnext = x;
    d->cost = d->differential->cost;
  }

  // Updating the cost value
  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>& x,
                                                   const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t& nv = differential_->get_state()->get_nv();

  // Static casting the data
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

  // Computing the derivatives for the time-continuous model (i.e. differential model)
  differential_->calcDiff(d->differential[0], d->y[0], u);
  d->dki_dx[0] = d->differential[0].Fx;

  for (std::size_t i = 1; i < 4; ++i) {
    differential_->calcDiff(d->differential[i], d->y[i], u);
    d->dy_dx[i].noalias() = d->dki_dx[i - 1] * rk4_c_[i] * time_step_;
    differential_->get_state()->JintegrateTransport(x, d->dx[i], d->dy_dx[i], second);
    differential_->get_state()->Jintegrate(x, d->dx[i], d->dy_dx[i], d->dy_dx[i], first, addto);
    d->dki_dx[i].noalias() = d->differential[i].Fx * d->dy_dx[i];
  }

  if (enable_integration_) {
    d->Fx.noalias() = time_step_ / 6 * (d->dki_dx[0] + 2 * d->dki_dx[1] + 2 * d->dki_dx[2] + d->dki_dx[3]);

    differential_->get_state()->JintegrateTransport(x, d->dx, d->Fx, second);
    differential_->get_state()->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);

  } else {
    // TODO
  }
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > IntegratedActionModelRK4Tpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool IntegratedActionModelRK4Tpl<Scalar>::checkData(const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (data != NULL) {
    return differential_->checkData(d->differential);
  } else {
    return false;
  }
}

template <typename Scalar>
const boost::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
IntegratedActionModelRK4Tpl<Scalar>::get_differential() const {
  return differential_;
}

template <typename Scalar>
const Scalar& IntegratedActionModelRK4Tpl<Scalar>::get_dt() const {
  return time_step_;
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::set_dt(const Scalar& dt) {
  if (dt < 0.) {
    throw_pretty("Invalid argument: "
                 << "dt has positive value");
  }
  time_step_ = dt;
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::set_differential(boost::shared_ptr<DifferentialActionModelAbstract> model) {
  const std::size_t& nu = model->get_nu();
  if (nu_ != nu) {
    nu_ = nu;
    unone_ = VectorXs::Zero(nu_);
  }
  nr_ = model->get_nr();
  state_ = model->get_state();
  differential_ = model;
  Base::set_u_lb(differential_->get_u_lb());
  Base::set_u_ub(differential_->get_u_ub());
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data,
                                                      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
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
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

  differential_->quasiStatic(d->differential, u, x, maxiter, tol);
}

}  // namespace crocoddyl
