///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/integrator/euler.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::IntegratedActionModelEulerTpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, const Scalar& time_step, const bool& with_cost_residual)
    : Base(model->get_state(), model->get_nu(), model->get_nr()),
      differential_(model),
      time_step_(time_step),
      time_step2_(time_step * time_step),
      with_cost_residual_(with_cost_residual),
      enable_integration_(true) {
  Base::set_u_lb(differential_->get_u_lb());
  Base::set_u_ub(differential_->get_u_ub());
  if (time_step_ < 0.) {
    time_step_ = 1e-3;
    time_step2_ = time_step_ * time_step_;
    std::cerr << "Warning: dt has positive value, set to 1e-3" << std::endl;
  }
  if (time_step == 0.) {
    enable_integration_ = false;
  }
}

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::~IntegratedActionModelEulerTpl() {}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
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

  // Static casting the data
  boost::shared_ptr<IntegratedActionDataEuler> d = boost::static_pointer_cast<IntegratedActionDataEuler>(data);

  // Computing the acceleration and cost
  differential_->calc(d->differential, x, u);

  // Computing the next state (discrete time)
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(differential_->get_state()->get_nv());
  const VectorXs& a = d->differential->xout;
  if (enable_integration_) {
    d->dx << v * time_step_ + a * time_step2_, a * time_step_;
    differential_->get_state()->integrate(x, d->dx, d->xnext);
    d->cost = time_step_ * d->differential->cost;
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
void IntegratedActionModelEulerTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
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
  boost::shared_ptr<IntegratedActionDataEuler> d = boost::static_pointer_cast<IntegratedActionDataEuler>(data);

  // Computing the derivatives for the time-continuous model (i.e. differential model)
  differential_->calcDiff(d->differential, x, u);
  differential_->get_state()->Jintegrate(x, d->dx, d->dxnext_dx, d->dxnext_ddx);

  d->Fx = d->dxnext_dx;
  if (enable_integration_) {
    const MatrixXs& da_dx = d->differential->Fx;
    const MatrixXs& da_du = d->differential->Fu;
    d->ddx_dx << da_dx * time_step_, da_dx;
    d->ddx_du << da_du * time_step_, da_du;
    for (std::size_t i = 0; i < nv; ++i) {
      d->ddx_dx(i, i + nv) += 1.;
    }
    d->Fx.noalias() += time_step_ * (d->dxnext_ddx * d->ddx_dx);
    d->Fu.noalias() = time_step_ * (d->dxnext_ddx * d->ddx_du);
    d->Lx = time_step_ * d->differential->Lx;
    d->Lu = time_step_ * d->differential->Lu;
    d->Lxx = time_step_ * d->differential->Lxx;
    d->Lxu = time_step_ * d->differential->Lxu;
    d->Luu = time_step_ * d->differential->Luu;
  } else {
    d->Fu.setZero();
    d->Lx = d->differential->Lx;
    d->Lu = d->differential->Lu;
    d->Lxx = d->differential->Lxx;
    d->Lxu = d->differential->Lxu;
    d->Luu = d->differential->Luu;
  }
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > IntegratedActionModelEulerTpl<Scalar>::createData() {
  return boost::make_shared<IntegratedActionDataEuler>(this);
}

template <typename Scalar>
const boost::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
IntegratedActionModelEulerTpl<Scalar>::get_differential() const {
  return differential_;
}

template <typename Scalar>
const Scalar& IntegratedActionModelEulerTpl<Scalar>::get_dt() const {
  return time_step_;
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::set_dt(const Scalar& dt) {
  if (dt < 0.) {
    throw_pretty("Invalid argument: "
                 << "dt has positive value");
  }
  time_step_ = dt;
  time_step2_ = dt * dt;
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::set_differential(
    boost::shared_ptr<DifferentialActionModelAbstract> model) {
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

}  // namespace crocoddyl
