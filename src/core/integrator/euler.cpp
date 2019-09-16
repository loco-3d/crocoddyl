///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/euler.hpp"

namespace crocoddyl {

IntegratedActionModelEuler::IntegratedActionModelEuler(DifferentialActionModelAbstract* const model,
                                                       const double& time_step, const bool& with_cost_residual)
    : ActionModelAbstract(model->get_state(), model->get_nu(), model->get_nr()),
      differential_(model),
      time_step_(time_step),
      time_step2_(time_step * time_step),
      with_cost_residual_(with_cost_residual) {}

IntegratedActionModelEuler::~IntegratedActionModelEuler() {}

void IntegratedActionModelEuler::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                      const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");

  // Static casting the data
  boost::shared_ptr<IntegratedActionDataEuler> d = boost::static_pointer_cast<IntegratedActionDataEuler>(data);

  // Computing the acceleration and cost
  differential_->calc(d->differential, x, u);

  // Computing the next state (discrete time)
  const Eigen::VectorXd& v = x.tail(differential_->get_state().get_nv());
  const Eigen::VectorXd& a = d->differential->xout;
  d->dx << v * time_step_ + a * time_step2_, a * time_step_;
  differential_->get_state().integrate(x, d->dx, d->xnext);

  // Updating the cost value
  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
  d->cost = d->differential->cost;
}

void IntegratedActionModelEuler::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");

  const unsigned int& nv = differential_->get_state().get_nv();
  if (recalc) {
    calc(data, x, u);
  }

  // Static casting the data
  boost::shared_ptr<IntegratedActionDataEuler> d = boost::static_pointer_cast<IntegratedActionDataEuler>(data);

  // Computing the derivatives for the time-continuous model (i.e. differential model)
  differential_->calcDiff(d->differential, x, u, false);
  differential_->get_state().Jintegrate(x, d->dx, d->dxnext_dx, d->dxnext_ddx);

  const Eigen::MatrixXd& da_dx = d->differential->Fx;
  const Eigen::MatrixXd& da_du = d->differential->Fu;
  d->ddx_dx << da_dx * time_step_, da_dx;
  d->ddx_du << da_du * time_step_, da_du;
  for (unsigned int i = 0; i < nv; ++i) {
    d->ddx_dx(i, i + nv) += 1.;
  }
  d->Fx = d->dxnext_dx + time_step_ * d->dxnext_ddx * d->ddx_dx;
  d->Fu = time_step_ * d->dxnext_ddx * d->ddx_du;
  d->Lx = d->differential->Lx;
  d->Lu = d->differential->Lu;
  d->Lxx = d->differential->Lxx;
  d->Lxu = d->differential->Lxu;
  d->Luu = d->differential->Luu;
}

boost::shared_ptr<ActionDataAbstract> IntegratedActionModelEuler::createData() {
  return boost::make_shared<IntegratedActionDataEuler>(this);
}

DifferentialActionModelAbstract* IntegratedActionModelEuler::get_differential() const { return differential_; }

const double& IntegratedActionModelEuler::get_dt() const { return time_step_; }

void IntegratedActionModelEuler::set_dt(double dt) {
  time_step_ = dt;
  time_step2_ = dt * dt;
}

}  // namespace crocoddyl
