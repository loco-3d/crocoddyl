///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, IRI: CSIC-UPC, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/integrator/rk4.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelRK4Tpl<Scalar>::IntegratedActionModelRK4Tpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, const Scalar time_step, const bool with_cost_residual)
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
  if (time_step == Scalar(0.)) {
    enable_integration_ = false;
  }
  rk4_c_.push_back(Scalar(0.));
  rk4_c_.push_back(Scalar(0.5));
  rk4_c_.push_back(Scalar(0.5));
  rk4_c_.push_back(Scalar(1.));
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

  const std::size_t nv = differential_->get_state()->get_nv();

  // Static casting the data
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

  // Computing the acceleration and cost
  differential_->calc(d->differential[0], x, u);

  // Computing the next state (discrete time)
  if (enable_integration_) {
    d->y[0] = x;
    d->ki[0].head(nv) = d->y[0].tail(nv);
    d->ki[0].tail(nv) = d->differential[0]->xout;
    d->integral[0] = d->differential[0]->cost;
    for (std::size_t i = 1; i < 4; ++i) {
      d->dx_rk4[i].noalias() = time_step_ * rk4_c_[i] * d->ki[i - 1];
      differential_->get_state()->integrate(x, d->dx_rk4[i], d->y[i]);
      differential_->calc(d->differential[i], d->y[i], u);
      d->ki[i].head(nv) = d->y[i].tail(nv);
      d->ki[i].tail(nv) = d->differential[i]->xout;
      d->integral[i] = d->differential[i]->cost;
    }
    d->dx = (d->ki[0] + Scalar(2.) * d->ki[1] + Scalar(2.) * d->ki[2] + d->ki[3]) * time_step_ / Scalar(6.);
    differential_->get_state()->integrate(x, d->dx, d->xnext);
    d->cost = (d->integral[0] + Scalar(2.) * d->integral[1] + Scalar(2.) * d->integral[2] + d->integral[3]) *
              time_step_ / Scalar(6.);
  } else {
    d->dx.setZero();
    d->xnext = x;
    d->cost = d->differential[0]->cost;
  }

  // Updating the cost value
  if (with_cost_residual_) {
    d->r = d->differential[0]->r;
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

  const std::size_t nv = differential_->get_state()->get_nv();

  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

  differential_->calcDiff(d->differential[0], x, u);

  if (enable_integration_) {
    d->dki_dy[0].bottomRows(nv) = d->differential[0]->Fx;
    d->dki_dx[0] = d->dki_dy[0];
    d->dki_du[0].bottomRows(nv) = d->differential[0]->Fu;

    d->dli_dx[0] = d->differential[0]->Lx;
    d->dli_du[0] = d->differential[0]->Lu;

    d->ddli_ddx[0] = d->differential[0]->Lxx;
    d->ddli_ddu[0] = d->differential[0]->Luu;
    d->ddli_dxdu[0] = d->differential[0]->Lxu;

    for (std::size_t i = 1; i < 4; ++i) {
      differential_->calcDiff(d->differential[i], d->y[i], u);
      d->dki_dy[i].bottomRows(nv) = d->differential[i]->Fx;

      d->dyi_dx[i].noalias() = d->dki_dx[i - 1] * rk4_c_[i] * time_step_;
      differential_->get_state()->JintegrateTransport(x, d->dx_rk4[i], d->dyi_dx[i], second);
      differential_->get_state()->Jintegrate(x, d->dx_rk4[i], d->dyi_dx[i], d->dyi_dx[i], first, addto);
      d->dki_dx[i].noalias() = d->dki_dy[i] * d->dyi_dx[i];

      d->dyi_du[i].noalias() = d->dki_du[i - 1] * rk4_c_[i] * time_step_;
      differential_->get_state()->JintegrateTransport(x, d->dx_rk4[i], d->dyi_du[i], second);
      d->dki_du[i].noalias() = d->dki_dy[i] * d->dyi_du[i];
      d->dki_du[i].bottomRows(nv) += d->differential[i]->Fu;

      d->dli_dx[i].noalias() = d->differential[i]->Lx.transpose() * d->dyi_dx[i];
      d->dli_du[i].noalias() = d->differential[i]->Lu.transpose();
      d->dli_du[i].noalias() += d->differential[i]->Lx.transpose() * d->dyi_du[i];

      d->Lxx_partialx[i].noalias() = d->differential[i]->Lxx * d->dyi_dx[i];
      d->ddli_ddx[i].noalias() = d->dyi_dx[i].transpose() * d->Lxx_partialx[i];

      d->Luu_partialx[i].noalias() = d->differential[i]->Lxu.transpose() * d->dyi_du[i];
      d->Lxx_partialu[i].noalias() = d->differential[i]->Lxx * d->dyi_du[i];
      d->ddli_ddu[i].noalias() = d->differential[i]->Luu + d->Luu_partialx[i].transpose() + d->Luu_partialx[i] +
                                 d->dyi_du[i].transpose() * d->Lxx_partialu[i];

      d->ddli_dxdu[i].noalias() = d->dyi_dx[i].transpose() * d->differential[i]->Lxu;
      d->ddli_dxdu[i].noalias() += d->dyi_dx[i].transpose() * d->Lxx_partialu[i];
    }

    d->Fx.noalias() = time_step_ / Scalar(6.) *
                      (d->dki_dx[0] + Scalar(2.) * d->dki_dx[1] + Scalar(2.) * d->dki_dx[2] + d->dki_dx[3]);
    differential_->get_state()->JintegrateTransport(x, d->dx, d->Fx, second);
    differential_->get_state()->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);

    d->Fu.noalias() = time_step_ / Scalar(6.) *
                      (d->dki_du[0] + Scalar(2.) * d->dki_du[1] + Scalar(2.) * d->dki_du[2] + d->dki_du[3]);
    differential_->get_state()->JintegrateTransport(x, d->dx, d->Fu, second);

    d->Lx.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_dx[0] + Scalar(2.) * d->dli_dx[1] + Scalar(2.) * d->dli_dx[2] + d->dli_dx[3]);
    d->Lu.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_du[0] + Scalar(2.) * d->dli_du[1] + Scalar(2.) * d->dli_du[2] + d->dli_du[3]);

    d->Lxx.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_ddx[0] + Scalar(2.) * d->ddli_ddx[1] + Scalar(2.) * d->ddli_ddx[2] + d->ddli_ddx[3]);
    d->Luu.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_ddu[0] + Scalar(2.) * d->ddli_ddu[1] + Scalar(2.) * d->ddli_ddu[2] + d->ddli_ddu[3]);
    d->Lxu.noalias() =
        time_step_ / Scalar(6.) *
        (d->ddli_dxdu[0] + Scalar(2.) * d->ddli_dxdu[1] + Scalar(2.) * d->ddli_dxdu[2] + d->ddli_dxdu[3]);
  } else {
    differential_->get_state()->Jintegrate(x, d->dx, d->Fx, d->Fx);
    d->Fu.setZero();
    d->Lx = d->differential[0]->Lx;
    d->Lu = d->differential[0]->Lu;
    d->Lxx = d->differential[0]->Lxx;
    d->Lxu = d->differential[0]->Lxu;
    d->Luu = d->differential[0]->Luu;
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
    return differential_->checkData(d->differential[0]) && differential_->checkData(d->differential[2]) &&
           differential_->checkData(d->differential[1]) && differential_->checkData(d->differential[3]);
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
const Scalar IntegratedActionModelRK4Tpl<Scalar>::get_dt() const {
  return time_step_;
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::set_dt(const Scalar dt) {
  if (dt < 0.) {
    throw_pretty("Invalid argument: "
                 << "dt has positive value");
  }
  time_step_ = dt;
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::set_differential(boost::shared_ptr<DifferentialActionModelAbstract> model) {
  const std::size_t nu = model->get_nu();
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
                                                      const std::size_t maxiter, const Scalar tol) {
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

  differential_->quasiStatic(d->differential[0], u, x, maxiter, tol);
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedActionModelRK4 {dt=" << time_step_ << ", " << *differential_ << "}";
}

}  // namespace crocoddyl
