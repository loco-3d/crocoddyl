///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          New York University, Max Planck Gesellschaft,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {

template <typename Scalar>
ActionModelNumDiffTpl<Scalar>::ActionModelNumDiffTpl(
    std::shared_ptr<Base> model, bool with_gauss_approx)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_nu(),
           internal::checkNumDiffModel(model)->get_nr(),
           internal::checkNumDiffModel(model)->get_ng(),
           internal::checkNumDiffModel(model)->get_nh(),
           internal::checkNumDiffModel(model)->get_ng_T(),
           internal::checkNumDiffModel(model)->get_nh_T()),
      model_(internal::checkNumDiffModel(model)),
      params_(nullptr),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())),
      with_gauss_approx_(with_gauss_approx) {
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
  this->set_u_lb(model_->get_u_lb());
  this->set_u_ub(model_->get_u_ub());
}

template <typename Scalar>
ActionModelNumDiffTpl<Scalar>::ActionModelNumDiffTpl(
    std::shared_ptr<Base> model, std::shared_ptr<ParameterManager> params,
    bool with_gauss_approx)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_nu(),
           internal::checkNumDiffModel(model)->get_nr(),
           internal::checkNumDiffModel(model)->get_ng(),
           internal::checkNumDiffModel(model)->get_nh(),
           internal::checkNumDiffModel(model)->get_ng_T(),
           internal::checkNumDiffModel(model)->get_nh_T(),
           params != nullptr ? params->get_np()
                             : internal::checkNumDiffModel(model)->get_np()),
      model_(internal::checkNumDiffModel(model)),
      params_(params),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())),
      with_gauss_approx_(with_gauss_approx) {
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
  this->set_u_lb(model_->get_u_lb());
  this->set_u_ub(model_->get_u_ub());
  if (params_ == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params_->get_state()->get_nx() != state_->get_nx() ||
      params_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  d->resize(this, true);
  model_->calc(d->data_0, x, u);
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  d->resize(this, false);
  model_->calc(d->data_0, x);
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  d->resize(this, true);
  std::size_t perturbed_ip = this->np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < this->np_) {
      model_->update_p(d->data_p[perturbed_ip], d->p);
    }
    d->dx.setZero();
    d->du.setZero();
    d->dp.setZero();
  });

  const VectorXs& x0 = d->data_0->xnext;
  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();
  const std::size_t nu = model_->get_nu();
  d->du.setZero();

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    // dynamics
    model_->get_state()->diff(x0, d->data_x[ix]->xnext, d->Fx.col(ix));
    // cost
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    if (get_with_gauss_approx()) {
      d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / d->xh_jac;
    }
    // constraint
    data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }
  data->Fx /= d->xh_jac;

  // Computing the d action(x,u) / du
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = d->uh_jac;
    d->up = u + d->du;
    model_->calc(d->data_u[iu], x, d->up);
    // dynamics
    model_->get_state()->diff(x0, d->data_u[iu]->xnext, d->Fu.col(iu));
    // cost
    data->Lu(iu) = (d->data_u[iu]->cost - c0) / d->uh_jac;
    if (get_with_gauss_approx()) {
      d->Ru.col(iu) = (d->data_u[iu]->r - d->data_0->r) / d->uh_jac;
    }
    // constraint
    d->Gu.col(iu) = (d->data_u[iu]->g - g0) / d->uh_jac;
    d->Hu.col(iu) = (d->data_u[iu]->h - h0) / d->uh_jac;
    d->du(iu) = Scalar(0.);
  }
  data->Fu /= d->uh_jac;

#ifdef NDEBUG
  // Computing the d^2 cost(x,u) / dx^2
  d->xh_hess = e_hess_ * std::max(Scalar(1.), d->x_norm);
  d->xh_hess_pow2 = d->xh_hess * d->xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    const Scalar cp = d->data_x[ix]->cost;
    d->dxn = -d->dx;
    model_->get_state()->integrate(x, d->dxn, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    const Scalar cm = d->data_x[ix]->cost;
    data->Lxx(ix, ix) = (cp - 2 * c0 + cm) / d->xh_hess_pow2;
    for (std::size_t jx = ix + 1; jx < ndx; ++jx) {
      d->dx(jx) = d->xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, u);
      const Scalar cpp =
          d->data_x[ix]
              ->cost;  // cost due to positive disturbance in both directions
      d->dx(ix) = Scalar(0.);
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, u);
      const Scalar czp =
          d->data_x[ix]->cost;  // cost due to zero disturbance in 'i' and
                                // positive disturbance in 'j' direction
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / d->xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = d->xh_hess;
      d->dx(jx) = Scalar(0.);
    }
    d->dx(ix) = Scalar(0.);
  }

  // Computing the d^2 cost(x,u) / du^2
  d->uh_hess = e_hess_ * std::max(Scalar(1.), u.norm());
  d->uh_hess_pow2 = d->uh_hess * d->uh_hess;
  for (std::size_t iu = 0; iu < nu; ++iu) {
    d->du(iu) = d->uh_hess;
    d->up = u + d->du;
    model_->calc(d->data_u[iu], x, d->up);
    const Scalar cp = d->data_u[iu]->cost;
    d->up = u - d->du;
    model_->calc(d->data_u[iu], x, d->up);
    const Scalar cm = d->data_u[iu]->cost;
    data->Luu(iu, iu) = (cp - 2 * c0 + cm) / d->uh_hess_pow2;
    for (std::size_t ju = iu + 1; ju < nu; ++ju) {
      d->du(ju) = d->uh_hess;
      d->up = u + d->du;
      model_->calc(d->data_u[iu], x, d->up);
      const Scalar cpp =
          d->data_u[iu]
              ->cost;  // cost due to positive disturbance in both directions
      d->du(iu) = Scalar(0.);
      d->up = u + d->du;
      model_->calc(d->data_u[iu], x, d->up);
      const Scalar czp =
          d->data_u[iu]->cost;  // cost due to zero disturbance in 'i' and
                                // positive disturbance in 'j' direction
      data->Luu(iu, ju) = (cpp - czp - cp + c0) / d->uh_hess_pow2;
      data->Luu(ju, iu) = data->Luu(iu, ju);
      d->du(iu) = d->uh_hess;
      d->du(ju) = Scalar(0.);
    }
    d->du(iu) = Scalar(0.);
  }

  // Computing the d^2 cost(x,u) / dxu
  d->xuh_hess_pow2 = Scalar(4.) * d->xh_hess * d->uh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    for (std::size_t ju = 0; ju < nu; ++ju) {
      d->dx(ix) = d->xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      d->du(ju) = d->uh_hess;
      d->up = u + d->du;
      model_->calc(d->data_x[ix], d->xp, d->up);
      const Scalar cpp = d->data_x[ix]->cost;
      d->up = u - d->du;
      model_->calc(d->data_x[ix], d->xp, d->up);
      const Scalar cpm = d->data_x[ix]->cost;
      d->dxn = -d->dx;
      model_->get_state()->integrate(x, d->dxn, d->xp);
      d->up = u + d->du;
      model_->calc(d->data_x[ix], d->xp, d->up);
      const Scalar cmp = d->data_x[ix]->cost;
      d->up = u - d->du;
      model_->calc(d->data_x[ix], d->xp, d->up);
      const Scalar cmm = d->data_x[ix]->cost;
      data->Lxu(ix, ju) = (cpp - cpm - cmp + cmm) / d->xuh_hess_pow2;
      d->dx(ix) = Scalar(0.);
      d->du(ju) = Scalar(0.);
    }
  }
#endif

  if (get_with_gauss_approx()) {
    data->Lxx = d->Rx.transpose() * d->Rx;
    data->Lxu = d->Rx.transpose() * d->Ru;
    data->Luu = d->Ru.transpose() * d->Ru;
  }

  if (this->np_ > 0) {
    d->dp.setZero();
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_jac;
      d->pp = d->p + d->dp;
      model_->update_p(d->data_p[ip], d->pp);
      model_->calc(d->data_p[ip], x, u);
      model_->get_state()->diff(x0, d->data_p[ip]->xnext, data->Fp.col(ip));
      data->Fp.col(ip) /= d->ph_jac;
      data->Lp(ip) = (d->data_p[ip]->cost - c0) / d->ph_jac;
      data->Gp.col(ip) = (d->data_p[ip]->g - g0) / d->ph_jac;
      data->Hp.col(ip) = (d->data_p[ip]->h - h0) / d->ph_jac;
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }

#ifdef NDEBUG
    d->ph_hess = e_hess_ * std::max(Scalar(1.), d->p.norm());
    d->ph_hess_pow2 = d->ph_hess * d->ph_hess;
    d->xph_hess_pow2 = Scalar(4.) * d->xh_hess * d->ph_hess;
    d->uph_hess_pow2 = Scalar(4.) * d->uh_hess * d->ph_hess;

    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_hess;
      d->pp = d->p + d->dp;
      model_->update_p(d->data_p[ip], d->pp);
      model_->calc(d->data_p[ip], x, u);
      const Scalar cp = d->data_p[ip]->cost;
      d->pp = d->p - d->dp;
      model_->update_p(d->data_p[ip], d->pp);
      model_->calc(d->data_p[ip], x, u);
      const Scalar cm = d->data_p[ip]->cost;
      data->Lpp(ip, ip) = (cp - Scalar(2.) * c0 + cm) / d->ph_hess_pow2;

      for (std::size_t jp = ip + 1; jp < this->np_; ++jp) {
        d->dp(jp) = d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        model_->calc(d->data_p[ip], x, u);
        const Scalar cpp = d->data_p[ip]->cost;
        d->dp(ip) = Scalar(0.);
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        model_->calc(d->data_p[ip], x, u);
        const Scalar czp = d->data_p[ip]->cost;
        data->Lpp(ip, jp) = (cpp - czp - cp + c0) / d->ph_hess_pow2;
        data->Lpp(jp, ip) = data->Lpp(ip, jp);
        d->dp(ip) = d->ph_hess;
        d->dp(jp) = Scalar(0.);
      }
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }

    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      for (std::size_t ix = 0; ix < ndx; ++ix) {
        d->dp(ip) = d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        d->dx(ix) = d->xh_hess;
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp, u);
        const Scalar cpp = d->data_p[ip]->cost;
        d->dxn = -d->dx;
        model_->get_state()->integrate(x, d->dxn, d->xp);
        model_->calc(d->data_p[ip], d->xp, u);
        const Scalar cpm = d->data_p[ip]->cost;
        d->dp(ip) = -d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp, u);
        const Scalar cmp = d->data_p[ip]->cost;
        d->dxn = -d->dx;
        model_->get_state()->integrate(x, d->dxn, d->xp);
        model_->calc(d->data_p[ip], d->xp, u);
        const Scalar cmm = d->data_p[ip]->cost;
        data->Lpx(ip, ix) = (cpp - cpm - cmp + cmm) / d->xph_hess_pow2;
        d->dx(ix) = Scalar(0.);
        d->dp(ip) = Scalar(0.);
      }
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }

    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      for (std::size_t iu = 0; iu < nu; ++iu) {
        d->dp(ip) = d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        d->du(iu) = d->uh_hess;
        d->up = u + d->du;
        model_->calc(d->data_p[ip], x, d->up);
        const Scalar cpp = d->data_p[ip]->cost;
        d->up = u - d->du;
        model_->calc(d->data_p[ip], x, d->up);
        const Scalar cpm = d->data_p[ip]->cost;
        d->dp(ip) = -d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        d->up = u + d->du;
        model_->calc(d->data_p[ip], x, d->up);
        const Scalar cmp = d->data_p[ip]->cost;
        d->up = u - d->du;
        model_->calc(d->data_p[ip], x, d->up);
        const Scalar cmm = d->data_p[ip]->cost;
        data->Lpu(ip, iu) = (cpp - cpm - cmp + cmm) / d->uph_hess_pow2;
        d->du(iu) = Scalar(0.);
        d->dp(ip) = Scalar(0.);
      }
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }
#endif
  }
  restore.restore();
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  d->resize(this, false);
  std::size_t perturbed_ip = this->np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < this->np_) {
      model_->update_p(d->data_p[perturbed_ip], d->p);
    }
    d->dx.setZero();
    d->dp.setZero();
  });

  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    // cost
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    if (get_with_gauss_approx()) {
      d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / d->xh_jac;
    }
    // constraint
    d->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    d->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }

#ifdef NDEBUG
  // Computing the d^2 cost(x,u) / dx^2
  d->xh_hess = e_hess_ * std::max(Scalar(1.), d->x_norm);
  d->xh_hess_pow2 = d->xh_hess * d->xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    // We can apply the same formulas for finite difference as above
    d->dx(ix) = d->xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar cp = d->data_x[ix]->cost;
    d->dxn = -d->dx;
    model_->get_state()->integrate(x, d->dxn, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar cm = d->data_x[ix]->cost;
    data->Lxx(ix, ix) = (cp - 2 * c0 + cm) / d->xh_hess_pow2;
    for (std::size_t jx = ix + 1; jx < ndx; ++jx) {
      d->dx(jx) = d->xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp);
      const Scalar cpp =
          d->data_x[ix]
              ->cost;  // cost due to positive disturbance in both directions
      d->dx(ix) = Scalar(0.);
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp);
      const Scalar czp =
          d->data_x[ix]->cost;  // cost due to zero disturbance in 'i' and
                                // positive disturbance in 'j' direction
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / d->xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = d->xh_hess;
      d->dx(jx) = Scalar(0.);
    }
    d->dx(ix) = Scalar(0.);
  }
#endif

  if (get_with_gauss_approx()) {
    data->Lxx = d->Rx.transpose() * d->Rx;
  }

  if (this->np_ > 0) {
    d->dp.setZero();
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_jac;
      d->pp = d->p + d->dp;
      model_->update_p(d->data_p[ip], d->pp);
      model_->calc(d->data_p[ip], x);
      data->Lp(ip) = (d->data_p[ip]->cost - c0) / d->ph_jac;
      data->Gp.col(ip) = (d->data_p[ip]->g - g0) / d->ph_jac;
      data->Hp.col(ip) = (d->data_p[ip]->h - h0) / d->ph_jac;
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }

#ifdef NDEBUG
    d->ph_hess = e_hess_ * std::max(Scalar(1.), d->p.norm());
    d->ph_hess_pow2 = d->ph_hess * d->ph_hess;
    d->xph_hess_pow2 = Scalar(4.) * d->xh_hess * d->ph_hess;

    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_hess;
      d->pp = d->p + d->dp;
      model_->update_p(d->data_p[ip], d->pp);
      model_->calc(d->data_p[ip], x);
      const Scalar cp = d->data_p[ip]->cost;
      d->pp = d->p - d->dp;
      model_->update_p(d->data_p[ip], d->pp);
      model_->calc(d->data_p[ip], x);
      const Scalar cm = d->data_p[ip]->cost;
      data->Lpp(ip, ip) = (cp - Scalar(2.) * c0 + cm) / d->ph_hess_pow2;

      for (std::size_t jp = ip + 1; jp < this->np_; ++jp) {
        d->dp(jp) = d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        model_->calc(d->data_p[ip], x);
        const Scalar cpp = d->data_p[ip]->cost;
        d->dp(ip) = Scalar(0.);
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        model_->calc(d->data_p[ip], x);
        const Scalar czp = d->data_p[ip]->cost;
        data->Lpp(ip, jp) = (cpp - czp - cp + c0) / d->ph_hess_pow2;
        data->Lpp(jp, ip) = data->Lpp(ip, jp);
        d->dp(ip) = d->ph_hess;
        d->dp(jp) = Scalar(0.);
      }
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }

    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      for (std::size_t ix = 0; ix < ndx; ++ix) {
        d->dp(ip) = d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        d->dx(ix) = d->xh_hess;
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cpp = d->data_p[ip]->cost;
        d->dxn = -d->dx;
        model_->get_state()->integrate(x, d->dxn, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cpm = d->data_p[ip]->cost;
        d->dp(ip) = -d->ph_hess;
        d->pp = d->p + d->dp;
        model_->update_p(d->data_p[ip], d->pp);
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cmp = d->data_p[ip]->cost;
        d->dxn = -d->dx;
        model_->get_state()->integrate(x, d->dxn, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cmm = d->data_p[ip]->cost;
        data->Lpx(ip, ix) = (cpp - cpm - cmp + cmm) / d->xph_hess_pow2;
        d->dx(ix) = Scalar(0.);
        d->dp(ip) = Scalar(0.);
      }
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = this->np_;
    }
#endif
  }
  restore.restore();
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ActionModelNumDiffTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ActionModelNumDiffTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  const std::shared_ptr<ActionDataAbstract> data = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, params_data);
  if (params_ != nullptr) {
    set_params(data, params_);
  }
  return data;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  this->np_ = params_->get_np();
  d->resize(this);
  model_->set_params(d->data_0, params_);
  for (std::size_t ix = 0; ix < d->data_x.size(); ++ix) {
    model_->set_params(d->data_x[ix], params_);
  }
  for (std::size_t iu = 0; iu < d->data_u.size(); ++iu) {
    model_->set_params(d->data_u[iu], params_);
  }
  for (std::size_t ip = 0; ip < d->data_p.size(); ++ip) {
    model_->set_params(d->data_p[ip], params_);
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(this->np_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  d->p = p;
  model_->update_p(d->data_0, p);
  for (std::size_t ix = 0; ix < d->data_x.size(); ++ix) {
    model_->update_p(d->data_x[ix], p);
  }
  for (std::size_t iu = 0; iu < d->data_u.size(); ++iu) {
    model_->update_p(d->data_u[iu], p);
  }
  for (std::size_t ip = 0; ip < d->data_p.size(); ++ip) {
    model_->update_p(d->data_p[ip], p);
  }
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
    const Scalar tol) {
  Data* d = static_cast<Data*>(data.get());
  model_->quasiStatic(d->data_0, u, x, maxiter, tol);
}

template <typename Scalar>
template <typename NewScalar>
ActionModelNumDiffTpl<NewScalar> ActionModelNumDiffTpl<Scalar>::cast() const {
  typedef ActionModelNumDiffTpl<NewScalar> ReturnType;
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  std::shared_ptr<ParameterManagerNew> params;
  if (params_ != nullptr) {
    params = std::make_shared<ParameterManagerNew>(
        params_->template cast<NewScalar>());
    return ReturnType(model_->template cast<NewScalar>(), params,
                      with_gauss_approx_);
  }
  return ReturnType(model_->template cast<NewScalar>(), with_gauss_approx_);
}

template <typename Scalar>
const std::shared_ptr<ActionModelAbstractTpl<Scalar> >&
ActionModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const std::shared_ptr<typename ActionModelNumDiffTpl<Scalar>::ParameterManager>&
ActionModelNumDiffTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const Scalar ActionModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
}

template <typename Scalar>
bool ActionModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return with_gauss_approx_;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelNumDiffTpl {action=" << *model_ << "}";
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /** x */) {
  // do nothing in the general case
}

}  // namespace crocoddyl
