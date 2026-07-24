///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <limits>

#include "crocoddyl/core/numdiff/observer.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ObserverModelNumDiffTpl<Scalar>::ObserverModelNumDiffTpl(
    std::shared_ptr<Base> model, bool with_gauss_approx)
    : Base(detail::check_observer_model(model)->get_state(),
           detail::check_observer_model(model)->get_ntau(),
           detail::check_observer_model(model)->get_nu(), 0,
           detail::check_observer_model(model)->get_ng(),
           detail::check_observer_model(model)->get_nh(),
           detail::check_observer_model(model)->get_ng_T(),
           detail::check_observer_model(model)->get_nh_T(),
           detail::check_observer_model(model)->get_np()),
      model_(detail::check_observer_model(model)),
      params_(nullptr),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())),
      with_gauss_approx_(with_gauss_approx) {
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
  Base::set_u_lb(model_->get_u_lb());
  Base::set_u_ub(model_->get_u_ub());
}

template <typename Scalar>
ObserverModelNumDiffTpl<Scalar>::ObserverModelNumDiffTpl(
    std::shared_ptr<Base> model, std::shared_ptr<ParameterManager> params,
    bool with_gauss_approx)
    : Base(detail::check_observer_model(model)->get_state(),
           detail::check_observer_model(model)->get_ntau(),
           detail::check_observer_model(model)->get_nu(), 0,
           detail::check_observer_model(model)->get_ng(),
           detail::check_observer_model(model)->get_nh(),
           detail::check_observer_model(model)->get_ng_T(),
           detail::check_observer_model(model)->get_nh_T(),
           params != nullptr ? params->get_np()
                             : detail::check_observer_model(model)->get_np()),
      model_(detail::check_observer_model(model)),
      params_(params),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())),
      with_gauss_approx_(with_gauss_approx) {
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
  Base::set_u_lb(model_->get_u_lb());
  Base::set_u_ub(model_->get_u_ub());
  if (params_ == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params_->get_state()->get_nx() != state_->get_nx() ||
      params_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& w) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(w.size()) != nu_) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  d->resize(this);
  model_->calc(d->data_0, x, w);
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  data->g = d->data_0->g;
  data->h = d->data_0->h;
  d->dissipative_E =
      static_cast<ObserverDataAbstractTpl<Scalar>*>(d->data_0.get())
          ->dissipative_E;
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  d->resize(this, false);
  model_->calc(d->data_0, x);
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  data->g = d->data_0->g;
  data->h = d->data_0->h;
  d->dissipative_E =
      static_cast<ObserverDataAbstractTpl<Scalar>*>(d->data_0.get())
          ->dissipative_E;
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& w) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(w.size()) != nu_) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  ObserverDataAbstractTpl<Scalar>* const data_obs = d.get();
  std::size_t perturbed_ip = np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < np_) {
      model_->update_p(d->data_p[perturbed_ip], d->p);
    }
    d->dx.setZero();
    d->dw.setZero();
    d->dp.setZero();
  });

  const VectorXs& x0 = d->data_0->xnext;
  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  const VectorXs& g0 = data->g;
  const VectorXs& h0 = data->h;
  const auto* d0_obs =
      static_cast<const ObserverDataAbstractTpl<Scalar>*>(d->data_0.get());
  const VectorXs e0 = d0_obs->dissipative_E;

  const std::size_t ndx = model_->get_state()->get_ndx();
  const std::size_t nv = model_->get_state()->get_nv();
  const std::size_t nu = model_->get_nu();
  const std::size_t ng = model_->get_ng();
  const std::size_t nh = model_->get_nh();

  // ---------------------------------------------------------------------------
  // d/dx
  // ---------------------------------------------------------------------------
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);

  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, w);

    model_->get_state()->diff(x0, d->data_x[ix]->xnext, d->Fx.col(ix));
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    if (ng != 0) {
      data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    }
    if (nh != 0) {
      data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    }
    // dissipative energy Jacobian w.r.t. velocity (velocity = x[nv:])
    if (ix >= nv) {
      const auto* dxi_obs = static_cast<const ObserverDataAbstractTpl<Scalar>*>(
          d->data_x[ix].get());
      d->dE_dv.col(ix - nv) = (dxi_obs->dissipative_E - e0) / d->xh_jac;
    }
    d->dx(ix) = Scalar(0.);
  }
  data->Fx /= d->xh_jac;

  // ---------------------------------------------------------------------------
  // d/dw
  // ---------------------------------------------------------------------------
  d->wh_jac = e_jac_ * std::max(Scalar(1.), w.norm());
  d->dw.setZero();
  for (std::size_t iw = 0; iw < nu; ++iw) {
    d->dw(iw) = d->wh_jac;
    model_->calc(d->data_w[iw], x, w + d->dw);

    model_->get_state()->diff(x0, d->data_w[iw]->xnext, d->Fu.col(iw));
    data->Lu(iw) = (d->data_w[iw]->cost - c0) / d->wh_jac;
    if (ng != 0) {
      data->Gu.col(iw) = (d->data_w[iw]->g - g0) / d->wh_jac;
    }
    if (nh != 0) {
      data->Hu.col(iw) = (d->data_w[iw]->h - h0) / d->wh_jac;
    }
    d->dw(iw) = Scalar(0.);
  }
  data->Fu /= d->wh_jac;

  if (np_ > 0) {
    d->dp.setZero();
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_jac;
      model_->update_p(d->data_p[ip], d->p + d->dp);
      model_->calc(d->data_p[ip], x, w);

      model_->get_state()->diff(x0, d->data_p[ip]->xnext, data->Fp.col(ip));
      data->Fp.col(ip) /= d->ph_jac;
      data->Lp(ip) = (d->data_p[ip]->cost - c0) / d->ph_jac;
      if (ng != 0) {
        data->Gp.col(ip) = (d->data_p[ip]->g - g0) / d->ph_jac;
      }
      if (nh != 0) {
        data->Hp.col(ip) = (d->data_p[ip]->h - h0) / d->ph_jac;
      }
      const auto* dpi_obs = static_cast<const ObserverDataAbstractTpl<Scalar>*>(
          d->data_p[ip].get());
      data_obs->dE_dp.col(ip) = (dpi_obs->dissipative_E - e0) / d->ph_jac;
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }

#ifdef NDEBUG
    d->xh_hess = e_hess_ * std::max(Scalar(1.), d->x_norm);
    d->wh_hess = e_hess_ * std::max(Scalar(1.), w.norm());
    d->ph_hess = e_hess_ * std::max(Scalar(1.), d->p.norm());
    d->ph_hess_pow2 = d->ph_hess * d->ph_hess;
    d->xph_hess_pow2 = Scalar(4.) * d->xh_hess * d->ph_hess;
    d->wph_hess_pow2 = Scalar(4.) * d->wh_hess * d->ph_hess;

    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_hess;
      model_->update_p(d->data_p[ip], d->p + d->dp);
      model_->calc(d->data_p[ip], x, w);
      const Scalar cp = d->data_p[ip]->cost;
      model_->update_p(d->data_p[ip], d->p - d->dp);
      model_->calc(d->data_p[ip], x, w);
      const Scalar cm = d->data_p[ip]->cost;
      data->Lpp(ip, ip) = (cp - Scalar(2.) * c0 + cm) / d->ph_hess_pow2;

      for (std::size_t jp = ip + 1; jp < np_; ++jp) {
        d->dp(jp) = d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->calc(d->data_p[ip], x, w);
        const Scalar cpp = d->data_p[ip]->cost;
        d->dp(ip) = Scalar(0.);
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->calc(d->data_p[ip], x, w);
        const Scalar czp = d->data_p[ip]->cost;
        data->Lpp(ip, jp) = (cpp - czp - cp + c0) / d->ph_hess_pow2;
        data->Lpp(jp, ip) = data->Lpp(ip, jp);
        d->dp(ip) = d->ph_hess;
        d->dp(jp) = Scalar(0.);
      }
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }

    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      for (std::size_t ix = 0; ix < ndx; ++ix) {
        d->dp(ip) = d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        d->dx(ix) = d->xh_hess;
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp, w);
        const Scalar cpp = d->data_p[ip]->cost;
        model_->get_state()->integrate(x, -d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp, w);
        const Scalar cpm = d->data_p[ip]->cost;
        d->dp(ip) = -d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp, w);
        const Scalar cmp = d->data_p[ip]->cost;
        model_->get_state()->integrate(x, -d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp, w);
        const Scalar cmm = d->data_p[ip]->cost;
        data->Lpx(ip, ix) = (cpp - cpm - cmp + cmm) / d->xph_hess_pow2;
        d->dx(ix) = Scalar(0.);
        d->dp(ip) = Scalar(0.);
      }
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }

    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      for (std::size_t iw = 0; iw < nu; ++iw) {
        d->dp(ip) = d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        d->dw(iw) = d->wh_hess;
        model_->calc(d->data_p[ip], x, w + d->dw);
        const Scalar cpp = d->data_p[ip]->cost;
        model_->calc(d->data_p[ip], x, w - d->dw);
        const Scalar cpm = d->data_p[ip]->cost;
        d->dp(ip) = -d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->calc(d->data_p[ip], x, w + d->dw);
        const Scalar cmp = d->data_p[ip]->cost;
        model_->calc(d->data_p[ip], x, w - d->dw);
        const Scalar cmm = d->data_p[ip]->cost;
        data->Lpu(ip, iw) = (cpp - cpm - cmp + cmm) / d->wph_hess_pow2;
        d->dw(iw) = Scalar(0.);
        d->dp(ip) = Scalar(0.);
      }
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }
#endif
  }

#ifdef NDEBUG
  // ---------------------------------------------------------------------------
  // d^2/dx^2
  // ---------------------------------------------------------------------------
  d->xh_hess = e_hess_ * std::max(Scalar(1.), d->x_norm);
  d->xh_hess_pow2 = d->xh_hess * d->xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, w);
    const Scalar cp = d->data_x[ix]->cost;
    model_->get_state()->integrate(x, -d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, w);
    const Scalar cm = d->data_x[ix]->cost;
    data->Lxx(ix, ix) = (cp - 2 * c0 + cm) / d->xh_hess_pow2;
    for (std::size_t jx = ix + 1; jx < ndx; ++jx) {
      d->dx(jx) = d->xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, w);
      const Scalar cpp = d->data_x[ix]->cost;
      d->dx(ix) = Scalar(0.);
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, w);
      const Scalar czp = d->data_x[ix]->cost;
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / d->xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = d->xh_hess;
      d->dx(jx) = Scalar(0.);
    }
    d->dx(ix) = Scalar(0.);
  }

  // ---------------------------------------------------------------------------
  // d^2/dw^2
  // ---------------------------------------------------------------------------
  d->wh_hess = e_hess_ * std::max(Scalar(1.), w.norm());
  d->wh_hess_pow2 = d->wh_hess * d->wh_hess;
  for (std::size_t iw = 0; iw < nu; ++iw) {
    d->dw(iw) = d->wh_hess;
    model_->calc(d->data_w[iw], x, w + d->dw);
    const Scalar cp = d->data_w[iw]->cost;
    model_->calc(d->data_w[iw], x, w - d->dw);
    const Scalar cm = d->data_w[iw]->cost;
    data->Luu(iw, iw) = (cp - 2 * c0 + cm) / d->wh_hess_pow2;
    for (std::size_t jw = iw + 1; jw < nu; ++jw) {
      d->dw(jw) = d->wh_hess;
      model_->calc(d->data_w[iw], x, w + d->dw);
      const Scalar cpp = d->data_w[iw]->cost;
      d->dw(iw) = Scalar(0.);
      model_->calc(d->data_w[iw], x, w + d->dw);
      const Scalar czp = d->data_w[iw]->cost;
      data->Luu(iw, jw) = (cpp - czp - cp + c0) / d->wh_hess_pow2;
      data->Luu(jw, iw) = data->Luu(iw, jw);
      d->dw(iw) = d->wh_hess;
      d->dw(jw) = Scalar(0.);
    }
    d->dw(iw) = Scalar(0.);
  }

  // ---------------------------------------------------------------------------
  // d^2/dxdw
  // ---------------------------------------------------------------------------
  d->xwh_hess_pow2 = Scalar(4.) * d->xh_hess * d->wh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    for (std::size_t jw = 0; jw < nu; ++jw) {
      d->dx(ix) = d->xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      d->dw(jw) = d->wh_hess;
      model_->calc(d->data_x[ix], d->xp, w + d->dw);
      const Scalar cpp = d->data_x[ix]->cost;
      model_->calc(d->data_x[ix], d->xp, w - d->dw);
      const Scalar cpm = d->data_x[ix]->cost;
      model_->get_state()->integrate(x, -d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, w + d->dw);
      const Scalar cmp = d->data_x[ix]->cost;
      model_->calc(d->data_x[ix], d->xp, w - d->dw);
      const Scalar cmm = d->data_x[ix]->cost;
      data->Lxu(ix, jw) = (cpp - cpm - cmp + cmm) / d->xwh_hess_pow2;
      d->dx(ix) = Scalar(0.);
      d->dw(jw) = Scalar(0.);
    }
  }
#endif
  restore.restore();
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  ObserverDataAbstractTpl<Scalar>* const data_obs = d.get();
  std::size_t perturbed_ip = np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < np_) {
      model_->update_p(d->data_p[perturbed_ip], d->p);
    }
    d->dx.setZero();
    d->dp.setZero();
  });

  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  const VectorXs& g0 = data->g;
  const VectorXs& h0 = data->h;

  const std::size_t ndx = model_->get_state()->get_ndx();
  const std::size_t ng_T = model_->get_ng_T();
  const std::size_t nh_T = model_->get_nh_T();

  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);

  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);

    data->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    if (ng_T != 0) {
      data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    }
    if (nh_T != 0) {
      data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    }
    d->dx(ix) = Scalar(0.);
  }

#ifdef NDEBUG
  d->xh_hess = e_hess_ * std::max(Scalar(1.), d->x_norm);
  d->xh_hess_pow2 = d->xh_hess * d->xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar cp = d->data_x[ix]->cost;
    model_->get_state()->integrate(x, -d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar cm = d->data_x[ix]->cost;
    data->Lxx(ix, ix) = (cp - 2 * c0 + cm) / d->xh_hess_pow2;
    for (std::size_t jx = ix + 1; jx < ndx; ++jx) {
      d->dx(jx) = d->xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp);
      const Scalar cpp = d->data_x[ix]->cost;
      d->dx(ix) = Scalar(0.);
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp);
      const Scalar czp = d->data_x[ix]->cost;
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / d->xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = d->xh_hess;
      d->dx(jx) = Scalar(0.);
    }
    d->dx(ix) = Scalar(0.);
  }
#endif

  if (np_ > 0) {
    const VectorXs& xnext0 = d->data_0->xnext;
    const auto* d0_obs =
        static_cast<const ObserverDataAbstractTpl<Scalar>*>(d->data_0.get());
    const VectorXs e0 = d0_obs->dissipative_E;
    d->dp.setZero();
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_jac;
      model_->update_p(d->data_p[ip], d->p + d->dp);
      model_->calc(d->data_p[ip], x);
      model_->get_state()->diff(xnext0, d->data_p[ip]->xnext, data->Fp.col(ip));
      data->Fp.col(ip) /= d->ph_jac;
      data->Lp(ip) = (d->data_p[ip]->cost - c0) / d->ph_jac;
      if (ng_T != 0) {
        data->Gp.col(ip) = (d->data_p[ip]->g - g0) / d->ph_jac;
      }
      if (nh_T != 0) {
        data->Hp.col(ip) = (d->data_p[ip]->h - h0) / d->ph_jac;
      }
      const auto* dpi_obs = static_cast<const ObserverDataAbstractTpl<Scalar>*>(
          d->data_p[ip].get());
      data_obs->dE_dp.col(ip) = (dpi_obs->dissipative_E - e0) / d->ph_jac;
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }

#ifdef NDEBUG
    d->ph_hess = e_hess_ * std::max(Scalar(1.), d->p.norm());
    d->ph_hess_pow2 = d->ph_hess * d->ph_hess;
    d->xph_hess_pow2 = Scalar(4.) * d->xh_hess * d->ph_hess;

    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_hess;
      model_->update_p(d->data_p[ip], d->p + d->dp);
      model_->calc(d->data_p[ip], x);
      const Scalar cp = d->data_p[ip]->cost;
      model_->update_p(d->data_p[ip], d->p - d->dp);
      model_->calc(d->data_p[ip], x);
      const Scalar cm = d->data_p[ip]->cost;
      data->Lpp(ip, ip) = (cp - Scalar(2.) * c0 + cm) / d->ph_hess_pow2;

      for (std::size_t jp = ip + 1; jp < np_; ++jp) {
        d->dp(jp) = d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->calc(d->data_p[ip], x);
        const Scalar cpp = d->data_p[ip]->cost;
        d->dp(ip) = Scalar(0.);
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->calc(d->data_p[ip], x);
        const Scalar czp = d->data_p[ip]->cost;
        data->Lpp(ip, jp) = (cpp - czp - cp + c0) / d->ph_hess_pow2;
        data->Lpp(jp, ip) = data->Lpp(ip, jp);
        d->dp(ip) = d->ph_hess;
        d->dp(jp) = Scalar(0.);
      }
      d->dp(ip) = Scalar(0.);
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }

    for (std::size_t ip = 0; ip < np_; ++ip) {
      perturbed_ip = ip;
      for (std::size_t ix = 0; ix < ndx; ++ix) {
        d->dp(ip) = d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        d->dx(ix) = d->xh_hess;
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cpp = d->data_p[ip]->cost;
        model_->get_state()->integrate(x, -d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cpm = d->data_p[ip]->cost;
        d->dp(ip) = -d->ph_hess;
        model_->update_p(d->data_p[ip], d->p + d->dp);
        model_->get_state()->integrate(x, d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cmp = d->data_p[ip]->cost;
        model_->get_state()->integrate(x, -d->dx, d->xp);
        model_->calc(d->data_p[ip], d->xp);
        const Scalar cmm = d->data_p[ip]->cost;
        data->Lpx(ip, ix) = (cpp - cpm - cmp + cmm) / d->xph_hess_pow2;
        d->dx(ix) = Scalar(0.);
        d->dp(ip) = Scalar(0.);
      }
      model_->update_p(d->data_p[ip], d->p);
      perturbed_ip = np_;
    }
#endif
  }
  restore.restore();
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ObserverModelNumDiffTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ObserverModelNumDiffTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  const std::shared_ptr<ActionDataAbstract> data = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, params_data);
  if (params_ != nullptr) {
    set_params(data, params_);
  }
  return data;
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::set_params(
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
  d->resize(this);
  model_->set_params(d->data_0, params_);
  for (std::size_t ix = 0; ix < d->data_x.size(); ++ix) {
    model_->set_params(d->data_x[ix], params_);
  }
  for (std::size_t iw = 0; iw < d->data_w.size(); ++iw) {
    model_->set_params(d->data_w[iw], params_);
  }
  for (std::size_t ip = 0; ip < d->data_p.size(); ++ip) {
    model_->set_params(d->data_p[ip], params_);
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
template <typename NewScalar>
ObserverModelNumDiffTpl<NewScalar> ObserverModelNumDiffTpl<Scalar>::cast()
    const {
  typedef ObserverModelNumDiffTpl<NewScalar> ReturnType;
  typedef ObserverModelAbstractTpl<NewScalar> ObserverBase;
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  std::shared_ptr<ParameterManagerNew> params;
  if (params_ != nullptr) {
    params = std::make_shared<ParameterManagerNew>(
        params_->template cast<NewScalar>());
    ReturnType ret(std::static_pointer_cast<ObserverBase>(
                       model_->template cast<NewScalar>()),
                   params, with_gauss_approx_);
    ret.update_tau(tau_meas_.template cast<NewScalar>());
    return ret;
  }
  ReturnType ret(std::static_pointer_cast<ObserverBase>(
                     model_->template cast<NewScalar>()),
                 with_gauss_approx_);
  ret.update_tau(tau_meas_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> w,
    const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
    const Scalar tol) {
  const std::shared_ptr<Data> d = cast_data(data);
  model_->quasiStatic(d->data_0, w, x, maxiter, tol);
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  Base::update_tau(tau_meas);
  model_->update_tau(tau_meas);
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(np_) + ")");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  d->p = p;
  model_->update_p(d->data_0, p);
  for (std::size_t ix = 0; ix < d->data_x.size(); ++ix) {
    model_->update_p(d->data_x[ix], p);
  }
  for (std::size_t iw = 0; iw < d->data_w.size(); ++iw) {
    model_->update_p(d->data_w[iw], p);
  }
  for (std::size_t ip = 0; ip < d->data_p.size(); ++ip) {
    model_->update_p(d->data_p[ip], p);
  }
}

template <typename Scalar>
std::size_t ObserverModelNumDiffTpl<Scalar>::get_ng() const {
  return model_->get_ng();
}

template <typename Scalar>
std::size_t ObserverModelNumDiffTpl<Scalar>::get_nh() const {
  return model_->get_nh();
}

template <typename Scalar>
std::size_t ObserverModelNumDiffTpl<Scalar>::get_ng_T() const {
  return model_->get_ng_T();
}

template <typename Scalar>
std::size_t ObserverModelNumDiffTpl<Scalar>::get_nh_T() const {
  return model_->get_nh_T();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ObserverModelNumDiffTpl<Scalar>::get_g_lb() const {
  return model_->get_g_lb();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ObserverModelNumDiffTpl<Scalar>::get_g_ub() const {
  return model_->get_g_ub();
}

template <typename Scalar>
std::shared_ptr<typename ObserverModelNumDiffTpl<Scalar>::Data>
ObserverModelNumDiffTpl<Scalar>::cast_data(
    const std::shared_ptr<ActionDataAbstract>& data) const {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data is not an observer numdiff data");
  }
  return d;
}

template <typename Scalar>
const std::shared_ptr<ObserverModelAbstractTpl<Scalar>>&
ObserverModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ObserverModelNumDiffTpl<Scalar>::ParameterManager>&
ObserverModelNumDiffTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const Scalar ObserverModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: disturbance constant must be non-negative");
  }
  e_jac_ = disturbance;
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
}

template <typename Scalar>
bool ObserverModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return with_gauss_approx_;
}

template <typename Scalar>
void ObserverModelNumDiffTpl<Scalar>::print(std::ostream& os) const {
  os << "ObserverModelNumDiffTpl {model=" << *model_ << "}";
}

}  // namespace crocoddyl
