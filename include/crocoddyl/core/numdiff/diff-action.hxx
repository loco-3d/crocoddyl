///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          New York University, Heriot-Watt University
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelNumDiffTpl<Scalar>::DifferentialActionModelNumDiffTpl(
    std::shared_ptr<Base> model, const bool with_gauss_approx)
    : Base(model->get_state(), model->get_nu(), model->get_nr(),
           model->get_ng(), model->get_nh(), model->get_ng_T(),
           model->get_nh_T()),
      model_(model),
      with_gauss_approx_(with_gauss_approx),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
  if (with_gauss_approx_ && nr_ == 1)
    throw_pretty("No Gauss approximation possible with nr = 1");
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
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
  model_->calc(d->data_0, x, u);
  data->xout = d->data_0->xout;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  data->xout = d->data_0->xout;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // For details about the finite difference formulas see
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
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

  const VectorXs& x0 = d->data_0->xout;
  const Scalar c0 = d->data_0->cost;
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = state_->get_ndx();
  const std::size_t nu = model_->get_nu();
  const std::size_t ng = model_->get_ng();
  const std::size_t nh = model_->get_nh();
  d->Gx.conservativeResize(ng, ndx);
  d->Gu.conservativeResize(ng, nu);
  d->Hx.conservativeResize(nh, ndx);
  d->Hu.conservativeResize(nh, nu);
  d->du.setZero();

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    // dynamics
    data->Fx.col(ix) = (d->data_x[ix]->xout - x0) / d->xh_jac;
    // constraint
    data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    // cost
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }

  // Computing the d action(x,u) / du
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (std::size_t iu = 0; iu < nu; ++iu) {
    d->du(iu) = d->uh_jac;
    model_->calc(d->data_u[iu], x, u + d->du);
    // dynamics
    data->Fu.col(iu) = (d->data_u[iu]->xout - x0) / d->uh_jac;
    // constraint
    data->Gu.col(iu) = (d->data_u[iu]->g - g0) / d->uh_jac;
    data->Hu.col(iu) = (d->data_u[iu]->h - h0) / d->uh_jac;
    // cost
    data->Lu(iu) = (d->data_u[iu]->cost - c0) / d->uh_jac;
    d->Ru.col(iu) = (d->data_u[iu]->r - d->data_0->r) / d->uh_jac;
    d->du(iu) = Scalar(0.);
  }

#ifdef NDEBUG
  // Computing the d^2 cost(x,u) / dx^2
  d->xh_hess = e_hess_ * std::max(Scalar(1.), d->x_norm);
  d->xh_hess_pow2 = d->xh_hess * d->xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    const Scalar cp = d->data_x[ix]->cost;
    model_->get_state()->integrate(x, -d->dx, d->xp);
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
          d->data_x[ix]->cost;  // cost due to zero disturance in 'i' and
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
    model_->calc(d->data_u[iu], x, u + d->du);
    const Scalar cp = d->data_u[iu]->cost;
    model_->calc(d->data_u[iu], x, u - d->du);
    const Scalar cm = d->data_u[iu]->cost;
    data->Luu(iu, iu) = (cp - 2 * c0 + cm) / d->uh_hess_pow2;
    for (std::size_t ju = iu + 1; ju < nu; ++ju) {
      d->du(ju) = d->uh_hess;
      model_->calc(d->data_u[iu], x, u + d->du);
      const Scalar cpp =
          d->data_u[iu]
              ->cost;  // cost due to positive disturbance in both directions
      d->du(iu) = Scalar(0.);
      model_->calc(d->data_u[iu], x, u + d->du);
      const Scalar czp =
          d->data_u[iu]->cost;  // cost due to zero disturance in 'i' and
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
      model_->calc(d->data_x[ix], d->xp, u + d->du);
      const Scalar cpp = d->data_x[ix]->cost;
      model_->calc(d->data_x[ix], d->xp, u - d->du);
      const Scalar cpm = d->data_x[ix]->cost;
      model_->get_state()->integrate(x, -d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, u + d->du);
      const Scalar cmp = d->data_x[ix]->cost;
      model_->calc(d->data_x[ix], d->xp, u - d->du);
      const Scalar cmm = d->data_x[ix]->cost;
      data->Lxu(ix, ju) = (cpp - cpm - cmp + cmm) / d->xuh_hess_pow2;
      d->dx(ix) = Scalar(0.);
      d->du(ju) = Scalar(0.);
    }
  }
#endif

  if (with_gauss_approx_) {
    data->Lxx = d->Rx.transpose() * d->Rx;
    data->Lxu = d->Rx.transpose() * d->Ru;
    data->Luu = d->Ru.transpose() * d->Ru;
  }
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  // For details about the finite difference formulas see
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const Scalar c0 = d->data_0->cost;
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = state_->get_ndx();
  d->Gx.conservativeResize(model_->get_ng_T(), ndx);
  d->Hx.conservativeResize(model_->get_nh_T(), ndx);

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    // cost
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / d->xh_jac;
    // constraint
    data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
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
    model_->get_state()->integrate(x, -d->dx, d->xp);
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
          d->data_x[ix]->cost;  // cost due to zero disturance in 'i' and
                                // positive disturbance in 'j' direction
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / d->xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = d->xh_hess;
      d->dx(jx) = Scalar(0.);
    }
    d->dx(ix) = Scalar(0.);
  }
#endif

  if (with_gauss_approx_) {
    data->Lxx = d->Rx.transpose() * d->Rx;
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelNumDiffTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
    const std::size_t maxiter, const Scalar tol) {
  Data* d = static_cast<Data*>(data.get());
  model_->quasiStatic(d->data_0, u, x, maxiter, tol);
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelNumDiffTpl<NewScalar>
DifferentialActionModelNumDiffTpl<Scalar>::cast() const {
  typedef DifferentialActionModelNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(model_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
const std::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
DifferentialActionModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar DifferentialActionModelNumDiffTpl<Scalar>::get_disturbance()
    const {
  return e_jac_;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
  e_hess_ = sqrt(Scalar(2.0) * e_jac_);
}

template <typename Scalar>
bool DifferentialActionModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return with_gauss_approx_;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelNumDiffTpl {action=" << *model_ << "}";
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /** x */) {
  // TODO(cmastalli): First we need to do it AMNumDiff and then to replicate it.
}

}  // namespace crocoddyl
