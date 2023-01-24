///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          New York University, Max Planck Gesellschaft,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {

template <typename Scalar>
ActionModelNumDiffTpl<Scalar>::ActionModelNumDiffTpl(boost::shared_ptr<Base> model, bool with_gauss_approx)
    : Base(model->get_state(), model->get_nu(), model->get_nr(), model->get_ng(), model->get_nh()),
      model_(model),
      e_jac_(std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon())),
      with_gauss_approx_(with_gauss_approx) {
  e_hess_ = std::sqrt(2.0 * e_jac_);
  this->set_u_lb(model_->get_u_lb());
  this->set_u_ub(model_->get_u_ub());
}

template <typename Scalar>
ActionModelNumDiffTpl<Scalar>::~ActionModelNumDiffTpl() {}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x, u);
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
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
  Data* d = static_cast<Data*>(data.get());

  const VectorXs& x0 = d->data_0->xnext;
  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();
  const std::size_t nu = model_->get_nu();
  const std::size_t ng = model_->get_ng();
  const std::size_t nh = model_->get_nh();
  d->Gx.resize(ng, ndx);
  d->Gu.resize(ng, nu);
  d->Hx.resize(nh, ndx);
  d->Hu.resize(nh, nu);
  d->dx.setZero();
  d->du.setZero();

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  const Scalar xh_jac = e_jac_ * std::max(1., x.norm());
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    // dynamics
    model_->get_state()->diff(x0, d->data_x[ix]->xnext, d->Fx.col(ix));
    // cost
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / xh_jac;
    if (get_with_gauss_approx() > 0) {
      d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / xh_jac;
    }
    // constraint
    data->Gx.col(ix) = (d->data_x[ix]->g - g0) / xh_jac;
    data->Hx.col(ix) = (d->data_x[ix]->h - h0) / xh_jac;
    d->dx(ix) = 0.;
  }
  data->Fx /= xh_jac;

  // Computing the d action(x,u) / du
  const Scalar uh_jac = e_jac_ * std::max(1., u.norm());
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = uh_jac;
    model_->calc(d->data_u[iu], x, u + d->du);
    // dynamics
    model_->get_state()->diff(x0, d->data_u[iu]->xnext, d->Fu.col(iu));
    // cost
    data->Lu(iu) = (d->data_u[iu]->cost - c0) / uh_jac;
    if (get_with_gauss_approx() > 0) {
      d->Ru.col(iu) = (d->data_u[iu]->r - d->data_0->r) / uh_jac;
    }
    // constraint
    d->Gu.col(iu) = (d->data_u[iu]->g - g0) / uh_jac;
    d->Hu.col(iu) = (d->data_u[iu]->h - h0) / uh_jac;
    d->du(iu) = 0.0;
  }
  data->Fu /= uh_jac;

  // Computing the d^2 cost(x,u) / dx^2
  const Scalar xh_hess = e_hess_ * std::max(1., x.norm());
  const Scalar xh_hess_pow2 = xh_hess * xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    const Scalar cp = d->data_x[ix]->cost;
    model_->get_state()->integrate(x, -d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    const Scalar cm = d->data_x[ix]->cost;
    data->Lxx(ix, ix) = (cp - 2 * c0 + cm) / xh_hess_pow2;
    for (std::size_t jx = ix + 1; jx < ndx; ++jx) {
      d->dx(jx) = xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, u);
      const Scalar cpp = d->data_x[ix]->cost;  // cost due to positive disturbance in both directions
      d->dx(ix) = 0.;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, u);
      const Scalar czp =
          d->data_x[ix]->cost;  // cost due to zero disturance in 'i' and positive disturbance in 'j' direction
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = xh_hess;
      d->dx(jx) = 0.;
    }
    d->dx(ix) = 0.;
  }

  // Computing the d^2 cost(x,u) / du^2
  const Scalar uh_hess = e_hess_ * std::max(1., u.norm());
  const Scalar uh_hess_pow2 = uh_hess * uh_hess;
  for (std::size_t iu = 0; iu < nu; ++iu) {
    d->du(iu) = uh_hess;
    model_->calc(d->data_u[iu], x, u + d->du);
    const Scalar cp = d->data_u[iu]->cost;
    model_->calc(d->data_u[iu], x, u - d->du);
    const Scalar cm = d->data_u[iu]->cost;
    data->Luu(iu, iu) = (cp - 2 * c0 + cm) / uh_hess_pow2;
    for (std::size_t ju = iu + 1; ju < nu; ++ju) {
      d->du(ju) = uh_hess;
      model_->calc(d->data_u[iu], x, u + d->du);
      const Scalar cpp = d->data_u[iu]->cost;  // cost due to positive disturbance in both directions
      d->du(iu) = 0.;
      model_->calc(d->data_u[iu], x, u + d->du);
      const Scalar czp =
          d->data_u[iu]->cost;  // cost due to zero disturance in 'i' and positive disturbance in 'j' direction
      data->Luu(iu, ju) = (cpp - czp - cp + c0) / uh_hess_pow2;
      data->Luu(ju, iu) = data->Luu(iu, ju);
      d->du(iu) = uh_hess;
      d->du(ju) = 0.;
    }
    d->du(iu) = 0.;
  }

  // Computing the d^2 cost(x,u) / dxu
  const Scalar xuh_hess_pow2 = 4. * xh_hess * uh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    for (std::size_t ju = 0; ju < nu; ++ju) {
      d->dx(ix) = xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      d->du(ju) = uh_hess;
      model_->calc(d->data_x[ix], d->xp, u + d->du);
      const Scalar cpp = d->data_x[ix]->cost;
      model_->calc(d->data_x[ix], d->xp, u - d->du);
      const Scalar cpm = d->data_x[ix]->cost;
      model_->get_state()->integrate(x, -d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp, u + d->du);
      const Scalar cmp = d->data_x[ix]->cost;
      model_->calc(d->data_x[ix], d->xp, u - d->du);
      const Scalar cmm = d->data_x[ix]->cost;
      data->Lxu(ix, ju) = (cpp - cpm - cmp + cmm) / xuh_hess_pow2;
      d->dx(ix) = 0.;
      d->du(ju) = 0.;
    }
  }

  if (get_with_gauss_approx() > 0) {
    data->Lxx = d->Rx.transpose() * d->Rx;
    data->Lxu = d->Rx.transpose() * d->Ru;
    data->Luu = d->Ru.transpose() * d->Ru;
  }
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();
  d->Gx.resize(model_->get_ng(), ndx);
  d->Hx.resize(model_->get_nh(), ndx);
  d->dx.setZero();

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  const Scalar xh_jac = e_jac_ * std::max(1., x.norm());
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    // cost
    data->Lx(ix) = (d->data_x[ix]->cost - c0) / xh_jac;
    if (get_with_gauss_approx() > 0) {
      d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / xh_jac;
    }
    // constraint
    d->Gx.col(ix) = (d->data_x[ix]->g - g0) / xh_jac;
    d->Hx.col(ix) = (d->data_x[ix]->h - h0) / xh_jac;
    d->dx(ix) = 0.;
  }

  // Computing the d^2 cost(x,u) / dx^2
  const Scalar xh_hess = e_hess_ * std::max(1., x.norm());
  const Scalar xh_hess_pow2 = xh_hess * xh_hess;
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    // We can apply the same formulas for finite difference as above
    d->dx(ix) = xh_hess;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar cp = d->data_x[ix]->cost;
    model_->get_state()->integrate(x, -d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar cm = d->data_x[ix]->cost;
    data->Lxx(ix, ix) = (cp - 2 * c0 + cm) / xh_hess_pow2;
    for (std::size_t jx = ix + 1; jx < ndx; ++jx) {
      d->dx(jx) = xh_hess;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp);
      const Scalar cpp = d->data_x[ix]->cost;  // cost due to positive disturbance in both directions
      d->dx(ix) = 0.;
      model_->get_state()->integrate(x, d->dx, d->xp);
      model_->calc(d->data_x[ix], d->xp);
      const Scalar czp =
          d->data_x[ix]->cost;  // cost due to zero disturance in 'i' and positive disturbance in 'j' direction
      data->Lxx(ix, jx) = (cpp - czp - cp + c0) / xh_hess_pow2;
      data->Lxx(jx, ix) = data->Lxx(ix, jx);
      d->dx(ix) = xh_hess;
      d->dx(jx) = 0.;
    }
    d->dx(ix) = 0.;
  }

  if (get_with_gauss_approx() > 0) {
    data->Lxx = d->Rx.transpose() * d->Rx;
  }
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > ActionModelNumDiffTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const boost::shared_ptr<ActionModelAbstractTpl<Scalar> >& ActionModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ActionModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
  e_hess_ = std::sqrt(2.0 * e_jac_);
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
void ActionModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /** x */) {
  // do nothing in the general case
}

}  // namespace crocoddyl
