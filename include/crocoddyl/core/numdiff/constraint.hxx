///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/constraint.hpp"

namespace crocoddyl {

template <typename Scalar>
ConstraintModelNumDiffTpl<Scalar>::ConstraintModelNumDiffTpl(
    const std::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_nu(), model->get_ng(),
           model->get_nh()),
      model_(model),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ConstraintDataAbstract>& data,
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
  d->data_0->g.setZero();
  d->data_0->h.setZero();
  model_->calc(d->data_0, x, u);
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  d->data_0->g.setZero();
  d->data_0->h.setZero();
  model_->calc(d->data_0, x);
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ConstraintDataAbstract>& data,
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
  d->du.setZero();

  assertStableStateFD(x);

  // Computing the d constraint(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, u);
    }
    model_->calc(d->data_x[ix], d->xp, u);
    d->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    d->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }

  // Computing the d constraint(x,u) / du
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = d->uh_jac;
    d->up = u + d->du;
    // call the update function
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, d->up);
    }
    model_->calc(d->data_u[iu], x, d->up);
    d->Gu.col(iu) = (d->data_u[iu]->g - g0) / d->uh_jac;
    d->Hu.col(iu) = (d->data_u[iu]->h - h0) / d->uh_jac;
    d->du(iu) = Scalar(0.);
  }
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();
  d->Gx.resize(model_->get_ng(), ndx);
  d->Hx.resize(model_->get_nh(), ndx);

  assertStableStateFD(x);

  // Computing the d constraint(x) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, unone_);
    }
    model_->calc(d->data_x[ix], d->xp);
    d->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    d->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }
}

template <typename Scalar>
std::shared_ptr<ConstraintDataAbstractTpl<Scalar> >
ConstraintModelNumDiffTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ConstraintModelNumDiffTpl<NewScalar> ConstraintModelNumDiffTpl<Scalar>::cast()
    const {
  typedef ConstraintModelNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(model_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelAbstractTpl<Scalar> >&
ConstraintModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ConstraintModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::set_reevals(
    const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
