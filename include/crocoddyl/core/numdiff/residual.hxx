///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/residual.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelNumDiffTpl<Scalar>::ResidualModelNumDiffTpl(
    const std::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_nr(), model->get_nu()),
      model_(model),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x, u);
  d->r = d->data_0->r;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  d->r = d->data_0->r;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());

  const VectorXs& r0 = d->r;
  d->dx.setZero();
  d->du.setZero();

  assertStableStateFD(x);

  // Computing the d residual(x,u) / dx
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
    d->Rx.col(ix) = (d->data_x[ix]->r - r0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }

  // Computing the d residual(x,u) / du
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = d->uh_jac;
    d->up = u + d->du;
    // call the update function
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, d->up);
    }
    model_->calc(d->data_u[iu], x, d->up);
    d->Ru.col(iu) = (d->data_u[iu]->r - r0) / d->uh_jac;
    d->du(iu) = Scalar(0.);
  }
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());

  const VectorXs& r0 = d->r;
  assertStableStateFD(x);

  // Computing the d residual(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, unone_);
    }
    model_->calc(d->data_x[ix], d->xp);
    d->Rx.col(ix) = (d->data_x[ix]->r - r0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelNumDiffTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelNumDiffTpl<NewScalar> ResidualModelNumDiffTpl<Scalar>::cast()
    const {
  typedef ResidualModelNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(model_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
const std::shared_ptr<ResidualModelAbstractTpl<Scalar> >&
ResidualModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ResidualModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::set_reevals(
    const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
