///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/cost.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelNumDiffTpl<Scalar>::CostModelNumDiffTpl(
    const std::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_activation(), model->get_nu()),
      model_(model),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());
  d->data_0->cost = Scalar(0.);
  model_->calc(d->data_0, x, u);
  d->cost = d->data_0->cost;
  d->residual->r = d->data_0->residual->r;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());
  d->data_0->cost = Scalar(0.);
  model_->calc(d->data_0, x);
  d->cost = d->data_0->cost;
  d->residual->r = d->data_0->residual->r;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());

  const Scalar c0 = d->cost;
  const VectorXs& r0 = d->residual->r;
  if (get_with_gauss_approx()) {
    model_->get_activation()->calc(d->data_0->activation, r0);
    model_->get_activation()->calcDiff(d->data_0->activation, r0);
  }
  d->du.setZero();

  assertStableStateFD(x);

  // Computing the d cost(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, u);
    }
    model_->calc(d->data_x[ix], d->xp, u);
    d->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    if (get_with_gauss_approx()) {
      d->residual->Rx.col(ix) = (d->data_x[ix]->residual->r - r0) / d->xh_jac;
    }
    d->dx(ix) = Scalar(0.);
  }

  // Computing the d cost(x,u) / du
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = d->uh_jac;
    d->up = u + d->du;
    // call the update function
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, d->up);
    }
    model_->calc(d->data_u[iu], x, d->up);
    d->Lu(iu) = (d->data_u[iu]->cost - c0) / d->uh_jac;
    if (get_with_gauss_approx()) {
      d->residual->Ru.col(iu) = (d->data_u[iu]->residual->r - r0) / d->uh_jac;
    }
    d->du(iu) = Scalar(0.);
  }

  if (get_with_gauss_approx()) {
    const MatrixXs& Arr = d->data_0->activation->Arr;
    d->Lxx = d->residual->Rx.transpose() * Arr * d->residual->Rx;
    d->Lxu = d->residual->Rx.transpose() * Arr * d->residual->Ru;
    d->Luu = d->residual->Ru.transpose() * Arr * d->residual->Ru;
  } else {
    d->Lxx.fill(Scalar(0.));
    d->Lxu.fill(Scalar(0.));
    d->Luu.fill(Scalar(0.));
  }
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());

  const Scalar c0 = d->cost;
  const VectorXs& r0 = d->residual->r;
  if (get_with_gauss_approx()) {
    model_->get_activation()->calc(d->data_0->activation, r0);
    model_->get_activation()->calcDiff(d->data_0->activation, r0);
  }
  d->dx.setZero();

  assertStableStateFD(x);

  // Computing the d cost(x,u) / dx
  d->xh_jac = e_jac_ * std::max(Scalar(1.), x.norm());
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, unone_);
    }
    model_->calc(d->data_x[ix], d->xp);
    d->Lx(ix) = (d->data_x[ix]->cost - c0) / d->xh_jac;
    if (get_with_gauss_approx()) {
      d->residual->Rx.col(ix) = (d->data_x[ix]->residual->r - r0) / d->xh_jac;
    }
    d->dx(ix) = Scalar(0.);
  }

  if (get_with_gauss_approx()) {
    const MatrixXs& Arr = d->data_0->activation->Arr;
    d->Lxx = d->residual->Rx.transpose() * Arr * d->residual->Rx;
  } else {
    d->Lxx.fill(Scalar(0.));
  }
}

template <typename Scalar>
std::shared_ptr<CostDataAbstractTpl<Scalar> >
CostModelNumDiffTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
CostModelNumDiffTpl<NewScalar> CostModelNumDiffTpl<Scalar>::cast() const {
  typedef CostModelNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(model_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
const std::shared_ptr<CostModelAbstractTpl<Scalar> >&
CostModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar CostModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

template <typename Scalar>
bool CostModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return activation_->get_nr() > 0;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::set_reevals(
    const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
