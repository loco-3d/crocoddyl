///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/numdiff/contact.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModelNumDiffTpl<Scalar>::ContactModelNumDiffTpl(
    const std::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_type(), model->get_nc(),
           model->get_nu()),
      model_(model),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  model_->calc(d->data_0, x);
  d->a0 = d->data_0->a0;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

  const VectorXs& a0 = d->a0;

  assertStableStateFD(x);

  // Computing the d contact(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, VectorXs::Zero(model_->get_nu()));
    }
    model_->calc(d->data_x[ix], d->xp);
    d->da0_dx.col(ix) = (d->data_x[ix]->a0 - a0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::updateForce(
    const std::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) != model_->get_nc()) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be "
                 << model_->get_nc() << ")");
  }

  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

  model_->updateForce(d->data_0, force);
}

template <typename Scalar>
std::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModelNumDiffTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModelNumDiffTpl<NewScalar> ContactModelNumDiffTpl<Scalar>::cast() const {
  typedef ContactModelNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(model_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
const std::shared_ptr<ContactModelAbstractTpl<Scalar> >&
ContactModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ContactModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::set_reevals(
    const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
