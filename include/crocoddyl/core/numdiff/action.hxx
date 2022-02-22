///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh,
//                          New York University, Max Planck Gesellschaft,
//                          University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {

template <typename Scalar>
ActionModelNumDiffTpl<Scalar>::ActionModelNumDiffTpl(boost::shared_ptr<Base> model, bool with_gauss_approx)
    : Base(model->get_state(), model->get_nu(), model->get_nr()),
      model_(model),
      disturbance_(std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon())),
      with_gauss_approx_(with_gauss_approx) {
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
  data->cost = d->data_0->cost;
  data->xnext = d->data_0->xnext;
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
  data->cost = d->data_0->cost;
  data->xnext = d->data_0->xnext;
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

  const VectorXs& xn0 = d->data_0->xnext;
  const Scalar c0 = d->data_0->cost;
  data->xnext = d->data_0->xnext;
  data->cost = d->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);

    const VectorXs& xn = d->data_x[ix]->xnext;
    const Scalar c = d->data_x[ix]->cost;
    model_->get_state()->diff(xn0, xn, d->Fx.col(ix));

    data->Lx(ix) = (c - c0) / disturbance_;
    if (get_with_gauss_approx() > 0) {
      d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / disturbance_;
    }
    d->dx(ix) = 0.0;
  }
  data->Fx /= disturbance_;

  // Computing the d action(x,u) / du
  d->du.setZero();
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = disturbance_;
    model_->calc(d->data_u[iu], x, u + d->du);

    const VectorXs& xn = d->data_u[iu]->xnext;
    const Scalar c = d->data_u[iu]->cost;
    model_->get_state()->diff(xn0, xn, d->Fu.col(iu));

    data->Lu(iu) = (c - c0) / disturbance_;
    if (get_with_gauss_approx() > 0) {
      d->Ru.col(iu) = (d->data_u[iu]->r - d->data_0->r) / disturbance_;
    }
    d->du(iu) = 0.0;
  }
  data->Fu /= disturbance_;

  if (get_with_gauss_approx() > 0) {
    data->Lxx = d->Rx.transpose() * d->Rx;
    data->Lxu = d->Rx.transpose() * d->Ru;
    data->Luu = d->Ru.transpose() * d->Ru;
  } else {
    data->Lxx.setZero();
    data->Lxu.setZero();
    data->Luu.setZero();
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

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar c = d->data_x[ix]->cost;

    data->Lx(ix) = (c - c0) / disturbance_;
    if (get_with_gauss_approx() > 0) {
      d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / disturbance_;
    }
    d->dx(ix) = 0.0;
  }

  if (get_with_gauss_approx() > 0) {
    data->Lxx = d->Rx.transpose() * d->Rx;
  } else {
    data->Lxx.setZero();
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
  return disturbance_;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

template <typename Scalar>
bool ActionModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return with_gauss_approx_;
}

template <typename Scalar>
void ActionModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /** x */) {
  // do nothing in the general case
}

}  // namespace crocoddyl
