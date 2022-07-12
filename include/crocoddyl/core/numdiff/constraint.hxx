///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/constraint.hpp"

namespace crocoddyl {

template <typename Scalar>
ConstraintModelNumDiffTpl<Scalar>::ConstraintModelNumDiffTpl(const boost::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_nu(), model->get_ng(), model->get_nh(), !model->is_state_only()),
      model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ConstraintModelNumDiffTpl<Scalar>::~ConstraintModelNumDiffTpl() {}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
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
  d->data_0->g.setZero();
  d->data_0->h.setZero();
  model_->calc(d->data_0, x, u);
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  d->data_0->g.setZero();
  d->data_0->h.setZero();
  model_->calc(d->data_0, x);
  d->g = d->data_0->g;
  d->h = d->data_0->h;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
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

  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();
  d->Gx.resize(model_->get_ng(), ndx);
  d->Hx.resize(model_->get_nh(), ndx);

  assertStableStateFD(x);

  // Computing the d constraint(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, u);
    }
    // constraints(x+dx, u)
    model_->calc(d->data_x[ix], d->xp, u);
    // Gx, Hx
    d->Gx.col(ix) = (d->data_x[ix]->g - g0) / disturbance_;
    d->Hx.col(ix) = (d->data_x[ix]->h - h0) / disturbance_;
    d->dx(ix) = 0.0;
  }

  // Computing the d constraint(x,u) / du
  d->du.setZero();
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    // up = u + du
    d->du(iu) = disturbance_;
    d->up = u + d->du;
    // call the update function
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, d->up);
    }
    // constraint(x, u+du)
    model_->calc(d->data_u[iu], x, d->up);
    // Gu, Hu
    d->Gu.col(iu) = (d->data_u[iu]->g - g0) / disturbance_;
    d->Hu.col(iu) = (d->data_u[iu]->h - h0) / disturbance_;
    d->du(iu) = 0.0;
  }
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const VectorXs& g0 = d->g;
  const VectorXs& h0 = d->h;
  const std::size_t ndx = model_->get_state()->get_ndx();
  d->Gx.resize(model_->get_ng(), ndx);
  d->Hx.resize(model_->get_nh(), ndx);

  assertStableStateFD(x);

  // Computing the d constraint(x) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    // call the update function
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](d->xp, unone_);
    }
    // constraints(x+dx)
    model_->calc(d->data_x[ix], d->xp);
    // Gx, Hx
    d->Gx.col(ix) = (d->data_x[ix]->g - g0) / disturbance_;
    d->Hx.col(ix) = (d->data_x[ix]->h - h0) / disturbance_;
    d->dx(ix) = 0.0;
  }
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> > ConstraintModelNumDiffTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelAbstractTpl<Scalar> >& ConstraintModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ConstraintModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  disturbance_ = disturbance;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::set_reevals(const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
