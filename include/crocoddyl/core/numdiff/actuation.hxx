///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/actuation.hpp"

namespace crocoddyl {

template <typename Scalar>
ActuationModelNumDiffTpl<Scalar>::ActuationModelNumDiffTpl(boost::shared_ptr<Base> model)
    : Base(model->get_state(), model->get_nu()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ActuationModelNumDiffTpl<Scalar>::~ActuationModelNumDiffTpl() {}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x, u);
  data->tau = d->data_0->tau;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(model_->get_state()->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  data->tau = d->data_0->tau;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>& x,
                                                const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const VectorXs& tau0 = d->data_0->tau;

  // Computing the d Actuation(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < model_->get_state()->get_ndx(); ++ix) {
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    d->dtau_dx.col(ix) = (d->data_x[ix]->tau - tau0) / disturbance_;
    d->dx(ix) = 0.0;
  }

  // Computing the d Actuation(x,u) / du
  d->du.setZero();
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = disturbance_;
    model_->calc(d->data_u[iu], x, u + d->du);
    d->dtau_du.col(iu) = (d->data_u[iu]->tau - tau0) / disturbance_;
    d->du(iu) = 0.0;
  }
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(model_->get_state()->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const VectorXs& tau0 = d->data_0->tau;

  // Computing the d Actuation(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < model_->get_state()->get_ndx(); ++ix) {
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    d->dtau_dx.col(ix) = (d->data_x[ix]->tau - tau0) / disturbance_;
    d->dx(ix) = 0.0;
  }
}

template <typename Scalar>
boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > ActuationModelNumDiffTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >& ActuationModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ActuationModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
