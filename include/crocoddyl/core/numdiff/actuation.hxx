///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/actuation.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"

namespace crocoddyl {
namespace internal {

template <typename Scalar>
void extractActuationFriction(
    const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& data,
    Eigen::Ref<typename MathBaseTpl<Scalar>::VectorXs> friction) {
  typedef ActuationDataMultibodyTpl<Scalar> ActuationDataMultibody;
  const std::shared_ptr<ActuationDataMultibody> multibody_data =
      std::dynamic_pointer_cast<ActuationDataMultibody>(data);
  if (multibody_data == nullptr) {
    friction.setZero();
  } else {
    friction = multibody_data->friction;
  }
}

}  // namespace internal

template <typename Scalar>
ActuationModelNumDiffTpl<Scalar>::ActuationModelNumDiffTpl(
    std::shared_ptr<Base> model)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_nu()),
      model_(internal::checkNumDiffModel(model)),
      params_(std::make_shared<ParameterManager>(model_->get_state())),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
ActuationModelNumDiffTpl<Scalar>::ActuationModelNumDiffTpl(
    std::shared_ptr<Base> model, std::shared_ptr<ParameterManager> params)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_nu()),
      model_(internal::checkNumDiffModel(model)),
      params_(params == nullptr
                  ? std::make_shared<ParameterManager>(model_->get_state())
                  : params),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {
  if (params_->get_np_action() != 0) {
    throw_pretty(
        "Invalid argument: ActuationModelNumDiff only supports "
        "dynamics parameters");
  }
  if (params_->get_state()->get_nx() != model_->get_state()->get_nx() ||
      params_->get_state()->get_ndx() != model_->get_state()->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x, u);
  data->tau = d->data_0->tau;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  data->tau = d->data_0->tau;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t np = params_->get_np_dynamics();
  d->p = d->parameter_data->params->p;
  std::size_t perturbed_ip = np;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < np) {
      params_->update(d->parameter_data, d->p);
    }
    d->dx.setZero();
    d->du.setZero();
    d->dp.setZero();
  });
  const VectorXs& tau0 = d->data_0->tau;
  internal::extractActuationFriction<Scalar>(d->data_0, d->friction_0);
  d->du.setZero();
  d->dp.setZero();
  d->dfriction_dx.setZero();
  d->dfriction_dp.setZero();
  d->dtau_dp.setZero();

  // Computing the d actuation(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < model_->get_state()->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    d->dtau_dx.col(ix) = (d->data_x[ix]->tau - tau0) / d->xh_jac;
    internal::extractActuationFriction<Scalar>(d->data_x[ix], d->friction_p);
    d->dfriction_dx.col(ix) = (d->friction_p - d->friction_0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }

  // Computing the d actuation(x,u) / du
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = d->uh_jac;
    d->up = u + d->du;
    model_->calc(d->data_u[iu], x, d->up);
    d->dtau_du.col(iu) = (d->data_u[iu]->tau - tau0) / d->uh_jac;
    d->du(iu) = Scalar(0.);
  }

  // Computing the d actuation(x,u) / dp
  if (np != 0) {
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < np; ++ip) {
      perturbed_ip = ip;
      d->dp(ip) = d->ph_jac;
      d->pp = d->p + d->dp;
      params_->update(d->parameter_data, d->pp);
      model_->calc(d->data_p[ip], x, u);
      d->dtau_dp.col(ip) = (d->data_p[ip]->tau - tau0) / d->ph_jac;
      internal::extractActuationFriction<Scalar>(d->data_p[ip], d->friction_p);
      d->dfriction_dp.col(ip) = (d->friction_p - d->friction_0) / d->ph_jac;
      d->dp(ip) = Scalar(0.);
      params_->update(d->parameter_data, d->p);
      perturbed_ip = np;
    }
  }
  restore.restore();
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  internal::NumDiffRestorationTpl restore([&]() { d->dx.setZero(); });
  const VectorXs& tau0 = d->data_0->tau;
  d->dx.setZero();
  d->dtau_dp.setZero();
  d->dfriction_dx.setZero();
  d->dfriction_dp.setZero();
  internal::extractActuationFriction<Scalar>(d->data_0, d->friction_0);

  // Computing the d actuation(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < model_->get_state()->get_ndx(); ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    d->dtau_dx.col(ix) = (d->data_x[ix]->tau - tau0) / d->xh_jac;
    internal::extractActuationFriction<Scalar>(d->data_x[ix], d->friction_p);
    d->dfriction_dx.col(ix) = (d->friction_p - d->friction_0) / d->xh_jac;
    d->dx(ix) = Scalar(0.);
  }
  restore.restore();
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::commands(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& tau) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(tau.size()) != model_->get_state()->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "tau has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nv()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->commands(d->data_0, x, tau);
  data->u = d->data_0->u;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::torqueTransform(
    const std::shared_ptr<ActuationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->torqueTransform(d->data_0, x, u);
  d->Mtau = d->data_0->Mtau;
}

template <typename Scalar>
std::shared_ptr<ActuationDataAbstractTpl<Scalar> >
ActuationModelNumDiffTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActuationDataAbstractTpl<Scalar> >
ActuationModelNumDiffTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& parameter_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    parameter_data);
}

template <typename Scalar>
template <typename NewScalar>
ActuationModelNumDiffTpl<NewScalar> ActuationModelNumDiffTpl<Scalar>::cast()
    const {
  typedef ActuationModelNumDiffTpl<NewScalar> ReturnType;
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  std::shared_ptr<ParameterManagerNew> params;
  if (params_ != nullptr) {
    params = std::make_shared<ParameterManagerNew>(
        params_->template cast<NewScalar>());
  }
  ReturnType res(model_->template cast<NewScalar>(), params);
  return res;
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
ActuationModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ActuationModelNumDiffTpl<Scalar>::ParameterManager>&
ActuationModelNumDiffTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
std::size_t ActuationModelNumDiffTpl<Scalar>::get_np() const {
  return params_->get_np_dynamics();
}

template <typename Scalar>
const Scalar ActuationModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

}  // namespace crocoddyl
