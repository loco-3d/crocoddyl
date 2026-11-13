///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/residual.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelNumDiffTpl<Scalar>::ResidualModelNumDiffTpl(
    const std::shared_ptr<Base>& model)
    : ResidualModelNumDiffTpl(
          internal::checkNumDiffModel(model),
          std::make_shared<ParameterManager>(
              internal::checkNumDiffModel(model)->get_state())) {}

template <typename Scalar>
ResidualModelNumDiffTpl<Scalar>::ResidualModelNumDiffTpl(
    const std::shared_ptr<Base>& model,
    std::shared_ptr<ParameterManager> params)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_nr(),
           internal::checkNumDiffModel(model)->get_nu(),
           internal::checkNumDiffModel(model)->get_q_dependent(),
           internal::checkNumDiffModel(model)->get_v_dependent(),
           internal::checkNumDiffModel(model)->get_u_dependent()),
      model_(internal::checkNumDiffModel(model)),
      params_(params),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {
  if (model_ == nullptr) {
    throw_pretty("Invalid argument: residual model is null");
  }
  if (params_ == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params_->get_state()->get_nx() != state_->get_nx() ||
      params_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
  this->np_ = model_->get_np();
  if (params_->get_np() != 0 && params_->get_np() != this->np_) {
    throw_pretty("Invalid argument: params dimension (" +
                 std::to_string(params_->get_np()) +
                 ") does not match residual np (" + std::to_string(this->np_) +
                 ")");
  }
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x, u);
  d->r = d->data_0->r;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  d->r = d->data_0->r;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  if (this->np_ != 0) {
    assertParameterData(d, params_);
    d->p = d->parameter_data->params->p;
  }
  std::size_t perturbed_ip = this->np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < this->np_) {
      params_->update(d->parameter_data, d->p);
    }
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, u);
    }
    d->dx.setZero();
    d->du.setZero();
    d->dp.setZero();
  });

  const VectorXs& r0 = d->r;
  d->dx.setZero();
  d->du.setZero();
  d->Rp.setZero();

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

  if (this->np_ != 0) {
    d->dp.setZero();
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      d->pp = d->p;
      d->pp(ip) += d->ph_jac;
      params_->update(d->parameter_data, d->pp);
      for (std::size_t i = 0; i < reevals_.size(); ++i) {
        reevals_[i](x, u);
      }
      model_->calc(d->data_p[ip], x, u);
      d->Rp.col(ip) = (d->data_p[ip]->r - r0) / d->ph_jac;
      params_->update(d->parameter_data, d->p);
      perturbed_ip = this->np_;
    }
  }
  restore.restore();
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  if (this->np_ != 0) {
    assertParameterData(d, params_);
    d->p = d->parameter_data->params->p;
  }
  std::size_t perturbed_ip = this->np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < this->np_) {
      params_->update(d->parameter_data, d->p);
    }
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, unone_);
    }
    d->dx.setZero();
    d->dp.setZero();
  });

  const VectorXs& r0 = d->r;
  assertStableStateFD(x);

  // Computing the d residual(x,u) / dx
  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->Rp.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
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

  if (this->np_ != 0) {
    d->dp.setZero();
    d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
    for (std::size_t ip = 0; ip < this->np_; ++ip) {
      perturbed_ip = ip;
      d->pp = d->p;
      d->pp(ip) += d->ph_jac;
      params_->update(d->parameter_data, d->pp);
      for (std::size_t i = 0; i < reevals_.size(); ++i) {
        reevals_[i](x, unone_);
      }
      model_->calc(d->data_p[ip], x);
      d->Rp.col(ip) = (d->data_p[ip]->r - r0) / d->ph_jac;
      params_->update(d->parameter_data, d->p);
      perturbed_ip = this->np_;
    }
  }
  restore.restore();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelNumDiffTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return createData(data, std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelNumDiffTpl<Scalar>::createData(
    DataCollectorAbstract* const data,
    const std::shared_ptr<ParameterDataManager>& parameter_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data, parameter_data);
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::set_params(
    const std::shared_ptr<ResidualDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
  if (params->get_np() != this->np_) {
    throw_pretty("Invalid argument: params dimension (" +
                 std::to_string(params->get_np()) +
                 ") does not match residual np (" + std::to_string(this->np_) +
                 ")");
  }
  if (data == nullptr) {
    throw_pretty("Invalid argument: data is null");
  }
  Data* d = static_cast<Data*>(data.get());
  assertParameterData(d, params);
  params_ = params;
  update_p(data, params_->zero());
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::update_p(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(this->np_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  assertParameterData(d, params_);
  params_->update(d->parameter_data, p);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelNumDiffTpl<NewScalar> ResidualModelNumDiffTpl<Scalar>::cast()
    const {
  typedef ResidualModelNumDiffTpl<NewScalar> ReturnType;
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
const std::shared_ptr<ResidualModelAbstractTpl<Scalar> >&
ResidualModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ResidualModelNumDiffTpl<Scalar>::ParameterManager>&
ResidualModelNumDiffTpl<Scalar>::get_params() const {
  return params_;
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

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::assertParameterData(
    const Data* const data,
    const std::shared_ptr<ParameterManager>& params) const {
  if (data == nullptr) {
    throw_pretty("Invalid argument: data is null");
  }
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_np() != this->np_) {
    throw_pretty("Invalid argument: params dimension (" +
                 std::to_string(params->get_np()) +
                 ") does not match residual np (" + std::to_string(this->np_) +
                 ")");
  }
  if (data->parameter_data == nullptr ||
      data->parameter_data->parameter_data != data->parameter_data.get() ||
      data->parameter_data->params == nullptr) {
    throw_pretty("Invalid argument: parameter data is null");
  }
  if (data->parameter_data->params->np != this->np_) {
    throw_pretty("Invalid argument: parameter data dimension (" +
                 std::to_string(data->parameter_data->params->np) +
                 ") does not match residual np (" + std::to_string(this->np_) +
                 ")");
  }
  const DataCollectorParamsTpl<Scalar>* collector =
      dynamic_cast<const DataCollectorParamsTpl<Scalar>*>(data->shared);
  if (this->np_ != 0 && collector == nullptr) {
    throw_pretty("Invalid argument: shared data must provide parameter data");
  }
  if (collector != nullptr &&
      (collector->params == nullptr || collector->parameter_data == nullptr ||
       collector->parameter_data != data->parameter_data.get() ||
       collector->params != data->parameter_data->params ||
       collector->parameter_data->parameter_data !=
           collector->parameter_data)) {
    throw_pretty("Invalid argument: collector parameter data is inconsistent");
  }
}

}  // namespace crocoddyl
