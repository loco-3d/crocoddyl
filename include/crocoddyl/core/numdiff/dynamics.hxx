///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/dynamics.hpp"

namespace crocoddyl {

template <typename Scalar>
DynamicsModelNumDiffTpl<Scalar>::DynamicsModelNumDiffTpl(
    std::shared_ptr<Base> model)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_dyn_type(),
           internal::checkNumDiffModel(model)->get_np(),
           internal::checkNumDiffModel(model)->get_nu(),
           internal::checkNumDiffModel(model)->get_ng(),
           internal::checkNumDiffModel(model)->get_nh()),
      model_(internal::checkNumDiffModel(model)),
      params_(nullptr),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
DynamicsModelNumDiffTpl<Scalar>::DynamicsModelNumDiffTpl(
    std::shared_ptr<Base> model, std::shared_ptr<ParameterManager> params)
    : Base(internal::checkNumDiffModel(model)->get_state(),
           internal::checkNumDiffModel(model)->get_dyn_type(),
           params != nullptr ? params->get_np()
                             : internal::checkNumDiffModel(model)->get_np(),
           internal::checkNumDiffModel(model)->get_nu(),
           internal::checkNumDiffModel(model)->get_ng(),
           internal::checkNumDiffModel(model)->get_nh()),
      model_(internal::checkNumDiffModel(model)),
      params_(params),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {
  if (params_ == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params_->get_state()->get_nx() != state_->get_nx() ||
      params_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
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
  model_->calc(d->data_0, x, u);
  data->vdot = d->data_0->vdot;
  data->dissipative_P = d->data_0->dissipative_P;
  data->g = d->data_0->g;
  data->h = d->data_0->h;
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  data->vdot = d->data_0->vdot;
  data->dissipative_P = d->data_0->dissipative_P;
  data->g = d->data_0->g;
  data->h = d->data_0->h;
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
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
  internal::NumDiffRestorationTpl restore([&]() {
    d->dx.setZero();
    d->du.setZero();
  });
  const VectorXs& vdot0 = d->data_0->vdot;
  const VectorXs& p0 = d->data_0->dissipative_P;
  const VectorXs& g0 = d->data_0->g;
  const VectorXs& h0 = d->data_0->h;
  const std::size_t nv = state_->get_nv();
  const std::size_t ndx = state_->get_ndx();
  data->dP_dv.setZero();

  assertStableStateFD(x);

  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);
    if (model_->get_dyn_type() == DynamicsType::DiscreteTime) {
      // vdot = [q+, v+] lives in the state manifold (size nx); use state->diff
      // to get the ndx-dimensional tangent-space difference.
      state_->diff(vdot0, d->data_x[ix]->vdot, d->dvdot);
      data->Fx.col(ix) = d->dvdot / d->xh_jac;
    } else {
      data->Fx.col(ix) = (d->data_x[ix]->vdot - vdot0) / d->xh_jac;
    }
    if (ix >= nv && ix - nv < static_cast<std::size_t>(data->dP_dv.cols())) {
      data->dP_dv.col(ix - nv) =
          (d->data_x[ix]->dissipative_P - p0) / d->xh_jac;
    }
    if (ng_ != 0) {
      data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    }
    if (nh_ != 0) {
      data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    }
    d->dx(ix) = Scalar(0.);
  }

  data->Fu.setZero();
  if (ng_ != 0) {
    data->Gu.setZero();
  }
  if (nh_ != 0) {
    data->Hu.setZero();
  }
  d->du.setZero();
  d->uh_jac = e_jac_ * std::max(Scalar(1.), u.norm());
  for (std::size_t iu = 0; iu < nu_; ++iu) {
    d->du(iu) = d->uh_jac;
    d->up = u + d->du;
    model_->calc(d->data_u[iu], x, d->up);
    if (model_->get_dyn_type() == DynamicsType::DiscreteTime) {
      state_->diff(vdot0, d->data_u[iu]->vdot, d->dvdot);
      data->Fu.col(iu) = d->dvdot / d->uh_jac;
    } else {
      data->Fu.col(iu) = (d->data_u[iu]->vdot - vdot0) / d->uh_jac;
    }
    if (ng_ != 0) {
      data->Gu.col(iu) = (d->data_u[iu]->g - g0) / d->uh_jac;
    }
    if (nh_ != 0) {
      data->Hu.col(iu) = (d->data_u[iu]->h - h0) / d->uh_jac;
    }
    d->du(iu) = Scalar(0.);
  }
  restore.restore();
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  internal::NumDiffRestorationTpl restore([&]() { d->dx.setZero(); });
  const VectorXs& vdot0 = d->data_0->vdot;
  const VectorXs& p0 = d->data_0->dissipative_P;
  const VectorXs& g0 = d->data_0->g;
  const VectorXs& h0 = d->data_0->h;
  const std::size_t nv = state_->get_nv();
  const std::size_t ndx = state_->get_ndx();
  data->dP_dv.setZero();
  data->Fu.setZero();
  if (ng_ != 0) {
    data->Gu.setZero();
  }
  if (nh_ != 0) {
    data->Hu.setZero();
  }

  assertStableStateFD(x);

  model_->get_state()->diff(model_->get_state()->zero(), x, d->dx);
  d->x_norm = d->dx.norm();
  d->dx.setZero();
  d->xh_jac = e_jac_ * std::max(Scalar(1.), d->x_norm);
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    d->dx(ix) = d->xh_jac;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    if (model_->get_dyn_type() == DynamicsType::DiscreteTime) {
      state_->diff(vdot0, d->data_x[ix]->vdot, d->dvdot);
      data->Fx.col(ix) = d->dvdot / d->xh_jac;
    } else {
      data->Fx.col(ix) = (d->data_x[ix]->vdot - vdot0) / d->xh_jac;
    }
    if (ix >= nv && ix - nv < static_cast<std::size_t>(data->dP_dv.cols())) {
      data->dP_dv.col(ix - nv) =
          (d->data_x[ix]->dissipative_P - p0) / d->xh_jac;
    }
    if (ng_ != 0) {
      data->Gx.col(ix) = (d->data_x[ix]->g - g0) / d->xh_jac;
    }
    if (nh_ != 0) {
      data->Hx.col(ix) = (d->data_x[ix]->h - h0) / d->xh_jac;
    }
    d->dx(ix) = Scalar(0.);
  }
  restore.restore();
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::calcDiff_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (this->np_ == 0) {
    return;
  }
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
  std::size_t perturbed_ip = this->np_;
  internal::NumDiffRestorationTpl restore([&]() {
    if (perturbed_ip < this->np_) {
      model_->update_p(d->data_p[perturbed_ip], d->p);
    }
    d->dp.setZero();
  });
  const VectorXs& vdot0 = d->data_0->vdot;
  const VectorXs& p0 = d->data_0->dissipative_P;
  const VectorXs& g0 = d->data_0->g;
  const VectorXs& h0 = d->data_0->h;
  data->Fp.setZero();
  data->dP_dp.setZero();
  if (ng_ != 0) {
    data->Gp.setZero();
  }
  if (nh_ != 0) {
    data->Hp.setZero();
  }
  d->dp.setZero();
  d->ph_jac = e_jac_ * std::max(Scalar(1.), d->p.norm());
  for (std::size_t ip = 0; ip < this->np_; ++ip) {
    perturbed_ip = ip;
    d->dp(ip) = d->ph_jac;
    d->pp = d->p + d->dp;
    model_->update_p(d->data_p[ip], d->pp);
    model_->calc(d->data_p[ip], x, u);
    if (model_->get_dyn_type() == DynamicsType::DiscreteTime) {
      state_->diff(vdot0, d->data_p[ip]->vdot, d->dvdot);
      data->Fp.col(ip) = d->dvdot / d->ph_jac;
    } else {
      data->Fp.col(ip) = (d->data_p[ip]->vdot - vdot0) / d->ph_jac;
    }
    data->dP_dp.col(ip) = (d->data_p[ip]->dissipative_P - p0) / d->ph_jac;
    if (ng_ != 0) {
      data->Gp.col(ip) = (d->data_p[ip]->g - g0) / d->ph_jac;
    }
    if (nh_ != 0) {
      data->Hp.col(ip) = (d->data_p[ip]->h - h0) / d->ph_jac;
    }
    d->dp(ip) = Scalar(0.);
    model_->update_p(d->data_p[ip], d->p);
    perturbed_ip = this->np_;
  }
  restore.restore();
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelNumDiffTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar> >
DynamicsModelNumDiffTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  const std::shared_ptr<DynamicsDataAbstract> data = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, params_data);
  if (params_ != nullptr) {
    set_params(data, params_);
  }
  return data;
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::set_params(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }

  Data* d = static_cast<Data*>(data.get());
  params_ = params;
  this->np_ = params_->get_np();
  d->resize(this);
  model_->set_params(d->data_0, params_);
  for (std::size_t ix = 0; ix < d->data_x.size(); ++ix) {
    model_->set_params(d->data_x[ix], params_);
  }
  for (std::size_t iu = 0; iu < d->data_u.size(); ++iu) {
    model_->set_params(d->data_u[iu], params_);
  }
  for (std::size_t ip = 0; ip < d->data_p.size(); ++ip) {
    model_->set_params(d->data_p[ip], params_);
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::update_p(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(this->np_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  d->p = p;
  model_->update_p(d->data_0, p);
  for (std::size_t ix = 0; ix < d->data_x.size(); ++ix) {
    model_->update_p(d->data_x[ix], p);
  }
  for (std::size_t iu = 0; iu < d->data_u.size(); ++iu) {
    model_->update_p(d->data_u[iu], p);
  }
  for (std::size_t ip = 0; ip < d->data_p.size(); ++ip) {
    model_->update_p(d->data_p[ip], p);
  }
}

template <typename Scalar>
template <typename NewScalar>
DynamicsModelNumDiffTpl<NewScalar> DynamicsModelNumDiffTpl<Scalar>::cast()
    const {
  typedef DynamicsModelNumDiffTpl<NewScalar> ReturnType;
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  std::shared_ptr<ParameterManagerNew> params;
  if (params_ != nullptr) {
    params = std::make_shared<ParameterManagerNew>(
        params_->template cast<NewScalar>());
    return ReturnType(model_->template cast<NewScalar>(), params);
  }
  return ReturnType(model_->template cast<NewScalar>());
}

template <typename Scalar>
const std::shared_ptr<DynamicsModelAbstractTpl<Scalar> >&
DynamicsModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const std::shared_ptr<
    typename DynamicsModelNumDiffTpl<Scalar>::ParameterManager>&
DynamicsModelNumDiffTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const Scalar DynamicsModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::print(std::ostream& os) const {
  os << "DynamicsModelNumDiffTpl {dynamics=" << *model_ << "}";
}

template <typename Scalar>
void DynamicsModelNumDiffTpl<Scalar>::assertStableStateFD(
    const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
