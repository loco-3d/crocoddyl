///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <limits>

#include "crocoddyl/core/integrator/dynamics-parameter-access.hxx"
#include "crocoddyl/core/observer/discretized.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
DiscretizedObserverModelTpl<Scalar>::DiscretizedObserverModelTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs, const std::size_t ntau,
    std::shared_ptr<ConstraintModelManager> constraints)
    : Base(
          detail::check_observer_model(dynamics)->get_state(), ntau,
          /*nu=*/detail::check_observer_model(dynamics)->get_state()->get_ndx(),
          /*nr=*/0,
          /*ng=*/detail::check_observer_model(dynamics)->get_ng() +
              (constraints != nullptr ? constraints->get_ng() : 0),
          /*nh=*/detail::check_observer_model(dynamics)->get_nh() +
              (constraints != nullptr ? constraints->get_nh() : 0),
          /*ng_T=*/constraints != nullptr ? constraints->get_ng_T() : 0,
          /*nh_T=*/constraints != nullptr ? constraints->get_nh_T() : 0,
          /*np=*/detail::check_observer_model(dynamics)->get_np()),
      dynamics_(detail::check_observer_model(dynamics)),
      costs_(costs),
      constraints_(constraints),
      params_(nullptr) {
  if (costs_ == nullptr) {
    throw_pretty("Invalid argument: costs is null");
  }
  if (dynamics_->get_dyn_type() != DynamicsType::DiscreteTime) {
    throw_pretty(
        "Invalid argument: "
        "DiscretizedObserverModel requires a DiscreteTime dynamics model");
  }
  if (ntau != dynamics_->get_nu()) {
    throw_pretty(
        "Invalid argument: ntau must match the discrete dynamics control "
        "dimension (it should be " +
        std::to_string(dynamics_->get_nu()) + ")");
  }
  if (costs_->get_state()->get_nx() != state_->get_nx() ||
      costs_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: costs has an incompatible state");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: costs doesn't have the observer control "
        "dimension (it should be " +
        std::to_string(nu_) + ")");
  }
  if (constraints_ != nullptr) {
    if (constraints_->get_state()->get_nx() != state_->get_nx() ||
        constraints_->get_state()->get_ndx() != state_->get_ndx()) {
      throw_pretty("Invalid argument: constraints has an incompatible state");
    }
    if (constraints_->get_nu() != nu_) {
      throw_pretty(
          "Invalid argument: constraints doesn't have the observer control "
          "dimension (it should be " +
          std::to_string(nu_) + ")");
    }
  }

  Base::set_u_lb(
      VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity()));
  Base::set_u_ub(
      VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity()));
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  g_lb_live_.resize(dynamics_->get_ng() + ng_c);
  g_ub_live_.resize(dynamics_->get_ng() + ng_c);
  refresh_constraint_bounds();
  Base::set_g_lb(g_lb_live_);
  Base::set_g_ub(g_ub_live_);
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& w) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(w.size()) != nu_) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  d->resize(this, true);
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t nh_c = constraints_ != nullptr ? constraints_->get_nh() : 0;

  // The discrete observer dynamics is driven by tau_meas (via update_tau);
  // the process noise w does not enter the mean dynamics.
  dynamics_->calc(d->dynamics, x, d->u_zero);
  d->xnext = d->dynamics->vdot;

  costs_->calc(d->costs, x, w);
  d->cost = d->costs->cost;

  d->g.setZero();
  d->h.setZero();
  if (ng_d != 0) {
    d->g.head(ng_d) = d->dynamics->g;
  }
  if (nh_d != 0) {
    d->h.head(nh_d) = d->dynamics->h;
  }
  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), true);
    constraints_->calc(d->constraints, x, w);
    if (ng_c != 0) {
      d->g.segment(ng_d, ng_c) = d->constraints->g;
    }
    if (nh_c != 0) {
      d->h.segment(nh_d, nh_c) = d->constraints->h;
    }
  }
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t nh_T =
      constraints_ != nullptr ? constraints_->get_nh_T() : 0;
  d->g.conservativeResize(ng_T);
  d->Gx.conservativeResize(ng_T, state_->get_ndx());
  d->Gp.conservativeResize(ng_T, np_);
  d->h.conservativeResize(nh_T);
  d->Hx.conservativeResize(nh_T, state_->get_ndx());
  d->Hp.conservativeResize(nh_T, np_);

  dynamics_->calc(d->dynamics, x);
  d->xnext = x;
  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  d->dissipative_E.setZero();
  d->dE_dv.setZero();
  d->dE_dp.setZero();

  d->g.setZero();
  d->h.setZero();
  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), false);
    constraints_->calc(d->constraints, x);
    if (ng_T != 0) {
      d->g.head(ng_T) = d->constraints->g;
    }
    if (nh_T != 0) {
      d->h.head(nh_T) = d->constraints->h;
    }
  }
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& w) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(w.size()) != nu_) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t nh_c = constraints_ != nullptr ? constraints_->get_nh() : 0;

  dynamics_->calcDiff(d->dynamics, x, d->u_zero);
  // Fx and Fp come directly from the dynamics; Fu = 0 (process noise is
  // independent of the mean discrete transition).
  d->Fx = d->dynamics->Fx;
  d->Fp = d->dynamics->Fp;
  d->Fu.setZero();

  costs_->calcDiff(d->costs, x, w);
  d->Lx = d->costs->Lx;
  d->Lu = d->costs->Lu;
  d->Lp = d->costs->Lp;
  d->Lxx = d->costs->Lxx;
  d->Lxu = d->costs->Lxu;
  d->Luu = d->costs->Luu;
  d->Lpx = d->costs->Lpx;
  d->Lpu = d->costs->Lpu;
  d->Lpp = d->costs->Lpp;

  d->Gx.setZero();
  d->Gu.setZero();
  d->Gp.setZero();
  d->Hx.setZero();
  d->Hu.setZero();
  d->Hp.setZero();
  if (ng_d != 0) {
    d->Gx.topRows(ng_d) = d->dynamics->Gx;
    if (dynamics_->get_np() != 0) {
      d->Gp.topRows(ng_d) = d->dynamics->Gp;
    }
  }
  if (nh_d != 0) {
    d->Hx.topRows(nh_d) = d->dynamics->Hx;
    if (dynamics_->get_np() != 0) {
      d->Hp.topRows(nh_d) = d->dynamics->Hp;
    }
  }
  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), true);
    constraints_->calcDiff(d->constraints, x, w);
    if (ng_c != 0) {
      d->Gx.middleRows(ng_d, ng_c) = d->constraints->Gx;
      d->Gu.middleRows(ng_d, ng_c) = d->constraints->Gu;
      if (constraints_->get_np() != 0) {
        d->Gp.middleRows(ng_d, ng_c) = d->constraints->Gp;
      }
    }
    if (nh_c != 0) {
      d->Hx.middleRows(nh_d, nh_c) = d->constraints->Hx;
      d->Hu.middleRows(nh_d, nh_c) = d->constraints->Hu;
      if (constraints_->get_np() != 0) {
        d->Hp.middleRows(nh_d, nh_c) = d->constraints->Hp;
      }
    }
  }
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t nh_T =
      constraints_ != nullptr ? constraints_->get_nh_T() : 0;

  dynamics_->calcDiff(d->dynamics, x);
  // xnext = x at terminal nodes, so Fx = I_{ndx} and Fp = 0.
  d->Fx.setIdentity();
  d->Fp.setZero();
  d->dissipative_E.setZero();
  d->dE_dv.setZero();
  d->dE_dp.setZero();

  costs_->calcDiff(d->costs, x);
  d->Lx = d->costs->Lx;
  d->Lp = d->costs->Lp;
  d->Lxx = d->costs->Lxx;
  d->Lpx = d->costs->Lpx;
  d->Lpp = d->costs->Lpp;

  d->Gx.setZero();
  d->Gp.setZero();
  d->Hx.setZero();
  d->Hp.setZero();
  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), false);
    constraints_->calcDiff(d->constraints, x);
    if (ng_T != 0) {
      d->Gx.topRows(ng_T) = d->constraints->Gx;
      if (constraints_->get_np() != 0) {
        d->Gp.topRows(ng_T) = d->constraints->Gp;
      }
    }
    if (nh_T != 0) {
      d->Hx.topRows(nh_T) = d->constraints->Hx;
      if (constraints_->get_np() != 0) {
        d->Hp.topRows(nh_T) = d->constraints->Hp;
      }
    }
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
DiscretizedObserverModelTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
DiscretizedObserverModelTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  const std::shared_ptr<ActionDataAbstract> data = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, params_data);
  if (params_ != nullptr) {
    set_params(data, params_);
  }
  return data;
}

template <typename Scalar>
template <typename NewScalar>
DiscretizedObserverModelTpl<NewScalar>
DiscretizedObserverModelTpl<Scalar>::cast() const {
  typedef DiscretizedObserverModelTpl<NewScalar> ReturnType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  std::shared_ptr<ConstraintType> constraints;
  if (constraints_ != nullptr) {
    constraints = std::make_shared<ConstraintType>(
        constraints_->template cast<NewScalar>());
  }
  const std::shared_ptr<DynamicsModelAbstractTpl<NewScalar> > dynamics =
      dynamics_->template cast<NewScalar>();
  ReturnType ret(dynamics,
                 std::make_shared<CostType>(costs_->template cast<NewScalar>()),
                 Base::ntau_, constraints);
  ret.update_tau(tau_meas_.template cast<NewScalar>());
  if (params_ != nullptr) {
    std::shared_ptr<ParameterManagerNew> params =
        internal::getDynamicsParameters(dynamics);
    if (params == nullptr) {
      params = std::make_shared<ParameterManagerNew>(
          params_->template cast<NewScalar>());
    }
    const std::shared_ptr<ActionDataAbstractTpl<NewScalar> > data =
        ret.createData(params->createData());
    ret.set_params(data, params);
  }
  return ret;
}

template <typename Scalar>
bool DiscretizedObserverModelTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    return false;
  }
  return dynamics_->checkData(d->dynamics);
}

template <typename Scalar>
std::size_t DiscretizedObserverModelTpl<Scalar>::get_ng() const {
  return dynamics_->get_ng() +
         (constraints_ != nullptr ? constraints_->get_ng() : 0);
}

template <typename Scalar>
std::size_t DiscretizedObserverModelTpl<Scalar>::get_nh() const {
  return dynamics_->get_nh() +
         (constraints_ != nullptr ? constraints_->get_nh() : 0);
}

template <typename Scalar>
std::size_t DiscretizedObserverModelTpl<Scalar>::get_ng_T() const {
  return constraints_ != nullptr ? constraints_->get_ng_T() : 0;
}

template <typename Scalar>
std::size_t DiscretizedObserverModelTpl<Scalar>::get_nh_T() const {
  return constraints_ != nullptr ? constraints_->get_nh_T() : 0;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DiscretizedObserverModelTpl<Scalar>::get_g_lb() const {
  refresh_constraint_bounds();
  return g_lb_live_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DiscretizedObserverModelTpl<Scalar>::get_g_ub() const {
  refresh_constraint_bounds();
  return g_ub_live_;
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::refresh_constraint_bounds() const {
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t ng_max = std::max(ng_d + ng_c, ng_T);
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  g_lb_live_.setConstant(ng_max, -inf);
  g_ub_live_.setConstant(ng_max, inf);
  if (ng_d != 0) {
    g_ub_live_.head(ng_d).setZero();
  }
  if (constraints_ != nullptr && ng_c != 0) {
    g_lb_live_.segment(ng_d, ng_c) = constraints_->get_lb().head(ng_c);
    g_ub_live_.segment(ng_d, ng_c) = constraints_->get_ub().head(ng_c);
  }
}

template <typename Scalar>
std::shared_ptr<typename DiscretizedObserverModelTpl<Scalar>::Data>
DiscretizedObserverModelTpl<Scalar>::cast_data(
    const std::shared_ptr<ActionDataAbstract>& data) const {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data is not a discretized observer data");
  }
  return d;
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> w,
    const Eigen::Ref<const VectorXs>& x, const std::size_t, const Scalar) {
  if (static_cast<std::size_t>(w.size()) != nu_) {
    throw_pretty("Invalid argument: w has wrong dimension (it should be " +
                 std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  cast_data(data);
  w.setZero();
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  params_ = params;
  np_ = params_->get_np();
  if (constraints_ != nullptr && constraints_->get_np() != 0 &&
      constraints_->get_np() != np_) {
    throw_pretty(
        "Invalid argument: constraints parameter dimension does not match "
        "discretized observer parameter dimension");
  }
  d->resize(this);
  dynamics_->set_params(d->dynamics, params);
  d->costs = costs_->createData(d->dynamics->shared);
  if (constraints_ != nullptr) {
    d->constraints = constraints_->createData(d->dynamics->shared);
  } else {
    d->constraints.reset();
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty("Invalid call: observer parameters are not set");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  dynamics_->update_p(d->dynamics, p);
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  Base::update_tau(tau_meas);
  dynamics_->update_tau(tau_meas_);
}

template <typename Scalar>
const std::shared_ptr<DynamicsModelAbstractTpl<Scalar> >&
DiscretizedObserverModelTpl<Scalar>::get_dynamics() const {
  return dynamics_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
DiscretizedObserverModelTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DiscretizedObserverModelTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
DiscretizedObserverModelTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
void DiscretizedObserverModelTpl<Scalar>::print(std::ostream& os) const {
  os << "DiscretizedObserverModel {" << *dynamics_ << "}";
}

}  // namespace crocoddyl
