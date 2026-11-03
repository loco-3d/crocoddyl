///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <limits>

#include "crocoddyl/core/integrator/discretized.hpp"
#include "crocoddyl/core/integrator/parameter-cast.hxx"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
DiscretizedActionModelTpl<Scalar>::DiscretizedActionModelTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints)
    : Base(dynamics != nullptr ? dynamics->get_state() : nullptr,
           dynamics != nullptr ? dynamics->get_nu() : 0,
           costs != nullptr ? costs->get_nr() : 0,
           (dynamics != nullptr ? dynamics->get_ng() : 0) +
               (constraints != nullptr ? constraints->get_ng() : 0),
           (dynamics != nullptr ? dynamics->get_nh() : 0) +
               (constraints != nullptr ? constraints->get_nh() : 0),
           constraints != nullptr ? constraints->get_ng_T() : 0,
           constraints != nullptr ? constraints->get_nh_T() : 0,
           dynamics != nullptr ? dynamics->get_np() : 0),
      dynamics_(dynamics),
      costs_(costs),
      constraints_(constraints),
      params_(nullptr) {
  if (dynamics_ == nullptr) {
    throw_pretty("Invalid argument: dynamics is null");
  }
  if (costs_ == nullptr) {
    throw_pretty("Invalid argument: costs is null");
  }
  if (dynamics_->get_dyn_type() != DynamicsType::DiscreteTime) {
    throw_pretty(
        "Invalid argument: "
        "DiscretizedActionModel requires a DiscreteTime dynamics model");
  }
  if (costs_->get_state()->get_nx() != state_->get_nx() ||
      costs_->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: costs has an incompatible state");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: costs has an incompatible control dimension");
  }
  if (constraints_ != nullptr) {
    if (constraints_->get_state()->get_nx() != state_->get_nx() ||
        constraints_->get_state()->get_ndx() != state_->get_ndx()) {
      throw_pretty("Invalid argument: constraints has an incompatible state");
    }
    if (constraints_->get_nu() != nu_) {
      throw_pretty(
          "Invalid argument: constraints has an incompatible control "
          "dimension");
    }
  }
  refresh_constraint_bounds();
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  nr_ = costs_->get_nr();
  static_cast<ActionDataAbstractTpl<Scalar>*>(d)->resize(this, true);
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t nh_c = constraints_ != nullptr ? constraints_->get_nh() : 0;

  dynamics_->calc(d->dynamics, x, u);
  d->xnext = d->dynamics->vdot;

  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  d->r.setZero();

  d->g.setZero();
  d->h.setZero();
  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), true);
    constraints_->calc(d->constraints, x, u);
    if (ng_c != 0) {
      d->g.segment(ng_d, ng_c) = d->constraints->g;
    }
    if (nh_c != 0) {
      d->h.segment(nh_d, nh_c) = d->constraints->h;
    }
  }
  if (ng_d != 0) {
    d->g.head(ng_d) = d->dynamics->g;
  }
  if (nh_d != 0) {
    d->h.head(nh_d) = d->dynamics->h;
  }
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  nr_ = costs_->get_nr();
  d->r.conservativeResize(this->nr_);
  d->g.conservativeResize(this->get_ng_T());
  d->Gx.conservativeResize(this->get_ng_T(), this->state_->get_ndx());
  d->Gp.conservativeResize(this->get_ng_T(), this->np_);
  d->h.conservativeResize(this->get_nh_T());
  d->Hx.conservativeResize(this->get_nh_T(), this->state_->get_ndx());
  d->Hp.conservativeResize(this->get_nh_T(), this->np_);
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t nh_T =
      constraints_ != nullptr ? constraints_->get_nh_T() : 0;

  d->xnext = x;
  dynamics_->calc(d->dynamics, x);
  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  d->r.setZero();

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
void DiscretizedActionModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  nr_ = costs_->get_nr();
  static_cast<ActionDataAbstractTpl<Scalar>*>(d)->resize(this, true);
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t nh_c = constraints_ != nullptr ? constraints_->get_nh() : 0;

  dynamics_->calcDiff(d->dynamics, x, u);
  // For DiscreteTime, Fx and Fu are already ndx x ndx and ndx x nu in tangent
  // space; pass them through directly without any integration chain rule.
  d->Fx = d->dynamics->Fx;
  d->Fu = d->dynamics->Fu;

  costs_->calcDiff(d->costs, x, u);
  d->Lx = d->costs->Lx;
  d->Lu = d->costs->Lu;
  d->Lxx = d->costs->Lxx;
  d->Lxu = d->costs->Lxu;
  d->Luu = d->costs->Luu;

  d->Gx.setZero();
  d->Gu.setZero();
  d->Gp.setZero();
  d->Hx.setZero();
  d->Hu.setZero();
  d->Hp.setZero();
  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), true);
    constraints_->calcDiff(d->constraints, x, u);
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
  if (ng_d != 0) {
    d->Gx.topRows(ng_d) = d->dynamics->Gx;
    d->Gu.topRows(ng_d) = d->dynamics->Gu;
    if (dynamics_->get_np() != 0) {
      d->Gp.topRows(ng_d) = d->dynamics->Gp;
    }
  }
  if (nh_d != 0) {
    d->Hx.topRows(nh_d) = d->dynamics->Hx;
    d->Hu.topRows(nh_d) = d->dynamics->Hu;
    if (dynamics_->get_np() != 0) {
      d->Hp.topRows(nh_d) = d->dynamics->Hp;
    }
  }

  if (np_ != 0) {
    d->Fp.setZero();
    d->Lp.setZero();
    d->Lpp.setZero();
    d->Lpx.setZero();
    d->Lpu.setZero();
    if (dynamics_->get_np() != 0) {
      d->Fp = d->dynamics->Fp;
    }
    if (costs_->get_np() != 0) {
      if (costs_->get_np() != np_) {
        throw_pretty(
            "Invalid argument: costs parameter dimension does not match "
            "discretized action parameter dimension");
      }
      d->Lp = d->costs->Lp;
      d->Lpp = d->costs->Lpp;
      d->Lpx = d->costs->Lpx;
      d->Lpu = d->costs->Lpu;
    }
  }
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  nr_ = costs_->get_nr();
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t nh_T =
      constraints_ != nullptr ? constraints_->get_nh_T() : 0;

  // xnext = x at terminal nodes, so Fx = I_{ndx} in tangent space.
  d->Fx.setIdentity();

  costs_->calcDiff(d->costs, x);
  d->Lx = d->costs->Lx;
  d->Lxx = d->costs->Lxx;

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

  if (np_ != 0) {
    // Terminal: xnext = x so Fp = 0; Lp from costs; no Gp/Hp from dynamics.
    d->Fp.setZero();
    d->Lp.setZero();
    d->Lpp.setZero();
    d->Lpx.setZero();
    if (costs_->get_np() != 0) {
      if (costs_->get_np() != np_) {
        throw_pretty(
            "Invalid argument: costs parameter dimension does not match "
            "discretized action parameter dimension");
      }
      d->Lp = d->costs->Lp;
      d->Lpp = d->costs->Lpp;
      d->Lpx = d->costs->Lpx;
    }
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
DiscretizedActionModelTpl<Scalar>::createData() {
  if (params_ == nullptr) {
    return createData(std::shared_ptr<ParameterDataManager>());
  }
  const std::shared_ptr<ParameterDataManager> params_data =
      params_->createData();
  return createData(params_data);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
DiscretizedActionModelTpl<Scalar>::createData(
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
DiscretizedActionModelTpl<NewScalar> DiscretizedActionModelTpl<Scalar>::cast()
    const {
  typedef DiscretizedActionModelTpl<NewScalar> ReturnType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  const std::shared_ptr<DynamicsModelAbstractTpl<NewScalar> > dynamics =
      dynamics_->template cast<NewScalar>();
  ReturnType ret(
      dynamics, std::make_shared<CostType>(costs_->template cast<NewScalar>()),
      constraints_ != nullptr ? std::make_shared<ConstraintType>(
                                    constraints_->template cast<NewScalar>())
                              : nullptr);
  if (params_ != nullptr) {
    const std::shared_ptr<ParameterManagerTpl<NewScalar> > casted_params =
        internal::getDynamicsParameters(dynamics);
    const std::shared_ptr<ParameterManagerTpl<NewScalar> > params =
        casted_params != nullptr
            ? casted_params
            : std::make_shared<ParameterManagerTpl<NewScalar> >(
                  params_->template cast<NewScalar>());
    ret.set_params(ret.createData(params->createData()), params);
  }
  return ret;
}

template <typename Scalar>
bool DiscretizedActionModelTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    return false;
  }
  return dynamics_->checkData(d->dynamics);
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
    const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  dynamics_->quasiStatic(d->dynamics, u, x, maxiter, tol);
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }

  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data has the wrong Discretized runtime type");
  }
  if (params->get_np_action() != 0) {
    throw_pretty(
        "Invalid argument: DiscretizedActionModel does not support active "
        "action parameters");
  }
  params_ = params;
  np_ = params_->get_np();
  if (constraints_ != nullptr && constraints_->get_np() != 0 &&
      constraints_->get_np() != this->np_) {
    throw_pretty(
        "Invalid argument: constraints parameter dimension does not match "
        "discretized action parameter dimension");
  }
  static_cast<ActionDataAbstractTpl<Scalar>*>(d.get())->resize(this);
  if (d->params == nullptr) {
    d->params = params_->createData();
  } else {
    d->params->resize(params_.get());
  }
  dynamics_->set_params(d->dynamics, params_);
  d->costs = costs_->createData(d->dynamics->shared);
  if (constraints_ != nullptr) {
    d->constraints = constraints_->createData(d->dynamics->shared);
  } else {
    d->constraints.reset();
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty("Invalid call: discretized action parameters are not set");
  }
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data has the wrong Discretized runtime type");
  }
  if (static_cast<std::size_t>(p.size()) != params_->get_np()) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(params_->get_np()) + ")");
  }
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: discretized action data has no parameter-manager "
        "payload");
  }
  params_->update(d->params, p);
  dynamics_->update_p(d->dynamics, p);
}

template <typename Scalar>
std::size_t DiscretizedActionModelTpl<Scalar>::get_ng() const {
  return dynamics_->get_ng() +
         (constraints_ != nullptr ? constraints_->get_ng() : 0);
}

template <typename Scalar>
std::size_t DiscretizedActionModelTpl<Scalar>::get_nh() const {
  return dynamics_->get_nh() +
         (constraints_ != nullptr ? constraints_->get_nh() : 0);
}

template <typename Scalar>
std::size_t DiscretizedActionModelTpl<Scalar>::get_ng_T() const {
  return constraints_ != nullptr ? constraints_->get_ng_T() : 0;
}

template <typename Scalar>
std::size_t DiscretizedActionModelTpl<Scalar>::get_nh_T() const {
  return constraints_ != nullptr ? constraints_->get_nh_T() : 0;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DiscretizedActionModelTpl<Scalar>::get_g_lb() const {
  refresh_constraint_bounds();
  return g_lb_live_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DiscretizedActionModelTpl<Scalar>::get_g_ub() const {
  refresh_constraint_bounds();
  return g_ub_live_;
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::refresh_constraint_bounds() const {
  const std::size_t ng_c = constraints_ != nullptr ? constraints_->get_ng() : 0;
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t ng_T =
      constraints_ != nullptr ? constraints_->get_ng_T() : 0;
  const std::size_t ng_max = std::max(ng_c + ng_d, ng_T);
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
const std::shared_ptr<DynamicsModelAbstractTpl<Scalar> >&
DiscretizedActionModelTpl<Scalar>::get_dynamics() const {
  return dynamics_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
DiscretizedActionModelTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DiscretizedActionModelTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const std::shared_ptr<ParameterManagerTpl<Scalar> >&
DiscretizedActionModelTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
void DiscretizedActionModelTpl<Scalar>::print(std::ostream& os) const {
  os << "DiscretizedActionModel {" << *dynamics_ << "}";
}

}  // namespace crocoddyl
