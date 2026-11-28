///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, University of Pisa,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/parameter-cast.hxx"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::IntegratedActionModelEulerTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model, control, time_step, with_cost_residual) {}

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::IntegratedActionModelEulerTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model, time_step, with_cost_residual) {}

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::IntegratedActionModelEulerTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    std::shared_ptr<IntegratorTime> integrator_time)
    : Base(dynamics, costs, constraints, control, integrator_time) {}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calc(
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
  refresh_integrator_time();

  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
    static_cast<ActionDataAbstractTpl<Scalar>*>(d)->resize(this, true);
  }
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  control_->calc(d->control, Scalar(0.), u);
  if (dynamics_ != nullptr) {
    const std::size_t ng_d = dynamics_->get_ng();
    const std::size_t nh_d = dynamics_->get_nh();
    const std::size_t ng_c =
        constraints_ != nullptr ? constraints_->get_ng() : 0;
    const std::size_t nh_c =
        constraints_ != nullptr ? constraints_->get_nh() : 0;

    dynamics_->calc(d->dynamics, x, d->control->w);
    const VectorXs& a = d->dynamics->vdot;
    d->dx.head(nv).noalias() = v * time_step_ + a * time_step2_;
    d->dx.tail(nv).noalias() = a * time_step_;
    state_->integrate(x, d->dx, d->xnext);

    costs_->calc(d->costs, x, d->control->w);
    d->cost = time_step_ * d->costs->cost;
    d->g.setZero();
    d->h.setZero();
    if (constraints_ != nullptr) {
      d->constraints->resize(constraints_.get());
      constraints_->calc(d->constraints, x, d->control->w);
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
    if (with_cost_residual_) {
      d->r.setZero();
    }
  } else {
    differential_->calc(d->differential, x, d->control->w);
    const VectorXs& a = d->differential->xout;
    d->dx.head(nv).noalias() = v * time_step_ + a * time_step2_;
    d->dx.tail(nv).noalias() = a * time_step_;
    differential_->get_state()->integrate(x, d->dx, d->xnext);
    d->cost = time_step_ * d->differential->cost;
    d->g = d->differential->g;
    d->h = d->differential->h;
    if (with_cost_residual_) {
      d->r = d->differential->r;
    }
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  refresh_integrator_time();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
    d->r.conservativeResize(this->nr_);
    d->g.conservativeResize(this->get_ng_T());
    d->Gx.conservativeResize(this->get_ng_T(), this->state_->get_ndx());
    d->Gp.conservativeResize(this->get_ng_T(), this->np_);
    d->h.conservativeResize(this->get_nh_T());
    d->Hx.conservativeResize(this->get_nh_T(), this->state_->get_ndx());
    d->Hp.conservativeResize(this->get_nh_T(), this->np_);
  }

  d->dx.setZero();
  d->xnext = x;
  if (dynamics_ != nullptr) {
    const std::size_t ng_T =
        constraints_ != nullptr ? constraints_->get_ng_T() : 0;
    const std::size_t nh_T =
        constraints_ != nullptr ? constraints_->get_nh_T() : 0;

    dynamics_->calc(d->dynamics, x);
    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
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
    if (with_cost_residual_) {
      d->r.setZero();
    }
  } else {
    differential_->calc(d->differential, x);
    d->cost = d->differential->cost;
    d->g = d->differential->g;
    d->h = d->differential->h;
    if (with_cost_residual_) {
      d->r = d->differential->r;
    }
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calcDiff(
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
  refresh_integrator_time();

  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
    static_cast<ActionDataAbstractTpl<Scalar>*>(d)->resize(this, true);
  }

  control_->calc(d->control, Scalar(0.), u);
  if (dynamics_ != nullptr) {
    const std::size_t ng_d = dynamics_->get_ng();
    const std::size_t nh_d = dynamics_->get_nh();
    const std::size_t ng_c =
        constraints_ != nullptr ? constraints_->get_ng() : 0;
    const std::size_t nh_c =
        constraints_ != nullptr ? constraints_->get_nh() : 0;
    const std::size_t np_action =
        params_ != nullptr ? params_->get_np_action() : 0;
    const std::size_t np_dynamics =
        params_ != nullptr ? params_->get_np_dynamics() : this->np_;

    dynamics_->calcDiff(d->dynamics, x, d->control->w);
    control_->multiplyByJacobian(d->control, d->dynamics->Fu, d->da_du);
    d->Fx.topRows(nv).noalias() = d->dynamics->Fx * time_step2_;
    d->Fx.bottomRows(nv).noalias() = d->dynamics->Fx * time_step_;
    d->Fx.topRightCorner(nv, nv).diagonal().array() += Scalar(time_step_);
    d->Fu.topRows(nv).noalias() = time_step2_ * d->da_du;
    d->Fu.bottomRows(nv).noalias() = time_step_ * d->da_du;
    state_->JintegrateTransport(x, d->dx, d->Fx, second);
    state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
    state_->JintegrateTransport(x, d->dx, d->Fu, second);
    if (this->np_ != 0) {
      d->Fp.setZero();
      if (np_dynamics != 0) {
        d->Fp.topRows(nv).middleCols(np_action, np_dynamics).noalias() =
            d->dynamics->Fp.middleCols(np_action, np_dynamics) * time_step2_;
        d->Fp.bottomRows(nv).middleCols(np_action, np_dynamics).noalias() =
            d->dynamics->Fp.middleCols(np_action, np_dynamics) * time_step_;
      }
      state_->JintegrateTransport(x, d->dx, d->Fp, second);
      if (params_ != nullptr && np_action != 0) {
        if (d->params == nullptr) {
          throw_pretty(
              "Invalid argument: Euler integrated action data has no "
              "parameter-manager payload");
        }
        params_->calcDiff_action(d->params, data, d->Fp.leftCols(np_action), x,
                                 u);
      }
    }

    costs_->calcDiff(d->costs, x, d->control->w);
    d->Lx.noalias() = time_step_ * d->costs->Lx;
    control_->multiplyJacobianTransposeBy(d->control, d->costs->Lu, d->Lu);
    d->Lu *= time_step_;
    d->Lxx.noalias() = time_step_ * d->costs->Lxx;
    control_->multiplyByJacobian(d->control, d->costs->Lxu, d->Lxu);
    d->Lxu *= time_step_;
    control_->multiplyByJacobian(d->control, d->costs->Luu, d->Lwu);
    control_->multiplyJacobianTransposeBy(d->control, d->Lwu, d->Luu);
    d->Luu *= time_step_;
    if (this->np_ != 0) {
      const std::size_t np_cost = costs_->get_np();
      d->Lp.setZero();
      d->Lpp.setZero();
      d->Lpx.setZero();
      d->Lpu.setZero();
      if (np_cost != 0) {
        if (np_cost != this->np_) {
          throw_pretty(
              "Invalid argument: costs parameter dimension does not match "
              "Euler integrated action parameter dimension");
        }
        d->Lp.noalias() = time_step_ * d->costs->Lp;
        d->Lpp.noalias() = time_step_ * d->costs->Lpp;
        d->Lpx.noalias() = time_step_ * d->costs->Lpx;
        control_->multiplyByJacobian(d->control, d->costs->Lpu, d->Lpu);
        d->Lpu *= time_step_;
      }
      if (integrator_time_->get_timeopt() && params_ != nullptr) {
        typename ParameterManager::ParameterContainer::const_iterator it, end;
        std::size_t offset = 0;
        for (it = params_->get_action_params().begin(),
            end = params_->get_action_params().end();
             it != end; ++it) {
          const std::shared_ptr<typename ParameterManager::ParameterItem>&
              item = it->second;
          if (!item->get_active()) {
            continue;
          }
          const std::size_t np_item = item->get_param()->get_np();
          if (std::dynamic_pointer_cast<IntegratorTimeoptParamsTpl<Scalar> >(
                  item->get_param()) != nullptr) {
            for (std::size_t ip = 0; ip < np_item; ++ip) {
              const std::size_t col = offset + ip;
              d->Lpp.row(col).noalias() += d->Lp.transpose();
              d->Lpp.col(col).noalias() += d->Lp;
              d->Lp(col) += d->cost;
              d->Lpp(col, col) += d->cost;
              d->Lpx.row(col).noalias() += d->Lx.transpose();
              d->Lpu.row(col).noalias() += d->Lu.transpose();
            }
          }
          offset += np_item;
        }
      }
    }

    d->Gx.setZero();
    d->Gu.setZero();
    d->Hx.setZero();
    d->Hu.setZero();
    if (this->np_ != 0) {
      d->Gp.setZero();
      d->Hp.setZero();
    }
    if (constraints_ != nullptr) {
      d->constraints->resize(constraints_.get());
      constraints_->calcDiff(d->constraints, x, d->control->w);
      if (ng_c != 0) {
        d->Gx.middleRows(ng_d, ng_c) = d->constraints->Gx;
        control_->multiplyByJacobian(d->control, d->constraints->Gu,
                                     d->Gu.middleRows(ng_d, ng_c));
        if (constraints_->get_np() != 0) {
          d->Gp.middleRows(ng_d, ng_c) = d->constraints->Gp;
        }
      }
      if (nh_c != 0) {
        d->Hx.middleRows(nh_d, nh_c) = d->constraints->Hx;
        control_->multiplyByJacobian(d->control, d->constraints->Hu,
                                     d->Hu.middleRows(nh_d, nh_c));
        if (constraints_->get_np() != 0) {
          d->Hp.middleRows(nh_d, nh_c) = d->constraints->Hp;
        }
      }
    }
    if (ng_d != 0) {
      d->Gx.topRows(ng_d) = d->dynamics->Gx;
      control_->multiplyByJacobian(d->control, d->dynamics->Gu,
                                   d->Gu.topRows(ng_d));
      if (np_dynamics != 0) {
        d->Gp.topRows(ng_d).middleCols(np_action, np_dynamics) =
            d->dynamics->Gp.middleCols(np_action, np_dynamics);
      }
    }
    if (nh_d != 0) {
      d->Hx.topRows(nh_d) = d->dynamics->Hx;
      control_->multiplyByJacobian(d->control, d->dynamics->Hu,
                                   d->Hu.topRows(nh_d));
      if (np_dynamics != 0) {
        d->Hp.topRows(nh_d).middleCols(np_action, np_dynamics) =
            d->dynamics->Hp.middleCols(np_action, np_dynamics);
      }
    }
  } else {
    differential_->calcDiff(d->differential, x, d->control->w);
    control_->multiplyByJacobian(d->control, d->differential->Fu, d->da_du);
    d->Fx.topRows(nv).noalias() = d->differential->Fx * time_step2_;
    d->Fx.bottomRows(nv).noalias() = d->differential->Fx * time_step_;
    d->Fx.topRightCorner(nv, nv).diagonal().array() += Scalar(time_step_);
    d->Fu.topRows(nv).noalias() = time_step2_ * d->da_du;
    d->Fu.bottomRows(nv).noalias() = time_step_ * d->da_du;
    state_->JintegrateTransport(x, d->dx, d->Fx, second);
    state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
    state_->JintegrateTransport(x, d->dx, d->Fu, second);

    d->Lx.noalias() = time_step_ * d->differential->Lx;
    control_->multiplyJacobianTransposeBy(d->control, d->differential->Lu,
                                          d->Lu);
    d->Lu *= time_step_;
    d->Lxx.noalias() = time_step_ * d->differential->Lxx;
    control_->multiplyByJacobian(d->control, d->differential->Lxu, d->Lxu);
    d->Lxu *= time_step_;
    control_->multiplyByJacobian(d->control, d->differential->Luu, d->Lwu);
    control_->multiplyJacobianTransposeBy(d->control, d->Lwu, d->Luu);
    d->Luu *= time_step_;
    d->Gx = d->differential->Gx;
    d->Hx = d->differential->Hx;
    d->Gu.setZero();
    d->Hu.setZero();
    if (differential_->get_ng() != 0) {
      control_->multiplyByJacobian(d->control, d->differential->Gu,
                                   d->Gu.topRows(differential_->get_ng()));
    }
    if (differential_->get_nh() != 0) {
      control_->multiplyByJacobian(d->control, d->differential->Hu,
                                   d->Hu.topRows(differential_->get_nh()));
    }
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  refresh_integrator_time();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
  }

  if (dynamics_ != nullptr) {
    const std::size_t ng_T =
        constraints_ != nullptr ? constraints_->get_ng_T() : 0;
    const std::size_t nh_T =
        constraints_ != nullptr ? constraints_->get_nh_T() : 0;

    d->Fx.setZero();
    state_->Jintegrate(x, d->dx, d->Fx, d->Fx);
    costs_->calcDiff(d->costs, x);
    d->Lx = d->costs->Lx;
    d->Lxx = d->costs->Lxx;
    d->Fp.setZero();
    d->Lp.setZero();
    d->Lpp.setZero();
    d->Lpx.setZero();
    if (this->np_ != 0 && costs_->get_np() != 0) {
      if (costs_->get_np() != this->np_) {
        throw_pretty(
            "Invalid argument: costs parameter dimension does not match "
            "Euler integrated action parameter dimension");
      }
      d->Lp = d->costs->Lp;
      d->Lpp = d->costs->Lpp;
      d->Lpx = d->costs->Lpx;
    }
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
  } else {
    differential_->calcDiff(d->differential, x);
    state_->Jintegrate(x, d->dx, d->Fx, d->Fx);
    d->Lx = d->differential->Lx;
    d->Lxx = d->differential->Lxx;
    d->Gx = d->differential->Gx;
    d->Hx = d->differential->Hx;
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (dynamics_ == nullptr) {
    throw_pretty(
        "Invalid call: Euler integrated actions built from a differential "
        "backend do not support set_params");
  }
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_state()->get_nx() != state_->get_nx() ||
      params->get_state()->get_ndx() != state_->get_ndx()) {
    throw_pretty("Invalid argument: params has an incompatible state");
  }
  std::size_t active_time_params = 0;
  typename ParameterManager::ParameterContainer::const_iterator it, end;
  for (it = params->get_action_params().begin(),
      end = params->get_action_params().end();
       it != end; ++it) {
    const std::shared_ptr<typename ParameterManager::ParameterItem>& item =
        it->second;
    if (!item->get_active()) {
      continue;
    }
    const std::shared_ptr<IntegratorTimeoptParamsTpl<Scalar> > time_param =
        std::dynamic_pointer_cast<IntegratorTimeoptParamsTpl<Scalar> >(
            item->get_param());
    if (time_param == nullptr) {
      throw_pretty(
          "Invalid argument: Euler supports only an active integration-time "
          "action parameter");
    }
    if (time_param->get_integrator_time() != integrator_time_) {
      throw_pretty(
          "Invalid argument: integration-time parameter must share the Euler "
          "integrator time");
    }
    if (++active_time_params > 1) {
      throw_pretty(
          "Invalid argument: Euler supports only one active integration-time "
          "parameter");
    }
  }

  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data has the wrong Euler runtime type");
  }
  params_ = params;
  this->np_ = params_->get_np();
  if (constraints_ != nullptr && constraints_->get_np() != 0 &&
      constraints_->get_np() != this->np_) {
    throw_pretty(
        "Invalid argument: constraints parameter dimension does not match "
        "Euler integrated action parameter dimension");
  }
  d->resize(this);
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
void IntegratedActionModelEulerTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (dynamics_ == nullptr) {
    throw_pretty(
        "Invalid call: Euler integrated actions built from a differential "
        "backend do not support update_p");
  }
  if (params_ == nullptr) {
    throw_pretty("Invalid call: integrated action parameters are not set");
  }
  if (static_cast<std::size_t>(p.size()) != params_->get_np()) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(params_->get_np()) + ")");
  }

  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data has the wrong Euler runtime type");
  }
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: Euler integrated action data has no "
        "parameter-manager payload");
  }
  params_->update(d->params, p);
  dynamics_->update_p(d->dynamics, p);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelEulerTpl<Scalar>::createData() {
  if (params_ == nullptr) {
    return createData(std::shared_ptr<ParameterDataManager>());
  }
  const std::shared_ptr<ParameterDataManager> params_data =
      params_->createData();
  return createData(params_data);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelEulerTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  const std::size_t nw =
      dynamics_ != nullptr ? dynamics_->get_nu() : differential_->get_nu();
  if (control_->get_nu() > nw) {
    std::cerr << "Warning: It is useless to use an Euler integrator with a "
                 "control parametrization larger than PolyZero"
              << std::endl;
  }
  const std::shared_ptr<ActionDataAbstract> data = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, params_data);
  if (params_ != nullptr) {
    const Scalar dt = integrator_time_->get_time_step();
    try {
      set_params(data, params_);
      integrator_time_->set_time_step(dt);
    } catch (...) {
      integrator_time_->set_time_step(dt);
      throw;
    }
  }
  return data;
}

template <typename Scalar>
template <typename NewScalar>
IntegratedActionModelEulerTpl<NewScalar>
IntegratedActionModelEulerTpl<Scalar>::cast() const {
  typedef IntegratedActionModelEulerTpl<NewScalar> ReturnType;
  if (dynamics_ != nullptr) {
    typedef CostModelSumTpl<NewScalar> CostType;
    typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
    const std::shared_ptr<IntegratorTimeTpl<NewScalar> > casted_time =
        std::make_shared<IntegratorTimeTpl<NewScalar> >(
            integrator_time_->template cast<NewScalar>());
    const std::shared_ptr<DynamicsModelAbstractTpl<NewScalar> > dynamics =
        dynamics_->template cast<NewScalar>();
    ReturnType ret(
        dynamics,
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        constraints_ != nullptr ? std::make_shared<ConstraintType>(
                                      constraints_->template cast<NewScalar>())
                                : nullptr,
        control_->template cast<NewScalar>(), casted_time);
    if (params_ != nullptr) {
      const std::shared_ptr<ParameterManagerTpl<NewScalar> > casted_params =
          detail::cast_integrated_action_params<Scalar, NewScalar>(
              params_, integrator_time_, casted_time, dynamics);
      ret.set_params(ret.createData(casted_params->createData()),
                     casted_params);
      casted_time->set_time_step(
          scalar_cast<NewScalar>(integrator_time_->get_time_step()));
    }
    return ret;
  } else if (control_) {
    ReturnType ret(differential_->template cast<NewScalar>(),
                   control_->template cast<NewScalar>(),
                   scalar_cast<NewScalar>(integrator_time_->get_time_step()),
                   with_cost_residual_);
    return ret;
  } else {
    ReturnType ret(differential_->template cast<NewScalar>(),
                   scalar_cast<NewScalar>(integrator_time_->get_time_step()),
                   with_cost_residual_);
    return ret;
  }
}

template <typename Scalar>
bool IntegratedActionModelEulerTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == NULL) {
    return false;
  }
  if (dynamics_ != nullptr) {
    return dynamics_->checkData(d->dynamics);
  } else {
    return differential_->checkData(d->differential);
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::quasiStatic(
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

  const std::shared_ptr<Data>& d = std::static_pointer_cast<Data>(data);

  d->control->w.setZero();
  if (dynamics_ != nullptr) {
    dynamics_->quasiStatic(d->dynamics, d->control->w, x, maxiter, tol);
  } else {
    differential_->quasiStatic(d->differential, d->control->w, x, maxiter, tol);
  }
  control_->params(d->control, Scalar(0.), d->control->w);
  u = d->control->u;
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedActionModelEuler {dt=" << integrator_time_->get_time_step()
     << ", ";
  if (dynamics_ != nullptr) {
    os << *dynamics_ << "}";
  } else {
    os << *differential_ << "}";
  }
}

}  // namespace crocoddyl
