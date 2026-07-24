///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/dynamics-parameter-access.hxx"

namespace crocoddyl {

template <typename Scalar>
IntegratedObserverModelEulerTpl<Scalar>::IntegratedObserverModelEulerTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints, const Scalar time_step)
    : Base(dynamics, costs, constraints, time_step) {}

template <typename Scalar>
void IntegratedObserverModelEulerTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(this->get_nu()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  d->resize(this, true);

  const std::size_t nv = state_->get_nv();
  const std::size_t ndx = state_->get_ndx();
  const std::size_t dynamics_nu = dynamics_->get_nu();

  if (dynamics_nu != 0u) {
    dynamics_->calc(d->dynamics, x, u.tail(dynamics_nu));
  } else {
    dynamics_->calc(d->dynamics, x, u_zero_);
  }

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const VectorXs& a = d->dynamics->vdot;
  d->dissipative_E.noalias() = d->dynamics->dissipative_P * time_step_;
  d->dx.head(nv).noalias() = v * time_step_ + a * time_step2_;
  d->dx.tail(nv).noalias() = a * time_step_;
  d->dx.noalias() += this->compute_projected_noise(d, x, u.head(ndx));
  state_->integrate(x, d->dx, d->xnext);

  costs_->calc(d->costs, x, u);
  d->cost = time_step_ * d->costs->cost;
  d->r.setZero();
  d->g.setZero();
  d->h.setZero();

  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();
  if (ng_d != 0u) {
    d->g.head(ng_d) = d->dynamics->g;
  }
  if (nh_d != 0u) {
    d->h.head(nh_d) = d->dynamics->h;
  }

  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), true);
    constraints_->calc(d->constraints, x, u);
    const std::size_t ng_c = constraints_->get_ng();
    const std::size_t nh_c = constraints_->get_nh();
    if (ng_c != 0u) {
      d->g.segment(ng_d, ng_c) = d->constraints->g;
    }
    if (nh_c != 0u) {
      d->h.segment(nh_d, nh_c) = d->constraints->h;
    }
  }
}

template <typename Scalar>
void IntegratedObserverModelEulerTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  d->g.conservativeResize(this->get_ng_T());
  d->Gx.conservativeResize(this->get_ng_T(), state_->get_ndx());
  d->Gp.conservativeResize(this->get_ng_T(), np_);
  d->h.conservativeResize(this->get_nh_T());
  d->Hx.conservativeResize(this->get_nh_T(), state_->get_ndx());
  d->Hp.conservativeResize(this->get_nh_T(), np_);

  dynamics_->calc(d->dynamics, x);
  costs_->calc(d->costs, x);

  d->xnext = x;
  d->dx.setZero();
  d->dissipative_E.setZero();
  d->cost = d->costs->cost;
  d->r.setZero();
  d->g.setZero();
  d->h.setZero();

  if (constraints_ != nullptr) {
    d->constraints_terminal->resize(constraints_.get(), false);
    constraints_->calc(d->constraints_terminal, x);
    const std::size_t ng_T = constraints_->get_ng_T();
    const std::size_t nh_T = constraints_->get_nh_T();
    if (ng_T != 0u) {
      d->g.head(ng_T) = d->constraints_terminal->g;
    }
    if (nh_T != 0u) {
      d->h.head(nh_T) = d->constraints_terminal->h;
    }
  }
}

template <typename Scalar>
void IntegratedObserverModelEulerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(this->get_nu()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);

  const std::size_t nv = state_->get_nv();
  const std::size_t ndx = state_->get_ndx();
  const std::size_t dynamics_nu = dynamics_->get_nu();
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();

  if (dynamics_nu != 0u) {
    dynamics_->calcDiff(d->dynamics, x, u.tail(dynamics_nu));
  } else {
    dynamics_->calcDiff(d->dynamics, x, u_zero_);
  }

  d->dE_dv.noalias() = d->dynamics->dP_dv * time_step_;
  d->dE_dp.noalias() = d->dynamics->dP_dp * time_step_;

  d->Fx.topRows(nv).noalias() = d->dynamics->Fx * time_step2_;
  d->Fx.bottomRows(nv).noalias() = d->dynamics->Fx * time_step_;
  d->Fx.topRightCorner(nv, nv).diagonal().array() += Scalar(time_step_);
  d->Fp.topRows(nv).noalias() = d->dynamics->Fp * time_step2_;
  d->Fp.bottomRows(nv).noalias() = d->dynamics->Fp * time_step_;
  d->Fx.noalias() += this->compute_projected_noise_jacobian(d, x, u.head(ndx));
  state_->JintegrateTransport(x, d->dx, d->Fx, second);
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
  state_->JintegrateTransport(x, d->dx, d->Fp, second);

  d->Fu.setZero();
  state_->Jintegrate(x, d->dx, d->Jfirst, d->Jsecond, second, setto);
  d->Fu.leftCols(ndx).noalias() = d->Jsecond * d->noise_projector;
  if (dynamics_nu != 0u) {
    d->ddx_du.topRows(nv).noalias() = d->dynamics->Fu * time_step2_;
    d->ddx_du.bottomRows(nv).noalias() = d->dynamics->Fu * time_step_;
    d->Fu.rightCols(dynamics_nu).noalias() = d->Jsecond * d->ddx_du;
  }

  costs_->calcDiff(d->costs, x, u);
  d->Lx.noalias() = time_step_ * d->costs->Lx;
  d->Lu.noalias() = time_step_ * d->costs->Lu;
  d->Lxx.noalias() = time_step_ * d->costs->Lxx;
  d->Lxu.noalias() = time_step_ * d->costs->Lxu;
  d->Luu.noalias() = time_step_ * d->costs->Luu;
  if (np_ != 0) {
    d->Lp.setZero();
    if (static_cast<std::size_t>(d->costs->Lp.size()) == np_) {
      d->Lp.noalias() = time_step_ * d->costs->Lp;
    }
    d->Lpp.setZero();
    if (static_cast<std::size_t>(d->costs->Lpp.rows()) == np_ &&
        static_cast<std::size_t>(d->costs->Lpp.cols()) == np_) {
      d->Lpp.noalias() = time_step_ * d->costs->Lpp;
    }
    d->Lpx.setZero();
    if (static_cast<std::size_t>(d->costs->Lpx.rows()) == np_ &&
        static_cast<std::size_t>(d->costs->Lpx.cols()) == ndx) {
      d->Lpx.noalias() = time_step_ * d->costs->Lpx;
    }
    d->Lpu.setZero();
    if (static_cast<std::size_t>(d->costs->Lpu.rows()) == np_ &&
        static_cast<std::size_t>(d->costs->Lpu.cols()) == this->get_nu()) {
      d->Lpu.noalias() = time_step_ * d->costs->Lpu;
    }
  } else {
    d->Lp.setZero();
    d->Lpp.setZero();
    d->Lpx.setZero();
    d->Lpu.setZero();
  }

  d->Gx.setZero();
  d->Gu.setZero();
  d->Gp.setZero();
  d->Hx.setZero();
  d->Hu.setZero();
  d->Hp.setZero();
  if (ng_d != 0u) {
    d->Gx.topRows(ng_d) = d->dynamics->Gx;
    d->Gp.topRows(ng_d) = d->dynamics->Gp;
    if (dynamics_nu != 0u) {
      d->Gu.block(0, ndx, ng_d, dynamics_nu) = d->dynamics->Gu;
    }
  }
  if (nh_d != 0u) {
    d->Hx.topRows(nh_d) = d->dynamics->Hx;
    d->Hp.topRows(nh_d) = d->dynamics->Hp;
    if (dynamics_nu != 0u) {
      d->Hu.block(0, ndx, nh_d, dynamics_nu) = d->dynamics->Hu;
    }
  }

  if (constraints_ != nullptr) {
    d->constraints->resize(constraints_.get(), true);
    constraints_->calcDiff(d->constraints, x, u);
    const std::size_t ng_c = constraints_->get_ng();
    const std::size_t nh_c = constraints_->get_nh();
    if (ng_c != 0u) {
      d->Gx.middleRows(ng_d, ng_c) = d->constraints->Gx;
      d->Gu.middleRows(ng_d, ng_c) = d->constraints->Gu;
      if (constraints_->get_np() != 0u) {
        d->Gp.middleRows(ng_d, ng_c) = d->constraints->Gp;
      }
    }
    if (nh_c != 0u) {
      d->Hx.middleRows(nh_d, nh_c) = d->constraints->Hx;
      d->Hu.middleRows(nh_d, nh_c) = d->constraints->Hu;
      if (constraints_->get_np() != 0u) {
        d->Hp.middleRows(nh_d, nh_c) = d->constraints->Hp;
      }
    }
  }
}

template <typename Scalar>
void IntegratedObserverModelEulerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);

  dynamics_->calcDiff(d->dynamics, x);

  d->Fx.setZero();
  d->Fp.setZero();
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx);
  d->dE_dv.setZero();
  d->dE_dp.setZero();

  costs_->calcDiff(d->costs, x);
  d->Lx = d->costs->Lx;
  d->Lxx = d->costs->Lxx;
  if (np_ != 0) {
    d->Lp.setZero();
    if (static_cast<std::size_t>(d->costs->Lp.size()) == np_) {
      d->Lp = d->costs->Lp;
    }
    d->Lpp.setZero();
    if (static_cast<std::size_t>(d->costs->Lpp.rows()) == np_ &&
        static_cast<std::size_t>(d->costs->Lpp.cols()) == np_) {
      d->Lpp = d->costs->Lpp;
    }
    d->Lpx.setZero();
    if (static_cast<std::size_t>(d->costs->Lpx.rows()) == np_ &&
        static_cast<std::size_t>(d->costs->Lpx.cols()) == state_->get_ndx()) {
      d->Lpx = d->costs->Lpx;
    }
  } else {
    d->Lp.setZero();
    d->Lpp.setZero();
    d->Lpx.setZero();
  }

  d->Gx.setZero();
  d->Gp.setZero();
  d->Hx.setZero();
  d->Hp.setZero();
  if (constraints_ != nullptr) {
    d->constraints_terminal->resize(constraints_.get(), false);
    constraints_->calcDiff(d->constraints_terminal, x);
    const std::size_t ng_T = constraints_->get_ng_T();
    const std::size_t nh_T = constraints_->get_nh_T();
    if (ng_T != 0u) {
      d->Gx.topRows(ng_T) = d->constraints_terminal->Gx;
      if (constraints_->get_np() != 0u) {
        d->Gp.topRows(ng_T) = d->constraints_terminal->Gp;
      }
    }
    if (nh_T != 0u) {
      d->Hx.topRows(nh_T) = d->constraints_terminal->Hx;
      if (constraints_->get_np() != 0u) {
        d->Hp.topRows(nh_T) = d->constraints_terminal->Hp;
      }
    }
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedObserverModelEulerTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedObserverModelEulerTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
bool IntegratedObserverModelEulerTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    return false;
  }
  return dynamics_->checkData(d->dynamics);
}

template <typename Scalar>
void IntegratedObserverModelEulerTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
    const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be " +
                 std::to_string(this->get_nu()) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  const std::size_t ndx = state_->get_ndx();
  const std::size_t dynamics_nu = dynamics_->get_nu();
  u.setZero();
  if (dynamics_nu != 0u) {
    dynamics_->quasiStatic(d->dynamics, u.tail(dynamics_nu), x, maxiter, tol);
  }
  if (ndx != 0u) {
    u.head(ndx).setZero();
  }
}

template <typename Scalar>
void IntegratedObserverModelEulerTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedObserverModelEuler {dt=" << time_step_ << ", " << *dynamics_
     << "}";
}

template <typename Scalar>
template <typename NewScalar>
IntegratedObserverModelEulerTpl<NewScalar>
IntegratedObserverModelEulerTpl<Scalar>::cast() const {
  typedef IntegratedObserverModelEulerTpl<NewScalar> ReturnType;
  typedef CostModelSumTpl<NewScalar> CostModelSumNew;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintModelManagerNew;
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;

  std::shared_ptr<ConstraintModelManagerNew> constraints;
  if (constraints_ != nullptr) {
    constraints = std::make_shared<ConstraintModelManagerNew>(
        constraints_->template cast<NewScalar>());
  }
  const std::shared_ptr<DynamicsModelAbstractTpl<NewScalar> > dynamics =
      dynamics_->template cast<NewScalar>();
  ReturnType ret(
      dynamics,
      std::make_shared<CostModelSumNew>(costs_->template cast<NewScalar>()),
      constraints, scalar_cast<NewScalar>(time_step_));
  if (this->get_tau_meas().size() != 0) {
    ret.update_tau(this->get_tau_meas().template cast<NewScalar>());
  }
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
std::shared_ptr<typename IntegratedObserverModelEulerTpl<Scalar>::Data>
IntegratedObserverModelEulerTpl<Scalar>::cast_data(
    const std::shared_ptr<ActionDataAbstract>& data) const {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data is not an integrated observer Euler data");
  }
  return d;
}

}  // namespace crocoddyl
