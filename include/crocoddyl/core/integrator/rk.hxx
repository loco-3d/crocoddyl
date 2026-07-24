///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, University of Trento,
//                          LAAS-CNRS, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/parameter-cast.hxx"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelRKTpl<Scalar>::IntegratedActionModelRKTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    const RKType rktype, const Scalar time_step, const bool with_cost_residual)
    : Base(model, control, time_step, with_cost_residual), rk_type_(rktype) {
  set_rk_type(rktype);
}

template <typename Scalar>
IntegratedActionModelRKTpl<Scalar>::IntegratedActionModelRKTpl(
    std::shared_ptr<DifferentialActionModelAbstract> model, const RKType rktype,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model, time_step, with_cost_residual), rk_type_(rktype) {
  set_rk_type(rktype);
}

template <typename Scalar>
IntegratedActionModelRKTpl<Scalar>::IntegratedActionModelRKTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    std::shared_ptr<IntegratorTime> integrator_time, const RKType rktype)
    : Base(dynamics, costs, constraints, control, integrator_time),
      rk_type_(rktype) {
  set_rk_type(rktype);
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::calc(
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
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
  }
  refresh_integrator_time();
  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
    static_cast<ActionDataAbstractTpl<Scalar>*>(d)->resize(this, true);
  }

  if (dynamics_ != nullptr) {
    const std::size_t ng_d = dynamics_->get_ng();
    const std::size_t nh_d = dynamics_->get_nh();
    const std::size_t ng_c = constraints_ != NULL ? constraints_->get_ng() : 0;
    const std::size_t nh_c = constraints_ != NULL ? constraints_->get_nh() : 0;

    const std::shared_ptr<DynamicsDataAbstract>& k0_data = d->dynamics[0];
    const std::shared_ptr<CostDataSum>& c0_data = d->costs[0];
    const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
        d->control[0];
    control_->calc(u0_data, rk_c_[0], u);
    d->ws[0] = u0_data->w;
    d->y[0] = x;
    dynamics_->calc(k0_data, x, d->ws[0]);
    costs_->calc(c0_data, x, d->ws[0]);
    d->ki[0].head(nv) = x.tail(nv);
    d->ki[0].tail(nv) = k0_data->vdot;
    d->integral[0] = c0_data->cost;
    for (std::size_t i = 1; i < ni_; ++i) {
      const std::shared_ptr<DynamicsDataAbstract>& ki_data = d->dynamics[i];
      const std::shared_ptr<CostDataSum>& ci_data = d->costs[i];
      const std::shared_ptr<ControlParametrizationDataAbstract>& ui_data =
          d->control[i];
      d->dx_rk[i].noalias() = time_step_ * rk_c_[i] * d->ki[i - 1];
      state_->integrate(x, d->dx_rk[i], d->y[i]);
      control_->calc(ui_data, rk_c_[i], u);
      d->ws[i] = ui_data->w;
      dynamics_->calc(ki_data, d->y[i], d->ws[i]);
      costs_->calc(ci_data, d->y[i], d->ws[i]);
      d->ki[i].head(nv) = d->y[i].tail(nv);
      d->ki[i].tail(nv) = ki_data->vdot;
      d->integral[i] = ci_data->cost;
    }

    if (ni_ == 2) {
      d->dx = d->ki[1] * time_step_;
      d->cost = d->integral[1] * time_step_;
    } else if (ni_ == 3) {
      d->dx = (d->ki[0] + Scalar(3.) * d->ki[2]) * time_step_ / Scalar(4.);
      d->cost = (d->integral[0] + Scalar(3.) * d->integral[2]) * time_step_ /
                Scalar(4.);
    } else {
      d->dx = (d->ki[0] + Scalar(2.) * d->ki[1] + Scalar(2.) * d->ki[2] +
               d->ki[3]) *
              time_step_ / Scalar(6.);
      d->cost = (d->integral[0] + Scalar(2.) * d->integral[1] +
                 Scalar(2.) * d->integral[2] + d->integral[3]) *
                time_step_ / Scalar(6.);
    }
    state_->integrate(x, d->dx, d->xnext);
    d->g.setZero();
    d->h.setZero();
    if (constraints_ != NULL) {
      d->constraints->resize(constraints_.get());
      constraints_->calc(d->constraints, x, d->ws[0]);
      if (ng_c != 0) {
        d->g.segment(ng_d, ng_c) = d->constraints->g;
      }
      if (nh_c != 0) {
        d->h.segment(nh_d, nh_c) = d->constraints->h;
      }
    }
    if (ng_d != 0) {
      d->g.head(ng_d) = k0_data->g;
    }
    if (nh_d != 0) {
      d->h.head(nh_d) = k0_data->h;
    }
    if (with_cost_residual_) {
      d->r.setZero();
    }
    return;
  }

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
      d->control[0];
  control_->calc(u0_data, rk_c_[0], u);
  d->ws[0] = u0_data->w;
  differential_->calc(k0_data, x, d->ws[0]);
  d->y[0] = x;
  d->ki[0].head(nv) = d->y[0].tail(nv);
  d->ki[0].tail(nv) = k0_data->xout;
  d->integral[0] = k0_data->cost;
  for (std::size_t i = 1; i < ni_; ++i) {
    const std::shared_ptr<DifferentialActionDataAbstract>& ki_data =
        d->differential[i];
    const std::shared_ptr<ControlParametrizationDataAbstract>& ui_data =
        d->control[i];
    d->dx_rk[i].noalias() = time_step_ * rk_c_[i] * d->ki[i - 1];
    state_->integrate(x, d->dx_rk[i], d->y[i]);
    control_->calc(ui_data, rk_c_[i], u);
    d->ws[i] = ui_data->w;
    differential_->calc(ki_data, d->y[i], d->ws[i]);
    d->ki[i].head(nv) = d->y[i].tail(nv);
    d->ki[i].tail(nv) = ki_data->xout;
    d->integral[i] = ki_data->cost;
  }

  if (ni_ == 2) {
    d->dx = d->ki[1] * time_step_;
    d->cost = d->integral[1] * time_step_;
  } else if (ni_ == 3) {
    d->dx = (d->ki[0] + Scalar(3.) * d->ki[2]) * time_step_ / Scalar(4.);
    d->cost = (d->integral[0] + Scalar(3.) * d->integral[2]) * time_step_ /
              Scalar(4.);
  } else {
    d->dx =
        (d->ki[0] + Scalar(2.) * d->ki[1] + Scalar(2.) * d->ki[2] + d->ki[3]) *
        time_step_ / Scalar(6.);
    d->cost = (d->integral[0] + Scalar(2.) * d->integral[1] +
               Scalar(2.) * d->integral[2] + d->integral[3]) *
              time_step_ / Scalar(6.);
  }
  state_->integrate(x, d->dx, d->xnext);
  d->g = k0_data->g;
  d->h = k0_data->h;
  if (with_cost_residual_) {
    d->r = k0_data->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
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
        constraints_ != NULL ? constraints_->get_ng_T() : 0;
    const std::size_t nh_T =
        constraints_ != NULL ? constraints_->get_nh_T() : 0;
    const std::shared_ptr<DynamicsDataAbstract>& k0_data = d->dynamics[0];
    const std::shared_ptr<CostDataSum>& c0_data = d->costs[0];

    dynamics_->calc(k0_data, x);
    costs_->calc(c0_data, x);
    d->cost = c0_data->cost;
    d->g.setZero();
    d->h.setZero();
    if (constraints_ != NULL) {
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
    return;
  }

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  differential_->calc(k0_data, x);
  d->cost = k0_data->cost;
  d->g = k0_data->g;
  d->h = k0_data->h;
  if (with_cost_residual_) {
    d->r = k0_data->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::calcDiff(
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
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
  }
  refresh_integrator_time();
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = control_->get_nu();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
    static_cast<ActionDataAbstractTpl<Scalar>*>(d)->resize(this, true);
  }
  assert_pretty(
      MatrixXs(d->dyi_dx[0])
          .isApprox(MatrixXs::Identity(state_->get_ndx(), state_->get_ndx())),
      "you have changed dyi_dx[0] values that supposed to be constant.");
  assert_pretty(
      MatrixXs(d->dki_dx[0])
          .topRightCorner(nv, nv)
          .isApprox(MatrixXs::Identity(nv, nv)),
      "you have changed dki_dx[0] values that supposed to be constant.");

  if (dynamics_ != nullptr) {
    const std::size_t ng_d = dynamics_->get_ng();
    const std::size_t nh_d = dynamics_->get_nh();
    const std::size_t ng_c = constraints_ != NULL ? constraints_->get_ng() : 0;
    const std::size_t nh_c = constraints_ != NULL ? constraints_->get_nh() : 0;
    const std::size_t np = this->np_;
    const std::size_t np_action =
        params_ != nullptr ? params_->get_np_action() : 0;
    const std::size_t np_dynamics =
        params_ != nullptr ? params_->get_np_dynamics() : this->np_;
    const std::size_t np_cost = costs_->get_np();
    if (np_cost != 0 && np_cost != np) {
      throw_pretty(
          "Invalid argument: costs parameter dimension does not match "
          "RK integrated action parameter dimension");
    }
    for (std::size_t i = 0; i < ni_; ++i) {
      dynamics_->calcDiff(d->dynamics[i], d->y[i], d->ws[i]);
      costs_->calcDiff(d->costs[i], d->y[i], d->ws[i]);
    }

    const std::shared_ptr<DynamicsDataAbstract>& k0_data = d->dynamics[0];
    const std::shared_ptr<CostDataSum>& c0_data = d->costs[0];
    const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
        d->control[0];
    d->dki_dx[0].bottomRows(nv) = k0_data->Fx;
    d->dki_du[0].setZero();
    control_->multiplyByJacobian(
        u0_data, k0_data->Fu,
        d->dki_du[0].bottomRows(nv));  // dki_du = dki_dw * dw_du
    d->dki_dp[0].setZero();
    if (np_dynamics != 0) {
      d->dki_dp[0].bottomRows(nv).middleCols(np_action, np_dynamics) =
          k0_data->Fp.middleCols(np_action, np_dynamics);
    }

    d->dli_dx[0] = c0_data->Lx;
    control_->multiplyJacobianTransposeBy(
        u0_data, c0_data->Lu, d->dli_du[0]);  // dli_du = dli_dw * dw_du
    d->dli_dp[0].setZero();
    d->ddli_ddp[0].setZero();
    d->ddli_dpdx[0].setZero();
    d->ddli_dpdu[0].setZero();
    if (np_cost != 0) {
      d->dli_dp[0] = c0_data->Lp;
      d->ddli_ddp[0] = c0_data->Lpp;
      d->ddli_dpdx[0] = c0_data->Lpx;
      control_->multiplyByJacobian(u0_data, c0_data->Lpu, d->ddli_dpdu[0]);
    }

    d->ddli_ddx[0] = c0_data->Lxx;
    d->ddli_ddw[0] = c0_data->Luu;
    control_->multiplyByJacobian(
        u0_data, d->ddli_ddw[0],
        d->ddli_dwdu[0]);  // ddli_dwdu = ddli_ddw * dw_du
    control_->multiplyJacobianTransposeBy(
        u0_data, d->ddli_dwdu[0],
        d->ddli_ddu[0]);  // ddli_ddu = dw_du.T * ddli_dwdu
    d->ddli_dxdw[0] = c0_data->Lxu;
    control_->multiplyByJacobian(
        u0_data, d->ddli_dxdw[0],
        d->ddli_dxdu[0]);  // ddli_dxdu = ddli_dxdw * dw_du
    for (std::size_t i = 1; i < ni_; ++i) {
      const std::shared_ptr<DynamicsDataAbstract>& ki_data = d->dynamics[i];
      const std::shared_ptr<CostDataSum>& ci_data = d->costs[i];
      const std::shared_ptr<ControlParametrizationDataAbstract>& ui_data =
          d->control[i];
      d->dyi_dx[i].noalias() = d->dki_dx[i - 1] * rk_c_[i] * time_step_;
      d->dyi_du[i].noalias() = d->dki_du[i - 1] * rk_c_[i] * time_step_;
      d->dyi_dp[i].noalias() = d->dki_dp[i - 1] * rk_c_[i] * time_step_;
      if (integrator_time_->get_timeopt()) {
        for (std::vector<std::size_t>::const_iterator it =
                 d->timeopt_param_cols.begin();
             it != d->timeopt_param_cols.end(); ++it) {
          d->dyi_dp[i].col(static_cast<Eigen::Index>(*it)).noalias() +=
              d->ki[i - 1] * (rk_c_[i] * time_step_);
        }
      }
      state_->JintegrateTransport(x, d->dx_rk[i], d->dyi_dx[i], second);
      state_->Jintegrate(x, d->dx_rk[i], d->dyi_dx[i], d->dyi_dx[i], first,
                         addto);
      state_->JintegrateTransport(x, d->dx_rk[i], d->dyi_du[i], second);
      state_->JintegrateTransport(x, d->dx_rk[i], d->dyi_dp[i], second);

      Eigen::Block<MatrixXs> dkvi_dq = d->dki_dx[i].bottomLeftCorner(nv, nv);
      Eigen::Block<MatrixXs> dkvi_dv = d->dki_dx[i].bottomRightCorner(nv, nv);
      Eigen::Block<MatrixXs> dkqi_du = d->dki_du[i].topLeftCorner(nv, nu);
      Eigen::Block<MatrixXs> dkvi_du = d->dki_du[i].bottomLeftCorner(nv, nu);
      const Eigen::Block<MatrixXs> dki_dqi =
          ki_data->Fx.bottomLeftCorner(nv, nv);
      const Eigen::Block<MatrixXs> dki_dvi =
          ki_data->Fx.bottomRightCorner(nv, nv);
      const Eigen::Block<MatrixXs> dqi_dq = d->dyi_dx[i].topLeftCorner(nv, nv);
      const Eigen::Block<MatrixXs> dqi_dv = d->dyi_dx[i].topRightCorner(nv, nv);
      const Eigen::Block<MatrixXs> dvi_dq =
          d->dyi_dx[i].bottomLeftCorner(nv, nv);
      const Eigen::Block<MatrixXs> dvi_dv =
          d->dyi_dx[i].bottomRightCorner(nv, nv);
      const Eigen::Block<MatrixXs> dqi_du = d->dyi_du[i].topLeftCorner(nv, nu);
      const Eigen::Block<MatrixXs> dvi_du =
          d->dyi_du[i].bottomLeftCorner(nv, nu);

      d->dki_dx[i].topRows(nv) = d->dyi_dx[i].bottomRows(nv);
      dkvi_dq.noalias() = dki_dqi * dqi_dq;
      if (i == 1) {
        dkvi_dv = time_step_ / Scalar(2.) * dki_dqi;
      } else {
        dkvi_dv.noalias() = dki_dqi * dqi_dv;
      }
      dkvi_dq.noalias() += dki_dvi * dvi_dq;
      dkvi_dv.noalias() += dki_dvi * dvi_dv;

      dkqi_du = dvi_du;
      dkvi_du.noalias() = dki_dqi * dqi_du;
      dkvi_du.noalias() += dki_dvi * dvi_du;
      control_->multiplyByJacobian(ui_data, ki_data->Fu,
                                   d->dki_du[i].bottomRows(nv), addto);
      d->dki_dp[i].topRows(nv) = d->dyi_dp[i].bottomRows(nv);
      d->dki_dp[i].bottomRows(nv).noalias() = ki_data->Fx * d->dyi_dp[i];
      if (np_dynamics != 0) {
        d->dki_dp[i].bottomRows(nv).middleCols(np_action, np_dynamics) +=
            ki_data->Fp.middleCols(np_action, np_dynamics);
      }

      d->dli_dx[i].noalias() = ci_data->Lx.transpose() * d->dyi_dx[i];
      control_->multiplyJacobianTransposeBy(ui_data, ci_data->Lu, d->dli_du[i]);
      d->dli_du[i].noalias() += ci_data->Lx.transpose() * d->dyi_du[i];
      d->dli_dp[i].setZero();
      if (np_cost != 0) {
        d->dli_dp[i] = ci_data->Lp;
      }
      if (np != 0) {
        d->dli_dp[i].noalias() += d->dyi_dp[i].transpose() * ci_data->Lx;
      }

      d->Lxx_partialx[i].noalias() = ci_data->Lxx * d->dyi_dx[i];
      d->ddli_ddx[i].noalias() = d->dyi_dx[i].transpose() * d->Lxx_partialx[i];

      control_->multiplyByJacobian(ui_data, ci_data->Lxu, d->Lxu_i[i]);
      d->Luu_partialx[i].noalias() = d->Lxu_i[i].transpose() * d->dyi_du[i];
      d->Lxx_partialu[i].noalias() = ci_data->Lxx * d->dyi_du[i];
      control_->multiplyByJacobian(ui_data, ci_data->Luu, d->ddli_dwdu[i]);
      control_->multiplyJacobianTransposeBy(ui_data, d->ddli_dwdu[i],
                                            d->ddli_ddu[i]);
      d->ddli_ddu[i].noalias() += d->Luu_partialx[i].transpose() +
                                  d->Luu_partialx[i] +
                                  d->dyi_du[i].transpose() * d->Lxx_partialu[i];

      d->ddli_dxdw[i].noalias() = d->dyi_dx[i].transpose() * ci_data->Lxu;
      control_->multiplyByJacobian(ui_data, d->ddli_dxdw[i], d->ddli_dxdu[i]);
      d->ddli_dxdu[i].noalias() +=
          d->dyi_dx[i].transpose() * d->Lxx_partialu[i];
      d->ddli_ddp[i].setZero();
      d->ddli_dpdx[i].setZero();
      d->ddli_dpdu[i].setZero();
      if (np != 0) {
        d->tmp_ndx_np.noalias() = ci_data->Lxx * d->dyi_dp[i];
        if (np_cost != 0) {
          d->Lpp_partialx[i].noalias() = ci_data->Lpx * d->dyi_dp[i];
          d->ddli_ddp[i] = ci_data->Lpp;
          d->ddli_ddp[i] += d->Lpp_partialx[i].transpose();
          d->ddli_ddp[i] += d->Lpp_partialx[i];
          d->ddli_dpdx[i] = ci_data->Lpx;
          control_->multiplyByJacobian(ui_data, ci_data->Lpu, d->ddli_dpdu[i]);
        }
        d->ddli_ddp[i].noalias() += d->dyi_dp[i].transpose() * d->tmp_ndx_np;
        d->ddli_dpdx[i].noalias() +=
            d->dyi_dp[i].transpose() * d->Lxx_partialx[i];
        d->ddli_dpdu[i].noalias() +=
            d->dyi_dp[i].transpose() * d->Lxx_partialu[i];
      }
    }

    if (ni_ == 2) {
      d->Fx.noalias() = time_step_ * d->dki_dx[1];
      d->Fu.noalias() = time_step_ * d->dki_du[1];
      d->Lx.noalias() = time_step_ * d->dli_dx[1];
      d->Lu.noalias() = time_step_ * d->dli_du[1];
      d->Lxx.noalias() = time_step_ * d->ddli_ddx[1];
      d->Luu.noalias() = time_step_ * d->ddli_ddu[1];
      d->Lxu.noalias() = time_step_ * d->ddli_dxdu[1];
      d->Fp.noalias() = time_step_ * d->dki_dp[1];
      d->Lp.noalias() = time_step_ * d->dli_dp[1];
      d->Lpp.noalias() = time_step_ * d->ddli_ddp[1];
      d->Lpx.noalias() = time_step_ * d->ddli_dpdx[1];
      d->Lpu.noalias() = time_step_ * d->ddli_dpdu[1];
    } else if (ni_ == 3) {
      d->Fx.noalias() =
          time_step_ / Scalar(4.) * (d->dki_dx[0] + Scalar(3.) * d->dki_dx[2]);
      d->Fu.noalias() =
          time_step_ / Scalar(4.) * (d->dki_du[0] + Scalar(3.) * d->dki_du[2]);
      d->Lx.noalias() =
          time_step_ / Scalar(4.) * (d->dli_dx[0] + Scalar(3.) * d->dli_dx[2]);
      d->Lu.noalias() =
          time_step_ / Scalar(4.) * (d->dli_du[0] + Scalar(3.) * d->dli_du[2]);
      d->Lxx.noalias() = time_step_ / Scalar(4.) *
                         (d->ddli_ddx[0] + Scalar(3.) * d->ddli_ddx[2]);
      d->Luu.noalias() = time_step_ / Scalar(4.) *
                         (d->ddli_ddu[0] + Scalar(3.) * d->ddli_ddu[2]);
      d->Lxu.noalias() = time_step_ / Scalar(4.) *
                         (d->ddli_dxdu[0] + Scalar(3.) * d->ddli_dxdu[2]);
      d->Fp.noalias() =
          time_step_ / Scalar(4.) * (d->dki_dp[0] + Scalar(3.) * d->dki_dp[2]);
      d->Lp.noalias() =
          time_step_ / Scalar(4.) * (d->dli_dp[0] + Scalar(3.) * d->dli_dp[2]);
      d->Lpp.noalias() = time_step_ / Scalar(4.) *
                         (d->ddli_ddp[0] + Scalar(3.) * d->ddli_ddp[2]);
      d->Lpx.noalias() = time_step_ / Scalar(4.) *
                         (d->ddli_dpdx[0] + Scalar(3.) * d->ddli_dpdx[2]);
      d->Lpu.noalias() = time_step_ / Scalar(4.) *
                         (d->ddli_dpdu[0] + Scalar(3.) * d->ddli_dpdu[2]);
    } else {
      d->Fx.noalias() = time_step_ / Scalar(6.) *
                        (d->dki_dx[0] + Scalar(2.) * d->dki_dx[1] +
                         Scalar(2.) * d->dki_dx[2] + d->dki_dx[3]);
      d->Fu.noalias() = time_step_ / Scalar(6.) *
                        (d->dki_du[0] + Scalar(2.) * d->dki_du[1] +
                         Scalar(2.) * d->dki_du[2] + d->dki_du[3]);
      d->Lx.noalias() = time_step_ / Scalar(6.) *
                        (d->dli_dx[0] + Scalar(2.) * d->dli_dx[1] +
                         Scalar(2.) * d->dli_dx[2] + d->dli_dx[3]);
      d->Lu.noalias() = time_step_ / Scalar(6.) *
                        (d->dli_du[0] + Scalar(2.) * d->dli_du[1] +
                         Scalar(2.) * d->dli_du[2] + d->dli_du[3]);
      d->Lxx.noalias() = time_step_ / Scalar(6.) *
                         (d->ddli_ddx[0] + Scalar(2.) * d->ddli_ddx[1] +
                          Scalar(2.) * d->ddli_ddx[2] + d->ddli_ddx[3]);
      d->Luu.noalias() = time_step_ / Scalar(6.) *
                         (d->ddli_ddu[0] + Scalar(2.) * d->ddli_ddu[1] +
                          Scalar(2.) * d->ddli_ddu[2] + d->ddli_ddu[3]);
      d->Lxu.noalias() = time_step_ / Scalar(6.) *
                         (d->ddli_dxdu[0] + Scalar(2.) * d->ddli_dxdu[1] +
                          Scalar(2.) * d->ddli_dxdu[2] + d->ddli_dxdu[3]);
      d->Fp.noalias() = time_step_ / Scalar(6.) *
                        (d->dki_dp[0] + Scalar(2.) * d->dki_dp[1] +
                         Scalar(2.) * d->dki_dp[2] + d->dki_dp[3]);
      d->Lp.noalias() = time_step_ / Scalar(6.) *
                        (d->dli_dp[0] + Scalar(2.) * d->dli_dp[1] +
                         Scalar(2.) * d->dli_dp[2] + d->dli_dp[3]);
      d->Lpp.noalias() = time_step_ / Scalar(6.) *
                         (d->ddli_ddp[0] + Scalar(2.) * d->ddli_ddp[1] +
                          Scalar(2.) * d->ddli_ddp[2] + d->ddli_ddp[3]);
      d->Lpx.noalias() = time_step_ / Scalar(6.) *
                         (d->ddli_dpdx[0] + Scalar(2.) * d->ddli_dpdx[1] +
                          Scalar(2.) * d->ddli_dpdx[2] + d->ddli_dpdx[3]);
      d->Lpu.noalias() = time_step_ / Scalar(6.) *
                         (d->ddli_dpdu[0] + Scalar(2.) * d->ddli_dpdu[1] +
                          Scalar(2.) * d->ddli_dpdu[2] + d->ddli_dpdu[3]);
    }
    if (integrator_time_->get_timeopt() && !d->timeopt_param_cols.empty()) {
      for (std::vector<std::size_t>::const_iterator it =
               d->timeopt_param_cols.begin();
           it != d->timeopt_param_cols.end(); ++it) {
        const Eigen::Index col = static_cast<Eigen::Index>(*it);
        d->Fp.col(col).noalias() += d->dx;
        d->Lpp.row(col).noalias() += d->Lp.transpose();
        d->Lpp.col(col).noalias() += d->Lp;
        d->Lp(col) += d->cost;
        d->Lpp(col, col) += d->cost;
        d->Lpx.row(col).noalias() += d->Lx.transpose();
        d->Lpu.row(col).noalias() += d->Lu.transpose();
      }
    }

    d->Gx.setZero();
    d->Gu.setZero();
    d->Gp.setZero();
    d->Hx.setZero();
    d->Hu.setZero();
    d->Hp.setZero();
    if (constraints_ != NULL) {
      d->constraints->resize(constraints_.get());
      constraints_->calcDiff(d->constraints, x, d->ws[0]);
      if (ng_c != 0) {
        d->Gx.middleRows(ng_d, ng_c) = d->constraints->Gx;
        control_->multiplyByJacobian(u0_data, d->constraints->Gu,
                                     d->Gu.middleRows(ng_d, ng_c));
        if (constraints_->get_np() != 0) {
          d->Gp.middleRows(ng_d, ng_c) = d->constraints->Gp;
        }
      }
      if (nh_c != 0) {
        d->Hx.middleRows(nh_d, nh_c) = d->constraints->Hx;
        control_->multiplyByJacobian(u0_data, d->constraints->Hu,
                                     d->Hu.middleRows(nh_d, nh_c));
        if (constraints_->get_np() != 0) {
          d->Hp.middleRows(nh_d, nh_c) = d->constraints->Hp;
        }
      }
    }
    if (ng_d != 0) {
      d->Gx.topRows(ng_d) = k0_data->Gx;
      control_->multiplyByJacobian(u0_data, k0_data->Gu, d->Gu.topRows(ng_d));
      if (np_dynamics != 0) {
        d->Gp.topRows(ng_d).middleCols(np_action, np_dynamics) =
            k0_data->Gp.middleCols(np_action, np_dynamics);
      }
    }
    if (nh_d != 0) {
      d->Hx.topRows(nh_d) = k0_data->Hx;
      control_->multiplyByJacobian(u0_data, k0_data->Hu, d->Hu.topRows(nh_d));
      if (np_dynamics != 0) {
        d->Hp.topRows(nh_d).middleCols(np_action, np_dynamics) =
            k0_data->Hp.middleCols(np_action, np_dynamics);
      }
    }

    state_->JintegrateTransport(x, d->dx, d->Fx, second);
    state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
    state_->JintegrateTransport(x, d->dx, d->Fu, second);
    state_->JintegrateTransport(x, d->dx, d->Fp, second);
    return;
  }

  for (std::size_t i = 0; i < ni_; ++i) {
    differential_->calcDiff(d->differential[i], d->y[i], d->ws[i]);
  }

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
      d->control[0];
  d->dki_dx[0].bottomRows(nv) = k0_data->Fx;
  control_->multiplyByJacobian(
      u0_data, k0_data->Fu,
      d->dki_du[0].bottomRows(nv));  // dki_du = dki_dw * dw_du

  d->dli_dx[0] = k0_data->Lx;
  control_->multiplyJacobianTransposeBy(
      u0_data, k0_data->Lu,
      d->dli_du[0]);  // dli_du = dli_dw * dw_du

  d->ddli_ddx[0] = k0_data->Lxx;
  d->ddli_ddw[0] = k0_data->Luu;
  control_->multiplyByJacobian(
      u0_data, d->ddli_ddw[0],
      d->ddli_dwdu[0]);  // ddli_dwdu = ddli_ddw * dw_du
  control_->multiplyJacobianTransposeBy(
      u0_data, d->ddli_dwdu[0],
      d->ddli_ddu[0]);  // ddli_ddu = dw_du.T * ddli_dwdu
  d->ddli_dxdw[0] = k0_data->Lxu;
  control_->multiplyByJacobian(
      u0_data, d->ddli_dxdw[0],
      d->ddli_dxdu[0]);  // ddli_dxdu = ddli_dxdw * dw_du

  for (std::size_t i = 1; i < ni_; ++i) {
    const std::shared_ptr<DifferentialActionDataAbstract>& ki_data =
        d->differential[i];
    const std::shared_ptr<ControlParametrizationDataAbstract>& ui_data =
        d->control[i];
    d->dyi_dx[i].noalias() = d->dki_dx[i - 1] * rk_c_[i] * time_step_;
    d->dyi_du[i].noalias() = d->dki_du[i - 1] * rk_c_[i] * time_step_;
    state_->JintegrateTransport(x, d->dx_rk[i], d->dyi_dx[i], second);
    state_->Jintegrate(x, d->dx_rk[i], d->dyi_dx[i], d->dyi_dx[i], first,
                       addto);
    state_->JintegrateTransport(x, d->dx_rk[i], d->dyi_du[i],
                                second);  // dyi_du = Jintegrate * dyi_du

    // Sparse matrix-matrix multiplication for computing:
    Eigen::Block<MatrixXs> dkvi_dq = d->dki_dx[i].bottomLeftCorner(nv, nv);
    Eigen::Block<MatrixXs> dkvi_dv = d->dki_dx[i].bottomRightCorner(nv, nv);
    Eigen::Block<MatrixXs> dkqi_du = d->dki_du[i].topLeftCorner(nv, nu);
    Eigen::Block<MatrixXs> dkvi_du = d->dki_du[i].bottomLeftCorner(nv, nu);
    const Eigen::Block<MatrixXs> dki_dqi = ki_data->Fx.bottomLeftCorner(nv, nv);
    const Eigen::Block<MatrixXs> dki_dvi =
        ki_data->Fx.bottomRightCorner(nv, nv);
    const Eigen::Block<MatrixXs> dqi_dq = d->dyi_dx[i].topLeftCorner(nv, nv);
    const Eigen::Block<MatrixXs> dqi_dv = d->dyi_dx[i].topRightCorner(nv, nv);
    const Eigen::Block<MatrixXs> dvi_dq = d->dyi_dx[i].bottomLeftCorner(nv, nv);
    const Eigen::Block<MatrixXs> dvi_dv =
        d->dyi_dx[i].bottomRightCorner(nv, nv);
    const Eigen::Block<MatrixXs> dqi_du = d->dyi_du[i].topLeftCorner(nv, nu);
    const Eigen::Block<MatrixXs> dvi_du = d->dyi_du[i].bottomLeftCorner(nv, nu);
    //   i. d->dki_dx[i].noalias() = d->dki_dy[i] * d->dyi_dx[i], where dki_dy
    //   is ki_data.Fx
    d->dki_dx[i].topRows(nv) = d->dyi_dx[i].bottomRows(nv);
    dkvi_dq.noalias() = dki_dqi * dqi_dq;
    if (i == 1) {
      dkvi_dv = time_step_ / Scalar(2.) * dki_dqi;
    } else {
      dkvi_dv.noalias() = dki_dqi * dqi_dv;
    }
    dkvi_dq.noalias() += dki_dvi * dvi_dq;
    dkvi_dv.noalias() += dki_dvi * dvi_dv;
    //  ii. d->dki_du[i].noalias() = d->dki_dy[i] * d->dyi_du[i], where dki_dy
    //  is ki_data.Fx
    dkqi_du = dvi_du;
    dkvi_du.noalias() = dki_dqi * dqi_du;
    dkvi_du.noalias() += dki_dvi * dvi_du;

    control_->multiplyByJacobian(ui_data, ki_data->Fu,
                                 d->dki_du[i].bottomRows(nv),
                                 addto);  // dfi_du = dki_dw * dw_du

    d->dli_dx[i].noalias() = ki_data->Lx.transpose() * d->dyi_dx[i];
    control_->multiplyJacobianTransposeBy(ui_data, ki_data->Lu,
                                          d->dli_du[i]);  // dli_du = Lu * dw_du
    d->dli_du[i].noalias() += ki_data->Lx.transpose() * d->dyi_du[i];

    d->Lxx_partialx[i].noalias() = ki_data->Lxx * d->dyi_dx[i];
    d->ddli_ddx[i].noalias() = d->dyi_dx[i].transpose() * d->Lxx_partialx[i];

    control_->multiplyByJacobian(ui_data, ki_data->Lxu,
                                 d->Lxu_i[i]);  // Lxu = Lxw * dw_du
    d->Luu_partialx[i].noalias() = d->Lxu_i[i].transpose() * d->dyi_du[i];
    d->Lxx_partialu[i].noalias() = ki_data->Lxx * d->dyi_du[i];
    control_->multiplyByJacobian(
        ui_data, ki_data->Luu,
        d->ddli_dwdu[i]);  // ddli_dwdu = ddli_ddw * dw_du
    control_->multiplyJacobianTransposeBy(
        ui_data, d->ddli_dwdu[i],
        d->ddli_ddu[i]);  // ddli_ddu = dw_du.T * ddli_dwdu
    d->ddli_ddu[i].noalias() += d->Luu_partialx[i].transpose() +
                                d->Luu_partialx[i] +
                                d->dyi_du[i].transpose() * d->Lxx_partialu[i];

    d->ddli_dxdw[i].noalias() = d->dyi_dx[i].transpose() * ki_data->Lxu;
    control_->multiplyByJacobian(
        ui_data, d->ddli_dxdw[i],
        d->ddli_dxdu[i]);  // ddli_dxdu = ddli_dxdw * dw_du
    d->ddli_dxdu[i].noalias() += d->dyi_dx[i].transpose() * d->Lxx_partialu[i];
  }

  if (ni_ == 2) {
    d->Fx.noalias() = time_step_ * d->dki_dx[1];
    d->Fu.noalias() = time_step_ * d->dki_du[1];
    d->Lx.noalias() = time_step_ * d->dli_dx[1];
    d->Lu.noalias() = time_step_ * d->dli_du[1];
    d->Lxx.noalias() = time_step_ * d->ddli_ddx[1];
    d->Luu.noalias() = time_step_ * d->ddli_ddu[1];
    d->Lxu.noalias() = time_step_ * d->ddli_dxdu[1];
  } else if (ni_ == 3) {
    d->Fx.noalias() =
        time_step_ / Scalar(4.) * (d->dki_dx[0] + Scalar(3.) * d->dki_dx[2]);
    d->Fu.noalias() =
        time_step_ / Scalar(4.) * (d->dki_du[0] + Scalar(3.) * d->dki_du[2]);
    d->Lx.noalias() =
        time_step_ / Scalar(4.) * (d->dli_dx[0] + Scalar(3.) * d->dli_dx[2]);
    d->Lu.noalias() =
        time_step_ / Scalar(4.) * (d->dli_du[0] + Scalar(3.) * d->dli_du[2]);
    d->Lxx.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_ddx[0] + Scalar(3.) * d->ddli_ddx[2]);
    d->Luu.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_ddu[0] + Scalar(3.) * d->ddli_ddu[2]);
    d->Lxu.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_dxdu[0] + Scalar(3.) * d->ddli_dxdu[2]);
  } else {
    d->Fx.noalias() = time_step_ / Scalar(6.) *
                      (d->dki_dx[0] + Scalar(2.) * d->dki_dx[1] +
                       Scalar(2.) * d->dki_dx[2] + d->dki_dx[3]);
    d->Fu.noalias() = time_step_ / Scalar(6.) *
                      (d->dki_du[0] + Scalar(2.) * d->dki_du[1] +
                       Scalar(2.) * d->dki_du[2] + d->dki_du[3]);
    d->Lx.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_dx[0] + Scalar(2.) * d->dli_dx[1] +
                       Scalar(2.) * d->dli_dx[2] + d->dli_dx[3]);
    d->Lu.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_du[0] + Scalar(2.) * d->dli_du[1] +
                       Scalar(2.) * d->dli_du[2] + d->dli_du[3]);
    d->Lxx.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_ddx[0] + Scalar(2.) * d->ddli_ddx[1] +
                        Scalar(2.) * d->ddli_ddx[2] + d->ddli_ddx[3]);
    d->Luu.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_ddu[0] + Scalar(2.) * d->ddli_ddu[1] +
                        Scalar(2.) * d->ddli_ddu[2] + d->ddli_ddu[3]);
    d->Lxu.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_dxdu[0] + Scalar(2.) * d->ddli_dxdu[1] +
                        Scalar(2.) * d->ddli_dxdu[2] + d->ddli_dxdu[3]);
  }
  d->Gx = k0_data->Gx;
  d->Hx = k0_data->Hx;
  d->Gu.setZero();
  d->Hu.setZero();
  control_->multiplyByJacobian(u0_data, k0_data->Gu, d->Gu);
  control_->multiplyByJacobian(u0_data, k0_data->Hu, d->Hu);

  state_->JintegrateTransport(x, d->dx, d->Fx, second);
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
  state_->JintegrateTransport(x, d->dx, d->Fu, second);
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
  }
  refresh_integrator_time();
  Data* d = static_cast<Data*>(data.get());
  if (dynamics_ != nullptr) {
    this->nr_ = costs_->get_nr();
  }

  if (dynamics_ != nullptr) {
    const std::size_t ng_T =
        constraints_ != NULL ? constraints_->get_ng_T() : 0;
    const std::size_t nh_T =
        constraints_ != NULL ? constraints_->get_nh_T() : 0;
    const std::shared_ptr<DynamicsDataAbstract>& k0_data = d->dynamics[0];
    const std::shared_ptr<CostDataSum>& c0_data = d->costs[0];

    dynamics_->calcDiff(k0_data, x);
    costs_->calcDiff(c0_data, x);
    d->Fx.setZero();
    state_->Jintegrate(x, d->dx, d->Fx, d->Fx);
    d->Lx = c0_data->Lx;
    d->Lxx = c0_data->Lxx;
    d->Fp.setZero();
    if (this->np_ != 0 && costs_->get_np() != 0) {
      if (costs_->get_np() != this->np_) {
        throw_pretty(
            "Invalid argument: costs parameter dimension does not match "
            "RK integrated action parameter dimension");
      }
      d->Lp = c0_data->Lp;
      d->Lpp = c0_data->Lpp;
      d->Lpx = c0_data->Lpx;
    } else {
      d->Lp.setZero();
      d->Lpp.setZero();
      d->Lpx.setZero();
    }
    d->Gx.setZero();
    d->Gp.setZero();
    d->Hx.setZero();
    d->Hp.setZero();
    if (constraints_ != NULL) {
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
    return;
  }

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  differential_->calcDiff(k0_data, x);
  d->Lx = k0_data->Lx;
  d->Lxx = k0_data->Lxx;
  d->Gx = k0_data->Gx;
  d->Hx = k0_data->Hx;
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (dynamics_ == nullptr) {
    throw_pretty(
        "Invalid call: RK integrated actions built from a differential "
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
          "Invalid argument: RK supports only an active integration-time "
          "action parameter");
    }
    if (time_param->get_integrator_time() != integrator_time_) {
      throw_pretty(
          "Invalid argument: integration-time parameter must share the RK "
          "integrator time");
    }
    if (++active_time_params > 1) {
      throw_pretty(
          "Invalid argument: RK supports only one active integration-time "
          "parameter");
    }
  }

  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty("Invalid argument: data has the wrong RK runtime type");
  }
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
  }
  params_ = params;
  this->np_ = params_->get_np();
  d->timeopt_param_cols.clear();
  if (integrator_time_->get_timeopt() && params_->get_np_action() != 0) {
    d->timeopt_param_cols.reserve(params_->get_np_action());
    std::size_t offset = 0;
    for (it = params_->get_action_params().begin(),
        end = params_->get_action_params().end();
         it != end; ++it) {
      const std::shared_ptr<typename ParameterManager::ParameterItem>& item =
          it->second;
      if (!item->get_active()) {
        continue;
      }
      const std::size_t np_item = item->get_param()->get_np();
      if (std::dynamic_pointer_cast<IntegratorTimeoptParamsTpl<Scalar> >(
              item->get_param()) != nullptr) {
        for (std::size_t ip = 0; ip < np_item; ++ip) {
          d->timeopt_param_cols.push_back(offset + ip);
        }
      }
      offset += np_item;
    }
  }
  if (constraints_ != nullptr && constraints_->get_np() != 0 &&
      constraints_->get_np() != this->np_) {
    throw_pretty(
        "Invalid argument: constraints parameter dimension does not match "
        "RK integrated action parameter dimension");
  }
  d->resize(this);
  if (d->params == nullptr) {
    d->params = params_->createData();
  } else {
    d->params->resize(params_.get());
  }
  for (std::size_t i = 0; i < ni_; ++i) {
    dynamics_->set_params(d->dynamics[i], params_);
    d->costs[i] = costs_->createData(d->dynamics[i]->shared);
  }
  if (constraints_ != NULL) {
    d->constraints = constraints_->createData(d->dynamics[0]->shared);
  } else {
    d->constraints.reset();
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (dynamics_ == nullptr) {
    throw_pretty(
        "Invalid call: RK integrated actions built from a differential "
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
    throw_pretty("Invalid argument: data has the wrong RK runtime type");
  }
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
  }
  if (d->params == nullptr) {
    throw_pretty(
        "Invalid argument: RK integrated action data has no "
        "parameter-manager payload");
  }
  params_->update(d->params, p);
  for (std::size_t i = 0; i < ni_; ++i) {
    dynamics_->update_p(d->dynamics[i], p);
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelRKTpl<Scalar>::createData() {
  if (params_ == nullptr) {
    return createData(std::shared_ptr<ParameterDataManager>());
  }
  const std::shared_ptr<ParameterDataManager> params_data =
      params_->createData();
  return createData(params_data);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelRKTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
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
IntegratedActionModelRKTpl<NewScalar> IntegratedActionModelRKTpl<Scalar>::cast()
    const {
  typedef IntegratedActionModelRKTpl<NewScalar> ReturnType;
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
        control_->template cast<NewScalar>(), casted_time, rk_type_);
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
                   control_->template cast<NewScalar>(), rk_type_,
                   scalar_cast<NewScalar>(integrator_time_->get_time_step()),
                   with_cost_residual_);
    return ret;
  } else {
    ReturnType ret(differential_->template cast<NewScalar>(), rk_type_,
                   scalar_cast<NewScalar>(integrator_time_->get_time_step()),
                   with_cost_residual_);
    return ret;
  }
}

template <typename Scalar>
bool IntegratedActionModelRKTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    return false;
  }
  std::size_t expected_ni = 0;
  switch (rk_type_) {
    case two:
      expected_ni = 2;
      break;
    case three:
      expected_ni = 3;
      break;
    case four:
      expected_ni = 4;
      break;
    default:
      return false;
  }
  if (ni_ != expected_ni || rk_c_.size() != ni_ || d->integral.size() != ni_ ||
      d->ki.size() != ni_ || d->y.size() != ni_ || d->ws.size() != ni_ ||
      d->dx_rk.size() != ni_ || d->dki_dx.size() != ni_ ||
      d->dki_du.size() != ni_ || d->dki_dp.size() != ni_ ||
      d->dyi_dx.size() != ni_ || d->dyi_du.size() != ni_ ||
      d->dyi_dp.size() != ni_ || d->dli_dx.size() != ni_ ||
      d->dli_du.size() != ni_ || d->dli_dp.size() != ni_ ||
      d->ddli_ddx.size() != ni_ || d->ddli_ddw.size() != ni_ ||
      d->ddli_ddu.size() != ni_ || d->ddli_ddp.size() != ni_ ||
      d->ddli_dxdw.size() != ni_ || d->ddli_dxdu.size() != ni_ ||
      d->ddli_dpdx.size() != ni_ || d->ddli_dpdu.size() != ni_ ||
      d->ddli_dwdu.size() != ni_ || d->Luu_partialx.size() != ni_ ||
      d->Lpp_partialx.size() != ni_ || d->Lxu_i.size() != ni_ ||
      d->Lxx_partialx.size() != ni_ || d->Lxx_partialu.size() != ni_ ||
      d->control.size() != ni_) {
    return false;
  }
  for (std::size_t i = 0; i < ni_; ++i) {
    if (d->control[i] == nullptr) {
      return false;
    }
  }
  if (dynamics_ != nullptr) {
    if (d->dynamics.size() != ni_ || d->costs.size() != ni_ ||
        (constraints_ != nullptr && d->constraints == nullptr)) {
      return false;
    }
    for (std::size_t i = 0; i < ni_; ++i) {
      if (d->dynamics[i] == nullptr || d->costs[i] == nullptr ||
          !dynamics_->checkData(d->dynamics[i])) {
        return false;
      }
    }
    return true;
  }
  if (d->differential.size() != ni_) {
    return false;
  }
  for (std::size_t i = 0; i < ni_; ++i) {
    if (d->differential[i] == nullptr ||
        !differential_->checkData(d->differential[i])) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::quasiStatic(
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
  if (!checkData(data)) {
    throw_pretty("Invalid argument: data is incompatible with this RK model");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
      d->control[0];
  u0_data->w *= Scalar(0.);
  if (dynamics_ != nullptr) {
    dynamics_->quasiStatic(d->dynamics[0], u0_data->w, x, maxiter, tol);
  } else {
    differential_->quasiStatic(d->differential[0], u0_data->w, x, maxiter, tol);
  }
  control_->params(u0_data, Scalar(0.), u0_data->w);
  u = u0_data->u;
}

template <typename Scalar>
std::size_t IntegratedActionModelRKTpl<Scalar>::get_ni() const {
  return ni_;
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::print(std::ostream& os) const {
  if (dynamics_ != nullptr) {
    os << "IntegratedActionModelRK {dt=" << integrator_time_->get_time_step()
       << ", " << *dynamics_ << "}";
  } else {
    os << "IntegratedActionModelRK {dt=" << integrator_time_->get_time_step()
       << ", " << *differential_ << "}";
  }
}

template <typename Scalar>
void IntegratedActionModelRKTpl<Scalar>::set_rk_type(const RKType rktype) {
  switch (rktype) {
    case two:
      ni_ = 2;
      rk_c_.resize(ni_);
      rk_c_[0] = Scalar(0.);
      rk_c_[1] = Scalar(0.5);
      break;
    case three:
      ni_ = 3;
      rk_c_.resize(ni_);
      rk_c_[0] = Scalar(0.);
      rk_c_[1] = Scalar(1. / 3.);
      rk_c_[2] = Scalar(2. / 3.);
      break;
    case four:
      ni_ = 4;
      rk_c_.resize(ni_);
      rk_c_[0] = Scalar(0.);
      rk_c_[1] = Scalar(0.5);
      rk_c_[2] = Scalar(0.5);
      rk_c_[3] = Scalar(1.);
      break;
  }
}

}  // namespace crocoddyl
