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
IntegratedObserverModelRKTpl<Scalar>::IntegratedObserverModelRKTpl(
    std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints, const Scalar time_step,
    const RKType rktype)
    : Base(dynamics, costs, constraints, time_step), rk_type_(rktype) {
  set_rk_type(rktype);
}

template <typename Scalar>
void IntegratedObserverModelRKTpl<Scalar>::calc(
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
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();
  const std::size_t ni1 = ni_ - 1;
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      dyn_u = u.tail(dynamics_nu);

  d->y[0] = x;
  for (std::size_t i = 0; i + 1 < ni_; ++i) {
    if (dynamics_nu != 0u) {
      dynamics_->calc(d->dynamics_stage[i], d->y[i], dyn_u);
    } else {
      dynamics_->calc(d->dynamics_stage[i], d->y[i], u_zero_);
    }
    costs_->calc(d->costs_stage[i], d->y[i], u);
    d->ki[i].head(nv) = d->y[i].tail(nv);
    d->ki[i].tail(nv) = d->dynamics_stage[i]->vdot;
    d->integral[i] = d->costs_stage[i]->cost;
    d->dx_rk[i + 1].noalias() = d->ki[i] * (rk_c_[i + 1] * time_step_);
    state_->integrate(x, d->dx_rk[i + 1], d->y[i + 1]);
  }

  if (dynamics_nu != 0u) {
    dynamics_->calc(d->dynamics_stage[ni1], d->y[ni1], dyn_u);
  } else {
    dynamics_->calc(d->dynamics_stage[ni1], d->y[ni1], u_zero_);
  }
  costs_->calc(d->costs_stage[ni1], d->y[ni1], u);
  d->ki[ni1].head(nv) = d->y[ni1].tail(nv);
  d->ki[ni1].tail(nv) = d->dynamics_stage[ni1]->vdot;
  d->integral[ni1] = d->costs_stage[ni1]->cost;

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
  d->dx.noalias() += this->compute_projected_noise(d, x, u.head(ndx));
  state_->integrate(x, d->dx, d->xnext);

  d->dissipative_E.setZero();
  d->r.setZero();
  d->g.setZero();
  d->h.setZero();

  if (ng_d != 0u) {
    d->g.head(ng_d) = d->dynamics_stage[0]->g;
  }
  if (nh_d != 0u) {
    d->h.head(nh_d) = d->dynamics_stage[0]->h;
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
void IntegratedObserverModelRKTpl<Scalar>::calc(
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
  dynamics_->calc(d->dynamics_stage[0], x);
  costs_->calc(d->costs_stage[0], x);

  d->xnext = x;
  d->dx.setZero();
  d->dissipative_E.setZero();
  d->cost = d->costs_stage[0]->cost;
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
void IntegratedObserverModelRKTpl<Scalar>::calcDiff(
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
  const std::size_t np = this->get_np();
  const std::size_t ng_d = dynamics_->get_ng();
  const std::size_t nh_d = dynamics_->get_nh();

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      dyn_u = u.tail(dynamics_nu);

  for (std::size_t i = 0; i < ni_; ++i) {
    if (dynamics_nu != 0u) {
      dynamics_->calcDiff(d->dynamics_stage[i], d->y[i], dyn_u);
    } else {
      dynamics_->calcDiff(d->dynamics_stage[i], d->y[i], u_zero_);
    }
    costs_->calcDiff(d->costs_stage[i], d->y[i], u);
  }

  d->dy_dx[0].setIdentity();
  d->dy_du[0].setZero();
  d->dy_dp[0].setZero();
  d->dki_dx[0].setZero();
  d->dki_dx[0].topRightCorner(nv, nv).diagonal().setOnes();
  d->dki_dx[0].bottomRows(nv) = d->dynamics_stage[0]->Fx;
  d->dki_du[0].setZero();
  if (dynamics_nu != 0u) {
    d->dki_du[0].block(nv, ndx, nv, dynamics_nu) = d->dynamics_stage[0]->Fu;
  }
  d->dki_dp[0].setZero();
  if (np != 0u) {
    d->dki_dp[0].bottomRows(nv) = d->dynamics_stage[0]->Fp;
  }
  d->dli_dx[0] = d->costs_stage[0]->Lx;
  d->dli_du[0] = d->costs_stage[0]->Lu;
  d->dli_dp[0].setZero();
  if (static_cast<std::size_t>(d->costs_stage[0]->Lp.size()) == np) {
    d->dli_dp[0] = d->costs_stage[0]->Lp;
  }
  d->ddli_ddx[0] = d->costs_stage[0]->Lxx;
  d->ddli_ddu[0] = d->costs_stage[0]->Luu;
  d->ddli_ddp[0].setZero();
  if (static_cast<std::size_t>(d->costs_stage[0]->Lpp.rows()) == np &&
      static_cast<std::size_t>(d->costs_stage[0]->Lpp.cols()) == np) {
    d->ddli_ddp[0] = d->costs_stage[0]->Lpp;
  }
  d->ddli_dxdu[0] = d->costs_stage[0]->Lxu;
  d->ddli_dpdx[0].setZero();
  if (static_cast<std::size_t>(d->costs_stage[0]->Lpx.rows()) == np &&
      static_cast<std::size_t>(d->costs_stage[0]->Lpx.cols()) == ndx) {
    d->ddli_dpdx[0] = d->costs_stage[0]->Lpx;
  }
  d->ddli_dpdu[0].setZero();
  if (static_cast<std::size_t>(d->costs_stage[0]->Lpu.rows()) == np &&
      static_cast<std::size_t>(d->costs_stage[0]->Lpu.cols()) ==
          this->get_nu()) {
    d->ddli_dpdu[0] = d->costs_stage[0]->Lpu;
  }

  for (std::size_t i = 1; i < ni_; ++i) {
    const Scalar c = rk_c_[i] * time_step_;
    d->dy_dx[i].noalias() = c * d->dki_dx[i - 1];
    d->dy_du[i].noalias() = c * d->dki_du[i - 1];
    d->dy_dp[i].noalias() = c * d->dki_dp[i - 1];
    state_->JintegrateTransport(x, d->dx_rk[i], d->dy_dx[i], second);
    state_->Jintegrate(x, d->dx_rk[i], d->dy_dx[i], d->dy_dx[i], first, addto);
    state_->JintegrateTransport(x, d->dx_rk[i], d->dy_du[i], second);
    state_->JintegrateTransport(x, d->dx_rk[i], d->dy_dp[i], second);

    d->dki_dx[i].topRows(nv) = d->dy_dx[i].bottomRows(nv);
    d->dki_dx[i].bottomRows(nv).noalias() =
        d->dynamics_stage[i]->Fx * d->dy_dx[i];
    d->dki_du[i].topRows(nv) = d->dy_du[i].bottomRows(nv);
    d->dki_du[i].bottomRows(nv).noalias() =
        d->dynamics_stage[i]->Fx * d->dy_du[i];
    if (dynamics_nu != 0u) {
      d->dki_du[i].block(nv, ndx, nv, dynamics_nu) += d->dynamics_stage[i]->Fu;
    }
    d->dki_dp[i].topRows(nv) = d->dy_dp[i].bottomRows(nv);
    d->dki_dp[i].bottomRows(nv).noalias() =
        d->dynamics_stage[i]->Fx * d->dy_dp[i];
    if (np != 0u) {
      d->dki_dp[i].bottomRows(nv) += d->dynamics_stage[i]->Fp;
    }

    d->dli_dx[i].noalias() = d->dy_dx[i].transpose() * d->costs_stage[i]->Lx;
    d->dli_du[i] = d->costs_stage[i]->Lu;
    d->dli_du[i].noalias() += d->dy_du[i].transpose() * d->costs_stage[i]->Lx;
    d->dli_dp[i].setZero();
    if (static_cast<std::size_t>(d->costs_stage[i]->Lp.size()) == np) {
      d->dli_dp[i] = d->costs_stage[i]->Lp;
    }
    if (np != 0u) {
      d->dli_dp[i].noalias() += d->dy_dp[i].transpose() * d->costs_stage[i]->Lx;
    }

    d->tmp_ndx_ndx.noalias() = d->costs_stage[i]->Lxx * d->dy_dx[i];
    d->ddli_ddx[i].noalias() = d->dy_dx[i].transpose() * d->tmp_ndx_ndx;
    d->tmp_ndx_nu.noalias() = d->costs_stage[i]->Lxx * d->dy_du[i];
    d->ddli_dxdu[i].noalias() =
        d->dy_dx[i].transpose() * (d->costs_stage[i]->Lxu + d->tmp_ndx_nu);
    d->Luu_partialx[i].noalias() =
        d->costs_stage[i]->Lxu.transpose() * d->dy_du[i];
    d->ddli_ddu[i] = d->costs_stage[i]->Luu;
    d->ddli_ddu[i] += d->Luu_partialx[i].transpose();
    d->ddli_ddu[i] += d->Luu_partialx[i];
    d->ddli_ddu[i].noalias() += d->dy_du[i].transpose() * d->tmp_ndx_nu;

    if (np != 0u) {
      d->tmp_ndx_np.noalias() = d->costs_stage[i]->Lxx * d->dy_dp[i];
      if (static_cast<std::size_t>(d->costs_stage[i]->Lpx.rows()) == np &&
          static_cast<std::size_t>(d->costs_stage[i]->Lpx.cols()) == ndx) {
        d->Lpp_partialx[i].noalias() = d->costs_stage[i]->Lpx * d->dy_dp[i];
      } else {
        d->Lpp_partialx[i].setZero();
      }
      d->ddli_ddp[i].setZero();
      if (static_cast<std::size_t>(d->costs_stage[i]->Lpp.rows()) == np &&
          static_cast<std::size_t>(d->costs_stage[i]->Lpp.cols()) == np) {
        d->ddli_ddp[i] = d->costs_stage[i]->Lpp;
      }
      d->ddli_ddp[i] += d->Lpp_partialx[i].transpose();
      d->ddli_ddp[i] += d->Lpp_partialx[i];
      d->ddli_ddp[i].noalias() += d->dy_dp[i].transpose() * d->tmp_ndx_np;
      d->ddli_dpdx[i].setZero();
      if (static_cast<std::size_t>(d->costs_stage[i]->Lpx.rows()) == np &&
          static_cast<std::size_t>(d->costs_stage[i]->Lpx.cols()) == ndx) {
        d->ddli_dpdx[i] = d->costs_stage[i]->Lpx;
      }
      d->ddli_dpdx[i].noalias() += d->dy_dp[i].transpose() * d->tmp_ndx_ndx;
      d->ddli_dpdu[i].setZero();
      if (static_cast<std::size_t>(d->costs_stage[i]->Lpu.rows()) == np &&
          static_cast<std::size_t>(d->costs_stage[i]->Lpu.cols()) ==
              this->get_nu()) {
        d->ddli_dpdu[i] = d->costs_stage[i]->Lpu;
      }
      d->ddli_dpdu[i].noalias() += d->dy_dp[i].transpose() * d->tmp_ndx_nu;
    } else {
      d->ddli_ddp[i].setZero();
      d->ddli_dpdx[i].setZero();
      d->ddli_dpdu[i].setZero();
    }
  }

  if (ni_ == 2) {
    d->ddx_dx.noalias() = time_step_ * d->dki_dx[1];
    d->ddx_du.noalias() = time_step_ * d->dki_du[1];
    d->ddx_dp.noalias() = time_step_ * d->dki_dp[1];
    d->Lx.noalias() = time_step_ * d->dli_dx[1];
    d->Lu.noalias() = time_step_ * d->dli_du[1];
    d->Lp.noalias() = time_step_ * d->dli_dp[1];
    d->Lxx.noalias() = time_step_ * d->ddli_ddx[1];
    d->Luu.noalias() = time_step_ * d->ddli_ddu[1];
    d->Lxu.noalias() = time_step_ * d->ddli_dxdu[1];
    d->Lpp.noalias() = time_step_ * d->ddli_ddp[1];
    d->Lpx.noalias() = time_step_ * d->ddli_dpdx[1];
    d->Lpu.noalias() = time_step_ * d->ddli_dpdu[1];
  } else if (ni_ == 3) {
    d->ddx_dx.noalias() =
        time_step_ / Scalar(4.) * (d->dki_dx[0] + Scalar(3.) * d->dki_dx[2]);
    d->ddx_du.noalias() =
        time_step_ / Scalar(4.) * (d->dki_du[0] + Scalar(3.) * d->dki_du[2]);
    d->ddx_dp.noalias() =
        time_step_ / Scalar(4.) * (d->dki_dp[0] + Scalar(3.) * d->dki_dp[2]);
    d->Lx.noalias() =
        time_step_ / Scalar(4.) * (d->dli_dx[0] + Scalar(3.) * d->dli_dx[2]);
    d->Lu.noalias() =
        time_step_ / Scalar(4.) * (d->dli_du[0] + Scalar(3.) * d->dli_du[2]);
    d->Lp.noalias() =
        time_step_ / Scalar(4.) * (d->dli_dp[0] + Scalar(3.) * d->dli_dp[2]);
    d->Lxx.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_ddx[0] + Scalar(3.) * d->ddli_ddx[2]);
    d->Luu.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_ddu[0] + Scalar(3.) * d->ddli_ddu[2]);
    d->Lxu.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_dxdu[0] + Scalar(3.) * d->ddli_dxdu[2]);
    d->Lpp.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_ddp[0] + Scalar(3.) * d->ddli_ddp[2]);
    d->Lpx.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_dpdx[0] + Scalar(3.) * d->ddli_dpdx[2]);
    d->Lpu.noalias() = time_step_ / Scalar(4.) *
                       (d->ddli_dpdu[0] + Scalar(3.) * d->ddli_dpdu[2]);
  } else {
    d->ddx_dx.noalias() = time_step_ / Scalar(6.) *
                          (d->dki_dx[0] + Scalar(2.) * d->dki_dx[1] +
                           Scalar(2.) * d->dki_dx[2] + d->dki_dx[3]);
    d->ddx_du.noalias() = time_step_ / Scalar(6.) *
                          (d->dki_du[0] + Scalar(2.) * d->dki_du[1] +
                           Scalar(2.) * d->dki_du[2] + d->dki_du[3]);
    d->ddx_dp.noalias() = time_step_ / Scalar(6.) *
                          (d->dki_dp[0] + Scalar(2.) * d->dki_dp[1] +
                           Scalar(2.) * d->dki_dp[2] + d->dki_dp[3]);
    d->Lx.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_dx[0] + Scalar(2.) * d->dli_dx[1] +
                       Scalar(2.) * d->dli_dx[2] + d->dli_dx[3]);
    d->Lu.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_du[0] + Scalar(2.) * d->dli_du[1] +
                       Scalar(2.) * d->dli_du[2] + d->dli_du[3]);
    d->Lp.noalias() = time_step_ / Scalar(6.) *
                      (d->dli_dp[0] + Scalar(2.) * d->dli_dp[1] +
                       Scalar(2.) * d->dli_dp[2] + d->dli_dp[3]);
    d->Lxx.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_ddx[0] + Scalar(2.) * d->ddli_ddx[1] +
                        Scalar(2.) * d->ddli_ddx[2] + d->ddli_ddx[3]);
    d->Luu.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_ddu[0] + Scalar(2.) * d->ddli_ddu[1] +
                        Scalar(2.) * d->ddli_ddu[2] + d->ddli_ddu[3]);
    d->Lxu.noalias() = time_step_ / Scalar(6.) *
                       (d->ddli_dxdu[0] + Scalar(2.) * d->ddli_dxdu[1] +
                        Scalar(2.) * d->ddli_dxdu[2] + d->ddli_dxdu[3]);
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

  state_->Jintegrate(x, d->dx, d->Jfirst, d->Jsecond);
  d->Fx.noalias() = d->Jsecond * d->ddx_dx;
  d->Fx += d->Jfirst;
  d->Fx.noalias() += this->compute_projected_noise_jacobian(d, x, u.head(ndx));
  d->Fp.noalias() = d->Jsecond * d->ddx_dp;
  d->Fu.noalias() = d->Jsecond * d->ddx_du;
  d->Fu.leftCols(ndx).noalias() = d->Jsecond * d->noise_projector;

  d->dE_dv.setZero();
  d->dE_dp.setZero();

  d->Gx.setZero();
  d->Gu.setZero();
  d->Gp.setZero();
  d->Hx.setZero();
  d->Hu.setZero();
  d->Hp.setZero();
  if (ng_d != 0u) {
    d->Gx.topRows(ng_d) = d->dynamics_stage[0]->Gx;
    d->Gp.topRows(ng_d) = d->dynamics_stage[0]->Gp;
    if (dynamics_nu != 0u) {
      d->Gu.block(0, ndx, ng_d, dynamics_nu) = d->dynamics_stage[0]->Gu;
    }
  }
  if (nh_d != 0u) {
    d->Hx.topRows(nh_d) = d->dynamics_stage[0]->Hx;
    d->Hp.topRows(nh_d) = d->dynamics_stage[0]->Hp;
    if (dynamics_nu != 0u) {
      d->Hu.block(0, ndx, nh_d, dynamics_nu) = d->dynamics_stage[0]->Hu;
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
void IntegratedObserverModelRKTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(state_->get_nx()) + ")");
  }

  const std::shared_ptr<Data> d = cast_data(data);
  dynamics_->calcDiff(d->dynamics_stage[0], x);
  costs_->calcDiff(d->costs_stage[0], x);

  d->Fx.setIdentity();
  d->Fp.setZero();
  d->dE_dv.setZero();
  d->dE_dp.setZero();
  d->Lx = d->costs_stage[0]->Lx;
  d->Lxx = d->costs_stage[0]->Lxx;
  if (np_ != 0u) {
    d->Lp = d->costs_stage[0]->Lp;
    d->Lpp = d->costs_stage[0]->Lpp;
    d->Lpx = d->costs_stage[0]->Lpx;
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
void IntegratedObserverModelRKTpl<Scalar>::set_params(
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
        "RK integrated observer parameter dimension");
  }
  d->resize(this);
  for (std::size_t i = 0; i < ni_; ++i) {
    dynamics_->set_params(d->dynamics_stage[i], params_);
    d->costs_stage[i] = costs_->createData(d->dynamics_stage[i]->shared);
  }
  d->dynamics = d->dynamics_stage[0];
  d->costs = d->costs_stage[0];
  if (constraints_ != nullptr) {
    d->constraints = constraints_->createData(d->dynamics_stage[0]->shared);
    d->constraints_terminal =
        constraints_->createData(d->dynamics_stage[0]->shared);
    d->constraints_terminal->resize(constraints_.get(), false);
  } else {
    d->constraints.reset();
    d->constraints_terminal.reset();
  }
  update_p(data, params_->zero());
}

template <typename Scalar>
void IntegratedObserverModelRKTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (params_ == nullptr) {
    throw_pretty("Invalid call: integrated observer parameters are not set");
  }
  if (static_cast<std::size_t>(p.size()) != params_->get_np()) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(params_->get_np()) + ")");
  }
  const std::shared_ptr<Data> d = cast_data(data);
  for (std::size_t i = 0; i < ni_; ++i) {
    dynamics_->update_p(d->dynamics_stage[i], p);
  }
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedObserverModelRKTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedObserverModelRKTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
bool IntegratedObserverModelRKTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr || d->dynamics_stage.size() != ni_ ||
      d->costs_stage.size() != ni_ || d->integral.size() != ni_ ||
      d->ki.size() != ni_ || d->y.size() != ni_ || d->dx_rk.size() != ni_ ||
      d->dki_dx.size() != ni_ || d->dki_du.size() != ni_ ||
      d->dki_dp.size() != ni_ || d->dy_dx.size() != ni_ ||
      d->dy_du.size() != ni_ || d->dy_dp.size() != ni_ ||
      d->dli_dx.size() != ni_ || d->dli_du.size() != ni_ ||
      d->dli_dp.size() != ni_ || d->ddli_ddx.size() != ni_ ||
      d->ddli_ddu.size() != ni_ || d->ddli_ddp.size() != ni_ ||
      d->ddli_dxdu.size() != ni_ || d->ddli_dpdx.size() != ni_ ||
      d->ddli_dpdu.size() != ni_ || d->Luu_partialx.size() != ni_ ||
      d->Lpp_partialx.size() != ni_) {
    return false;
  }
  for (std::size_t i = 0; i < ni_; ++i) {
    if (d->dynamics_stage[i] == nullptr || d->costs_stage[i] == nullptr ||
        !dynamics_->checkData(d->dynamics_stage[i])) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void IntegratedObserverModelRKTpl<Scalar>::quasiStatic(
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
    dynamics_->quasiStatic(d->dynamics_stage[0], u.tail(dynamics_nu), x,
                           maxiter, tol);
  }
  if (ndx != 0u) {
    u.head(ndx).setZero();
  }
}

template <typename Scalar>
std::size_t IntegratedObserverModelRKTpl<Scalar>::get_ni() const {
  return ni_;
}

template <typename Scalar>
RKType IntegratedObserverModelRKTpl<Scalar>::get_rk_type() const {
  return rk_type_;
}

template <typename Scalar>
void IntegratedObserverModelRKTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedObserverModelRK {dt=" << time_step_ << ", " << *dynamics_
     << "}";
}

template <typename Scalar>
template <typename NewScalar>
IntegratedObserverModelRKTpl<NewScalar>
IntegratedObserverModelRKTpl<Scalar>::cast() const {
  typedef IntegratedObserverModelRKTpl<NewScalar> ReturnType;
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
      constraints, scalar_cast<NewScalar>(time_step_), rk_type_);
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
std::shared_ptr<typename IntegratedObserverModelRKTpl<Scalar>::Data>
IntegratedObserverModelRKTpl<Scalar>::cast_data(
    const std::shared_ptr<ActionDataAbstract>& data) {
  const std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data is not an integrated observer RK data");
  }
  if (!checkData(d)) {
    throw_pretty(
        "Invalid argument: integrated observer RK data has an incompatible "
        "scheme or invalid stage storage");
  }
  return d;
}

template <typename Scalar>
void IntegratedObserverModelRKTpl<Scalar>::set_rk_type(const RKType rktype) {
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
    default:
      throw_pretty("Invalid argument: unsupported RK type");
  }
  rk_type_ = rktype;
}

template <typename Scalar>
template <template <typename S> class Model>
IntegratedObserverDataRKTpl<Scalar>::IntegratedObserverDataRKTpl(
    Model<Scalar>* const model,
    const std::shared_ptr<ParameterDataManager>& params_data)
    : Base(model, params_data) {
  const std::size_t ni = model->get_ni();
  dynamics_stage.reserve(ni);
  costs_stage.reserve(ni);
  dynamics_stage.push_back(dynamics);
  costs_stage.push_back(costs);
  for (std::size_t i = 1; i < ni; ++i) {
    dynamics_stage.push_back(
        params_data != nullptr ? model->get_dynamics()->createData(params_data)
                               : model->get_dynamics()->createData());
    costs_stage.push_back(
        model->get_costs()->createData(dynamics_stage.back()->shared));
  }
  resize(model);
}

}  // namespace crocoddyl
