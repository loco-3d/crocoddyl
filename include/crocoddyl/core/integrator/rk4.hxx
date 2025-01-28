///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, IRI: CSIC-UPC, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/integrator/rk4.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelRK4Tpl<Scalar>::IntegratedActionModelRK4Tpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    std::shared_ptr<ControlParametrizationModelAbstract> control,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model, control, time_step, with_cost_residual) {
  rk4_c_ = {Scalar(0.), Scalar(0.5), Scalar(0.5), Scalar(1.)};
  std::cerr
      << "Deprecated IntegratedActionModelRK4: Use IntegratedActionModelRK"
      << std::endl;
}

template <typename Scalar>
IntegratedActionModelRK4Tpl<Scalar>::IntegratedActionModelRK4Tpl(
    std::shared_ptr<DifferentialActionModelAbstract> model,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model, time_step, with_cost_residual) {
  rk4_c_ = {Scalar(0.), Scalar(0.5), Scalar(0.5), Scalar(1.)};
  std::cerr
      << "Deprecated IntegratedActionModelRK4: Use IntegratedActionModelRK"
      << std::endl;
}

template <typename Scalar>
IntegratedActionModelRK4Tpl<Scalar>::~IntegratedActionModelRK4Tpl() {}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::calc(
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
  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
      d->control[0];
  control_->calc(u0_data, rk4_c_[0], u);
  d->ws[0] = u0_data->w;
  differential_->calc(k0_data, x, d->ws[0]);
  d->y[0] = x;
  d->ki[0].head(nv) = d->y[0].tail(nv);
  d->ki[0].tail(nv) = k0_data->xout;
  d->integral[0] = k0_data->cost;
  for (std::size_t i = 1; i < 4; ++i) {
    const std::shared_ptr<DifferentialActionDataAbstract>& ki_data =
        d->differential[i];
    const std::shared_ptr<ControlParametrizationDataAbstract>& ui_data =
        d->control[i];
    d->dx_rk4[i].noalias() = time_step_ * rk4_c_[i] * d->ki[i - 1];
    state_->integrate(x, d->dx_rk4[i], d->y[i]);
    control_->calc(ui_data, rk4_c_[i], u);
    d->ws[i] = ui_data->w;
    differential_->calc(ki_data, d->y[i], d->ws[i]);
    d->ki[i].head(nv) = d->y[i].tail(nv);
    d->ki[i].tail(nv) = ki_data->xout;
    d->integral[i] = ki_data->cost;
  }
  d->dx =
      (d->ki[0] + Scalar(2.) * d->ki[1] + Scalar(2.) * d->ki[2] + d->ki[3]) *
      time_step_ / Scalar(6.);
  state_->integrate(x, d->dx, d->xnext);
  d->cost = (d->integral[0] + Scalar(2.) * d->integral[1] +
             Scalar(2.) * d->integral[2] + d->integral[3]) *
            time_step_ / Scalar(6.);
  d->g = k0_data->g;
  d->h = k0_data->h;
  if (with_cost_residual_) {
    d->r = k0_data->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  differential_->calc(k0_data, x);
  d->dx.setZero();
  d->xnext = x;
  d->cost = k0_data->cost;
  d->g = k0_data->g;
  d->h = k0_data->h;
  if (with_cost_residual_) {
    d->r = k0_data->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::calcDiff(
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
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = control_->get_nu();
  Data* d = static_cast<Data*>(data.get());
  assert_pretty(
      MatrixXs(d->dyi_dx[0])
          .isApprox(MatrixXs::Identity(state_->get_ndx(), state_->get_ndx())),
      "you have changed dyi_dx[0] values that supposed to be constant.");
  assert_pretty(
      MatrixXs(d->dki_dx[0])
          .topRightCorner(nv, nv)
          .isApprox(MatrixXs::Identity(nv, nv)),
      "you have changed dki_dx[0] values that supposed to be constant.");

  for (std::size_t i = 0; i < 4; ++i) {
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

  for (std::size_t i = 1; i < 4; ++i) {
    const std::shared_ptr<DifferentialActionDataAbstract>& ki_data =
        d->differential[i];
    const std::shared_ptr<ControlParametrizationDataAbstract>& ui_data =
        d->control[i];
    d->dyi_dx[i].noalias() = d->dki_dx[i - 1] * rk4_c_[i] * time_step_;
    d->dyi_du[i].noalias() = d->dki_du[i - 1] * rk4_c_[i] * time_step_;
    state_->JintegrateTransport(x, d->dx_rk4[i], d->dyi_dx[i], second);
    state_->Jintegrate(x, d->dx_rk4[i], d->dyi_dx[i], d->dyi_dx[i], first,
                       addto);
    state_->JintegrateTransport(x, d->dx_rk4[i], d->dyi_du[i],
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

  d->Fx.noalias() = time_step_ / Scalar(6.) *
                    (d->dki_dx[0] + Scalar(2.) * d->dki_dx[1] +
                     Scalar(2.) * d->dki_dx[2] + d->dki_dx[3]);
  state_->JintegrateTransport(x, d->dx, d->Fx, second);
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);

  d->Fu.noalias() = time_step_ / Scalar(6.) *
                    (d->dki_du[0] + Scalar(2.) * d->dki_du[1] +
                     Scalar(2.) * d->dki_du[2] + d->dki_du[3]);
  state_->JintegrateTransport(x, d->dx, d->Fu, second);

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
  d->Gx = k0_data->Gx;
  d->Hx = k0_data->Hx;
  d->Gu.resize(differential_->get_ng(), nu_);
  d->Hu.resize(differential_->get_nh(), nu_);
  control_->multiplyByJacobian(u0_data, k0_data->Gu, d->Gu);
  control_->multiplyByJacobian(u0_data, k0_data->Hu, d->Hu);
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const std::shared_ptr<DifferentialActionDataAbstract>& k0_data =
      d->differential[0];
  differential_->calcDiff(k0_data, x);
  d->Lx = k0_data->Lx;
  d->Lxx = k0_data->Lxx;
  d->Gx = k0_data->Gx;
  d->Hx = k0_data->Hx;
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelRK4Tpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool IntegratedActionModelRK4Tpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (data != NULL) {
    return differential_->checkData(d->differential[0]) &&
           differential_->checkData(d->differential[2]) &&
           differential_->checkData(d->differential[1]) &&
           differential_->checkData(d->differential[3]);
  } else {
    return false;
  }
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::quasiStatic(
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
  const std::shared_ptr<ControlParametrizationDataAbstract>& u0_data =
      d->control[0];
  u0_data->w *= 0.;
  differential_->quasiStatic(d->differential[0], u0_data->w, x, maxiter, tol);
  control_->params(u0_data, 0., u0_data->w);
  u = u0_data->u;
}

template <typename Scalar>
void IntegratedActionModelRK4Tpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedActionModelRK4 {dt=" << time_step_ << ", " << *differential_
     << "}";
}

}  // namespace crocoddyl
