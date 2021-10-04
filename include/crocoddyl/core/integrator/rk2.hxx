///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/rk2.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelRK2Tpl<Scalar>::IntegratedActionModelRK2Tpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model,
    boost::shared_ptr<ControlParametrizationModelAbstract> control, const Scalar time_step,
    const bool with_cost_residual)
    : Base(model, control, time_step, with_cost_residual) {
  rk2_c_ = {Scalar(0.), Scalar(0.5)};
}

template <typename Scalar>
IntegratedActionModelRK2Tpl<Scalar>::IntegratedActionModelRK2Tpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, const Scalar time_step, const bool with_cost_residual)
    : Base(model, time_step, with_cost_residual) {
  rk2_c_ = {Scalar(0.), Scalar(0.5)};
}

template <typename Scalar>
IntegratedActionModelRK2Tpl<Scalar>::~IntegratedActionModelRK2Tpl() {}

template <typename Scalar>
void IntegratedActionModelRK2Tpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x,
                                               const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());

  const boost::shared_ptr<DifferentialActionDataAbstract>& k0_data = d->differential[0];
  const boost::shared_ptr<ControlParametrizationDataAbstract>& u0_data = d->control[0];
  control_->calc(u0_data, rk2_c_[0], u);
  d->ws[0] = u0_data->w;
  differential_->calc(k0_data, x, d->ws[0]);
  d->y[0] = x;
  d->ki[0].head(nv) = d->y[0].tail(nv);
  d->ki[0].tail(nv) = k0_data->xout;
  d->integral[0] = k0_data->cost;

  const boost::shared_ptr<DifferentialActionDataAbstract>& k1_data = d->differential[1];
  const boost::shared_ptr<ControlParametrizationDataAbstract>& u1_data = d->control[1];
  d->dx_rk2[1] = time_step_ * rk2_c_[1] * d->ki[0];
  state_->integrate(x, d->dx_rk2[1], d->y[1]);
  control_->calc(u1_data, rk2_c_[1], u);
  d->ws[1] = u1_data->w;
  differential_->calc(k1_data, d->y[1], d->ws[1]);
  d->ki[1].head(nv) = d->y[1].tail(nv);
  d->ki[1].tail(nv) = k1_data->xout;
  d->integral[1] = k1_data->cost;

  d->dx = d->ki[1] * time_step_;
  state_->integrate(x, d->dx, d->xnext);
  d->cost = d->integral[1] * time_step_;

  if (with_cost_residual_) {
    d->r = k0_data->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRK2Tpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const boost::shared_ptr<DifferentialActionDataAbstract>& k0_data = d->differential[0];
  differential_->calc(k0_data, x);
  d->dx.setZero();
  d->cost = k0_data->cost;
  if (with_cost_residual_) {
    d->r = k0_data->r;
  }
}

template <typename Scalar>
void IntegratedActionModelRK2Tpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>& x,
                                                   const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = control_->get_nu();
  Data* d = static_cast<Data*>(data.get());
  assert_pretty(MatrixXs(d->dyi_dx[0]).isApprox(MatrixXs::Identity(state_->get_ndx(), state_->get_ndx())),
                "you have changed dyi_dx[0] values that supposed to be constant.");
  assert_pretty(MatrixXs(d->dki_dx[0]).topRightCorner(nv, nv).isApprox(MatrixXs::Identity(nv, nv)),
                "you have changed dki_dx[0] values that supposed to be constant.");

  for (std::size_t i = 0; i < 2; ++i) {
    differential_->calcDiff(d->differential[i], d->y[i], d->ws[i]);
  }

  const boost::shared_ptr<DifferentialActionDataAbstract>& k0_data = d->differential[0];
  const boost::shared_ptr<ControlParametrizationDataAbstract>& u0_data = d->control[0];
  d->dki_dx[0].bottomRows(nv) = k0_data->Fx;
  control_->multiplyByJacobian(u0_data, k0_data->Fu,
                               d->dki_du[0].bottomRows(nv));  // dki_du = dki_dw * dw_du

  d->dli_dx[0] = k0_data->Lx;
  control_->multiplyJacobianTransposeBy(u0_data, k0_data->Lu,
                                        d->dli_du[0]);  // dli_du = dli_dw * dw_du

  d->ddli_ddx[0] = k0_data->Lxx;
  d->ddli_ddw[0] = k0_data->Luu;
  control_->multiplyByJacobian(u0_data, d->ddli_ddw[0], d->ddli_dwdu[0]);  // ddli_dwdu = ddli_ddw * dw_du
  control_->multiplyJacobianTransposeBy(u0_data, d->ddli_dwdu[0],
                                        d->ddli_ddu[0]);  // ddli_ddu = dw_du.T * ddli_dwdu
  d->ddli_dxdw[0] = k0_data->Lxu;
  control_->multiplyByJacobian(u0_data, d->ddli_dxdw[0], d->ddli_dxdu[0]);  // ddli_dxdu = ddli_dxdw * dw_du

  const boost::shared_ptr<DifferentialActionDataAbstract>& k1_data = d->differential[1];
  const boost::shared_ptr<ControlParametrizationDataAbstract>& u1_data = d->control[1];
  d->dyi_dx[1].noalias() = d->dki_dx[0] * rk2_c_[1] * time_step_;
  d->dyi_du[1].noalias() = d->dki_du[0] * rk2_c_[1] * time_step_;
  state_->JintegrateTransport(x, d->dx_rk2[1], d->dyi_dx[1], second);
  state_->Jintegrate(x, d->dx_rk2[1], d->dyi_dx[1], d->dyi_dx[1], first, addto);
  state_->JintegrateTransport(x, d->dx_rk2[1], d->dyi_du[1], second);  // dyi_du = Jintegrate * dyi_du

  // Sparse matrix-matrix multiplication for computing:
  Eigen::Block<MatrixXs> dkv1_dq = d->dki_dx[1].bottomLeftCorner(nv, nv);
  Eigen::Block<MatrixXs> dkv1_dv = d->dki_dx[1].bottomRightCorner(nv, nv);
  Eigen::Block<MatrixXs> dkq1_du = d->dki_du[1].topLeftCorner(nv, nu);
  Eigen::Block<MatrixXs> dkv1_du = d->dki_du[1].bottomLeftCorner(nv, nu);
  const Eigen::Block<MatrixXs> dk1_dq1 = k1_data->Fx.bottomLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> dk1_dv1 = k1_data->Fx.bottomRightCorner(nv, nv);
  const Eigen::Block<MatrixXs> dqi_dq = d->dyi_dx[1].topLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> dvi_dq = d->dyi_dx[1].bottomLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> dvi_dv = d->dyi_dx[1].bottomRightCorner(nv, nv);
  const Eigen::Block<MatrixXs> dqi_du = d->dyi_du[1].topLeftCorner(nv, nu);
  const Eigen::Block<MatrixXs> dvi_du = d->dyi_du[1].bottomLeftCorner(nv, nu);
  //   i. d->dki_dx[i].noalias() = d->dki_dy[i] * d->dyi_dx[i], where dki_dy is ki_data.Fx
  d->dki_dx[1].topRows(nv) = d->dyi_dx[1].bottomRows(nv);
  dkv1_dq.noalias() = dk1_dq1 * dqi_dq;
  dkv1_dv = time_step_ / Scalar(2.) * dk1_dq1;
  dkv1_dq.noalias() += dk1_dv1 * dvi_dq;
  dkv1_dv.noalias() += dk1_dv1 * dvi_dv;
  //  ii. d->dki_du[i].noalias() = d->dki_dy[i] * d->dyi_du[i], where dki_dy is ki_data.Fx
  dkq1_du = dvi_du;
  dkv1_du.noalias() = dk1_dq1 * dqi_du;
  dkv1_du.noalias() += dk1_dv1 * dvi_du;

  control_->multiplyByJacobian(u1_data, k1_data->Fu, d->dki_du[1].bottomRows(nv),
                               addto);  // dfi_du = dki_dw * dw_du

  d->dli_dx[1].noalias() = k1_data->Lx.transpose() * d->dyi_dx[1];
  control_->multiplyJacobianTransposeBy(u1_data, k1_data->Lu,
                                        d->dli_du[1]);  // dli_du = Lu * dw_du
  d->dli_du[1].noalias() += k1_data->Lx.transpose() * d->dyi_du[1];

  d->Lxx_partialx[1].noalias() = k1_data->Lxx * d->dyi_dx[1];
  d->ddli_ddx[1].noalias() = d->dyi_dx[1].transpose() * d->Lxx_partialx[1];

  control_->multiplyByJacobian(u1_data, k1_data->Lxu, d->Lxu_i[1]);  // Lxu = Lxw * dw_du
  d->Luu_partialx[1].noalias() = d->Lxu_i[1].transpose() * d->dyi_du[1];
  d->Lxx_partialu[1].noalias() = k1_data->Lxx * d->dyi_du[1];
  control_->multiplyByJacobian(u1_data, k1_data->Luu,
                               d->ddli_dwdu[1]);  // ddli_dwdu = ddli_ddw * dw_du
  control_->multiplyJacobianTransposeBy(u1_data, d->ddli_dwdu[1],
                                        d->ddli_ddu[1]);  // ddli_ddu = dw_du.T * ddli_dwdu
  d->ddli_ddu[1].noalias() +=
      d->Luu_partialx[1].transpose() + d->Luu_partialx[1] + d->dyi_du[1].transpose() * d->Lxx_partialu[1];

  d->ddli_dxdw[1].noalias() = d->dyi_dx[1].transpose() * d->differential[1]->Lxu;
  control_->multiplyByJacobian(u1_data, d->ddli_dxdw[1],
                               d->ddli_dxdu[1]);  // ddli_dxdu = ddli_dxdw * dw_du
  d->ddli_dxdu[1].noalias() += d->dyi_dx[1].transpose() * d->Lxx_partialu[1];

  d->Fx.noalias() = time_step_ * d->dki_dx[1];
  state_->JintegrateTransport(x, d->dx, d->Fx, second);
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);

  d->Fu.noalias() = time_step_ * d->dki_du[1];
  state_->JintegrateTransport(x, d->dx, d->Fu, second);

  d->Lx.noalias() = time_step_ * d->dli_dx[1];
  d->Lu.noalias() = time_step_ * d->dli_du[1];

  d->Lxx.noalias() = time_step_ * d->ddli_ddx[1];
  d->Luu.noalias() = time_step_ * d->ddli_ddu[1];
  d->Lxu.noalias() = time_step_ * d->ddli_dxdu[1];
}

template <typename Scalar>
void IntegratedActionModelRK2Tpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const boost::shared_ptr<DifferentialActionDataAbstract>& k0_data = d->differential[0];
  differential_->calcDiff(k0_data, x);
  d->Lx = k0_data->Lx;
  d->Lxx = k0_data->Lxx;
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > IntegratedActionModelRK2Tpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool IntegratedActionModelRK2Tpl<Scalar>::checkData(const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (data != NULL) {
    return differential_->checkData(d->differential[0]) && differential_->checkData(d->differential[1]);
  } else {
    return false;
  }
}

template <typename Scalar>
void IntegratedActionModelRK2Tpl<Scalar>::quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data,
                                                      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                                                      const std::size_t maxiter, const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const boost::shared_ptr<ControlParametrizationDataAbstract>& u0_data = d->control[0];
  u0_data->w *= 0.;
  differential_->quasiStatic(d->differential[0], u0_data->w, x, maxiter, tol);
  control_->params(u0_data, 0., u0_data->w);
  u = u0_data->u;
}

template <typename Scalar>
void IntegratedActionModelRK2Tpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedActionModelRK2 {dt=" << time_step_ << ", " << *differential_ << "}";
}

}  // namespace crocoddyl
