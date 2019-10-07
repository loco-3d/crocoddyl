///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {

DifferentialActionModelNumDiff::DifferentialActionModelNumDiff(DifferentialActionModelAbstract& model,
                                                               bool with_gauss_approx)
    : DifferentialActionModelAbstract(model.get_state(), model.get_nu(), model.get_nr()), model_(model) {
  with_gauss_approx_ = with_gauss_approx;
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  assert((!with_gauss_approx_ || nr_ > 1) && "No Gauss approximation possible with nr = 1");
}

DifferentialActionModelNumDiff::~DifferentialActionModelNumDiff() {}

void DifferentialActionModelNumDiff::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");
  DifferentialActionDataNumDiff* data_nd = static_cast<DifferentialActionDataNumDiff*>(data.get());
  model_.calc(data_nd->data_0, x, u);
  data->cost = data_nd->data_0->cost;
  data->xout = data_nd->data_0->xout;
}

void DifferentialActionModelNumDiff::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                              const Eigen::Ref<const Eigen::VectorXd>& x,
                                              const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");
  boost::shared_ptr<DifferentialActionDataNumDiff> data_nd =
      boost::static_pointer_cast<DifferentialActionDataNumDiff>(data);

  if (recalc) {
    model_.calc(data_nd->data_0, x, u);
  }
  const Eigen::VectorXd& xn0 = data_nd->data_0->xout;
  const double& c0 = data_nd->data_0->cost;
  data->xout = data_nd->data_0->xout;
  data->cost = data_nd->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_.get_ndx(); ++ix) {
    data_nd->dx(ix) = disturbance_;
    model_.get_state().integrate(x, data_nd->dx, data_nd->xp);
    model_.calc(data_nd->data_x[ix], data_nd->xp, u);

    const Eigen::VectorXd& xn = data_nd->data_x[ix]->xout;
    const double& c = data_nd->data_x[ix]->cost;
    data->Fx.col(ix) = (xn - xn0) / disturbance_;

    data->Lx(ix) = (c - c0) / disturbance_;
    data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - data_nd->data_0->r) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d action(x,u) / du
  data_nd->du.setZero();
  for (unsigned iu = 0; iu < model_.get_nu(); ++iu) {
    data_nd->du(iu) = disturbance_;
    model_.calc(data_nd->data_u[iu], x, u + data_nd->du);

    const Eigen::VectorXd& xn = data_nd->data_u[iu]->xout;
    const double& c = data_nd->data_u[iu]->cost;
    data->Fu.col(iu) = (xn - xn0) / disturbance_;

    data->Lu(iu) = (c - c0) / disturbance_;
    data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - data_nd->data_0->r) / disturbance_;
    data_nd->du(iu) = 0.0;
  }

  if (with_gauss_approx_) {
    data->Lxx = data_nd->Rx.transpose() * data_nd->Rx;
    data->Lxu = data_nd->Rx.transpose() * data_nd->Ru;
    data->Luu = data_nd->Ru.transpose() * data_nd->Ru;
  }
}

DifferentialActionModelAbstract& DifferentialActionModelNumDiff::get_model() const { return model_; }

const double& DifferentialActionModelNumDiff::get_disturbance() const { return disturbance_; }

bool DifferentialActionModelNumDiff::get_with_gauss_approx() { return with_gauss_approx_; }

void DifferentialActionModelNumDiff::assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& /** x */) {
  // TODO(cmastalli): First we need to do it AMNumDiff and then to replicate it.
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelNumDiff::createData() {
  return boost::make_shared<DifferentialActionDataNumDiff>(this);
}

}  // namespace crocoddyl
