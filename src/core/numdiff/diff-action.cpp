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

  dx_.resize(state_.get_ndx());
  dx_.setZero();
  du_.resize(model.get_nu());
  du_.setZero();
  tmp_x_.resize(state_.get_nx());
  tmp_x_.setZero();
}

DifferentialActionModelNumDiff::~DifferentialActionModelNumDiff() {}

void DifferentialActionModelNumDiff::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");
  model_.calc(data, x, u);
}

void DifferentialActionModelNumDiff::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                              const Eigen::Ref<const Eigen::VectorXd>& x,
                                              const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");
  boost::shared_ptr<DifferentialActionDataNumDiff> data_num_diff =
      boost::static_pointer_cast<DifferentialActionDataNumDiff>(data);

  if (recalc) {
    model_.calc(data_num_diff->data_0, x, u);
  }
  Eigen::VectorXd& xn0 = data_num_diff->data_0->xout;
  double& c0 = data_num_diff->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  dx_.setZero();
  for (unsigned ix = 0; ix < state_.get_ndx(); ++ix) {
    dx_(ix) = disturbance_;
    model_.get_state().integrate(x, dx_, tmp_x_);
    calc(data_num_diff->data_x[ix], tmp_x_, u);

    Eigen::VectorXd& xn = data_num_diff->data_x[ix]->xout;
    double& c = data_num_diff->data_x[ix]->cost;
    model_.get_state().diff(xn0, xn, data_num_diff->Fx.col(ix));

    data_num_diff->Lx(ix) = (c - c0) / disturbance_;
    data_num_diff->Rx.col(ix) = (data_num_diff->data_x[ix]->r - data_num_diff->data_0->r) / disturbance_;
    dx_(ix) = 0.0;
  }
  data_num_diff->Fx /= disturbance_;

  // Computing the d action(x,u) / du
  du_.setZero();
  for (unsigned iu = 0; iu < model_.get_nu(); ++iu) {
    du_(iu) = disturbance_;
    calc(data_num_diff->data_u[iu], x, u + du_);

    Eigen::VectorXd& xn = data_num_diff->data_u[iu]->xout;
    double& c = data_num_diff->data_u[iu]->cost;
    model_.get_state().diff(xn0, xn, data_num_diff->Fu.col(iu));

    data_num_diff->Lu(iu) = (c - c0) / disturbance_;
    data_num_diff->Ru.col(iu) = (data_num_diff->data_u[iu]->r - data_num_diff->data_0->r) / disturbance_;
    du_(iu) = 0.0;
  }
  data_num_diff->Fu /= disturbance_;

  if (with_gauss_approx_) {
    data_num_diff->Lxx = data_num_diff->Rx.transpose() * data_num_diff->Rx;
    data_num_diff->Lxu = data_num_diff->Rx.transpose() * data_num_diff->Ru;
    data_num_diff->Luu = data_num_diff->Ru.transpose() * data_num_diff->Ru;
  }
}

void DifferentialActionModelNumDiff::assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& /** x */) {
  // TODO(cmastalli): First we need to do it AMNumDiff and then to replicate it.
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelNumDiff::createData() {
  return boost::make_shared<DifferentialActionDataNumDiff>(this);
}

}  // namespace crocoddyl
