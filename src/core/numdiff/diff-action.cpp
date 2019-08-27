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
  data_num_diff->dx.setZero();
  for (unsigned int ix = 0; ix < state_.get_ndx(); ++ix) {
    data_num_diff->dx(ix) = disturbance_;
    model_.get_state().integrate(x, data_num_diff->dx, data_num_diff->xp);
    calc(data_num_diff->data_x[ix], data_num_diff->xp, u);

    const Eigen::VectorXd& xn = data_num_diff->data_x[ix]->xout;
    const double& c = data_num_diff->data_x[ix]->cost;
    data_num_diff->Fx.col(ix) = (xn - xn0) / disturbance_;

    data_num_diff->Lx(ix) = (c - c0) / disturbance_;
    data_num_diff->Rx.col(ix) = (data_num_diff->data_x[ix]->r - data_num_diff->data_0->r) / disturbance_;
    data_num_diff->dx(ix) = 0.0;
  }

  // Computing the d action(x,u) / du
  data_num_diff->du.setZero();
  for (unsigned iu = 0; iu < model_.get_nu(); ++iu) {
    data_num_diff->du(iu) = disturbance_;
    calc(data_num_diff->data_u[iu], x, u + data_num_diff->du);

    const Eigen::VectorXd& xn = data_num_diff->data_u[iu]->xout;
    const double& c = data_num_diff->data_u[iu]->cost;
    data_num_diff->Fu.col(iu) = (xn - xn0) / disturbance_;

    data_num_diff->Lu(iu) = (c - c0) / disturbance_;
    data_num_diff->Ru.col(iu) = (data_num_diff->data_u[iu]->r - data_num_diff->data_0->r) / disturbance_;
    data_num_diff->du(iu) = 0.0;
  }

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
