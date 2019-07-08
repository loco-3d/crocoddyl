///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/action-num-diff.hpp"
// #include <cmath>
// #include <limits>

namespace crocoddyl {

ActionModelNumDiff::ActionModelNumDiff(ActionModelAbstract& model, bool with_gauss_approx)
    : ActionModelAbstract(model.get_state(), model.get_nu(), model.get_ncost()), model_(model) {
  with_gauss_approx_ = with_gauss_approx;
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  assert((!with_gauss_approx_ || ncost_ > 1) && "No Gauss approximation possible with ncost = 1");

  dx_.resize(model.get_ndx());
  dx_.setZero();
  du_.resize(model.get_nu());
  du_.setZero();
  tmp_x_.resize(model.get_nx());
  tmp_x_.setZero();
}

ActionModelNumDiff::~ActionModelNumDiff() {}

void ActionModelNumDiff::calc(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& u) {
  model_.calc(data, x, u);
}

void ActionModelNumDiff::calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                  const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  std::shared_ptr<ActionDataNumDiff> data_num_diff = std::static_pointer_cast<ActionDataNumDiff>(data);

  if (recalc) {
    model_.calc(data_num_diff->data_0, x, u);
  }
  Eigen::VectorXd& xn0 = data_num_diff->data_0->xnext;
  double& c0 = data_num_diff->data_0->cost;

  // std::cout << "assert stability" << std::endl;
  // assert_stable_state_finite_differences(x);

  // Computing the d action(x,u) / dx
  dx_.setZero();
  for (unsigned ix = 0; ix < model_.get_ndx(); ++ix) {
    dx_(ix) = disturbance_;
    model_.get_state()->integrate(x, dx_, tmp_x_);
    calc(data_num_diff->data_x[ix], tmp_x_, u);
    // data->Fx
    Eigen::VectorXd& xn = data_num_diff->data_x[ix]->xnext;
    double& c = data_num_diff->data_x[ix]->cost;
    model_.get_state()->diff(xn0, xn, data_num_diff->Fx.col(ix));
    // data->Lx
    data_num_diff->Lx(ix) = (c - c0) / disturbance_;

    // data->Rx
    if (model_.get_ncost() > 1) {
      // data_num_diff->Rx.col(ix) = data_num_diff->data_x[ix]->cost_residual -
      //                             data_num_diff->data_0[ix]->cost_residual;
    }
    dx_(ix) = 0.0;
  }
  data_num_diff->Fx /= disturbance_;

  // Computing the d action(x,u) / du
  du_.setZero();
  for (unsigned iu = 0; iu < model_.get_nu(); ++iu) {
    du_(iu) = disturbance_;
    calc(data_num_diff->data_u[iu], x, u + du_);
    // data->Fu
    Eigen::VectorXd& xn = data_num_diff->data_u[iu]->xnext;
    double& c = data_num_diff->data_u[iu]->cost;
    model_.get_state()->diff(xn0, xn, data_num_diff->Fu.col(iu));
    // data->Lu
    data_num_diff->Lu(iu) = (c - c0) / disturbance_;
    // data->Ru
    if (model_.get_ncost() > 1) {
      // data_num_diff->Ru.col(iu) = data_num_diff->data_u[iu]->cost_residual -
      //                             data_num_diff->data_0[iu]->cost_residual;
    }
    du_(iu) = 0.0;
  }
  data_num_diff->Fu /= disturbance_;

  if (with_gauss_approx_) {
    data_num_diff->Lxx = data_num_diff->Rx.transpose() * data_num_diff->Rx;
    data_num_diff->Lxu = data_num_diff->Ru.transpose() * data_num_diff->Ru;
    data_num_diff->Luu = data_num_diff->Ru.transpose() * data_num_diff->Ru;
  }
}

void ActionModelNumDiff::assert_stable_state_finite_differences(Eigen::Ref<Eigen::VectorXd> /** x */) {
  // md = model_.differential_;
  // if isinstance(md, DifferentialActionModelFloatingInContact)
  // {
  //   if hasattr(md, "costs")
  //   {
  //     mc = md.costs;
  //     if isinstance(mc, CostModelState)
  //     {
  //       assert (~np.isclose(model.State.diff(mc.ref, x)[3:6], np.ones(3) * np.pi, atol=1e-6).any())
  //       assert (~np.isclose(model.State.diff(mc.ref, x)[3:6], -np.ones(3) * np.pi, atol=1e-6).any())
  //     }
  //     else if isinstance(mc, CostModelSum)
  //     {
  //       for (key, cost) in mc.costs.items()
  //       {
  //         if isinstance(cost.cost, CostModelState)
  //         {
  //           assert (~np.isclose(
  //               model.State.diff(cost.cost.ref, x)[3:6], np.ones(3) * np.pi, atol=1e-6).any())
  //           assert (~np.isclose(
  //               model.State.diff(cost.cost.ref, x)[3:6], -np.ones(3) * np.pi, atol=1e-6).any())
  //         }
  //       }
  //     }
  //   }
  // }
}

std::shared_ptr<ActionDataAbstract> ActionModelNumDiff::createData() {
  return std::make_shared<ActionDataNumDiff>(this);
}

}  // namespace crocoddyl