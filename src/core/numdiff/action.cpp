///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {

ActionModelNumDiff::ActionModelNumDiff(ActionModelAbstract& model, bool with_gauss_approx)
    : ActionModelAbstract(model.get_state(), model.get_nu(), model.get_nr()), model_(model) {
  with_gauss_approx_ = with_gauss_approx;
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  assert((!with_gauss_approx_ || nr_ > 1) && "No Gauss approximation possible with nr = 1");
}

ActionModelNumDiff::~ActionModelNumDiff() {}

void ActionModelNumDiff::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");
  boost::shared_ptr<ActionDataNumDiff> data_nd = boost::static_pointer_cast<ActionDataNumDiff>(data);
  model_.calc(data_nd->data_0, x, u);
  data->cost = data_nd->data_0->cost;
  data->xnext = data_nd->data_0->xnext;
}

void ActionModelNumDiff::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                  const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");
  boost::shared_ptr<ActionDataNumDiff> data_nd = boost::static_pointer_cast<ActionDataNumDiff>(data);

  if (recalc) {
    model_.calc(data_nd->data_0, x, u);
  }
  const Eigen::VectorXd& xn0 = data_nd->data_0->xnext;
  double& c0 = data_nd->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  data_nd->dx.setZero();
  for (unsigned int ix = 0; ix < state_.get_ndx(); ++ix) {
    data_nd->dx(ix) = disturbance_;
    model_.get_state().integrate(x, data_nd->dx, data_nd->xp);
    calc(data_nd->data_x[ix], data_nd->xp, u);

    const Eigen::VectorXd& xn = data_nd->data_x[ix]->xnext;
    const double& c = data_nd->data_x[ix]->cost;
    model_.get_state().diff(xn0, xn, data_nd->Fx.col(ix));

    data_nd->Lx(ix) = (c - c0) / disturbance_;
    data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - data_nd->data_0->r) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }
  data_nd->Fx /= disturbance_;

  // Computing the d action(x,u) / du
  data_nd->du.setZero();
  for (unsigned iu = 0; iu < model_.get_nu(); ++iu) {
    data_nd->du(iu) = disturbance_;
    calc(data_nd->data_u[iu], x, u + data_nd->du);

    const Eigen::VectorXd& xn = data_nd->data_u[iu]->xnext;
    const double& c = data_nd->data_u[iu]->cost;
    model_.get_state().diff(xn0, xn, data_nd->Fu.col(iu));

    data_nd->Lu(iu) = (c - c0) / disturbance_;
    data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - data_nd->data_0->r) / disturbance_;
    data_nd->du(iu) = 0.0;
  }
  data_nd->Fu /= disturbance_;

  if (with_gauss_approx_) {
    data_nd->Lxx = data_nd->Rx.transpose() * data_nd->Rx;
    data_nd->Lxu = data_nd->Rx.transpose() * data_nd->Ru;
    data_nd->Luu = data_nd->Ru.transpose() * data_nd->Ru;
  }
}

ActionModelAbstract& ActionModelNumDiff::get_model() const { return model_; }

const double& ActionModelNumDiff::get_disturbance() const { return disturbance_; }

bool ActionModelNumDiff::get_with_gauss_approx() { return with_gauss_approx_; }

void ActionModelNumDiff::assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& /** x */) {
  // TODO(mnaveau): make this method virtual and this one should do nothing, update the documentation.
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

boost::shared_ptr<ActionDataAbstract> ActionModelNumDiff::createData() {
  return boost::make_shared<ActionDataNumDiff>(this);
}

}  // namespace crocoddyl
