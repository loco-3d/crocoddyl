#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {

ActionModelNumDiff::ActionModelNumDiff(ActionModelAbstract& model, bool with_gauss_approx)
    : ActionModelAbstract(model.get_state(), model.get_nu(), model.get_nr()), model_(model) {
  with_gauss_approx_ = with_gauss_approx;
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  assert((!with_gauss_approx_ || nr_ > 1) && "No Gauss approximation possible with nr = 1");

  dx_.resize(state_->get_ndx());
  dx_.setZero();
  du_.resize(model.get_nu());
  du_.setZero();
  tmp_x_.resize(state_->get_nx());
  tmp_x_.setZero();
}

ActionModelNumDiff::~ActionModelNumDiff() {}

void ActionModelNumDiff::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_->get_nx() && "ActionModelNumDiff::calc: x has wrong dimension");
  assert(u.size() == nu_ && "ActionModelNumDiff::calc: u has wrong dimension");
  model_.calc(data, x, u);
}

void ActionModelNumDiff::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                  const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == state_->get_nx() && "ActionModelNumDiff::calcDiff: x has wrong dimension");
  assert(u.size() == nu_ && "ActionModelNumDiff::calcDiff: u has wrong dimension");
  boost::shared_ptr<ActionDataNumDiff> data_num_diff = boost::static_pointer_cast<ActionDataNumDiff>(data);

  if (recalc) {
    model_.calc(data_num_diff->data_0, x, u);
  }
  Eigen::VectorXd& xn0 = data_num_diff->data_0->xnext;
  double& c0 = data_num_diff->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  dx_.setZero();
  for (unsigned ix = 0; ix < state_->get_ndx(); ++ix) {
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
    if (model_.get_nr() > 1) {
      // TODO: @mnaveau manage the gaussian approximation
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
    if (model_.get_nr() > 1) {
      // TODO: @mnaveau manage the gaussian approximation
      // data_num_diff->Ru.col(iu) = data_num_diff->data_u[iu]->cost_residual -
      //                             data_num_diff->data_0[iu]->cost_residual;
    }
    du_(iu) = 0.0;
  }
  data_num_diff->Fu /= disturbance_;

  if (with_gauss_approx_) {
    data_num_diff->Lxx = data_num_diff->Rx.transpose() * data_num_diff->Rx;
    data_num_diff->Lxu = data_num_diff->Rx.transpose() * data_num_diff->Ru;
    data_num_diff->Luu = data_num_diff->Ru.transpose() * data_num_diff->Ru;
  }
}

void ActionModelNumDiff::assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& /** x */) {
  // TODO: @mnaveau make this method virtual and this one should do nothing, update the documentation.
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
