///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {

ActionModelNumDiff::ActionModelNumDiff(boost::shared_ptr<ActionModelAbstract> model)
    : ActionModelAbstract(model->get_state(), model->get_nu(), model->get_nr()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
}

ActionModelNumDiff::~ActionModelNumDiff() {}

void ActionModelNumDiff::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  boost::shared_ptr<ActionDataNumDiff> data_nd = boost::static_pointer_cast<ActionDataNumDiff>(data);
  model_->calc(data_nd->data_0, x, u);
  data->cost = data_nd->data_0->cost;
  data->xnext = data_nd->data_0->xnext;
}

void ActionModelNumDiff::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                  const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  boost::shared_ptr<ActionDataNumDiff> data_nd = boost::static_pointer_cast<ActionDataNumDiff>(data);

  const Eigen::VectorXd& xn0 = data_nd->data_0->xnext;
  const double& c0 = data_nd->data_0->cost;
  data->xnext = data_nd->data_0->xnext;
  data->cost = data_nd->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);

    const Eigen::VectorXd& xn = data_nd->data_x[ix]->xnext;
    const double& c = data_nd->data_x[ix]->cost;
    model_->get_state()->diff(xn0, xn, data_nd->Fx.col(ix));

    data->Lx(ix) = (c - c0) / disturbance_;
    if (model_->get_nr() > 0) {
      data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - data_nd->data_0->r) / disturbance_;
    }
    data_nd->dx(ix) = 0.0;
  }
  data->Fx /= disturbance_;

  // Computing the d action(x,u) / du
  data_nd->du.setZero();
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    data_nd->du(iu) = disturbance_;
    model_->calc(data_nd->data_u[iu], x, u + data_nd->du);

    const Eigen::VectorXd& xn = data_nd->data_u[iu]->xnext;
    const double& c = data_nd->data_u[iu]->cost;
    model_->get_state()->diff(xn0, xn, data_nd->Fu.col(iu));

    data->Lu(iu) = (c - c0) / disturbance_;
    if (model_->get_nr() > 0) {
      data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - data_nd->data_0->r) / disturbance_;
    }
    data_nd->du(iu) = 0.0;
  }
  data->Fu /= disturbance_;

  if (model_->get_nr() > 0) {
    data->Lxx = data_nd->Rx.transpose() * data_nd->Rx;
    data->Lxu = data_nd->Rx.transpose() * data_nd->Ru;
    data->Luu = data_nd->Ru.transpose() * data_nd->Ru;
  } else {
    data->Lxx.fill(0.0);
    data->Lxu.fill(0.0);
    data->Luu.fill(0.0);
  }
}

boost::shared_ptr<ActionDataAbstract> ActionModelNumDiff::createData() {
  return boost::make_shared<ActionDataNumDiff>(this);
}

const boost::shared_ptr<ActionModelAbstract>& ActionModelNumDiff::get_model() const { return model_; }

const double& ActionModelNumDiff::get_disturbance() const { return disturbance_; }

void ActionModelNumDiff::set_disturbance(const double& disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

bool ActionModelNumDiff::get_with_gauss_approx() { return model_->get_nr() > 0; }

void ActionModelNumDiff::assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& /** x */) {
  // do nothing in the general case
}

}  // namespace crocoddyl
