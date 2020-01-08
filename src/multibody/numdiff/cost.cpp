///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/numdiff/cost.hpp"

namespace crocoddyl {

CostNumDiffModel::CostNumDiffModel(const boost::shared_ptr<CostModelAbstract>& model)
    : CostModelAbstract(model->get_state(), model->get_activation(), model->get_nu(), model->get_with_residuals()),
      model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
}

CostNumDiffModel::~CostNumDiffModel() {}

void CostNumDiffModel::calc(const boost::shared_ptr<CostDataAbstract>& data,
                            const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  boost::shared_ptr<CostDataNumDiff> data_nd = boost::static_pointer_cast<CostDataNumDiff>(data);
  data_nd->data_0->cost = 0.0;
  model_->calc(data_nd->data_0, x, u);
  data_nd->cost = data_nd->data_0->cost;
}

void CostNumDiffModel::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                                const bool& recalc) {
  boost::shared_ptr<CostDataNumDiff> data_nd = boost::static_pointer_cast<CostDataNumDiff>(data);

  if (recalc) {
    model_->calc(data_nd->data_0, x, u);
  }

  const double& c0 = data_nd->data_0->cost;
  data_nd->cost = c0;

  assertStableStateFD(x);

  // Computing the d cost(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](data_nd->xp);
    }
    // cost(x+dx, u)
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);
    // Lx
    data_nd->Lx(ix) = (data_nd->data_x[ix]->cost - c0) / disturbance_;
    // Check if we need to/can compute the Gauss approximation of the Hessian.
    if (get_with_gauss_approx()) {
      data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - data_nd->data_0->r) / disturbance_;
    }
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d cost(x,u) / du
  data_nd->du.setZero();
  // call the update function on the pinocchio data
  for (size_t i = 0; i < reevals_.size(); ++i) {
    reevals_[i](x);
  }
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    // up = u + du
    data_nd->du(iu) = disturbance_;
    data_nd->up = u + data_nd->du;
    // cost(x, u+du)
    model_->calc(data_nd->data_u[iu], x, data_nd->up);
    // Lu
    data_nd->Lu(iu) = (data_nd->data_u[iu]->cost - c0) / disturbance_;
    // Check if we need to/can compute the Gauss approximation of the Hessian.
    if (get_with_gauss_approx()) {
      data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - data_nd->data_0->r) / disturbance_;
    }
    data_nd->du(iu) = 0.0;
  }

  if (get_with_gauss_approx()) {
    data_nd->Lxx = data_nd->Rx.transpose() * data_nd->Rx;
    data_nd->Lxu = data_nd->Rx.transpose() * data_nd->Ru;
    data_nd->Luu = data_nd->Ru.transpose() * data_nd->Ru;
  } else {
    data_nd->Lxx.fill(0.0);
    data_nd->Lxu.fill(0.0);
    data_nd->Luu.fill(0.0);
  }
}

boost::shared_ptr<CostDataAbstract> CostNumDiffModel::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataNumDiff>(this, data);
}

const boost::shared_ptr<CostModelAbstract>& CostNumDiffModel::get_model() const { return model_; }

const double& CostNumDiffModel::get_disturbance() const { return disturbance_; }

void CostNumDiffModel::set_disturbance(const double& disturbance) { disturbance_ = disturbance; }

bool CostNumDiffModel::get_with_gauss_approx() { return activation_->get_nr() > 0; }

void CostNumDiffModel::set_reevals(const std::vector<ReevaluationFunction>& reevals) { reevals_ = reevals; }

}  // namespace crocoddyl
