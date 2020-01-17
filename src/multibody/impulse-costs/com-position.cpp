///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulse-costs/com-position.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace crocoddyl {

ImpulseCostModelCoM::ImpulseCostModelCoM(boost::shared_ptr<StateMultibody> state,
                                         boost::shared_ptr<ActivationModelAbstract> activation)
    : ImpulseCostModelAbstract(state, activation) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

ImpulseCostModelCoM::ImpulseCostModelCoM(boost::shared_ptr<StateMultibody> state)
    : ImpulseCostModelAbstract(state, 3) {}

ImpulseCostModelCoM::~ImpulseCostModelCoM() {}

void ImpulseCostModelCoM::calc(const boost::shared_ptr<ImpulseCostDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  // Compute the cost residual give the reference CoM position
  ImpulseCostDataCoM* d = static_cast<ImpulseCostDataCoM*>(data.get());
  const std::size_t& nq = state_->get_nq();
  const std::size_t& nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(nv);

  pinocchio::centerOfMass(state_->get_pinocchio(), d->pinocchio_dv, q, d->impulses->vnext - v);
  data->r = d->pinocchio_dv.vcom[0];

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void ImpulseCostModelCoM::calcDiff(const boost::shared_ptr<ImpulseCostDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }

  ImpulseCostDataCoM* d = static_cast<ImpulseCostDataCoM*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  const std::size_t& ndx = state_->get_ndx();
  activation_->calcDiff(data->activation, data->r, recalc);

  pinocchio::getCenterOfMassVelocityDerivatives(state_->get_pinocchio(), d->pinocchio_dv, d->dvc_dq);
  pinocchio::jacobianCenterOfMass(state_->get_pinocchio(), d->pinocchio_dv, false);
  data->Rx.leftCols(nv) = d->dvc_dq + d->pinocchio_dv.Jcom * data->impulses->dvnext_dx.leftCols(nv);
  d->ddv_dv = data->impulses->dvnext_dx.rightCols(ndx - nv);
  d->ddv_dv.diagonal().array() -= 1;
  data->Rx.rightCols(ndx - nv).noalias() = d->pinocchio_dv.Jcom * d->ddv_dv;

  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

boost::shared_ptr<ImpulseCostDataAbstract> ImpulseCostModelCoM::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<ImpulseCostDataCoM>(this, data);
}

}  // namespace crocoddyl
