///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace crocoddyl {

CostModelImpulseCoM::CostModelImpulseCoM(boost::shared_ptr<StateMultibody> state,
                                         boost::shared_ptr<ActivationModelAbstract> activation)
    : CostModelAbstract(state, activation, 0, true) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelImpulseCoM::CostModelImpulseCoM(boost::shared_ptr<StateMultibody> state)
    : CostModelAbstract(state, 3, 0, true) {}

CostModelImpulseCoM::~CostModelImpulseCoM() {}

void CostModelImpulseCoM::calc(const boost::shared_ptr<CostDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>&) {
  // Compute the cost residual give the reference CoM position
  CostDataImpulseCoM* d = static_cast<CostDataImpulseCoM*>(data.get());
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

void CostModelImpulseCoM::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x,
                                   const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }

  CostDataImpulseCoM* d = static_cast<CostDataImpulseCoM*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  const std::size_t& ndx = state_->get_ndx();
  activation_->calcDiff(data->activation, data->r, recalc);

  pinocchio::getCenterOfMassVelocityDerivatives(state_->get_pinocchio(), d->pinocchio_dv, d->dvc_dq);
  pinocchio::jacobianCenterOfMass(state_->get_pinocchio(), d->pinocchio_dv, false);
  data->Rx.leftCols(nv) = d->dvc_dq;
  data->Rx.leftCols(nv).noalias() += d->pinocchio_dv.Jcom * d->impulses->dvnext_dx.leftCols(nv);
  d->ddv_dv = d->impulses->dvnext_dx.rightCols(ndx - nv);
  d->ddv_dv.diagonal().array() -= 1;
  data->Rx.rightCols(ndx - nv).noalias() = d->pinocchio_dv.Jcom * d->ddv_dv;

  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

boost::shared_ptr<CostDataAbstract> CostModelImpulseCoM::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataImpulseCoM>(this, data);
}

}  // namespace crocoddyl
