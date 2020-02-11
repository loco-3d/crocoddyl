///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/costs/frame-rotation.hpp"

namespace crocoddyl {

CostModelFrameRotation::CostModelFrameRotation(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                               const FrameRotation& Rref, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), Rref_(Rref), oRf_inv_(Rref.oRf.transpose()) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelFrameRotation::CostModelFrameRotation(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                               const FrameRotation& Rref)
    : CostModelAbstract(state, activation), Rref_(Rref), oRf_inv_(Rref.oRf.transpose()) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelFrameRotation::CostModelFrameRotation(boost::shared_ptr<StateMultibody> state, const FrameRotation& Rref,
                                               const std::size_t& nu)
    : CostModelAbstract(state, 3, nu), Rref_(Rref), oRf_inv_(Rref.oRf.transpose()) {}

CostModelFrameRotation::CostModelFrameRotation(boost::shared_ptr<StateMultibody> state, const FrameRotation& Rref)
    : CostModelAbstract(state, 3), Rref_(Rref), oRf_inv_(Rref.oRf.transpose()) {}

CostModelFrameRotation::~CostModelFrameRotation() {}

void CostModelFrameRotation::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&) {
  CostDataFrameRotation* d = static_cast<CostDataFrameRotation*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(state_->get_pinocchio(), *d->pinocchio, Rref_.frame);
  d->rRf.noalias() = oRf_inv_ * d->pinocchio->oMf[Rref_.frame].rotation();
  d->r = pinocchio::log3(d->rRf);
  data->r = d->r;  // this is needed because we overwrite it

  // Compute the cost
  activation_->calc(d->activation, d->r);
  d->cost = d->activation->a_value;
}

void CostModelFrameRotation::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                      const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // Update the frame placements
  CostDataFrameRotation* d = static_cast<CostDataFrameRotation*>(data.get());

  // // Compute the frame Jacobian at the error point
  pinocchio::Jlog3(d->rRf, d->rJf);
  pinocchio::getFrameJacobian(state_->get_pinocchio(), *d->pinocchio, Rref_.frame, pinocchio::LOCAL, d->fJf);
  d->J.noalias() = d->rJf * d->fJf.topRows<3>();

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r, recalc);
  data->Rx.leftCols(nv) = d->J;
  data->Lx.head(nv).noalias() = d->J.transpose() * data->activation->Ar;
  d->Arr_J.noalias() = data->activation->Arr * d->J;
  data->Lxx.topLeftCorner(nv, nv).noalias() = d->J.transpose() * d->Arr_J;
}

boost::shared_ptr<CostDataAbstract> CostModelFrameRotation::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataFrameRotation>(this, data);
}

const FrameRotation& CostModelFrameRotation::get_Rref() const { return Rref_; }

void CostModelFrameRotation::set_Rref(const FrameRotation& Rref_in) {
  Rref_ = Rref_in;
  oRf_inv_ = Rref_.oRf.transpose();
}

}  // namespace crocoddyl
