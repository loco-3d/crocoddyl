///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/costs/frame-translation.hpp"

namespace crocoddyl {

CostModelFrameTranslation::CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state,
                                                     boost::shared_ptr<ActivationModelAbstract> activation,
                                                     const FrameTranslation& xref, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), xref_(xref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelFrameTranslation::CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state,
                                                     boost::shared_ptr<ActivationModelAbstract> activation,
                                                     const FrameTranslation& xref)
    : CostModelAbstract(state, activation), xref_(xref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelFrameTranslation::CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state,
                                                     const FrameTranslation& xref, const std::size_t& nu)
    : CostModelAbstract(state, 3, nu), xref_(xref) {}

CostModelFrameTranslation::CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state,
                                                     const FrameTranslation& xref)
    : CostModelAbstract(state, 3), xref_(xref) {}

CostModelFrameTranslation::~CostModelFrameTranslation() {}

void CostModelFrameTranslation::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>&,
                                     const Eigen::Ref<const Eigen::VectorXd>&) {
  // Compute the frame translation w.r.t. the reference frame
  CostDataFrameTranslation* d = static_cast<CostDataFrameTranslation*>(data.get());
  pinocchio::updateFramePlacement(state_->get_pinocchio(), *d->pinocchio, xref_.frame);
  data->r = d->pinocchio->oMf[xref_.frame].translation() - xref_.oxf;

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelFrameTranslation::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                         const Eigen::Ref<const Eigen::VectorXd>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&) {
  // Update the frame placements
  CostDataFrameTranslation* d = static_cast<CostDataFrameTranslation*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::getFrameJacobian(state_->get_pinocchio(), *d->pinocchio, xref_.frame, pinocchio::LOCAL, d->fJf);
  d->J = d->pinocchio->oMf[xref_.frame].rotation() * d->fJf.topRows<3>();

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(d->activation, d->r);
  d->Rx.leftCols(nv) = d->J;
  d->Lx.head(nv) = d->J.transpose() * d->activation->Ar;
  d->Lxx.topLeftCorner(nv, nv) = d->J.transpose() * d->activation->Arr * d->J;
}

boost::shared_ptr<CostDataAbstract> CostModelFrameTranslation::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataFrameTranslation>(this, data);
}

const FrameTranslation& CostModelFrameTranslation::get_xref() const { return xref_; }

void CostModelFrameTranslation::set_xref(const FrameTranslation& xref_in) { xref_ = xref_in; }

}  // namespace crocoddyl
