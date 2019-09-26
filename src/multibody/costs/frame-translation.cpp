///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/costs/frame-translation.hpp"

namespace crocoddyl {

CostModelFrameTranslation::CostModelFrameTranslation(StateMultibody& state, ActivationModelAbstract& activation,
                                                     const FrameTranslation& xref, unsigned int const& nu)
    : CostModelAbstract(state, activation, nu), xref_(xref) {
  assert(activation_.get_nr() == 3 && "activation::nr is not equals to 3");
}

CostModelFrameTranslation::CostModelFrameTranslation(StateMultibody& state, ActivationModelAbstract& activation,
                                                     const FrameTranslation& xref)
    : CostModelAbstract(state, activation), xref_(xref) {
  assert(activation_.get_nr() == 3 && "activation::nr is not equals to 3");
}

CostModelFrameTranslation::CostModelFrameTranslation(StateMultibody& state, const FrameTranslation& xref,
                                                     unsigned int const& nu)
    : CostModelAbstract(state, 3, nu), xref_(xref) {}

CostModelFrameTranslation::CostModelFrameTranslation(StateMultibody& state, const FrameTranslation& xref)
    : CostModelAbstract(state, 3), xref_(xref) {}

CostModelFrameTranslation::~CostModelFrameTranslation() {}

void CostModelFrameTranslation::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>&,
                                     const Eigen::Ref<const Eigen::VectorXd>&) {
  // Compute the frame translation w.r.t. the reference frame
  data->r = data->pinocchio->oMf[xref_.frame].translation() - xref_.oxf;

  // Compute the cost
  activation_.calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelFrameTranslation::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                         const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // Update the frame placements
  CostDataFrameTranslation* d = static_cast<CostDataFrameTranslation*>(data.get());
  pinocchio::updateFramePlacements(state_.get_pinocchio(), *d->pinocchio);

  // Compute the frame Jacobian at the error point
  pinocchio::getFrameJacobian(state_.get_pinocchio(), *d->pinocchio, xref_.frame, pinocchio::LOCAL, d->fJf);
  d->J = d->pinocchio->oMf[xref_.frame].rotation() * d->fJf.topRows<3>();

  // Compute the derivatives of the frame placement
  unsigned int const& nv = state_.get_nv();
  activation_.calcDiff(d->activation, d->r, recalc);
  d->Rx.leftCols(nv) = d->J;
  d->Lx.head(nv) = d->J.transpose() * d->activation->Ar;
  d->Lxx.topLeftCorner(nv, nv) = d->J.transpose() * d->activation->Arr * d->J;
}

boost::shared_ptr<CostDataAbstract> CostModelFrameTranslation::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataFrameTranslation>(this, data);
}

const FrameTranslation& CostModelFrameTranslation::get_xref() const { return xref_; }

}  // namespace crocoddyl
