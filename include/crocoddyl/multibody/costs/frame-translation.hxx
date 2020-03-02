///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameTranslation& xref, const std::size_t& nu)
    : Base(state, activation, nu), xref_(xref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameTranslation& xref)
    : Base(state, activation), xref_(xref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const FrameTranslation& xref, const std::size_t& nu)
    : Base(state, 3, nu), xref_(xref) {}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const FrameTranslation& xref)
    : Base(state, 3), xref_(xref) {}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::~CostModelFrameTranslationTpl() {}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Compute the frame translation w.r.t. the reference frame
  CostDataFrameTranslationTpl<Scalar>* d = static_cast<CostDataFrameTranslationTpl<Scalar>*>(data.get());
  pinocchio::updateFramePlacement(state_->get_pinocchio(), *d->pinocchio, xref_.frame);
  data->r = d->pinocchio->oMf[xref_.frame].translation() - xref_.oxf;

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                    const Eigen::Ref<const VectorXs>&,
                                                    const Eigen::Ref<const VectorXs>&) {
  // Update the frame placements
  CostDataFrameTranslationTpl<Scalar>* d = static_cast<CostDataFrameTranslationTpl<Scalar>*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::getFrameJacobian(state_->get_pinocchio(), *d->pinocchio, xref_.frame, pinocchio::LOCAL, d->fJf);
  d->J = d->pinocchio->oMf[xref_.frame].rotation() * d->fJf.template topRows<3>();

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(d->activation, d->r);
  d->Rx.leftCols(nv) = d->J;
  d->Lx.head(nv) = d->J.transpose() * d->activation->Ar;
  d->Lxx.topLeftCorner(nv, nv) = d->J.transpose() * d->activation->Arr * d->J;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameTranslationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataFrameTranslationTpl<Scalar> >(this, data);
}

template <typename Scalar>
const FrameTranslationTpl<Scalar>& CostModelFrameTranslationTpl<Scalar>::get_xref() const {
  return xref_;
}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::set_xref(const FrameTranslation& xref_in) {
  xref_ = xref_in;
}

}  // namespace crocoddyl
