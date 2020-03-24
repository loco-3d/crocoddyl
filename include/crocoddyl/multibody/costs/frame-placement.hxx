///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FramePlacement& Mref, const std::size_t& nu)
    : Base(state, activation, nu), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FramePlacement& Mref)
    : Base(state, activation), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FramePlacement& Mref, const std::size_t& nu)
    : Base(state, 6, nu), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()) {}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FramePlacement& Mref)
    : Base(state, 6), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()) {}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::~CostModelFramePlacementTpl() {}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  CostDataFramePlacementTpl<Scalar>* d = static_cast<CostDataFramePlacementTpl<Scalar>*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, Mref_.frame);
  d->rMf = oMf_inv_ * d->pinocchio->oMf[Mref_.frame];
  d->r = pinocchio::log6(d->rMf);
  data->r = d->r;  // this is needed because we overwrite it

  // Compute the cost
  activation_->calc(d->activation, d->r);
  d->cost = d->activation->a_value;
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  // Update the frame placements
  CostDataFramePlacementTpl<Scalar>* d = static_cast<CostDataFramePlacementTpl<Scalar>*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, Mref_.frame, pinocchio::LOCAL, d->fJf);
  d->J.noalias() = d->rJf * d->fJf;

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r);
  data->Rx.leftCols(nv) = d->J;
  data->Lx.head(nv).noalias() = d->J.transpose() * data->activation->Ar;
  d->Arr_J.noalias() = data->activation->Arr * d->J;
  data->Lxx.topLeftCorner(nv, nv).noalias() = d->J.transpose() * d->Arr_J;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFramePlacementTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataFramePlacementTpl<Scalar> >(this, data);
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FramePlacement)) {
    Mref_ = *static_cast<const FramePlacement*>(pv);
  } else {
    throw_pretty("Invalid argument: "
                 << "incorrect type (it should be FramePlacement)");
  }
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FramePlacement)) {
    FramePlacement& ref_map = *static_cast<FramePlacement*>(pv);
    ref_map = Mref_;
  } else {
    throw_pretty("Invalid argument: "
                 << "incorrect type (it should be FramePlacement)");
  }
}

template <typename Scalar>
const FramePlacementTpl<Scalar>& CostModelFramePlacementTpl<Scalar>::get_Mref() const {
  return Mref_;
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::set_Mref(const FramePlacement& Mref_in) {
  Mref_ = Mref_in;
  oMf_inv_ = Mref_.oMf.inverse();
}

}  // namespace crocoddyl
