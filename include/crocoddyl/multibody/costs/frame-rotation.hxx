///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameRotation& Rref, std::size_t nu)
    : Base(state, activation, nu),
      Rref_(Rref),
      oRf_inv_(Rref.rotation.transpose()),
      pin_model_(state->get_pinocchio()) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameRotation& Rref)
    : Base(state, activation), Rref_(Rref), oRf_inv_(Rref.rotation.transpose()), pin_model_(state->get_pinocchio()) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameRotation& Rref, std::size_t nu)
    : Base(state, 3, nu), Rref_(Rref), oRf_inv_(Rref.rotation.transpose()), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameRotation& Rref)
    : Base(state, 3), Rref_(Rref), oRf_inv_(Rref.rotation.transpose()), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::~CostModelFrameRotationTpl() {}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, Rref_.id);
  d->rRf.noalias() = oRf_inv_ * d->pinocchio->oMf[Rref_.id].rotation();
  d->r = pinocchio::log3(d->rRf);
  data->r = d->r;  // this is needed because we overwrite it

  // Compute the cost
  activation_->calc(d->activation, d->r);
  d->cost = d->activation->a_value;
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>&,
                                                 const Eigen::Ref<const VectorXs>&) {
  // Update the frame placements
  Data* d = static_cast<Data*>(data.get());

  // // Compute the frame Jacobian at the error point
  pinocchio::Jlog3(d->rRf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, Rref_.id, pinocchio::LOCAL, d->fJf);
  d->J.noalias() = d->rJf * d->fJf.template bottomRows<3>();

  // Compute the derivatives of the frame placement
  std::size_t nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r);
  data->Rx.leftCols(nv) = d->J;
  data->Lx.head(nv).noalias() = d->J.transpose() * data->activation->Ar;
  d->Arr_J.noalias() = data->activation->Arr * d->J;
  data->Lxx.topLeftCorner(nv, nv).noalias() = d->J.transpose() * d->Arr_J;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameRotationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameRotation)) {
    Rref_ = *static_cast<const FrameRotation*>(pv);
    oRf_inv_ = Rref_.rotation.transpose();
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameRotation)");
  }
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameRotation)) {
    FrameRotation& ref_map = *static_cast<FrameRotation*>(pv);
    ref_map = Rref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameRotation)");
  }
}

template <typename Scalar>
const FrameRotationTpl<Scalar>& CostModelFrameRotationTpl<Scalar>::get_Rref() const {
  return Rref_;
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::set_Rref(const FrameRotation& Rref_in) {
  Rref_ = Rref_in;
  oRf_inv_ = Rref_.rotation.transpose();
}

}  // namespace crocoddyl
