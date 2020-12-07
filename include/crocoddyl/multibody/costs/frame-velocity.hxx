///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameMotion& vref, std::size_t nu)
    : Base(state, activation, nu), vref_(vref), pin_model_(state->get_pinocchio()) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameMotion& vref)
    : Base(state, activation), vref_(vref), pin_model_(state->get_pinocchio()) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameMotion& vref, std::size_t nu)
    : Base(state, 6, nu), vref_(vref), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameMotion& vref)
    : Base(state, 6), vref_(vref), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::~CostModelFrameVelocityTpl() {}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  data->r = (pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, vref_.id, vref_.reference) - vref_.motion)
                .toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>&,
                                                 const Eigen::Ref<const VectorXs>&) {
  // Get the partial derivatives of the local frame velocity
  Data* d = static_cast<Data*>(data.get());
  std::size_t nv = state_->get_nv();
  pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio, vref_.id, vref_.reference,
                                         data->Rx.leftCols(nv), data->Rx.rightCols(nv));

  // Compute the derivatives of the frame velocity
  activation_->calcDiff(data->activation, data->r);
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameVelocityTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameMotion)) {
    vref_ = *static_cast<const FrameMotion*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameMotion)) {
    FrameMotion& ref_map = *static_cast<FrameMotion*>(pv);
    ref_map = vref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

template <typename Scalar>
const FrameMotionTpl<Scalar>& CostModelFrameVelocityTpl<Scalar>::get_vref() const {
  return vref_;
}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::set_vref(const FrameMotion& vref_in) {
  vref_ = vref_in;
}

}  // namespace crocoddyl
