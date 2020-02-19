///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include "crocoddyl/multibody/costs/frame-velocity.hpp"

namespace crocoddyl {
  
template<typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                               const FrameMotion& vref, const std::size_t& nu)
    : Base(state, activation, nu), vref_(vref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template<typename Scalar>  
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                               const FrameMotion& vref)
    : Base(state, activation), vref_(vref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

  
template<typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state, const FrameMotion& vref,
                                               const std::size_t& nu)
    : Base(state, 6, nu), vref_(vref) {}

template<typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state, const FrameMotion& vref)
    : Base(state, 6), vref_(vref) {}

template<typename Scalar>
CostModelFrameVelocityTpl<Scalar>::~CostModelFrameVelocityTpl() {}

template<typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                  const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  CostDataFrameVelocityTpl<Scalar>* d = static_cast<CostDataFrameVelocityTpl<Scalar>* >(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  d->vr = pinocchio::getFrameVelocity(state_->get_pinocchio(), *d->pinocchio, vref_.frame) - vref_.oMf;
  data->r = d->vr.toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template<typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                      const Eigen::Ref<const VectorXs>&,
                                      const Eigen::Ref<const VectorXs>&) {
  // Get the partial derivatives of the local frame velocity
  CostDataFrameVelocityTpl<Scalar>* d =
    static_cast<CostDataFrameVelocityTpl<Scalar>* >(data.get());
  pinocchio::getJointVelocityDerivatives(state_->get_pinocchio(), *d->pinocchio, d->joint, pinocchio::LOCAL, d->dv_dq,
                                         d->dv_dv);

  // Compute the derivatives of the frame velocity
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r);
  data->Rx.leftCols(nv).noalias() = d->fXj * d->dv_dq;
  data->Rx.rightCols(nv).noalias() = d->fXj * d->dv_dv;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

template<typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameVelocityTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataFrameVelocityTpl<Scalar> >(this, data);
}

template<typename Scalar>
const FrameMotionTpl<Scalar>& CostModelFrameVelocityTpl<Scalar>::get_vref() const { return vref_; }

template<typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::set_vref(const FrameMotion& vref_in) { vref_ = vref_in; }

}  // namespace crocoddyl
