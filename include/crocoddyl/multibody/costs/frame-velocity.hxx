///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/frame-velocity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameMotion& vref, const std::size_t nu)
    : Base(state, activation,
           boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference, nu)),
      vref_(vref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameMotion& vref)
    : Base(state, activation,
           boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference)),
      vref_(vref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameMotion& vref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference, nu)),
      vref_(vref) {}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameMotion& vref)
    : Base(state, boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference)),
      vref_(vref) {}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::~CostModelFrameVelocityTpl() {}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& x,
                                             const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference frame velocity
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, d->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and frame velocity residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(data->activation, d->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = d->residual->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * d->residual->Rx;
  data->Lxx.noalias() = d->residual->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameVelocityTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
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
void CostModelFrameVelocityTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameMotion)) {
    vref_ = *static_cast<const FrameMotion*>(pv);
    ResidualModelFrameVelocity* residual = static_cast<ResidualModelFrameVelocity*>(residual_.get());
    residual->set_id(vref_.id);
    residual->set_reference(vref_.motion);
    residual->set_type(vref_.reference);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

}  // namespace crocoddyl
