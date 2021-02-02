///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::CostModelImpulseWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone& fref)
    : Base(state, activation, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone, 0)),
      fref_(fref) {
  if (activation_->get_nr() != fref_.cone.get_nf() + 13) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::CostModelImpulseWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const FrameWrenchCone& fref)
    : Base(state, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone, 0)), fref_(fref) {}

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::~CostModelImpulseWrenchConeTpl() {}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference impulse wrench
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(d->activation, d->residual->r);
  d->cost = d->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x,
                                                     const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and impulse wrench cone residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelImpulseWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameWrenchCone)) {
    fref_ = *static_cast<const FrameWrenchCone*>(pv);
    ResidualModelContactWrenchCone* residual = static_cast<ResidualModelContactWrenchCone*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.cone);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameWrenchCone)) {
    FrameWrenchCone& ref_map = *static_cast<FrameWrenchCone*>(pv);
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

}  // namespace crocoddyl
