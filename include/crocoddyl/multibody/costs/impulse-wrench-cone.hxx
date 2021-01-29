///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::CostModelImpulseWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone &fref)
    : Base(state, activation, 0), fref_(fref) {
  if (activation_->get_nr() != fref_.cone.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::CostModelImpulseWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, const FrameWrenchCone &fref)
    : Base(state, fref.cone.get_nf() + 1, 0), fref_(fref) {}

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::~CostModelImpulseWrenchConeTpl() {}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  // Compute the residual of the wrench cone. Note that we need to transform the
  // force to the contact frame
  data->r.noalias() =
      fref_.cone.get_A() * d->impulse->jMf.actInv(d->impulse->f).toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const MatrixXs &df_dx = d->impulse->df_dx;
  const MatrixX6s &A = fref_.cone.get_A();

  activation_->calcDiff(data->activation, data->r);
  data->Rx.noalias() = A * df_dx;

  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelImpulseWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::set_referenceImpl(
    const std::type_info &ti, const void *pv) {
  if (ti == typeid(FrameWrenchCone)) {
    fref_ = *static_cast<const FrameWrenchCone *>(pv);
  } else {
    throw_pretty(
        "Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::get_referenceImpl(
    const std::type_info &ti, void *pv) const {
  if (ti == typeid(FrameWrenchCone)) {
    FrameWrenchCone &ref_map = *static_cast<FrameWrenchCone *>(pv);
    ref_map = fref_;
  } else {
    throw_pretty(
        "Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

} // namespace crocoddyl
