///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-friction-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseFrictionConeTpl<Scalar>::CostModelImpulseFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameFrictionCone& fref)
    : Base(state, activation, 0), fref_(fref) {
  if (activation_->get_nr() != fref_.oRf.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.oRf.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelImpulseFrictionConeTpl<Scalar>::CostModelImpulseFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                         const FrameFrictionCone& fref)
    : Base(state, fref.oRf.get_nf() + 1, 0), fref_(fref) {}

template <typename Scalar>
CostModelImpulseFrictionConeTpl<Scalar>::~CostModelImpulseFrictionConeTpl() {}

template <typename Scalar>
void CostModelImpulseFrictionConeTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>&,
                                                   const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual of the friction cone. Note that we need to transform the force
  // to the contact frame
  data->r.noalias() = fref_.oRf.get_A() * d->impulse->jMf.actInv(d->impulse->f).linear();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseFrictionConeTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                       const Eigen::Ref<const VectorXs>&,
                                                       const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const std::size_t& nv = state_->get_nv();
  const MatrixXs& df_dx = d->impulse->df_dx;
  const MatrixX3s& A = fref_.oRf.get_A();

  activation_->calcDiff(data->activation, data->r);
  if (d->more_than_3_constraints) {
    data->Rx.noalias() = A * df_dx.template topRows<3>();
  } else {
    data->Rx.noalias() = A * df_dx;
  }
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelImpulseFrictionConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelImpulseFrictionConeTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameFrictionCone)) {
    fref_ = *static_cast<const FrameFrictionCone*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameFrictionCone)");
  }
}

template <typename Scalar>
void CostModelImpulseFrictionConeTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameFrictionCone)) {
    FrameFrictionCone& ref_map = *static_cast<FrameFrictionCone*>(pv);
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameFrictionCone)");
  }
}

template <typename Scalar>
const FrameFrictionConeTpl<Scalar>& CostModelImpulseFrictionConeTpl<Scalar>::get_fref() const {
  return fref_;
}

template <typename Scalar>
void CostModelImpulseFrictionConeTpl<Scalar>::set_fref(const FrameFrictionCone& fref_in) {
  fref_ = fref_in;
}

}  // namespace crocoddyl
