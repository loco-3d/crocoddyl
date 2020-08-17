///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone& fref, const std::size_t& nu)
    : Base(state, activation, nu), fref_(fref) {
  if (activation_->get_nr() != fref_.oRf.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.oRf.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone& fref)
    : Base(state, activation), fref_(fref) {
  if (activation_->get_nr() != fref_.oRf.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.oRf.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                         const FrameWrenchCone& fref,
                                                                         const std::size_t& nu)
    : Base(state, fref.oRf.get_nf() + 1, nu), fref_(fref) {}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                         const FrameWrenchCone& fref)
    : Base(state, fref.oRf.get_nf() + 1), fref_(fref) {}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::~CostModelContactWrenchConeTpl() {}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>&,
                                                   const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual of the wrench cone. Note that we need to transform the wrench
  // to the contact frame
  data->r.noalias() = fref_.oRf.get_A() * d->contact->jMf.actInv(d->contact->f).toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                       const Eigen::Ref<const VectorXs>&,
                                                       const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const MatrixX6s& A = fref_.oRf.get_A();

  activation_->calcDiff(data->activation, data->r);
  
  data->Rx.noalias() = A * df_dx;
  data->Ru.noalias() = A * df_du;

  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->Ru.transpose() * data->activation->Ar;

  d->Arr_Ru.noalias() = data->activation->Arr * data->Ru;

  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
  data->Lxu.noalias() = data->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameWrenchCone)) {
    fref_ = *static_cast<const FrameWrenchCone*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameWrenchCone)) {
    FrameWrenchCone& ref_map = *static_cast<FrameWrenchCone*>(pv);
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

template <typename Scalar>
const FrameWrenchConeTpl<Scalar>& CostModelContactWrenchConeTpl<Scalar>::get_fref() const {
  return fref_;
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::set_fref(const FrameWrenchCone& fref_in) {
  fref_ = fref_in;
}

}  // namespace crocoddyl
