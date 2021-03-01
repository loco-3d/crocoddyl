///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameFrictionCone& fref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelContactFrictionCone>(state, fref.id, fref.cone, nu)),
      fref_(fref) {
  std::cerr << "Deprecated CostModelContactFrictionCone: Use ResidualModelContactFrictionCone with "
               "CostModelResidual class"
            << std::endl;
  if (activation_->get_nr() != fref_.cone.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameFrictionCone& fref)
    : Base(state, activation, boost::make_shared<ResidualModelContactFrictionCone>(state, fref.id, fref.cone)),
      fref_(fref) {
  std::cerr << "Deprecated CostModelContactFrictionCone: Use ResidualModelContactFrictionCone with "
               "CostModelResidual class"
            << std::endl;
  if (activation_->get_nr() != fref_.cone.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                         const FrameFrictionCone& fref,
                                                                         const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelContactFrictionCone>(state, fref.id, fref.cone, nu)), fref_(fref) {
  std::cerr << "Deprecated CostModelContactFrictionCone: Use ResidualModelContactFrictionCone with "
               "CostModelResidual class"
            << std::endl;
}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                         const FrameFrictionCone& fref)
    : Base(state, boost::make_shared<ResidualModelContactFrictionCone>(state, fref.id, fref.cone)), fref_(fref) {
  std::cerr << "Deprecated CostModelContactFrictionCone: Use ResidualModelContactFrictionCone with "
               "CostModelResidual class"
            << std::endl;
}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::~CostModelContactFrictionConeTpl() {}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>& x,
                                                   const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference contact friction
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                       const Eigen::Ref<const VectorXs>& x,
                                                       const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and contact friction cone residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->residual->Ru.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  d->Arr_Ru.noalias() = data->activation->Arr * data->residual->Ru;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
  data->Lxu.noalias() = data->residual->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactFrictionConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameFrictionCone)) {
    fref_ = *static_cast<const FrameFrictionCone*>(pv);
    ResidualModelContactFrictionCone* residual = static_cast<ResidualModelContactFrictionCone*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.cone);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameFrictionCone)");
  }
}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameFrictionCone)) {
    FrameFrictionCone& ref_map = *static_cast<FrameFrictionCone*>(pv);
    ResidualModelContactFrictionCone* residual = static_cast<ResidualModelContactFrictionCone*>(residual_.get());
    fref_.id = residual->get_id();
    fref_.cone = residual->get_reference();
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameFrictionCone)");
  }
}

}  // namespace crocoddyl
