///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/impulse-cop-position.hpp"

namespace crocoddyl {

template <typename _Scalar>
CostModelImpulseCoPPositionTpl<_Scalar>::CostModelImpulseCoPPositionTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameCoPSupport& cref)
    : Base(state, activation,
           boost::make_shared<ResidualModelContactCoPPosition>(state, cref.get_id(),
                                                               CoPSupport(Matrix3s::Identity(), cref.get_box()), 0)),
      cop_support_(cref) {
  std::cerr << "Deprecated CostModelImpulseCoPPosition: Use CostModelContactCoPPosition" << std::endl;
}

template <typename _Scalar>
CostModelImpulseCoPPositionTpl<_Scalar>::CostModelImpulseCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                                        const FrameCoPSupport& cref)
    : Base(state,
           boost::make_shared<ActivationModelQuadraticBarrier>(
               ActivationBounds(VectorXs::Zero(4), std::numeric_limits<_Scalar>::max() * VectorXs::Ones(4))),
           boost::make_shared<ResidualModelContactCoPPosition>(state, cref.get_id(),
                                                               CoPSupport(Matrix3s::Identity(), cref.get_box()), 0)),
      cop_support_(cref) {
  std::cerr << "Deprecated CostModelImpulseCoPPosition: Use CostModelContactCoPPosition" << std::endl;
}

template <typename Scalar>
CostModelImpulseCoPPositionTpl<Scalar>::~CostModelImpulseCoPPositionTpl() {}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>& x,
                                                  const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference CoP support
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>& x,
                                                      const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and CoP support residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelImpulseCoPPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameCoPSupport)) {
    cop_support_ = *static_cast<const FrameCoPSupport*>(pv);
    ResidualModelContactCoPPosition* residual = static_cast<ResidualModelContactCoPPosition*>(residual_.get());
    residual->set_id(cop_support_.get_id());
    residual->set_reference(CoPSupport(Matrix3s::Identity(), cop_support_.get_box()));
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameCoPSupport)");
  }
}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameCoPSupport)) {
    FrameCoPSupport& ref_map = *static_cast<FrameCoPSupport*>(pv);
    ResidualModelContactCoPPosition* residual = static_cast<ResidualModelContactCoPPosition*>(residual_.get());
    cop_support_.set_id(residual->get_id());
    cop_support_.set_box(residual->get_reference().get_box());
    ref_map = cop_support_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameCoPSupport)");
  }
}

}  // namespace crocoddyl
