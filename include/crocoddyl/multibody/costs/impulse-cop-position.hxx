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
    const FrameCoPSupport& cop_support)
    : Base(state, activation, 0), cop_support_(cop_support) {
  std::cerr << "Deprecated CostModelImpulseCoPPosition class: Use CostModelContactCoPPosition class" << std::endl;
}

template <typename _Scalar>
CostModelImpulseCoPPositionTpl<_Scalar>::CostModelImpulseCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                                        const FrameCoPSupport& cop_support)
    : Base(state,
           boost::make_shared<ActivationModelQuadraticBarrier>(
               ActivationBounds(VectorXs::Zero(4), std::numeric_limits<_Scalar>::max() * VectorXs::Ones(4))),
           0),
      cop_support_(cop_support) {
  std::cerr << "Deprecated CostModelImpulseCoPPosition class: Use CostModelContactCoPPosition class" << std::endl;
}

template <typename Scalar>
CostModelImpulseCoPPositionTpl<Scalar>::~CostModelImpulseCoPPositionTpl() {}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the cost residual r =  A * f
  data->r.noalias() = cop_support_.get_A() * d->impulse->jMf.actInv(d->impulse->f).toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  // Update all data
  Data* d = static_cast<Data*>(data.get());

  // Get the derivatives of the local impulse wrench
  const MatrixXs& df_dx = d->impulse->df_dx;
  const Matrix46& A = cop_support_.get_A();

  // Compute the derivatives of the activation function
  activation_->calcDiff(data->activation, data->r);

  // Compute the derivative of the cost residual
  data->Rx.noalias() = A * df_dx;

  // Compute the first order derivative of the cost function
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;

  // Compute the second order derivative of the cost function
  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
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
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameCoPSupport)");
  }
}

template <typename Scalar>
void CostModelImpulseCoPPositionTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameCoPSupport)) {
    FrameCoPSupport& ref_map = *static_cast<FrameCoPSupport*>(pv);
    ref_map = cop_support_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameCoPSupport)");
  }
}

}  // namespace crocoddyl
