///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state,
                                                       boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, 0, true) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, 3, 0, true) {}

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::~CostModelImpulseCoMTpl() {}

template <typename Scalar>
void CostModelImpulseCoMTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                          const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  // Compute the cost residual give the reference CoM position
  CostDataImpulseCoMTpl<Scalar>* d = static_cast<CostDataImpulseCoMTpl<Scalar>*>(data.get());
  const std::size_t& nq = state_->get_nq();
  const std::size_t& nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  pinocchio::centerOfMass(state_->get_pinocchio(), d->pinocchio_internal, q, d->impulses->vnext - v);
  data->r = d->pinocchio_internal.vcom[0];

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseCoMTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  CostDataImpulseCoMTpl<Scalar>* d = static_cast<CostDataImpulseCoMTpl<Scalar>*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  const std::size_t& ndx = state_->get_ndx();
  activation_->calcDiff(data->activation, data->r);

  pinocchio::getCenterOfMassVelocityDerivatives(state_->get_pinocchio(), d->pinocchio_internal, d->dvc_dq);
  pinocchio::jacobianCenterOfMass(state_->get_pinocchio(), d->pinocchio_internal, false);
  data->Rx.leftCols(nv) = d->dvc_dq;
  data->Rx.leftCols(nv).noalias() += d->pinocchio_internal.Jcom * d->impulses->dvnext_dx.leftCols(nv);
  d->ddv_dv = d->impulses->dvnext_dx.rightCols(ndx - nv);
  d->ddv_dv.diagonal().array() -= 1;
  data->Rx.rightCols(ndx - nv).noalias() = d->pinocchio_internal.Jcom * d->ddv_dv;

  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelImpulseCoMTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataImpulseCoMTpl<Scalar> >(this, data);
}

}  // namespace crocoddyl
