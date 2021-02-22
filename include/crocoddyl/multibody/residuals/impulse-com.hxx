///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ResidualModelImpulseCoMTpl<Scalar>::ResidualModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, 3, 0), pin_model_(state->get_pinocchio()) {
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelImpulseCoMTpl<Scalar>::~ResidualModelImpulseCoMTpl() {}

template <typename Scalar>
void ResidualModelImpulseCoMTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  // Compute the residual residual give the reference CoM position
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  pinocchio::centerOfMass(*pin_model_.get(), d->pinocchio_internal, q, d->impulses->vnext - v);
  data->r = d->pinocchio_internal.vcom[0];
}

template <typename Scalar>
void ResidualModelImpulseCoMTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  const std::size_t ndx = state_->get_ndx();
  pinocchio::getCenterOfMassVelocityDerivatives(*pin_model_.get(), d->pinocchio_internal, d->dvc_dq);
  pinocchio::jacobianCenterOfMass(*pin_model_.get(), d->pinocchio_internal, false);
  d->ddv_dv = d->impulses->dvnext_dx.rightCols(ndx - nv);
  d->ddv_dv.diagonal().array() -= 1;
  data->Rx.leftCols(nv) = d->dvc_dq;
  data->Rx.leftCols(nv).noalias() += d->pinocchio_internal.Jcom * d->impulses->dvnext_dx.leftCols(nv);
  data->Rx.rightCols(ndx - nv).noalias() = d->pinocchio_internal.Jcom * d->ddv_dv;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelImpulseCoMTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

}  // namespace crocoddyl
