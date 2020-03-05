///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ImpulseModel6DTpl<Scalar>::ImpulseModel6DTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& frame)
    : Base(state, 6), frame_(frame) {}

template <typename Scalar>
ImpulseModel6DTpl<Scalar>::~ImpulseModel6DTpl() {}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  boost::shared_ptr<ImpulseData6D> d = boost::static_pointer_cast<ImpulseData6D>(data);

  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, frame_, pinocchio::LOCAL, d->Jc);
}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  boost::shared_ptr<ImpulseData6D> d = boost::static_pointer_cast<ImpulseData6D>(data);
  pinocchio::getJointVelocityDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->dv0_dq.noalias() = d->fXj * d->v_partial_dq;
}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 6) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 6)");
  }
  boost::shared_ptr<ImpulseData6D> d = boost::static_pointer_cast<ImpulseData6D>(data);
  data->f = d->jMf.act(pinocchio::ForceTpl<Scalar>(force));
}

template <typename Scalar>
boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > ImpulseModel6DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::make_shared<ImpulseData6D>(this, data);
}
template <typename Scalar>
const std::size_t& ImpulseModel6DTpl<Scalar>::get_frame() const {
  return frame_;
}

}  // namespace crocoddyl
