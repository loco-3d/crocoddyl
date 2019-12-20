///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

ImpulseModel6D::ImpulseModel6D(boost::shared_ptr<StateMultibody> state, const std::size_t& frame)
    : ImpulseModelAbstract(state, 6), frame_(frame) {}

ImpulseModel6D::~ImpulseModel6D() {}

void ImpulseModel6D::calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  boost::shared_ptr<ImpulseData6D> d = boost::static_pointer_cast<ImpulseData6D>(data);

  pinocchio::getFrameJacobian(state_->get_pinocchio(), *d->pinocchio, frame_, pinocchio::LOCAL, d->Jc);
}

void ImpulseModel6D::calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }

  boost::shared_ptr<ImpulseData6D> d = boost::static_pointer_cast<ImpulseData6D>(data);
  pinocchio::getJointVelocityDerivatives(state_->get_pinocchio(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->dv0_dq.noalias() = d->fXj * d->v_partial_dq;
}

void ImpulseModel6D::updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& force) {
  if (force.size() != 6) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 6)");
  }
  boost::shared_ptr<ImpulseData6D> d = boost::static_pointer_cast<ImpulseData6D>(data);
  data->f = d->jMf.act(pinocchio::Force(force));
}

boost::shared_ptr<ImpulseDataAbstract> ImpulseModel6D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseData6D>(this, data);
}

const std::size_t& ImpulseModel6D::get_frame() const { return frame_; }

}  // namespace crocoddyl
