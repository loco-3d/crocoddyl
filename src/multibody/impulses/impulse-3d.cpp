///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

ImpulseModel3D::ImpulseModel3D(boost::shared_ptr<StateMultibody> state, const std::size_t& frame)
    : ImpulseModelAbstract(state, 3), frame_(frame) {}

ImpulseModel3D::~ImpulseModel3D() {}

void ImpulseModel3D::calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  boost::shared_ptr<ImpulseData3D> d = boost::static_pointer_cast<ImpulseData3D>(data);

  pinocchio::getFrameJacobian(state_->get_pinocchio(), *d->pinocchio, frame_, pinocchio::LOCAL, d->fJf);
  d->Jc = d->fJf.topRows<3>();
}

void ImpulseModel3D::calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool&) {
  boost::shared_ptr<ImpulseData3D> d = boost::static_pointer_cast<ImpulseData3D>(data);
  pinocchio::getJointVelocityDerivatives(state_->get_pinocchio(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->dv0_dq.noalias() = d->fXj.topRows<3>() * d->v_partial_dq;
}

void ImpulseModel3D::updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& force) {
  if (force.size() != 3) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 3)");
  }
  boost::shared_ptr<ImpulseData3D> d = boost::static_pointer_cast<ImpulseData3D>(data);
  data->f = d->jMf.act(pinocchio::Force(force, Eigen::Vector3d::Zero()));
}

boost::shared_ptr<ImpulseDataAbstract> ImpulseModel3D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseData3D>(this, data);
}

const std::size_t& ImpulseModel3D::get_frame() const { return frame_; }

}  // namespace crocoddyl
