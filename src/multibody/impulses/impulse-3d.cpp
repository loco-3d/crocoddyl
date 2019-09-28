///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

ImpulseModel3D::ImpulseModel3D(StateMultibody& state, unsigned int const& frame)
    : ImpulseModelAbstract(state, 3), frame_(frame) {}

ImpulseModel3D::~ImpulseModel3D() {}

void ImpulseModel3D::calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  ImpulseData3D* d = static_cast<ImpulseData3D*>(data.get());

  pinocchio::getFrameJacobian(state_.get_pinocchio(), *d->pinocchio, frame_, pinocchio::LOCAL, d->fJf);
  d->Jc = d->fJf.topRows<3>();
}

void ImpulseModel3D::calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }

  ImpulseData3D* d = static_cast<ImpulseData3D*>(data.get());
  pinocchio::getJointVelocityDerivatives(state_.get_pinocchio(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->Vq.noalias() = d->fXj.topRows<3>() * d->v_partial_dq;
}

void ImpulseModel3D::updateLagrangian(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                      const Eigen::VectorXd& lambda) {
  assert(lambda.size() == 3 && "lambda has wrong dimension, it should be 3d vector");
  ImpulseData3D* d = static_cast<ImpulseData3D*>(data.get());
  data->f = d->jMf.act(pinocchio::Force(lambda, Eigen::Vector3d::Zero()));
}

boost::shared_ptr<ImpulseDataAbstract> ImpulseModel3D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseData3D>(this, data);
}

unsigned int const& ImpulseModel3D::get_frame() const { return frame_; }

}  // namespace crocoddyl
