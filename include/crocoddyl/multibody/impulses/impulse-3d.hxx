///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ImpulseModel3DTpl<Scalar>::ImpulseModel3DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id)
    : Base(state, 3) {
  id_ = id;
}

template <typename Scalar>
ImpulseModel3DTpl<Scalar>::~ImpulseModel3DTpl() {}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  d->Jc = d->fJf.template topRows<3>();
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  pinocchio::getJointVelocityDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->dv0_dq.noalias() = d->fXj.template topRows<3>() * d->v_partial_dq;
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 3) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 3)");
  }
  data->f.linear() = force;
  data->f.angular().setZero();
  data->fext = data->jMf.act(data->f);
}

template <typename Scalar>
boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > ImpulseModel3DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::print(std::ostream& os) const {
  os << "ImpulseModel3D {frame=" << state_->get_pinocchio()->frames[id_].name << "}";
}

}  // namespace crocoddyl
