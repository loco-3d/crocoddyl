///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ImpulseModel6DTpl<Scalar>::ImpulseModel6DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const pinocchio::ReferenceFrame type)
    : Base(state, type, 6) {
  id_ = id;
}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::calc(
    const std::shared_ptr<ImpulseDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  pinocchio::updateFramePlacement<Scalar>(*state_->get_pinocchio().get(),
                                          *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_, pinocchio::LOCAL, d->fJf);
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->Jc = d->fJf;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->lwaMl.rotation(d->pinocchio->oMf[id_].rotation());
      data->Jc.noalias() = d->lwaMl.toActionMatrix() * d->fJf;
      break;
  }
}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::calcDiff(
    const std::shared_ptr<ImpulseDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parentJoint;
  pinocchio::getJointVelocityDerivatives(*state_->get_pinocchio().get(),
                                         *d->pinocchio, joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->dv0_local_dq.noalias() = d->fXj * d->v_partial_dq;

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->dv0_dq = d->dv0_local_dq;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
      d->v0 = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                          *d->pinocchio, id_, type_);
      pinocchio::skew(d->v0.linear(), d->vv_skew);
      pinocchio::skew(d->v0.angular(), d->vw_skew);
      d->vv_world_skew.noalias() = d->vv_skew * oRf;
      d->vw_world_skew.noalias() = d->vw_skew * oRf;
      data->dv0_dq.noalias() = d->lwaMl.toActionMatrix() * d->dv0_local_dq;
      d->dv0_dq.template topRows<3>().noalias() -=
          d->vv_world_skew * d->fJf.template bottomRows<3>();
      d->dv0_dq.template bottomRows<3>().noalias() -=
          d->vw_world_skew * d->fJf.template bottomRows<3>();
      break;
  }
}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::updateForce(
    const std::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 6) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 6)");
  }
  Data* d = static_cast<Data*>(data.get());
  data->f = pinocchio::ForceTpl<Scalar>(force);
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->fext = data->jMf.act(data->f);
      data->dtau_dq.setZero();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->f_local = d->lwaMl.actInv(data->f);
      data->fext = data->jMf.act(d->f_local);
      pinocchio::skew(d->f_local.linear(), d->fv_skew);
      pinocchio::skew(d->f_local.angular(), d->fw_skew);
      d->fJf_df.template topRows<3>().noalias() =
          d->fv_skew * d->fJf.template bottomRows<3>();
      d->fJf_df.template bottomRows<3>().noalias() =
          d->fw_skew * d->fJf.template bottomRows<3>();
      d->dtau_dq.noalias() = -d->fJf.transpose() * d->fJf_df;
      break;
  }
}

template <typename Scalar>
std::shared_ptr<ImpulseDataAbstractTpl<Scalar> >
ImpulseModel6DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ImpulseModel6DTpl<NewScalar> ImpulseModel6DTpl<Scalar>::cast() const {
  typedef ImpulseModel6DTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      type_);
  return ret;
}

template <typename Scalar>
void ImpulseModel6DTpl<Scalar>::print(std::ostream& os) const {
  os << "ImpulseModel^D {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", type=" << type_ << "}";
}

}  // namespace crocoddyl
