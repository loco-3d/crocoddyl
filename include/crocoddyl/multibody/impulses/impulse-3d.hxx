///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ImpulseModel3DTpl<Scalar>::ImpulseModel3DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const pinocchio::ReferenceFrame type)
    : Base(state, type, 3) {
  id_ = id;
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::calc(
    const std::shared_ptr<ImpulseDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  pinocchio::updateFramePlacement<Scalar>(*state_->get_pinocchio().get(),
                                          *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_, pinocchio::LOCAL, d->fJf);

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->Jc = d->fJf.template topRows<3>();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      data->Jc.noalias() =
          d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
      break;
  }
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::calcDiff(
    const std::shared_ptr<ImpulseDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parentJoint;
  pinocchio::getJointVelocityDerivatives(*state_->get_pinocchio().get(),
                                         *d->pinocchio, joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);
  d->dv0_local_dq.noalias() = d->fXj.template topRows<3>() * d->v_partial_dq;

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->dv0_dq = d->dv0_local_dq;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
      d->v0 = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                          *d->pinocchio, id_,
                                          pinocchio::LOCAL_WORLD_ALIGNED)
                  .linear();
      pinocchio::skew(d->v0, d->v0_skew);
      d->v0_world_skew.noalias() = d->v0_skew * oRf;
      data->dv0_dq.noalias() = oRf * d->dv0_local_dq;
      data->dv0_dq.noalias() -=
          d->v0_world_skew * d->fJf.template bottomRows<3>();
      break;
  }
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::updateForce(
    const std::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 3) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 3)");
  }
  Data* d = static_cast<Data*>(data.get());
  data->f.linear() = force;
  data->f.angular().setZero();
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->fext = d->jMf.act(data->f);
      data->dtau_dq.setZero();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
      d->f_local.linear().noalias() = oRf.transpose() * force;
      d->f_local.angular().setZero();
      data->fext = data->jMf.act(d->f_local);
      pinocchio::skew(d->f_local.linear(), d->f_skew);
      d->fJf_df.noalias() = d->f_skew * d->fJf.template bottomRows<3>();
      data->dtau_dq.noalias() =
          -d->fJf.template topRows<3>().transpose() * d->fJf_df;
      break;
  }
}

template <typename Scalar>
std::shared_ptr<ImpulseDataAbstractTpl<Scalar> >
ImpulseModel3DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ImpulseModel3DTpl<NewScalar> ImpulseModel3DTpl<Scalar>::cast() const {
  typedef ImpulseModel3DTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      type_);
  return ret;
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::print(std::ostream& os) const {
  os << "ImpulseModel3D {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", type=" << type_ << "}";
}

}  // namespace crocoddyl
