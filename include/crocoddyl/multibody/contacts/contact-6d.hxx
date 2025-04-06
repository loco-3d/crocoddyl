
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
ContactModel6DTpl<Scalar>::ContactModel6DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const pinocchio::ReferenceFrame type, const std::size_t nu,
    const Vector2s& gains)
    : Base(state, type, 6, nu), pref_(pref), gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const pinocchio::ReferenceFrame type,
    const Vector2s& gains)
    : Base(state, type, 6), pref_(pref), gains_(gains) {
  id_ = id;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has
                             // been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6, nu),
      pref_(pref),
      gains_(gains) {
  id_ = id;
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6),
      pref_(pref),
      gains_(gains) {
  id_ = id;
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

#pragma GCC diagnostic pop

template <typename Scalar>
void ContactModel6DTpl<Scalar>::calc(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement<Scalar>(*state_->get_pinocchio().get(),
                                          *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_, pinocchio::LOCAL, d->fJf);
  d->a0_local = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(),
                                                *d->pinocchio, id_);

  if (gains_[0] != 0.) {
    d->rMf = pref_.actInv(d->pinocchio->oMf[id_]);
    d->a0_local += gains_[0] * pinocchio::log6(d->rMf);
  }
  if (gains_[1] != 0.) {
    d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                       *d->pinocchio, id_);
    d->a0_local += gains_[1] * d->v;
  }
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->Jc = d->fJf;
      data->a0 = d->a0_local.toVector();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->lwaMl.rotation(d->pinocchio->oMf[id_].rotation());
      data->Jc.noalias() = d->lwaMl.toActionMatrix() * d->fJf;
      data->a0.noalias() = d->lwaMl.act(d->a0_local).toVector();
      break;
  }
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::calcDiff(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parentJoint;
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
      d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  d->da0_local_dx.leftCols(nv).noalias() = d->fXj * d->a_partial_dq;
  d->da0_local_dx.rightCols(nv).noalias() = d->fXj * d->a_partial_dv;

  if (gains_[0] != 0.) {
    pinocchio::Jlog6(d->rMf, d->rMf_Jlog6);
    d->da0_local_dx.leftCols(nv).noalias() += gains_[0] * d->rMf_Jlog6 * d->fJf;
  }
  if (gains_[1] != 0.) {
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[1] * d->fXj * d->v_partial_dq;
    d->da0_local_dx.rightCols(nv).noalias() += gains_[1] * d->fJf;
  }
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->da0_dx = d->da0_local_dx;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      // Recalculate the constrained accelerations after imposing contact
      // constraints. This is necessary for the forward-dynamics case.
      d->a0_local = pinocchio::getFrameAcceleration(
          *state_->get_pinocchio().get(), *d->pinocchio, id_);
      if (gains_[0] != 0.) {
        d->a0_local += gains_[0] * pinocchio::log6(d->rMf);
      }
      if (gains_[1] != 0.) {
        d->a0_local += gains_[1] * d->v;
      }
      data->a0.noalias() = d->lwaMl.act(d->a0_local).toVector();

      const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
      pinocchio::skew(d->a0.template head<3>(), d->av_skew);
      pinocchio::skew(d->a0.template tail<3>(), d->aw_skew);
      d->av_world_skew.noalias() = d->av_skew * oRf;
      d->aw_world_skew.noalias() = d->aw_skew * oRf;
      d->da0_dx.noalias() = d->lwaMl.toActionMatrix() * d->da0_local_dx;
      d->da0_dx.leftCols(nv).template topRows<3>().noalias() -=
          d->av_world_skew * d->fJf.template bottomRows<3>();
      d->da0_dx.leftCols(nv).template bottomRows<3>().noalias() -=
          d->aw_world_skew * d->fJf.template bottomRows<3>();
      break;
  }
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::updateForce(
    const std::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
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
std::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModel6DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModel6DTpl<NewScalar> ContactModel6DTpl<Scalar>::cast() const {
  typedef ContactModel6DTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      pref_.template cast<NewScalar>(), type_, nu_,
      gains_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel6D {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", type=" << type_ << "}";
}

template <typename Scalar>
const pinocchio::SE3Tpl<Scalar>& ContactModel6DTpl<Scalar>::get_reference()
    const {
  return pref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ContactModel6DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::set_reference(const SE3& reference) {
  pref_ = reference;
}

}  // namespace crocoddyl
