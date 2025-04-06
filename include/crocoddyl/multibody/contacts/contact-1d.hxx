///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar xref, const pinocchio::ReferenceFrame type,
    const Matrix3s& rotation, const std::size_t nu, const Vector2s& gains)
    : Base(state, type, 1, nu), xref_(xref), Raxis_(rotation), gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar xref, const pinocchio::ReferenceFrame type,
    const Vector2s& gains)
    : Base(state, type, 1), xref_(xref), gains_(gains) {
  id_ = id;
  Raxis_ = Matrix3s::Identity();
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has
                             // been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar xref, const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 1, nu),
      xref_(xref),
      gains_(gains) {
  id_ = id;
  Raxis_ = Matrix3s::Identity();
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar xref, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 1),
      xref_(xref),
      gains_(gains) {
  id_ = id;
  Raxis_ = Matrix3s::Identity();
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

#pragma GCC diagnostic pop

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calc(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio,
                                  id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_, pinocchio::LOCAL, d->fJf);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                     *d->pinocchio, id_);

  d->a0_local =
      pinocchio::getFrameClassicalAcceleration(
          *state_->get_pinocchio().get(), *d->pinocchio, id_, pinocchio::LOCAL)
          .linear();

  const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
  if (gains_[0] != Scalar(0.)) {
    d->dp = d->pinocchio->oMf[id_].translation() -
            (xref_ * Raxis_ * Vector3s::UnitZ());
    d->dp_local.noalias() = oRf.transpose() * d->dp;
    d->a0_local += gains_[0] * d->dp_local;
  }
  if (gains_[1] != Scalar(0.)) {
    d->a0_local += gains_[1] * d->v.linear();
  }
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->Jc.row(0) = (Raxis_ * d->fJf.template topRows<3>()).row(2);
      d->a0[0] = (Raxis_ * d->a0_local)[2];
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->Jc.row(0) = (Raxis_ * oRf * d->fJf.template topRows<3>()).row(2);
      d->a0[0] = (Raxis_ * oRf * d->a0_local)[2];
      break;
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calcDiff(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parentJoint;
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
      d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->v.linear(), d->vv_skew);
  pinocchio::skew(d->v.angular(), d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;
  d->da0_local_dx.leftCols(nv) = d->fXjda_dq.template topRows<3>();
  d->da0_local_dx.leftCols(nv).noalias() +=
      d->vw_skew * d->fXjdv_dq.template topRows<3>();
  d->da0_local_dx.leftCols(nv).noalias() -=
      d->vv_skew * d->fXjdv_dq.template bottomRows<3>();
  d->da0_local_dx.rightCols(nv) = d->fXjda_dv.template topRows<3>();
  d->da0_local_dx.rightCols(nv).noalias() +=
      d->vw_skew * d->fJf.template topRows<3>();
  d->da0_local_dx.rightCols(nv).noalias() -=
      d->vv_skew * d->fJf.template bottomRows<3>();
  const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();

  if (gains_[0] != Scalar(0.)) {
    pinocchio::skew(d->dp_local, d->dp_skew);
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[0] * d->dp_skew * d->fJf.template bottomRows<3>();
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[0] * d->fJf.template topRows<3>();
  }
  if (gains_[1] != Scalar(0.)) {
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[1] * d->fXjdv_dq.template topRows<3>();
    d->da0_local_dx.rightCols(nv).noalias() +=
        gains_[1] * d->fJf.template topRows<3>();
  }
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->da0_dx.row(0) = (Raxis_ * d->da0_local_dx).row(2);
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      // Recalculate the constrained accelerations after imposing contact
      // constraints. This is necessary for the forward-dynamics case.
      d->a0_local = pinocchio::getFrameClassicalAcceleration(
                        *state_->get_pinocchio().get(), *d->pinocchio, id_,
                        pinocchio::LOCAL)
                        .linear();
      if (gains_[0] != Scalar(0.)) {
        d->a0_local += gains_[0] * d->dp_local;
      }
      if (gains_[1] != Scalar(0.)) {
        d->a0_local += gains_[1] * d->v.linear();
      }
      d->a0[0] = (Raxis_ * oRf * d->a0_local)[2];

      pinocchio::skew((Raxis_ * oRf * d->a0_local).template head<3>(),
                      d->a0_skew);
      d->a0_world_skew.noalias() = d->a0_skew * Raxis_ * oRf;
      d->da0_dx.row(0) = (Raxis_ * oRf * d->da0_local_dx).row(2);
      d->da0_dx.leftCols(nv).row(0) -=
          (d->a0_world_skew * d->fJf.template bottomRows<3>()).row(2);
      break;
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::updateForce(
    const std::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 1) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 1)");
  }
  Data* d = static_cast<Data*>(data.get());
  const Eigen::Ref<const Matrix3s> R = d->jMf.rotation();
  data->f.linear()[2] = force[0];
  data->f.linear().template head<2>().setZero();
  data->f.angular().setZero();
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->fext.linear() = (R * Raxis_.transpose()).col(2) * force[0];
      data->fext.angular() = d->jMf.translation().cross(data->fext.linear());
      data->dtau_dq.setZero();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
      d->f_local.linear().noalias() =
          (oRf.transpose() * Raxis_.transpose()).col(2) * force[0];
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
std::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModel1DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModel1DTpl<NewScalar> ContactModel1DTpl<Scalar>::cast() const {
  typedef ContactModel1DTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      scalar_cast<NewScalar>(xref_), type_, Raxis_.template cast<NewScalar>(),
      nu_, gains_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel1D {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", axis=" << (Raxis_ * Vector3s::UnitZ()).transpose() << "}";
}

template <typename Scalar>
const Scalar ContactModel1DTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ContactModel1DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s&
ContactModel1DTpl<Scalar>::get_axis_rotation() const {
  return Raxis_;
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::set_reference(const Scalar reference) {
  xref_ = reference;
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::set_axis_rotation(const Matrix3s& rotation) {
  Raxis_ = rotation;
}

}  // namespace crocoddyl
