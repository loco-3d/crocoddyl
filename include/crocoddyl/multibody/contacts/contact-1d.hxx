///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    Scalar xref, const pinocchio::ReferenceFrame type, const Matrix3s& rotation,
    const std::size_t nu, const Vector2s& gains)
    : Base(state, type, 1, 1, nu),
      xref_(xref),
      Raxis_(rotation),
      gains_(gains) {
  id_[0] = id;
  type_ = type;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    Scalar xref, const pinocchio::ReferenceFrame type, const Vector2s& gains)
    : Base(state, type, 1, 1), xref_(xref), gains_(gains) {
  id_[0] = id;
  type_= type;
  Raxis_ = Matrix3s::Identity();
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has
                             // been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    Scalar xref, const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 1, 1, nu),
      xref_(xref),
      gains_(gains) {
  id_[0] = id;
  Raxis_ = Matrix3s::Identity();
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    Scalar xref, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 1, 1),
      xref_(xref),
      gains_(gains) {
  id_[0] = id;
  Raxis_ = Matrix3s::Identity();
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

#pragma GCC diagnostic pop

template <typename Scalar>
ContactModel1DTpl<Scalar>::~ContactModel1DTpl() {}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calc(
    const boost::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  ForceDataAbstract& fdata = d->force_datas[0];  // there's only one force data
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio,
                                  id_[0]);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_[0], pinocchio::LOCAL, fdata.fJf);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                     *d->pinocchio, id_[0]);

  d->a0_local = pinocchio::getFrameClassicalAcceleration(
                    *state_->get_pinocchio().get(), *d->pinocchio, id_[0],
                    pinocchio::LOCAL)
                    .linear();

  const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_[0]].rotation();
  if (gains_[0] != 0.) {
    d->dp = d->pinocchio->oMf[id_[0]].translation() -
            (xref_ * Raxis_ * Vector3s::UnitZ());
    d->dp_local.noalias() = oRf.transpose() * d->dp;
    d->a0_local += gains_[0] * d->dp_local;
  }
  if (gains_[1] != 0.) {
    d->a0_local += gains_[1] * d->v.linear();
  }
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->Jc.row(0) = (Raxis_ * fdata.fJf.template topRows<3>()).row(2);
      d->a0[0] = (Raxis_ * d->a0_local)[2];
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->Jc.row(0) = (Raxis_ * oRf * fdata.fJf.template topRows<3>()).row(2);
      d->a0[0] = (Raxis_ * oRf * d->a0_local)[2];
      break;
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  ForceDataAbstract& fdata = d->force_datas[0];  // there's only one force data
#if PINOCCHIO_VERSION_AT_LEAST(3, 0, 0)
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[fdata.frame].parentJoint;
#else
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[fdata.frame].parent;
#endif
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
      fdata.v_partial_dq, fdata.a_partial_dq, fdata.a_partial_dv, fdata.a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->v.linear(), d->vv_skew);
  pinocchio::skew(d->v.angular(), d->vw_skew);
  d->fXjdv_dq.noalias() = fdata.fXj * fdata.v_partial_dq;
  d->fXjda_dq.noalias() = fdata.fXj * fdata.a_partial_dq;
  d->fXjda_dv.noalias() = fdata.fXj * fdata.a_partial_dv;
  d->da0_local_dx.leftCols(nv) = d->fXjda_dq.template topRows<3>();
  d->da0_local_dx.leftCols(nv).noalias() +=
      d->vw_skew * d->fXjdv_dq.template topRows<3>();
  d->da0_local_dx.leftCols(nv).noalias() -=
      d->vv_skew * d->fXjdv_dq.template bottomRows<3>();
  d->da0_local_dx.rightCols(nv) = d->fXjda_dv.template topRows<3>();
  d->da0_local_dx.rightCols(nv).noalias() +=
      d->vw_skew * fdata.fJf.template topRows<3>();
  d->da0_local_dx.rightCols(nv).noalias() -=
      d->vv_skew * fdata.fJf.template bottomRows<3>();
  const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_[0]].rotation();

  if (gains_[0] != 0.) {
    pinocchio::skew(d->dp_local, d->dp_skew);
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[0] * d->dp_skew * fdata.fJf.template bottomRows<3>();
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[0] * fdata.fJf.template topRows<3>();
  }
  if (gains_[1] != 0.) {
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[1] * d->fXjdv_dq.template topRows<3>();
    d->da0_local_dx.rightCols(nv).noalias() +=
        gains_[1] * fdata.fJf.template topRows<3>();
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
                        *state_->get_pinocchio().get(), *d->pinocchio, id_[0],
                        pinocchio::LOCAL)
                        .linear();
      if (gains_[0] != 0.) {
        d->a0_local += gains_[0] * d->dp_local;
      }
      if (gains_[1] != 0.) {
        d->a0_local += gains_[1] * d->v.linear();
      }
      d->a0[0] = (Raxis_ * oRf * d->a0_local)[2];

      pinocchio::skew((Raxis_ * oRf * d->a0_local).template head<3>(),
                      d->a0_skew);
      d->a0_world_skew.noalias() = d->a0_skew * Raxis_ * oRf;
      d->da0_dx.row(0) = (Raxis_ * oRf * d->da0_local_dx).row(2);
      d->da0_dx.leftCols(nv).row(0) -=
          (d->a0_world_skew * fdata.fJf.template bottomRows<3>()).row(2);
      break;
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::updateForce(
    const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 1) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 1)");
  }
  Data* d = static_cast<Data*>(data.get());
  ForceDataAbstract& fdata =
      data->force_datas[0];  // there's only one force data
  const Eigen::Ref<const Matrix3s> R = fdata.jMf.rotation();
  fdata.f.linear()[2] = force[0];
  fdata.f.linear().template head<2>().setZero();
  fdata.f.angular().setZero();
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      fdata.fext.linear() = (R * Raxis_.transpose()).col(2) * force[0];
      fdata.fext.angular() = fdata.jMf.translation().cross(fdata.fext.linear());
      data->dtau_dq.setZero();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Eigen::Ref<const Matrix3s> oRf =
          d->pinocchio->oMf[id_[0]].rotation();
      d->f_local.linear().noalias() =
          (oRf.transpose() * Raxis_.transpose()).col(2) * force[0];
      d->f_local.angular().setZero();
      fdata.fext = fdata.jMf.act(d->f_local);
      pinocchio::skew(d->f_local.linear(), d->f_skew);
      d->fJf_df.noalias() = d->f_skew * fdata.fJf.template bottomRows<3>();
      data->dtau_dq.noalias() =
          -fdata.fJf.template topRows<3>().transpose() * d->fJf_df;
      break;
  }
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModel1DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel1D {frame=" << state_->get_pinocchio()->frames[id_[0]].name
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
