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
ContactModel2DTpl<Scalar>::ContactModel2DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector2s& xref, const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 2, nu),
      xref_(xref),
      gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector2s& xref, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 2),
      xref_(xref),
      gains_(gains) {
  id_ = id;
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::calc(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio,
                                  id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_, pinocchio::LOCAL, d->fJf);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                     *d->pinocchio, id_);
  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(),
                                         *d->pinocchio, id_);

  d->Jc.row(0) = d->fJf.row(0);
  d->Jc.row(1) = d->fJf.row(2);

  d->vw = d->v.angular();
  d->vv = d->v.linear();

  d->a0[0] = d->a.linear()[0] + d->vw[1] * d->vv[2] - d->vw[2] * d->vv[1];
  d->a0[1] = d->a.linear()[2] + d->vw[0] * d->vv[1] - d->vw[1] * d->vv[0];

  if (gains_[0] != 0.) {
    d->a0[0] +=
        gains_[0] * (d->pinocchio->oMf[id_].translation()[0] - xref_[0]);
    d->a0[1] +=
        gains_[0] * (d->pinocchio->oMf[id_].translation()[2] - xref_[1]);
  }
  if (gains_[1] != 0.) {
    d->a0[0] += gains_[1] * d->vv[0];
    d->a0[1] += gains_[1] * d->vv[2];
  }
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::calcDiff(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parentJoint;
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
      d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;

  d->da0_dx.leftCols(nv).row(0) = d->fXjda_dq.row(0);
  d->da0_dx.leftCols(nv).row(0).noalias() +=
      d->vw_skew.row(0) * d->fXjdv_dq.template topRows<3>();
  d->da0_dx.leftCols(nv).row(0).noalias() -=
      d->vv_skew.row(0) * d->fXjdv_dq.template bottomRows<3>();

  d->da0_dx.leftCols(nv).row(1) = d->fXjda_dq.row(2);
  d->da0_dx.leftCols(nv).row(1).noalias() +=
      d->vw_skew.row(2) * d->fXjdv_dq.template topRows<3>();
  d->da0_dx.leftCols(nv).row(1).noalias() -=
      d->vv_skew.row(2) * d->fXjdv_dq.template bottomRows<3>();

  d->da0_dx.rightCols(nv).row(0) = d->fXjda_dv.row(0);
  d->da0_dx.rightCols(nv).row(0).noalias() +=
      d->vw_skew.row(0) * d->fJf.template topRows<3>();
  d->da0_dx.rightCols(nv).row(0).noalias() -=
      d->vv_skew.row(0) * d->fJf.template bottomRows<3>();

  d->da0_dx.rightCols(nv).row(1) = d->fXjda_dv.row(2);
  d->da0_dx.rightCols(nv).row(1).noalias() +=
      d->vw_skew.row(2) * d->fJf.template topRows<3>();
  d->da0_dx.rightCols(nv).row(1).noalias() -=
      d->vv_skew.row(2) * d->fJf.template bottomRows<3>();

  if (gains_[0] != 0.) {
    const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
    d->oRf(0, 0) = oRf(0, 0);
    d->oRf(1, 0) = oRf(2, 0);
    d->oRf(0, 1) = oRf(0, 2);
    d->oRf(1, 1) = oRf(2, 2);
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * d->oRf * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv).row(0).noalias() +=
        gains_[1] * d->fXj.row(0) * d->v_partial_dq;
    d->da0_dx.leftCols(nv).row(1).noalias() +=
        gains_[1] * d->fXj.row(2) * d->v_partial_dq;
    d->da0_dx.rightCols(nv).row(0).noalias() +=
        gains_[1] * d->fXj.row(0) * d->a_partial_da;
    d->da0_dx.rightCols(nv).row(1).noalias() +=
        gains_[1] * d->fXj.row(2) * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::updateForce(
    const std::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 2) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 2)");
  }
  Data* d = static_cast<Data*>(data.get());
  const Eigen::Ref<const Matrix3s> R = d->jMf.rotation();
  data->f.linear() = R.col(0) * force[0] + R.col(2) * force[1];
  data->f.angular().setZero();
  data->fext.linear() = R.col(0) * force[0] + R.col(2) * force[1];
  data->fext.angular() = d->jMf.translation().cross(data->fext.linear());
}

template <typename Scalar>
std::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModel2DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModel2DTpl<NewScalar> ContactModel2DTpl<Scalar>::cast() const {
  typedef ContactModel2DTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      xref_.template cast<NewScalar>(), nu_, gains_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel2D {frame=" << state_->get_pinocchio()->frames[id_].name
     << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ContactModel2DTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ContactModel2DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::set_reference(const Vector2s& reference) {
  xref_ = reference;
}

}  // namespace crocoddyl
