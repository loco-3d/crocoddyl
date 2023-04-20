///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar xref, const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 1, nu),
      xref_(xref),
      gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar xref, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 1),
      xref_(xref),
      gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::~ContactModel1DTpl() {}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calc(
    const boost::shared_ptr<ContactDataAbstract>& data,
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

  d->Jc.row(0) = d->fJf.row(2);

  d->vw = d->v.angular();
  d->vv = d->v.linear();

  d->a0[0] = d->a.linear()[2] + d->vw[0] * d->vv[1] - d->vw[1] * d->vv[0];

  if (gains_[0] != 0.) {
    d->a0[0] += gains_[0] * (d->pinocchio->oMf[id_].translation()[2] - xref_);
  }
  if (gains_[1] != 0.) {
    d->a0[0] += gains_[1] * d->vv[2];
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parent;
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
      d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;

  d->da0_dx.leftCols(nv).row(0).noalias() = d->fXjda_dq.row(2);
  d->da0_dx.leftCols(nv).row(0).noalias() +=
      d->vw_skew.row(2) * d->fXjdv_dq.template topRows<3>();
  d->da0_dx.leftCols(nv).row(0).noalias() -=
      d->vv_skew.row(2) * d->fXjdv_dq.template bottomRows<3>();

  d->da0_dx.rightCols(nv).row(0).noalias() = d->fXjda_dv.row(2);
  d->da0_dx.rightCols(nv).row(0).noalias() +=
      d->vw_skew.row(2) * d->fJf.template topRows<3>();
  d->da0_dx.rightCols(nv).row(0).noalias() -=
      d->vv_skew.row(2) * d->fJf.template bottomRows<3>();

  if (gains_[0] != 0.) {
    const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
    d->oRf(0, 0) = oRf(2, 2);
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * d->oRf * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv).row(0).noalias() +=
        gains_[1] * d->fXj.row(2) * d->v_partial_dq;
    d->da0_dx.rightCols(nv).row(0).noalias() +=
        gains_[1] * d->fXj.row(2) * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::updateForce(
    const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 1) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 1)");
  }
  Data* d = static_cast<Data*>(data.get());
  const Eigen::Ref<const Matrix3s> R = d->jMf.rotation();
  data->f.linear()[0] = force[0];
  data->f.linear().template tail<2>().setZero();
  data->f.angular().setZero();
  data->fext.linear() = R.col(2) * force[0];
  data->fext.angular() = d->jMf.translation().cross(data->fext.linear());
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModel1DTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel1D {frame=" << state_->get_pinocchio()->frames[id_].name
     << "}";
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
void ContactModel1DTpl<Scalar>::set_reference(const Scalar reference) {
  xref_ = reference;
}

}  // namespace crocoddyl
