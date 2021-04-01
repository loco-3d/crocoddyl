///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel3DTpl<Scalar>::ContactModel3DTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                                             const std::size_t nu, const Vector2s& gains)
    : Base(state, 3, nu), xref_(xref), gains_(gains) {}

template <typename Scalar>
ContactModel3DTpl<Scalar>::ContactModel3DTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                                             const Vector2s& gains)
    : Base(state, 3), xref_(xref), gains_(gains) {}

template <typename Scalar>
ContactModel3DTpl<Scalar>::~ContactModel3DTpl() {}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, xref_.id);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, xref_.id, pinocchio::LOCAL, d->fJf);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, xref_.id);
  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, xref_.id);

  d->Jc = d->fJf.template topRows<3>();
  d->vw = d->v.angular();
  d->vv = d->v.linear();
  d->a0 = d->a.linear() + d->vw.cross(d->vv);

  if (gains_[0] != 0.) {
    d->a0 += gains_[0] * (d->pinocchio->oMf[xref_.id].translation() - xref_.translation);
  }
  if (gains_[1] != 0.) {
    d->a0 += gains_[1] * d->vv;
  }
}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;
  d->da0_dx.leftCols(nv).noalias() = d->fXjda_dq.template topRows<3>() +
                                     d->vw_skew * d->fXjdv_dq.template topRows<3>() -
                                     d->vv_skew * d->fXjdv_dq.template bottomRows<3>();
  d->da0_dx.rightCols(nv).noalias() =
      d->fXjda_dv.template topRows<3>() + d->vw_skew * d->Jc - d->vv_skew * d->fJf.template bottomRows<3>();

  if (gains_[0] != 0.) {
    d->oRf = d->pinocchio->oMf[xref_.id].rotation();
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * d->oRf * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv).noalias() += gains_[1] * d->fXj.template topRows<3>() * d->v_partial_dq;
    d->da0_dx.rightCols(nv).noalias() += gains_[1] * d->fXj.template topRows<3>() * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 3) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 3)");
  }
  Data* d = static_cast<Data*>(data.get());
  data->f = d->jMf.act(pinocchio::ForceTpl<Scalar>(force, Vector3s::Zero()));
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModel3DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const FrameTranslationTpl<Scalar>& ContactModel3DTpl<Scalar>::get_xref() const {
  return xref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel3DTpl<Scalar>::get_gains() const {
  return gains_;
}

}  // namespace crocoddyl
