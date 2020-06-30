///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                                             const std::size_t& nu, const Vector2s& gains)
    : Base(state, 2, nu), xref_(xref), gains_(gains) {}

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                                             const Vector2s& gains)
    : Base(state, 2), xref_(xref), gains_(gains) {}

template <typename Scalar>
ContactModel2DTpl<Scalar>::~ContactModel2DTpl() {}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, xref_.frame);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, xref_.frame);
  d->vw = d->v.angular();
  d->vv = d->v.linear();

  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, xref_.frame, pinocchio::LOCAL, d->fJf);
  d->Jc.row(0) = d->fJf.template row(0);
  d->Jc.row(1) = d->fJf.template row(2);

  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, xref_.frame);
  Vector3s a03D = d->a.linear() + d->vw.cross(d->vv);
  d->a0[0] = a03D[0];
  d->a0[1] = a03D[2];

  if (gains_[0] != 0.) {
    Vector3s gains1 = gains_[0] * (d->pinocchio->oMf[xref_.frame].translation() - xref_.oxf);
    d->a0[0] += gains1[0];
    d->a0[1] += gains1[2];
  }
  if (gains_[1] != 0.) {
    Vector3s gains2 = gains_[1] * d->vv;
    d->a0[0] += gains2[0];
    d->a0[1] += gains2[2];
  }
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t& nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;
  MathBase::MatrixXs m(3,nv); 
  m = d->fXjda_dq.template topRows<3>() +
      d->vw_skew * d->fXjdv_dq.template topRows<3>() -
      d->vv_skew * d->fXjdv_dq.template bottomRows<3>();
  d->da0_dx.leftCols(nv).row(0) = m.row(0);
  d->da0_dx.leftCols(nv).row(1) = m.row(2);
  MathBase::MatrixXs vw2D(3,2);
  vw2D.col(0) = d->vw_skew.col(0);
  vw2D.col(1) = d->vw_skew.col(2);
  m = d->fXjda_dv.template topRows<3>() + vw2D * d->Jc - d->vv_skew * d->fJf.template bottomRows<3>();
  d->da0_dx.rightCols(nv).row(0) = m.row(0);
  d->da0_dx.rightCols(nv).row(1) = m.row(2);

  if (gains_[0] != 0.) {
    d->oRf = d->pinocchio->oMf[xref_.frame].rotation();
    MathBase::MatrixXs oRf2D(2,2);
    oRf2D(0,0) = d->oRf(0,0);
    oRf2D(1,0) = d->oRf(2,0);
    oRf2D(0,1) = d->oRf(0,2);
    oRf2D(1,1) = d->oRf(2,2);
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * oRf2D * d->Jc;
  }
  if (gains_[1] != 0.) {
	MathBase::MatrixXs fXj2D(2,nv);
	fXj2D.row(0) = d->fXj.template row(0);
	fXj2D.row(1) = d->fXj.template row(2);
    d->da0_dx.leftCols(nv).noalias() += gains_[1] * fXj2D * d->v_partial_dq;
    d->da0_dx.rightCols(nv).noalias() += gains_[1] * fXj2D * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 2) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 2)");
  }
  Data* d = static_cast<Data*>(data.get());
  data->f = d->jMf.act(pinocchio::ForceTpl<Scalar>(force, Vector2s::Zero()));
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModel2DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const FrameTranslationTpl<Scalar>& ContactModel2DTpl<Scalar>::get_xref() const {
  return xref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel2DTpl<Scalar>::get_gains() const {
  return gains_;
}

}  // namespace crocoddyl
