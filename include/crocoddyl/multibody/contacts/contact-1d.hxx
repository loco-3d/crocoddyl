///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const Scalar xref, const std::size_t nu, const Vector2s& gains,
                                             const Contact1DMaskType& mask, const pinocchio::ReferenceFrame type)
    : Base(state, 1, nu), xref_(xref), gains_(gains), mask_(mask), type_(type) {
  id_ = id;
  if (mask_ < 0 || mask_ > 2) {
    throw_pretty("Invalid argument: "
                 << "Contact1D mask must be in {0,1,2} for {x,y,z}");
  }
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::ContactModel1DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const Scalar xref, const Vector2s& gains,
                                             const pinocchio::ReferenceFrame type)
    : Base(state, 1), xref_(xref), gains_(gains), type_(type) {
  id_ = id;
  // Default mask is z
  mask_ = Z_MASK;
}

template <typename Scalar>
ContactModel1DTpl<Scalar>::~ContactModel1DTpl() {}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, id_, type_, d->fJf);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, id_, type_);
  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, id_, type_);

  d->Jc.row(0) = d->fJf.row(mask_);
  d->vw = d->v.angular();
  d->vv = d->v.linear();
  d->a0[0] = (d->a.linear() + d->vw.cross(d->vv))[mask_];

  if (gains_[0] != 0.) {
    if (type_ == pinocchio::WORLD) {
      d->a0[0] += gains_[0] * (d->pinocchio->oMf[id_].translation()[mask_] - xref_);
    } else if (type_ == pinocchio::LOCAL || type_ == pinocchio::LOCAL_WORLD_ALIGNED) {
      d->a0[0] += -gains_[0] * xref_;
    }
  }
  if (gains_[1] != 0.) {
    d->a0[0] += gains_[1] * d->vv[mask_];
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, type_,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;

  d->da0_dx.leftCols(nv).row(0) = d->fXjda_dq.row(mask_);
  d->da0_dx.leftCols(nv).row(0).noalias() += d->vw_skew.row(mask_) * d->fXjdv_dq.template topRows<3>();
  d->da0_dx.leftCols(nv).row(0).noalias() -= d->vv_skew.row(mask_) * d->fXjdv_dq.template bottomRows<3>();

  d->da0_dx.rightCols(nv).row(0) = d->fXjda_dv.row(mask_);
  d->da0_dx.rightCols(nv).row(0).noalias() += d->vw_skew.row(mask_) * d->fJf.template topRows<3>();
  d->da0_dx.rightCols(nv).row(0).noalias() -= d->vv_skew.row(mask_) * d->fJf.template bottomRows<3>();

  if (gains_[0] != 0.) {
    const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
    d->oRf(0, 0) = oRf(mask_, mask_);
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * d->oRf * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv).row(0).noalias() += gains_[1] * d->fXj.row(mask_) * d->v_partial_dq;
    d->da0_dx.rightCols(nv).row(0).noalias() += gains_[1] * d->fXj.row(mask_) * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 1) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 1)");
  }
  Data* d = static_cast<Data*>(data.get());
  pinocchio::SE3 jMc;
  switch (type_) {
    case pinocchio::LOCAL: {
      jMc = d->jMf;
      break;
    }
    case pinocchio::WORLD: {
      jMc = d->jMf.act(d->pinocchio->oMf[id_].inverse());
      break;
    }
    case pinocchio::LOCAL_WORLD_ALIGNED: {
      jMc = pinocchio::SE3(Matrix3s::Identity(), d->jMf.translation());
      break;
    }
  }
  data->f.linear() = jMc.rotation().col(mask_) * force[0];
  data->f.angular() = jMc.translation().cross(data->f.linear());
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModel1DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel1D {frame=" << state_->get_pinocchio()->frames[id_].name << "}";
}

template <typename Scalar>
const Scalar ContactModel1DTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel1DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::set_reference(const Scalar reference) {
  xref_ = reference;
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::set_type(const pinocchio::ReferenceFrame type) {
  type_ = type;
}

template <typename Scalar>
const pinocchio::ReferenceFrame ContactModel1DTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void ContactModel1DTpl<Scalar>::set_mask(const Contact1DMaskType mask) {
  mask_ = mask;
}

template <typename Scalar>
const Contact1DMaskType ContactModel1DTpl<Scalar>::get_mask() const {
  return mask_;
}

}  // namespace crocoddyl
