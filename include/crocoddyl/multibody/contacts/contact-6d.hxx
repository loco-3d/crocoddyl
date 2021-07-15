
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const SE3& pref, const std::size_t nu, const Vector2s& gains)
    : Base(state, 6, nu), pref_(pref), gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const SE3& pref, const Vector2s& gains)
    : Base(state, 6), pref_(pref), gains_(gains) {
  id_ = id;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(boost::shared_ptr<StateMultibody> state,
                                             const FramePlacementTpl<Scalar>& Mref, const std::size_t nu,
                                             const Vector2s& gains)
    : Base(state, 6, nu), pref_(Mref.placement), gains_(gains) {
  id_ = Mref.id;
  std::cerr << "Deprecated: Use constructor which is not based on FramePlacement." << std::endl;
}

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(boost::shared_ptr<StateMultibody> state,
                                             const FramePlacementTpl<Scalar>& Mref, const Vector2s& gains)
    : Base(state, 6), pref_(Mref.placement), gains_(gains) {
  id_ = Mref.id;
  std::cerr << "Deprecated: Use constructor which is not based on FramePlacement." << std::endl;
}

#pragma GCC diagnostic pop

template <typename Scalar>
ContactModel6DTpl<Scalar>::~ContactModel6DTpl() {}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement<Scalar>(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, id_, pinocchio::LOCAL, d->Jc);

  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  d->a0 = d->a.toVector();

  if (gains_[0] != 0.) {
    d->rMf = pref_.inverse() * d->pinocchio->oMf[id_];
    d->a0 += gains_[0] * pinocchio::log6(d->rMf).toVector();
  }
  if (gains_[1] != 0.) {
    d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, id_);
    d->a0 += gains_[1] * d->v.toVector();
  }
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  d->da0_dx.leftCols(nv).noalias() = d->fXj * d->a_partial_dq;
  d->da0_dx.rightCols(nv).noalias() = d->fXj * d->a_partial_dv;

  if (gains_[0] != 0.) {
    pinocchio::Jlog6(d->rMf, d->rMf_Jlog6);
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * d->rMf_Jlog6 * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv).noalias() += gains_[1] * d->fXj * d->v_partial_dq;
    d->da0_dx.rightCols(nv).noalias() += gains_[1] * d->fXj * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 6) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 6)");
  }
  Data* d = static_cast<Data*>(data.get());
  data->f = d->jMf.act(pinocchio::ForceTpl<Scalar>(force));
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModel6DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel6D {frame=" << state_->get_pinocchio()->frames[id_].name << "}";
}

template <typename Scalar>
const pinocchio::SE3Tpl<Scalar>& ContactModel6DTpl<Scalar>::get_reference() const {
  return pref_;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
FramePlacementTpl<Scalar> ContactModel6DTpl<Scalar>::get_Mref() const {
  return FramePlacementTpl<Scalar>(id_, pref_);
}

#pragma GCC diagnostic pop

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel6DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::set_reference(const SE3& reference) {
  pref_ = reference;
}

}  // namespace crocoddyl
