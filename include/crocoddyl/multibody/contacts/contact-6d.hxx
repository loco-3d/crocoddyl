
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& Mref,
                                             const std::size_t& nu, const Vector2s& gains)
    : Base(state, 6, nu), Mref_(Mref), gains_(gains) {}

template <typename Scalar>
ContactModel6DTpl<Scalar>::ContactModel6DTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& Mref,
                                             const Vector2s& gains)
    : Base(state, 6), Mref_(Mref), gains_(gains) {}

template <typename Scalar>
ContactModel6DTpl<Scalar>::~ContactModel6DTpl() {}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, Mref_.id);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, Mref_.id, pinocchio::LOCAL, d->Jc);

  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, Mref_.id);
  d->a0 = d->a.toVector();

  if (gains_[0] != 0.) {
    d->rMf = Mref_.placement.inverse() * d->pinocchio->oMf[Mref_.id];
    d->a0 += gains_[0] * pinocchio::log6(d->rMf).toVector();
  }
  if (gains_[1] != 0.) {
    d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, Mref_.id);
    d->a0 += gains_[1] * d->v.toVector();
  }
}

template <typename Scalar>
void ContactModel6DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t& nv = state_->get_nv();
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
const FramePlacementTpl<Scalar>& ContactModel6DTpl<Scalar>::get_Mref() const {
  return Mref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel6DTpl<Scalar>::get_gains() const {
  return gains_;
}

}  // namespace crocoddyl
