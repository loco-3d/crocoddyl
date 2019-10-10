
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

ContactModel6D::ContactModel6D(boost::shared_ptr<StateMultibody> state, const FramePlacement& Mref, const std::size_t& nu,
                               const Eigen::Vector2d& gains)
    : ContactModelAbstract(state, 6, nu), Mref_(Mref), gains_(gains) {}

ContactModel6D::ContactModel6D(boost::shared_ptr<StateMultibody> state, const FramePlacement& Mref, const Eigen::Vector2d& gains)
    : ContactModelAbstract(state, 6), Mref_(Mref), gains_(gains) {}

ContactModel6D::~ContactModel6D() {}

void ContactModel6D::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  ContactData6D* d = static_cast<ContactData6D*>(data.get());

  pinocchio::getFrameJacobian(state_->get_pinocchio(), *d->pinocchio, Mref_.frame, pinocchio::LOCAL, d->Jc);

  d->a = pinocchio::getFrameAcceleration(state_->get_pinocchio(), *d->pinocchio, Mref_.frame);
  d->a0 = d->a.toVector();

  if (gains_[0] != 0.) {
    d->rMf = Mref_.oMf.inverse() * d->pinocchio->oMf[Mref_.frame];
    d->a0 += gains_[0] * pinocchio::log6(d->rMf).toVector();
  }
  if (gains_[1] != 0.) {
    d->v = pinocchio::getFrameVelocity(state_->get_pinocchio(), *d->pinocchio, Mref_.frame);
    d->a0 += gains_[1] * d->v.toVector();
  }
}

void ContactModel6D::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }

  ContactData6D* d = static_cast<ContactData6D*>(data.get());
  pinocchio::getJointAccelerationDerivatives(state_->get_pinocchio(), *d->pinocchio, d->joint, pinocchio::LOCAL,
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

void ContactModel6D::updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& force) {
  assert(force.size() == 6 && "force has wrong dimension, it should be 6d vector");
  ContactData6D* d = static_cast<ContactData6D*>(data.get());
  data->f = d->jMf.act(pinocchio::Force(force));
}

boost::shared_ptr<ContactDataAbstract> ContactModel6D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactData6D>(this, data);
}

const FramePlacement& ContactModel6D::get_Mref() const { return Mref_; }

const Eigen::Vector2d& ContactModel6D::get_gains() const { return gains_; }

}  // namespace crocoddyl
