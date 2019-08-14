///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

ContactModel3D::ContactModel3D(StateMultibody& state, const FrameTranslation& xref, const Eigen::Vector2d& gains)
    : ContactModelAbstract(state, 3), xref_(xref), gains_(gains) {}

ContactModel3D::~ContactModel3D() {}

void ContactModel3D::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  ContactData3D* d = static_cast<ContactData3D*>(data.get());
  d->v = pinocchio::getFrameVelocity(state_.get_pinocchio(), *d->pinocchio, xref_.frame);
  d->vw = d->v.angular();
  d->vv = d->v.linear();

  pinocchio::getFrameJacobian(state_.get_pinocchio(), *d->pinocchio, xref_.frame, pinocchio::LOCAL, d->fJf);
  d->Jc = d->fJf.topRows<3>();

  d->a = pinocchio::getFrameAcceleration(state_.get_pinocchio(), *d->pinocchio, xref_.frame);
  d->a0 = d->a.linear() + d->vw.cross(d->vv);

  if (gains_[0] != 0.) {
    d->a0 += gains_[0] * (d->pinocchio->oMf[xref_.frame].translation() - xref_.oxf);
  }
  if (gains_[1] != 0.) {
    d->a0 += gains_[1] * d->vv;
  }
}

void ContactModel3D::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }

  ContactData3D* d = static_cast<ContactData3D*>(data.get());
  pinocchio::getJointAccelerationDerivatives(state_.get_pinocchio(), *d->pinocchio, d->joint, pinocchio::LOCAL,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const unsigned int& nv = state_.get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;
  d->Ax.leftCols(nv).noalias() =
      d->fXjda_dq.topRows<3>() + d->vw_skew * d->fXjdv_dq.topRows<3>() - d->vv_skew * d->fXjdv_dq.bottomRows<3>();
  d->Ax.rightCols(nv).noalias() = d->fXjda_dv.topRows<3>() + d->vw_skew * d->Jc - d->vv_skew * d->fJf.bottomRows<3>();

  if (gains_[0] != 0.) {
    d->oRf = d->pinocchio->oMf[xref_.frame].rotation();
    d->Ax.leftCols(nv).noalias() += gains_[0] * d->oRf * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->Ax.leftCols(nv).noalias() += gains_[1] * d->fXj.topRows<3>() * d->v_partial_dq;
    d->Ax.rightCols(nv).noalias() += gains_[1] * d->fXj.topRows<3>() * d->a_partial_da;
  }
}

boost::shared_ptr<ContactDataAbstract> ContactModel3D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactData3D>(this, data);
}

const FrameTranslation& ContactModel3D::get_xref() const { return xref_; }

const Eigen::Vector2d& ContactModel3D::get_gains() const { return gains_; }

}  // namespace crocoddyl
