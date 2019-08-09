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
  // vw = data.v.angular
  // vv = data.v.linear

  // J6 = pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame, pinocchio.ReferenceFrame.LOCAL)
  // data.J[:, :] = J6[:3, :]
  // data.Jw[:, :] = J6[3:, :]

  // data.a0[:] = (pinocchio.getFrameAcceleration(self.pinocchio, data.pinocchio, self.frame).linear +
  //               cross(vw, vv)).flat
  // if self.gains[0] != 0.:
  //     data.a0[:] += self.gains[0] * (m2a(data.pinocchio.oMf[self.frame].translation) - self.ref)
  // if self.gains[1] != 0.:
  //     data.a0[:] += self.gains[1] * m2a(vv)
}

void ContactModel3D::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }

  ContactData3D* d = static_cast<ContactData3D*>(data.get());
  pinocchio::getJointAccelerationDerivatives(state_.get_pinocchio(), *d->pinocchio, xref_.frame, pinocchio::LOCAL,
                                             d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);

  const unsigned int& nv = state_.get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->Ax.leftCols(nv) = (d->fXj * d->a_partial_dq).topRows<3>() + d->vw_skew * (d->fXj * d->v_partial_dq).topRows<3>() -
                       d->vv_skew * (d->fXj * d->v_partial_dq).bottomRows<3>();
  d->Ax.rightCols(nv) =
      (d->fXj * d->a_partial_dv).topRows<3>() + d->vw_skew * d->Jc - d->vv_skew * d->fJf.bottomRows<3>();

  if (gains_[0] != 0.) {
    d->oRf = d->pinocchio->oMf[xref_.frame].rotation();
    d->Ax.leftCols(nv) += gains_[0] * d->oRf * d->Jc;
  }
  if (gains_[1] != 0.) {
    d->Ax.leftCols(nv) += gains_[1] * d->fXj.topRows<3>() * d->a_partial_dq;
    d->Ax.rightCols(nv) += gains_[1] * d->fXj.topRows<3>() * d->a_partial_da;
  }

  // if recalc:
  //     self.calc(data, x)
  // dv_dq, da_dq, da_dv, da_da = pinocchio.getJointAccelerationDerivatives(self.pinocchio, data.pinocchio,
  //                                                                        data.joint,
  //                                                                        pinocchio.ReferenceFrame.LOCAL)
  // dv_dq, dv_dvq = pinocchio.getJointVelocityDerivatives(self.pinocchio, data.pinocchio, data.joint,
  //                                                       pinocchio.ReferenceFrame.LOCAL)
  // note that dv_dvq = da_da
  // vw = data.v.angular
  // vv = data.v.linear

  // data.Aq[:, :] = (data.fXj *
  //                  da_dq)[:3, :] + skew(vw) * (data.fXj * dv_dq)[:3, :] - skew(vv) * (data.fXj * dv_dq)[3:, :]
  // data.Av[:, :] = (data.fXj * da_dv)[:3, :] + skew(vw) * data.J - skew(vv) * data.Jw
  // R = data.pinocchio.oMf[self.frame].rotation

  // if self.gains[0] != 0.:
  //     data.Aq[:, :] += self.gains[0] * R * pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame,
  //                                                                     pinocchio.ReferenceFrame.LOCAL)[:3, :]
  // if self.gains[1] != 0.:
  //     data.Aq[:, :] += self.gains[1] * (data.fXj[:3, :] * dv_dq)
  //     data.Av[:, :] += self.gains[1] * (data.fXj[:3, :] * dv_dvq)
}

boost::shared_ptr<ContactDataAbstract> ContactModel3D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactData3D>(this, data);
}

}  // namespace crocoddyl
