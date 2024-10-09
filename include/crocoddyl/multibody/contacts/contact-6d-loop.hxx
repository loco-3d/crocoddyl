
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::ContactModel6DLoopTpl(
    boost::shared_ptr<StateMultibody> state, const int joint1_id,
    const SE3 &joint1_placement, const int joint2_id,
    const SE3 &joint2_placement, const pinocchio::ReferenceFrame ref,
    const std::size_t nu, const Vector2s &gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6, nu),
      joint1_id_(joint1_id),
      joint2_id_(joint2_id),
      joint1_placement_(joint1_placement),
      joint2_placement_(joint2_placement),
      gains_(gains) {
  if (ref != pinocchio::ReferenceFrame::LOCAL) {
    std::cerr << "Warning: Only reference frame pinocchio::LOCAL is supported "
                 "for 6D loop "
                 "contacts\n"
              << std::endl;
  }
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::ContactModel6DLoopTpl(
    boost::shared_ptr<StateMultibody> state, const int joint1_id,
    const SE3 &joint1_placement, const int joint2_id,
    const SE3 &joint2_placement, const pinocchio::ReferenceFrame ref,
    const Vector2s &gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6),
      joint1_id_(joint1_id),
      joint2_id_(joint2_id),
      joint1_placement_(joint1_placement),
      joint2_placement_(joint2_placement),
      gains_(gains) {
  if (ref != pinocchio::ReferenceFrame::LOCAL) {
    std::cerr << "Warning: Only reference frame pinocchio::LOCAL is supported "
                 "for 6D loop "
                 "contacts\n"
              << std::endl;
  }
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::~ContactModel6DLoopTpl() {}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calc(
    const boost::shared_ptr<ContactDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  pinocchio::updateFramePlacements<Scalar>(*state_->get_pinocchio().get(),
                                           *d->pinocchio);
  d->j1Xf1.noalias() = joint1_placement_.toActionMatrix();
  d->j2Xf2.noalias() = joint2_placement_.toActionMatrix();

  pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              joint1_id_, pinocchio::LOCAL, d->j1Jj1);
  pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              joint2_id_, pinocchio::LOCAL, d->j2Jj2);
  d->f1Jf1.noalias() = d->j1Xf1.inverse() * d->j1Jj1;
  d->f2Jf2.noalias() = d->j2Xf2.inverse() * d->j2Jj2;

  d->oMf1 = d->pinocchio->oMi[joint1_id_].act(joint1_placement_);
  d->oMf2 = d->pinocchio->oMi[joint2_id_].act(joint2_placement_);
  d->f1Mf2 = d->oMf1.actInv(d->oMf2);
  d->f1Xf2.noalias() = d->f1Mf2.toActionMatrix();

  d->Jc.noalias() = d->f1Jf1 - d->f1Xf2 * d->f2Jf2;
  // Compute the acceleration drift
  if (joint1_id_ > 0) {
    d->f1vf1 = joint1_placement_.actInv(d->pinocchio->v[joint1_id_]);
    d->f1af1 = joint1_placement_.actInv(d->pinocchio->a[joint1_id_]);
  }
  if (joint2_id_ > 0) {
    d->f2vf2 = joint2_placement_.actInv(d->pinocchio->v[joint2_id_]);
    d->f2af2 = joint2_placement_.actInv(d->pinocchio->a[joint2_id_]);
    d->f1vf2 = d->f1Mf2.act(d->f2vf2);
    d->f1af2 = d->f1Mf2.act(d->f2af2);
  }
  d->a0.noalias() =
      (d->f1af1 - d->f1Mf2.act(d->f2af2) + d->f1vf1.cross(d->f1vf2)).toVector();

  if (std::abs<Scalar>(gains_[0]) > std::numeric_limits<Scalar>::epsilon()) {
    d->a0.noalias() -= gains_[0] * pinocchio::log6(d->f1Mf2).toVector();
  }
  if (std::abs<Scalar>(gains_[1]) > std::numeric_limits<Scalar>::epsilon()) {
    d->a0 += gains_[1] * (d->f1vf1 - d->f1vf2).toVector();
  }
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ContactDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  const std::size_t nv = state_->get_nv();

  if (joint1_id_ > 0) {
    d->f1af1 = joint1_placement_.actInv(d->pinocchio->a[joint1_id_]);
  }
  if (joint2_id_ > 0) {
    d->f2af2 = joint2_placement_.actInv(d->pinocchio->a[joint2_id_]);
    d->f1af2 = d->f1Mf2.act(d->f2af2);
  }

  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint1_id_,
      pinocchio::LOCAL, d->v1_partial_dq, d->a1_partial_dq, d->a1_partial_dv,
      d->a1_partial_da);
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint2_id_,
      pinocchio::LOCAL, d->v2_partial_dq, d->a2_partial_dq, d->a2_partial_dv,
      d->a2_partial_da);

  d->da0_dq_t1 = joint1_placement_.toActionMatrixInverse() * d->a1_partial_dq;
  d->da0_dq_t2 = (d->f1af2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2) +
                  d->f1Xf2 * (joint2_placement_.toActionMatrixInverse() *
                              d->a2_partial_dq));
  d->da0_dq_t3 =
      -d->f1vf2.toActionMatrix() *
          (joint1_placement_.toActionMatrixInverse() * d->v1_partial_dq) +
      d->f1vf1.toActionMatrix() * d->f1vf2.toActionMatrix() *
          (d->f1Jf1 - d->f1Xf2 * d->f2Jf2) +
      d->f1vf1.toActionMatrix() * d->f1Xf2 *
          (joint2_placement_.toActionMatrixInverse() * d->v2_partial_dq);

  d->da0_dx.leftCols(nv).noalias() = d->da0_dq_t1 - d->da0_dq_t2 + d->da0_dq_t3;
  d->da0_dx.rightCols(nv) =
      joint1_placement_.toActionMatrixInverse() * d->a1_partial_dv -
      d->f1Xf2 *
          (joint2_placement_.toActionMatrixInverse() * d->a2_partial_dv) -
      d->f1vf2.toActionMatrix() * d->f1Jf1 +
      d->f1vf1.toActionMatrix() * d->f1Xf2 * d->f2Jf2;
  if (std::abs<Scalar>(gains_[0]) > std::numeric_limits<Scalar>::epsilon()) {
    Matrix6s f1Mf2_log6;
    pinocchio::Jlog6(d->f1Mf2, f1Mf2_log6);
    d->da0_dx.leftCols(nv).noalias() +=
        gains_[0] * f1Mf2_log6 *
        (d->oMf2.toActionMatrixInverse() * d->oMf1.toActionMatrix() * d->f1Jf1 -
         d->f2Jf2);
  }
  if (std::abs<Scalar>(gains_[1]) > std::numeric_limits<Scalar>::epsilon()) {
    d->da0_dx.leftCols(nv).noalias() +=
        gains_[1] *
        (joint1_placement_.toActionMatrixInverse() * d->v1_partial_dq -
         d->f1Mf2.act(d->f2vf2).toActionMatrix() *
             (d->f1Jf1 - d->f1Xf2 * d->f2Jf2) -
         d->f1Xf2 * joint2_placement_.toActionMatrixInverse() *
             d->v2_partial_dq);
    d->da0_dx.rightCols(nv).noalias() +=
        gains_[1] * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);
  }
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::updateForce(
    const boost::shared_ptr<ContactDataAbstract> &data, const VectorXs &force) {
  if (force.size() != 6) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 6)");
  }
  Data *d = static_cast<Data *>(data.get());
  d->f = pinocchio::ForceTpl<Scalar>(-force);
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL: {
      d->fext = joint1_placement_.act(d->f);
      d->joint1_f = -joint1_placement_.act(d->f);
      d->joint2_f = (joint2_placement_ * d->f1Mf2.inverse()).act(d->f);

      Matrix6s f_cross = Matrix6s::Zero(6, 6);
      f_cross.template topRightCorner<3, 3>() =
          pinocchio::skew(d->joint2_f.linear());
      f_cross.template bottomLeftCorner<3, 3>() =
          pinocchio::skew(d->joint2_f.linear());
      f_cross.template bottomRightCorner<3, 3>() =
          pinocchio::skew(d->joint2_f.angular());

      SE3 j2Mj1 =
          joint2_placement_.act(d->f1Mf2.actInv(joint1_placement_.inverse()));

      d->dtau_dq.noalias() =
          d->j2Jj2.transpose() *
          (-f_cross * (d->j2Jj2 - j2Mj1.toActionMatrix() * d->j1Jj1));
      break;
    }
    case pinocchio::ReferenceFrame::WORLD:
      throw_pretty(
          "Reference frame pinocchio::WORLD is not implemented, please use "
          "pinocchio::LOCAL");
      break;
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      throw_pretty(
          "Reference frame pinocchio::LOCAL_WORLD_ALIGNED is not implemented, "
          "please use pinocchio::LOCAL");
      break;
  }
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar>>
ContactModel6DLoopTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar> *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::print(std::ostream &os) const {
  os << "ContactModel6D {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", type=" << type_ << "}";
}

template <typename Scalar>
const int ContactModel6DLoopTpl<Scalar>::get_joint1_id() const {
  return joint1_id_;
}

template <typename Scalar>
const int ContactModel6DLoopTpl<Scalar>::get_joint2_id() const {
  return joint2_id_;
}

template <typename Scalar>
const typename pinocchio::SE3Tpl<Scalar> &
ContactModel6DLoopTpl<Scalar>::get_joint1_placement() const {
  return joint1_placement_;
}

template <typename Scalar>
const typename pinocchio::SE3Tpl<Scalar> &
ContactModel6DLoopTpl<Scalar>::get_joint2_placement() const {
  return joint2_placement_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s &
ContactModel6DLoopTpl<Scalar>::get_gains() const {
  return gains_;
}

}  // namespace crocoddyl
