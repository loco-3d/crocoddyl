///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace crocoddyl {

template <typename Scalar>
typename KinematicLoopModelTpl<Scalar>::MaskArray
KinematicLoopModelTpl<Scalar>::DefaultMask() {
  return {{true, true, true, true, true, true}};
}

template <typename Scalar>
KinematicLoopModelTpl<Scalar>::KinematicLoopModelTpl(
    std::shared_ptr<StateMultibody> state,
    const pinocchio::JointIndex joint1_id, const SE3& joint1_placement,
    const pinocchio::JointIndex joint2_id, const SE3& joint2_placement,
    const pinocchio::ReferenceFrame type, const std::size_t nu,
    const Vector2s& gains, const MaskArray& mask)
    : Base(state, type,
           static_cast<std::size_t>(std::count(mask.begin(), mask.end(), true)),
           nu),
      joint1_id_(joint1_id),
      joint1_placement_(joint1_placement),
      joint2_id_(joint2_id),
      joint2_placement_(joint2_placement),
      gains_(gains),
      mask_(mask),
      S_(6, nc_) {
  if (nc_ == 0) {
    throw_pretty("Invalid argument: kinematic-loop mask cannot be empty");
  }
  const pinocchio::JointIndex njoints =
      static_cast<pinocchio::JointIndex>(state_->get_pinocchio()->njoints);
  if (joint1_id_ >= njoints || joint2_id_ >= njoints) {
    throw_pretty("Invalid argument: kinematic-loop joint id is out of range");
  }
  if (type != pinocchio::ReferenceFrame::LOCAL) {
    throw_pretty("Invalid argument: "
                 << "KinematicLoopModel only supports pinocchio::LOCAL");
  }

  S_.setZero();
  std::size_t nc = 0;
  for (std::size_t i = 0; i < 6; ++i) {
    if (mask_[i]) {
      S_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(nc)) =
          Scalar(1);
      ++nc;
    }
  }
}

template <typename Scalar>
KinematicLoopModelTpl<Scalar>::KinematicLoopModelTpl(
    std::shared_ptr<StateMultibody> state,
    const pinocchio::JointIndex joint1_id, const SE3& joint1_placement,
    const pinocchio::JointIndex joint2_id, const SE3& joint2_placement,
    const pinocchio::ReferenceFrame type, const Vector2s& gains,
    const MaskArray& mask)
    : KinematicLoopModelTpl(
          state, joint1_id, joint1_placement, joint2_id, joint2_placement, type,
          state != nullptr ? state->get_nv() : 0, gains, mask) {}

template <typename Scalar>
void KinematicLoopModelTpl<Scalar>::calc(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              joint1_id_, pinocchio::LOCAL, d->j1Jj1);
  pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              joint2_id_, pinocchio::LOCAL, d->j2Jj2);
  d->f1Jf1.noalias() = d->f1Xj1 * d->j1Jj1;
  d->f2Jf2.noalias() = d->f2Xj2 * d->j2Jj2;

  d->oMf1 = d->pinocchio->oMi[joint1_id_].act(joint1_placement_);
  d->oMf2 = d->pinocchio->oMi[joint2_id_].act(joint2_placement_);
  d->f1Mf2 = d->oMf1.actInv(d->oMf2);
  d->f1Xf2 = d->f1Mf2.toActionMatrix();

  d->Jc.noalias() = S_.transpose() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);

  d->f1vf1.setZero();
  d->f1af1.setZero();
  d->f2vf2.setZero();
  d->f2af2.setZero();
  d->f1vf2.setZero();
  d->f1af2.setZero();
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

  d->a0_6d = (d->f1af1 - d->f1af2 + d->f1vf1.cross(d->f1vf2)).toVector();
  if (gains_[0] != Scalar(0)) {
    d->a0_6d.noalias() -= gains_[0] * pinocchio::log6(d->f1Mf2).toVector();
  }
  if (gains_[1] != Scalar(0)) {
    d->a0_6d.noalias() += gains_[1] * (d->f1vf1 - d->f1vf2).toVector();
  }
  data->a0.noalias() = S_.transpose() * d->a0_6d;
}

template <typename Scalar>
void KinematicLoopModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  d->f1af1.setZero();
  d->f2af2.setZero();
  d->f1af2.setZero();
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

  d->da0_dq_t1.noalias() = d->f1Xj1 * d->a1_partial_dq;
  d->da0_dq_t2.noalias() =
      d->f1af2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);
  d->f2_a2_partial_dq.noalias() = d->f2Xj2 * d->a2_partial_dq;
  d->da0_dq_t2.noalias() += d->f1Xf2 * d->f2_a2_partial_dq;

  d->f1_v1_partial_dq.noalias() = d->f1Xj1 * d->v1_partial_dq;
  d->da0_dq_t3.noalias() = -d->f1vf2.toActionMatrix() * d->f1_v1_partial_dq;
  d->da0_dq_t3_tmp.noalias() =
      d->f1vf2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);
  d->da0_dq_t3.noalias() += d->f1vf1.toActionMatrix() * d->da0_dq_t3_tmp;
  d->da0_dq_t3_tmp.noalias() = d->f2Xj2 * d->v2_partial_dq;
  d->da0_dq_t3.noalias() +=
      d->f1vf1.toActionMatrix() * d->f1Xf2 * d->da0_dq_t3_tmp;

  d->da0_dx_6d.leftCols(nv).noalias() =
      d->da0_dq_t1 - d->da0_dq_t2 + d->da0_dq_t3;

  d->f2_a2_partial_dv.noalias() = d->f2Xj2 * d->a2_partial_dv;
  d->f1Jf2.noalias() = d->f1Xf2 * d->f2Jf2;
  d->da0_dx_6d.rightCols(nv).noalias() = d->f1Xj1 * d->a1_partial_dv;
  d->da0_dx_6d.rightCols(nv).noalias() -= d->f1Xf2 * d->f2_a2_partial_dv;
  d->da0_dx_6d.rightCols(nv).noalias() -= d->f1vf2.toActionMatrix() * d->f1Jf1;
  d->da0_dx_6d.rightCols(nv).noalias() += d->f1vf1.toActionMatrix() * d->f1Jf2;

  if (gains_[0] != Scalar(0)) {
    pinocchio::Jlog6(d->f1Mf2, d->Jlog6);
    d->dpos_dq.noalias() = d->oMf2.inverse().toActionMatrix() *
                           d->oMf1.toActionMatrix() * d->f1Jf1;
    d->dpos_dq.noalias() -= d->f2Jf2;
    d->da0_dx_6d.leftCols(nv).noalias() += gains_[0] * d->Jlog6 * d->dpos_dq;
  }

  d->f2_v2_partial_dq.noalias() = d->f2Xj2 * d->v2_partial_dq;
  if (gains_[1] != Scalar(0)) {
    d->f1_v2_partial_dq.noalias() = d->f1Xf2 * d->f2_v2_partial_dq;
    d->dvel_dq.noalias() = d->f1_v1_partial_dq - d->f1_v2_partial_dq;
    d->dvel_dq.noalias() -=
        d->f1vf2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);
    d->da0_dx_6d.leftCols(nv).noalias() += gains_[1] * d->dvel_dq;
    d->da0_dx_6d.rightCols(nv).noalias() +=
        gains_[1] * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);
  }
  d->dv0_dq_6d.noalias() = d->f1_v1_partial_dq;
  d->dv0_dq_6d.noalias() -=
      d->f1vf2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2);
  d->dv0_dq_6d.noalias() -= d->f1Xf2 * d->f2_v2_partial_dq;

  data->da0_dx.noalias() = S_.transpose() * d->da0_dx_6d;
  data->dv0_dq.noalias() = S_.transpose() * d->dv0_dq_6d;
}

template <typename Scalar>
void KinematicLoopModelTpl<Scalar>::updateForce(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) != nc_) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be " +
                                    std::to_string(nc_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  d->force_6d.setZero();
  std::size_t nc = 0;
  for (std::size_t i = 0; i < 6; ++i) {
    if (mask_[i]) {
      d->force_6d[static_cast<Eigen::Index>(i)] = -force[nc];
      ++nc;
    }
  }

  data->f = pinocchio::ForceTpl<Scalar>(d->force_6d);
  data->fext = joint1_placement_.act(data->f);
  d->joint1_f = -data->fext;
  d->joint2_f = (joint2_placement_ * d->f1Mf2.inverse()).act(data->f);

  pinocchio::skew(d->joint2_f.linear(),
                  d->f_cross.template topRightCorner<3, 3>());
  d->f_cross.template topLeftCorner<3, 3>().setZero();
  d->f_cross.template bottomLeftCorner<3, 3>() =
      d->f_cross.template topRightCorner<3, 3>();
  pinocchio::skew(d->joint2_f.angular(),
                  d->f_cross.template bottomRightCorner<3, 3>());

  d->j2Mj1 =
      joint2_placement_.act(d->f1Mf2.actInv(joint1_placement_.inverse()));
  d->j2Jj1.noalias() = d->j2Mj1.toActionMatrix() * d->j1Jj1;
  d->dtau_dq_tmp.noalias() = -d->f_cross * (d->j2Jj2 - d->j2Jj1);
  data->dtau_dq.noalias() = d->j2Jj2.transpose() * d->dtau_dq_tmp;
}

template <typename Scalar>
void KinematicLoopModelTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const Eigen::Ref<const MatrixXs>& df_dx,
    const Eigen::Ref<const MatrixXs>& df_du) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx()) {
    throw_pretty("df_dx has wrong dimension");
  }
  if (static_cast<std::size_t>(df_du.rows()) != nc_ ||
      static_cast<std::size_t>(df_du.cols()) != nu_) {
    throw_pretty("df_du has wrong dimension");
  }

  data->df_dx.noalias() = -df_dx;
  data->df_du.noalias() = -df_du;
}

template <typename Scalar>
std::shared_ptr<ImplicitConstraintDataAbstractTpl<Scalar> >
KinematicLoopModelTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  if (data == nullptr) {
    throw_pretty("Invalid argument: Pinocchio data cannot be null");
  }
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
KinematicLoopModelTpl<NewScalar> KinematicLoopModelTpl<Scalar>::cast() const {
  typedef KinematicLoopModelTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()),
      joint1_id_, joint1_placement_.template cast<NewScalar>(), joint2_id_,
      joint2_placement_.template cast<NewScalar>(), type_, nu_,
      gains_.template cast<NewScalar>(), mask_);
  return ret;
}

template <typename Scalar>
const pinocchio::JointIndex& KinematicLoopModelTpl<Scalar>::get_joint1_id()
    const {
  return joint1_id_;
}

template <typename Scalar>
const pinocchio::JointIndex& KinematicLoopModelTpl<Scalar>::get_joint2_id()
    const {
  return joint2_id_;
}

template <typename Scalar>
const typename KinematicLoopModelTpl<Scalar>::SE3&
KinematicLoopModelTpl<Scalar>::get_joint1_placement() const {
  return joint1_placement_;
}

template <typename Scalar>
const typename KinematicLoopModelTpl<Scalar>::SE3&
KinematicLoopModelTpl<Scalar>::get_joint2_placement() const {
  return joint2_placement_;
}

template <typename Scalar>
const typename KinematicLoopModelTpl<Scalar>::Vector2s&
KinematicLoopModelTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void KinematicLoopModelTpl<Scalar>::set_gains(const Vector2s& gains) {
  gains_ = gains;
}

template <typename Scalar>
const typename KinematicLoopModelTpl<Scalar>::MaskArray&
KinematicLoopModelTpl<Scalar>::get_mask() const {
  return mask_;
}

template <typename Scalar>
const typename KinematicLoopModelTpl<Scalar>::Matrix6xs&
KinematicLoopModelTpl<Scalar>::get_selection_matrix() const {
  return S_;
}

template <typename Scalar>
void KinematicLoopModelTpl<Scalar>::print(std::ostream& os) const {
  os << "KinematicLoopModel {joint1="
     << state_->get_pinocchio()->names[joint1_id_]
     << ", joint2=" << state_->get_pinocchio()->names[joint2_id_]
     << ", nc=" << nc_ << "}";
}

}  // namespace crocoddyl
