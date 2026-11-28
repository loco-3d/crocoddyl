///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace crocoddyl {

template <typename Scalar>
typename ContactModelTpl<Scalar>::MaskArray
ContactModelTpl<Scalar>::DefaultMask() {
  return {{true, true, true, true, true, true}};
}

template <typename Scalar>
ContactModelTpl<Scalar>::ContactModelTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& reference, const pinocchio::ReferenceFrame type,
    const std::size_t nu, const Vector2s& gains, const MaskArray& mask)
    : Base(state, type,
           static_cast<std::size_t>(std::count(mask.begin(), mask.end(), true)),
           nu),
      reference_(reference),
      gains_(gains),
      mask_(mask),
      spatial_(mask[3] || mask[4] || mask[5]),
      S_(6, nc_) {
  if (nc_ == 0) {
    throw_pretty("Invalid argument: contact mask cannot be empty");
  }
  if (id >= state_->get_pinocchio()->frames.size()) {
    throw_pretty("Invalid argument: contact frame id is out of range");
  }
  if (type != pinocchio::LOCAL && type != pinocchio::WORLD &&
      type != pinocchio::LOCAL_WORLD_ALIGNED) {
    throw_pretty("Invalid argument: unsupported contact reference frame");
  }
  id_ = id;
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
ContactModelTpl<Scalar>::ContactModelTpl(std::shared_ptr<StateMultibody> state,
                                         const pinocchio::FrameIndex id,
                                         const SE3& reference,
                                         const pinocchio::ReferenceFrame type,
                                         const Vector2s& gains,
                                         const MaskArray& mask)
    : ContactModelTpl(state, id, reference, type,
                      state != nullptr ? state->get_nv() : 0, gains, mask) {}

template <typename Scalar>
void ContactModelTpl<Scalar>::calc(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio,
                                  id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                              id_, pinocchio::LOCAL, d->fJf);
  if (spatial_) {
    d->a0_local = pinocchio::getFrameAcceleration(
        *state_->get_pinocchio().get(), *d->pinocchio, id_);
    if (gains_[0] != Scalar(0)) {
      d->rMf = reference_.actInv(d->pinocchio->oMf[id_]);
      d->a0_local += gains_[0] * pinocchio::log6(d->rMf);
    }
    if (gains_[1] != Scalar(0)) {
      d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                         *d->pinocchio, id_);
      d->a0_local += gains_[1] * d->v;
    }
  } else {
    d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                       *d->pinocchio, id_);
    d->a0_local.setZero();
    d->a0_local.linear() = pinocchio::getFrameClassicalAcceleration(
                               *state_->get_pinocchio().get(), *d->pinocchio,
                               id_, pinocchio::LOCAL)
                               .linear();
    if (gains_[0] != Scalar(0)) {
      d->dp = d->pinocchio->oMf[id_].translation() - reference_.translation();
      d->dp_local.noalias() =
          d->pinocchio->oMf[id_].rotation().transpose() * d->dp;
      d->a0_local.linear() += gains_[0] * d->dp_local;
    }
    if (gains_[1] != Scalar(0)) {
      d->a0_local.linear() += gains_[1] * d->v.linear();
    }
  }

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->Jc_6d = d->fJf;
      d->a0_6d = d->a0_local.toVector();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->lwaMl.rotation(d->pinocchio->oMf[id_].rotation());
      d->Jc_6d.noalias() = d->lwaMl.toActionMatrix() * d->fJf;
      d->a0_6d.noalias() = d->lwaMl.act(d->a0_local).toVector();
      break;
  }

  data->Jc.noalias() = S_.transpose() * d->Jc_6d;
  data->a0.noalias() = S_.transpose() * d->a0_6d;
}

template <typename Scalar>
void ContactModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint =
      state_->get_pinocchio()->frames[d->frame].parentJoint;
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
      d->v_partial_dq, d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();

  d->da0_local_dx.leftCols(nv).noalias() = d->fXj * d->a_partial_dq;
  d->da0_local_dx.rightCols(nv).noalias() = d->fXj * d->a_partial_dv;
  d->dv0_local_dq.noalias() = d->fXj * d->v_partial_dq;

  if (!spatial_) {
    pinocchio::skew(d->v.linear(), d->vv_skew);
    pinocchio::skew(d->v.angular(), d->vw_skew);
    d->da0_local_dx.leftCols(nv).template topRows<3>().noalias() +=
        d->vw_skew * d->dv0_local_dq.template topRows<3>();
    d->da0_local_dx.leftCols(nv).template topRows<3>().noalias() -=
        d->vv_skew * d->dv0_local_dq.template bottomRows<3>();
    d->da0_local_dx.rightCols(nv).template topRows<3>().noalias() +=
        d->vw_skew * d->fJf.template topRows<3>();
    d->da0_local_dx.rightCols(nv).template topRows<3>().noalias() -=
        d->vv_skew * d->fJf.template bottomRows<3>();
  }

  if (gains_[0] != Scalar(0)) {
    if (spatial_) {
      pinocchio::Jlog6(d->rMf, d->rMf_Jlog6);
      d->da0_local_dx.leftCols(nv).noalias() +=
          gains_[0] * d->rMf_Jlog6 * d->fJf;
    } else {
      pinocchio::skew(d->dp_local, d->dp_skew);
      d->da0_local_dx.leftCols(nv).template topRows<3>().noalias() +=
          gains_[0] * d->dp_skew * d->fJf.template bottomRows<3>();
      d->da0_local_dx.leftCols(nv).template topRows<3>().noalias() +=
          gains_[0] * d->fJf.template topRows<3>();
    }
  }
  if (gains_[1] != Scalar(0)) {
    d->da0_local_dx.leftCols(nv).noalias() +=
        gains_[1] * d->fXj * d->v_partial_dq;
    d->da0_local_dx.rightCols(nv).noalias() += gains_[1] * d->fJf;
  }
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->da0_dx_6d = d->da0_local_dx;
      d->dv0_dq_6d = d->dv0_local_dq;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      // Recalculate the constrained accelerations after imposing contact
      // constraints. This is necessary for the forward-dynamics case.
      if (spatial_) {
        d->a0_local = pinocchio::getFrameAcceleration(
            *state_->get_pinocchio().get(), *d->pinocchio, id_);
        if (gains_[0] != Scalar(0)) {
          d->a0_local += gains_[0] * pinocchio::log6(d->rMf);
        }
        if (gains_[1] != Scalar(0)) {
          d->a0_local += gains_[1] * d->v;
        }
      } else {
        d->a0_local.setZero();
        d->a0_local.linear() = pinocchio::getFrameClassicalAcceleration(
                                   *state_->get_pinocchio().get(),
                                   *d->pinocchio, id_, pinocchio::LOCAL)
                                   .linear();
        if (gains_[0] != Scalar(0)) {
          d->a0_local.linear() += gains_[0] * d->dp_local;
        }
        if (gains_[1] != Scalar(0)) {
          d->a0_local.linear() += gains_[1] * d->v.linear();
        }
      }
      d->a0_6d.noalias() = d->lwaMl.act(d->a0_local).toVector();

      pinocchio::skew(d->a0_6d.template head<3>(), d->av_skew);
      pinocchio::skew(d->a0_6d.template tail<3>(), d->aw_skew);
      d->av_world_skew.noalias() =
          d->av_skew * d->pinocchio->oMf[id_].rotation();
      d->aw_world_skew.noalias() =
          d->aw_skew * d->pinocchio->oMf[id_].rotation();
      d->da0_dx_6d.noalias() = d->lwaMl.toActionMatrix() * d->da0_local_dx;
      d->da0_dx_6d.leftCols(nv).template topRows<3>().noalias() -=
          d->av_world_skew * d->fJf.template bottomRows<3>();
      d->da0_dx_6d.leftCols(nv).template bottomRows<3>().noalias() -=
          d->aw_world_skew * d->fJf.template bottomRows<3>();

      d->v0 = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(),
                                          *d->pinocchio, id_, type_);
      pinocchio::skew(d->v0.linear(), d->vv_skew);
      pinocchio::skew(d->v0.angular(), d->vw_skew);
      d->vv_world_skew.noalias() =
          d->vv_skew * d->pinocchio->oMf[id_].rotation();
      d->vw_world_skew.noalias() =
          d->vw_skew * d->pinocchio->oMf[id_].rotation();
      d->dv0_dq_6d.noalias() = d->lwaMl.toActionMatrix() * d->dv0_local_dq;
      d->dv0_dq_6d.template topRows<3>().noalias() -=
          d->vv_world_skew * d->fJf.template bottomRows<3>();
      d->dv0_dq_6d.template bottomRows<3>().noalias() -=
          d->vw_world_skew * d->fJf.template bottomRows<3>();
      data->a0.noalias() = S_.transpose() * d->a0_6d;
      break;
  }

  data->da0_dx.noalias() = S_.transpose() * d->da0_dx_6d;
  data->dv0_dq.noalias() = S_.transpose() * d->dv0_dq_6d;
}

template <typename Scalar>
void ContactModelTpl<Scalar>::updateForce(
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
      d->force_6d[static_cast<Eigen::Index>(i)] =
          force[static_cast<Eigen::Index>(nc)];
      ++nc;
    }
  }
  data->f = pinocchio::ForceTpl<Scalar>(d->force_6d);

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->fext = data->jMf.act(data->f);
      data->dtau_dq.setZero();
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->f_local = d->lwaMl.actInv(data->f);
      data->fext = data->jMf.act(d->f_local);
      pinocchio::skew(d->f_local.linear(), d->fv_skew);
      pinocchio::skew(d->f_local.angular(), d->fw_skew);
      d->fJf_df.template topRows<3>().noalias() =
          d->fv_skew * d->fJf.template bottomRows<3>();
      d->fJf_df.template bottomRows<3>().noalias() =
          d->fw_skew * d->fJf.template bottomRows<3>();
      data->dtau_dq.noalias() = -d->fJf.transpose() * d->fJf_df;
      break;
  }
}

template <typename Scalar>
std::shared_ptr<ImplicitConstraintDataAbstractTpl<Scalar> >
ContactModelTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  if (data == nullptr) {
    throw_pretty("Invalid argument: Pinocchio data cannot be null");
  }
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModelTpl<NewScalar> ContactModelTpl<Scalar>::cast() const {
  typedef ContactModelTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      reference_.template cast<NewScalar>(), type_, nu_,
      gains_.template cast<NewScalar>(), mask_);
  return ret;
}

template <typename Scalar>
const typename ContactModelTpl<Scalar>::SE3&
ContactModelTpl<Scalar>::get_reference() const {
  return reference_;
}

template <typename Scalar>
void ContactModelTpl<Scalar>::set_reference(const SE3& reference) {
  reference_ = reference;
}

template <typename Scalar>
const typename ContactModelTpl<Scalar>::Vector2s&
ContactModelTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModelTpl<Scalar>::set_gains(const Vector2s& gains) {
  gains_ = gains;
}

template <typename Scalar>
const typename ContactModelTpl<Scalar>::MaskArray&
ContactModelTpl<Scalar>::get_mask() const {
  return mask_;
}

template <typename Scalar>
const typename ContactModelTpl<Scalar>::Matrix6xs&
ContactModelTpl<Scalar>::get_selection_matrix() const {
  return S_;
}

template <typename Scalar>
void ContactModelTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", nc=" << nc_ << ", type=" << type_ << "}";
}

}  // namespace crocoddyl
