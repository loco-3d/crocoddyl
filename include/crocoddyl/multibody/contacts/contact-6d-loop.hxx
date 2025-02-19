
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
    const SE3 &joint2_placement, const pinocchio::ReferenceFrame type,
    const std::size_t nu, const Vector2s &gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 2, 6, nu), gains_(gains) {
  if (type != pinocchio::ReferenceFrame::LOCAL)
    throw_pretty(
        "Only reference frame pinocchio::LOCAL is supported  "
        "for 6D loop contacts");
  if (joint1_id == 0 || joint2_id == 0)
    throw_pretty(
        "Either joint1_id or joint2_id is set to 0, cannot use form a "
        "kinematic loop with the world");

  id_[0] = joint1_id;
  id_[1] = joint2_id;
  placements_[0] = joint1_placement;
  placements_[1] = joint2_placement;
  type_ = type;
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::ContactModel6DLoopTpl(
    boost::shared_ptr<StateMultibody> state, const int joint1_id,
    const SE3 &joint1_placement, const int joint2_id,
    const SE3 &joint2_placement, const pinocchio::ReferenceFrame type,
    const Vector2s &gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 2, 6), gains_(gains) {
  if (type != pinocchio::ReferenceFrame::LOCAL)
    throw_pretty(
        "Only reference frame pinocchio::LOCAL is supported  "
        "for 6D loop contacts");
  if (joint1_id == 0 || joint2_id == 0)
    throw_pretty(
        "Either joint1_id or joint2_id is set to 0, cannot use form a "
        "kinematic loop with the world");

  id_[0] = joint1_id;
  id_[1] = joint2_id;
  placements_[0] = joint1_placement;
  placements_[1] = joint2_placement;
  type_ = type;
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::~ContactModel6DLoopTpl() {}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calc(
    const boost::shared_ptr<ContactDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  // TODO(jfoster): is this frame placement update call needed?
  pinocchio::updateFramePlacements(*state_->get_pinocchio().get(),
                                   *d->pinocchio);

  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstract &fdata_i = d->force_datas[i];
    pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                                fdata_i.frame, pinocchio::LOCAL, fdata_i.jJj);
    fdata_i.fJf = fdata_i.fXj * fdata_i.jJj;
    fdata_i.oMf = d->pinocchio->oMi[id_[i]].act(fdata_i.jMf);
  }

  ForceDataAbstract &fdata_1 = d->force_datas[0];
  ForceDataAbstract &fdata_2 = d->force_datas[1];

  d->f1Mf2 = fdata_1.oMf.actInv(fdata_2.oMf);
  d->f1Xf2 = d->f1Mf2.toActionMatrix();
  d->Jc = fdata_1.fJf - d->f1Xf2 * fdata_2.fJf;

  // Compute the acceleration drift
  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstract &fdata_i = d->force_datas[i];
    fdata_i.fvf = fdata_i.jMf.actInv(d->pinocchio->v[id_[i]]);
    fdata_i.faf = fdata_i.jMf.actInv(d->pinocchio->a[id_[i]]);
  }
  d->f1vf2 = d->f1Mf2.act(fdata_2.fvf);
  d->f1af2 = d->f1Mf2.act(fdata_2.faf);

  d->a0 =
      (fdata_1.faf - d->f1Mf2.act(fdata_2.faf) + fdata_1.fvf.cross(d->f1vf2))
          .toVector();

  if (gains_[0] != 0.0)
    d->a0 += gains_[0] * -pinocchio::log6(d->f1Mf2).toVector();
  if (gains_[1] != 0.0)
    d->a0 += gains_[1] * (fdata_1.fvf - d->f1vf2).toVector();
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ContactDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  const std::size_t nv = state_->get_nv();
  ForceDataAbstract &fdata_1 = d->force_datas[0];
  ForceDataAbstract &fdata_2 = d->force_datas[1];

  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstract &fdata_i = d->force_datas[i];
    fdata_i.faf = fdata_i.jMf.actInv(d->pinocchio->a[id_[i]]);
  }
  d->f1af2 = d->f1Mf2.act(fdata_2.faf);

  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstract &fdata_i = d->force_datas[i];

    pinocchio::getJointAccelerationDerivatives(
        *state_->get_pinocchio().get(), *d->pinocchio, fdata_i.frame,
        pinocchio::LOCAL, fdata_i.v_partial_dq, fdata_i.a_partial_dq,
        fdata_i.a_partial_dv, fdata_i.a_partial_da);
  }

  d->da0_dq_t1 = fdata_1.jMf.toActionMatrixInverse() * fdata_1.a_partial_dq;

  d->da0_dq_t2 =
      d->f1af2.toActionMatrix() * (fdata_1.fJf - d->f1Xf2 * fdata_2.fJf) +
      d->f1Xf2 * (fdata_2.jMf.toActionMatrixInverse() * fdata_2.a_partial_dq);
  d->da0_dq_t3 =
      -d->f1vf2.toActionMatrix() * (fdata_1.jMf.toActionMatrixInverse() *
                                    fdata_1.v_partial_dq)  // part 1
      + fdata_1.fvf.toActionMatrix() * d->f1vf2.toActionMatrix() *
            (fdata_1.fJf - d->f1Xf2 * fdata_2.fJf)  // part 2
      + fdata_1.fvf.toActionMatrix() * d->f1Xf2 *
            (fdata_2.jMf.toActionMatrixInverse() *
             fdata_2.v_partial_dq);  // part 3

  d->da0_dx.leftCols(nv) = d->da0_dq_t1 - d->da0_dq_t2 + d->da0_dq_t3;
  d->da0_dx.rightCols(nv) =
      fdata_1.jMf.toActionMatrixInverse() * fdata_1.a_partial_dv -
      d->f1Xf2 * (fdata_2.jMf.toActionMatrixInverse() * fdata_2.a_partial_dv) -
      d->f1vf2.toActionMatrix() * fdata_1.fJf +
      fdata_1.fvf.toActionMatrix() * d->f1Xf2 * fdata_2.fJf;

  if (gains_[0] != 0.0) {
    Matrix6s f1Mf2_log6;
    pinocchio::Jlog6(d->f1Mf2, f1Mf2_log6);
    d->da0_dx.leftCols(nv) +=
        gains_[0] *
        (-f1Mf2_log6 * (-fdata_2.oMf.toActionMatrixInverse() *
                            fdata_1.oMf.toActionMatrix() * fdata_1.fJf +
                        fdata_2.fJf));
  }
  if (gains_[1] != 0.0) {
    d->da0_dx.leftCols(nv) +=
        gains_[1] *
        (fdata_1.jMf.toActionMatrixInverse() * fdata_1.v_partial_dq -
         d->f1Mf2.act(fdata_2.fvf).toActionMatrix() *
             (fdata_1.fJf - d->f1Xf2 * fdata_2.fJf) -
         d->f1Xf2 * fdata_2.jMf.toActionMatrixInverse() * fdata_2.v_partial_dq);
    d->da0_dx.rightCols(nv) +=
        gains_[1] * (fdata_1.fJf - d->f1Xf2 * fdata_2.fJf);
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
  ForceDataAbstract &fdata_1 = d->force_datas[0];
  ForceDataAbstract &fdata_2 = d->force_datas[1];

  pinocchio::ForceTpl<Scalar> f = pinocchio::ForceTpl<Scalar>(-force);
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL: {
      // TODO(jfoster): unsure if this logic is correct
      fdata_1.f = fdata_1.jMf.act(f);
      fdata_1.fext = -fdata_1.jMf.act(f);
      fdata_2.f = -(fdata_2.jMf * d->f1Mf2.inverse()).act(f);
      fdata_2.fext = (fdata_2.jMf * d->f1Mf2.inverse()).act(f);

      d->dtau_dq.setZero();
      d->f_cross.setZero();
      d->f_cross.topRightCorner(3, 3) = pinocchio::skew(fdata_2.fext.linear());
      d->f_cross.bottomLeftCorner(3, 3) =
          pinocchio::skew(fdata_2.fext.linear());
      d->f_cross.bottomRightCorner(3, 3) =
          pinocchio::skew(fdata_2.fext.angular());

      SE3 j2Mj1 = fdata_2.jMf.act(d->f1Mf2.actInv(fdata_1.jMf.inverse()));
      d->dtau_dq = fdata_2.jJj.transpose() * (-d->f_cross * (fdata_2.jJj - j2Mj1.toActionMatrix() * fdata_1.jJj));
      break;
    }
    case pinocchio::ReferenceFrame::WORLD:
      throw_pretty(
          "Reference frame WORLD is not implemented for kinematic loops");
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      throw_pretty(
          "Reference frame LOCAL_WORLD_ALIGNED is not implemented for "
          "kinematic loops");
  }
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::updateForceDiff(
    const boost::shared_ptr<ContactDataAbstract> &data, const MatrixXs &df_dx,
    const MatrixXs &df_du) {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx())
    throw_pretty("df_dx has wrong dimension");

  if (static_cast<std::size_t>(df_du.rows()) != nc_ ||
      static_cast<std::size_t>(df_du.cols()) != nu_)
    throw_pretty("df_du has wrong dimension");

  data->df_dx = -df_dx;
  data->df_du = -df_du;
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
  os << "ContactModel6D {frame 1 = "
     << state_->get_pinocchio()->frames[id_[0]].name
     << ", frame 2 = " << state_->get_pinocchio()->frames[id_[1]].name
     << ", type = " << type_ << ", gains = " << gains_.transpose() << "}";
}

template <typename Scalar>
const typename pinocchio::SE3Tpl<Scalar> &
ContactModel6DLoopTpl<Scalar>::get_placement(const int force_index) const {
  return placements_[force_index];
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s &
ContactModel6DLoopTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::set_gains(
    const typename MathBaseTpl<Scalar>::Vector2s &gains) {
  gains_ = gains;
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::set_placement(
    const int force_index,
    const typename pinocchio::SE3Tpl<Scalar> &placement) {
  placements_[force_index] = placement;
}

}  // namespace crocoddyl
