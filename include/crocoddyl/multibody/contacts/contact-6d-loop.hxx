
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
  if (type != pinocchio::ReferenceFrame::LOCAL) {
    std::cerr << "Warning: Only reference frame pinocchio::LOCAL is supported "
                 "for 6D loop contacts"
              << std::endl;
  }
  if (joint1_id == 0 || joint2_id == 0) {
    std::cerr << "Warning: At least one of the parents joints id is zero"
                 "you should use crocoddyl::ContactModel6D instead"
              << std::endl;
  }

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
  if (type != pinocchio::ReferenceFrame::LOCAL) {
    std::cerr << "Warning: Only reference frame pinocchio::LOCAL is supported "
                 "for 6D loop contacts"
              << std::endl;
  }
  if (joint1_id == 0 || joint2_id == 0) {
    std::cerr << "Warning: At least one of the parents joints id is zero"
                 "you should use crocoddyl::ContactModel6D instead"
              << std::endl;
  }

  id_[0].frame = joint1_id;
  id_[1].frame = joint2_id;
  placements_[0].jMf = joint1_placement;
  placements_[1].jMf = joint2_placement;
  type_ = type;
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::~ContactModel6DLoopTpl() {}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calc(
    const boost::shared_ptr<ContactDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  ForceDataAbstractTpl<Scalar> &fdata_1 = d->force_datas[0];
  ForceDataAbstractTpl<Scalar> &fdata_2 = d->force_datas[1];

  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstractTpl<Scalar> &fdata_i = d->force_datas[i];
    pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                                fdata_i.frame, pinocchio::LOCAL, fdata_i.jJj);
    fdata_i.fJf.noalias() = fdata_i.fXj * fdata_i.jJj;
    fdata_i.oMf = d->pinocchio->oMi[id_[i]].act(fdata_i.jMf);
  }

  d->f1Mf2 = fdata_1.oMf.actInv(fdata_2.oMf);
  d->f1Xf2.noalias() = d->f1Mf2.toActionMatrix();
  d->Jc.noalias() = fdata_1.fJf - d->f1Xf2 * fdata_2.fJf;

  // Compute the acceleration drift
  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstractTpl<Scalar> &fdata_i = d->force_datas[i];
    fdata_i.fvf = fdata_i.jMf.actInv(d->pinocchio->v[id_[i]]);
    fdata_i.faf = fdata_i.jMf.actInv(d->pinocchio->a[id_[i]]);
  }
  d->f1vf2 = d->f1Mf2.act(fdata_2.fvf);
  d->f1af2 = d->f1Mf2.act(fdata_2.faf);

  d->a0.noalias() =
      (fdata_1.faf - d->f1af2 + fdata_1.fvf.cross(d->f1vf2)).toVector();

  if (std::abs<Scalar>(gains_[0]) > std::numeric_limits<Scalar>::epsilon()) {
    d->a0.noalias() -= gains_[0] * pinocchio::log6(d->f1Mf2).toVector();
  }
  if (std::abs<Scalar>(gains_[1]) > std::numeric_limits<Scalar>::epsilon()) {
    d->a0.noalias() += gains_[1] * (fdata_1.fvf - d->f1vf2).toVector();
  }
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ContactDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  const std::size_t nv = state_->get_nv();
  ForceDataAbstractTpl<Scalar> &fdata_1 = d->force_datas[0];
  ForceDataAbstractTpl<Scalar> &fdata_2 = d->force_datas[1];

  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstractTpl<Scalar> &fdata_i = d->force_datas[i];
    fdata_i.faf = fdata_i.jMf.actInv(d->pinocchio->a[id_[i]]);
  }
  d->f1af2 = d->f1Mf2.act(d->force_datas[1].faf);

  for (int i = 0; i < nf_; ++i) {
    ForceDataAbstractTpl<Scalar> &fdata_i = d->force_datas[i];

    pinocchio::getJointAccelerationDerivatives(
        *state_->get_pinocchio().get(), *d->pinocchio, fdata_i.frame,
        pinocchio::LOCAL, fdata_i.v_partial_dq, fdata_i.a_partial_dq,
        fdata_i.a_partial_dv, fdata_i.a_partial_da);
  }

  d->da0_dq_t1.noalias() =
      fdata_1.jMf.toActionMatrixInverse() * fdata_1.a_partial_dq;

  d->da0_dq_t2.noalias() = d->f1af2.toActionMatrix() * d->Jc;
  fdata_2.f_a_partial_dq.noalias() =
      fdata_2.jMf.toActionMatrixInverse() * fdata_2.a_partial_dq;
  d->da0_dq_t2.noalias() += d->f1Xf2 * fdata_2.f_a_partial_dq;

  fdata_1.f_v_partial_dq.noalias() =
      fdata_1.jMf.toActionMatrixInverse() * fdata_1.v_partial_dq;
  d->da0_dq_t3.noalias() = -d->f1vf2.toActionMatrix() * fdata_1.f_v_partial_dq;
  d->da0_dq_t3_tmp.noalias() = d->f1vf2.toActionMatrix() * d->Jc;
  d->da0_dq_t3.noalias() += fdata_1.fvf.toActionMatrix() * d->da0_dq_t3_tmp;
  d->da0_dq_t3_tmp.noalias() =
      fdata_2.jMf.toActionMatrixInverse() * fdata_2.v_partial_dq;
  d->da0_dq_t3.noalias() +=
      fdata_1.fvf.toActionMatrix() * d->f1Xf2 * d->da0_dq_t3_tmp;
  d->da0_dx.leftCols(nv).noalias() = d->da0_dq_t1 - d->da0_dq_t2 + d->da0_dq_t3;

  fdata_2.f_a_partial_dv.noalias() =
      fdata_2.jMf.toActionMatrixInverse() * fdata_2.a_partial_dv;
  d->f1Jf2.noalias() = d->f1Xf2 * fdata_2.fJf;
  d->da0_dx.rightCols(nv).noalias() =
      fdata_1.jMf.toActionMatrixInverse() * fdata_1.a_partial_dv;
  d->da0_dx.rightCols(nv).noalias() -= d->f1Xf2 * fdata_2.f_a_partial_dv;
  d->da0_dx.rightCols(nv).noalias() -= d->f1vf2.toActionMatrix() * fdata_1.fJf;
  d->da0_dx.rightCols(nv).noalias() += fdata_1.fvf.toActionMatrix() * d->f1Jf2;

  if (std::abs<Scalar>(gains_[0]) > std::numeric_limits<Scalar>::epsilon()) {
    Matrix6s f1Mf2_log6;
    pinocchio::Jlog6(d->f1Mf2, f1Mf2_log6);
    d->dpos_dq.noalias() = fdata_2.oMf.toActionMatrixInverse() *
                           fdata_1.oMf.toActionMatrix() * fdata_1.fJf;
    d->dpos_dq.noalias() -= fdata_2.fJf;
    d->da0_dx.leftCols(nv).noalias() += gains_[0] * f1Mf2_log6 * d->dpos_dq;
  }
  if (std::abs<Scalar>(gains_[1]) > std::numeric_limits<Scalar>::epsilon()) {
    fdata_2.f_v_partial_dq.noalias() =
        fdata_2.jMf.toActionMatrixInverse() * fdata_2.v_partial_dq;
    d->f1_v2_partial_dq.noalias() = d->f1Xf2 * fdata_2.f_v_partial_dq;
    d->dvel_dq.noalias() = fdata_1.f_v_partial_dq - d->f1_v2_partial_dq;
    d->dvel_dq.noalias() -= d->f1vf2.toActionMatrix() * d->Jc;
    d->da0_dx.leftCols(nv).noalias() += gains_[1] * d->dvel_dq;
    d->da0_dx.rightCols(nv).noalias() += gains_[1] * d->Jc;
  }
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::updateForce(
    const boost::shared_ptr<ContactDataAbstract> &data, const VectorXs &force) {
  if (force.size() != 6) {
    throw_pretty("Contact force vector has wrong dimension (expected 6 got "
                 << force.size() << ")");
  }
  Data *d = static_cast<Data *>(data.get());
  ForceDataAbstractTpl<Scalar> &fdata_1 = d->force_datas[0];
  ForceDataAbstractTpl<Scalar> &fdata_2 = d->force_datas[1];

  fdata_1.f = pinocchio::ForceTpl<Scalar>(-force);
  fdata_1.fext = fdata_1.jMf.act(fdata_1.f);
  fdata_2.f = -fdata_1.jMf.act(fdata_1.f);
  fdata_2.fext = (fdata_2.jMf * d->f1Mf2.inverse()).act(fdata_1.f);

  Matrix6s f_cross = Matrix6s::Zero(6, 6);
  f_cross.template topRightCorner<3, 3>() = pinocchio::skew(fdata_2.f.linear());
  f_cross.template bottomLeftCorner<3, 3>() =
      pinocchio::skew(fdata_2.f.linear());
  f_cross.template bottomRightCorner<3, 3>() =
      pinocchio::skew(fdata_2.f.angular());

  SE3 j2Mj1 = fdata_2.jMf.act(d->f1Mf2.actInv(fdata_1.jMf.inverse()));
  d->j2Jj1.noalias() = j2Mj1.toActionMatrix() * fdata_1.jJj;
  d->dtau_dq_tmp.noalias() = -f_cross * (fdata_2.jJj - d->j2Jj1);
  d->dtau_dq.noalias() = fdata_2.jJj.transpose() * d->dtau_dq_tmp;
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
