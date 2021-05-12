///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/frame-rotation.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameRotationTpl<Scalar>::ResidualModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Matrix3s& Rref, const std::size_t nu)
    : Base(state, 3, nu, true, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFrameRotationTpl<Scalar>::ResidualModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Matrix3s& Rref)
    : Base(state, 3, true, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFrameRotationTpl<Scalar>::~ResidualModelFrameRotationTpl() {}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>&,
                                                 const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame rotation w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rRf.noalias() = oRf_inv_ * d->pinocchio->oMf[id_].rotation();
  data->r = pinocchio::log3(d->rRf);
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>&,
                                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog3(d->rRf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);

  // Compute the derivatives of the frame rotation
  const std::size_t nv = state_->get_nv();
  data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf.template bottomRows<3>();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFrameRotationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameRotationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s& ResidualModelFrameRotationTpl<Scalar>::get_reference() const {
  return Rref_;
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::set_reference(const Matrix3s& rotation) {
  Rref_ = rotation;
  oRf_inv_ = rotation.transpose();
}

}  // namespace crocoddyl
