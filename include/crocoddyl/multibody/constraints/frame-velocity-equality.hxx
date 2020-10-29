///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh, IRI:CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/constraints/frame-velocity-equality.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ConstraintModelFrameVelocityEqualityTpl<Scalar>::ConstraintModelFrameVelocityEqualityTpl(
    boost::shared_ptr<StateMultibody> state, const FrameMotion& vref, const std::size_t& nu)
    : Base(state, nu, 0, 6), vref_(vref), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ConstraintModelFrameVelocityEqualityTpl<Scalar>::ConstraintModelFrameVelocityEqualityTpl(
    boost::shared_ptr<StateMultibody> state, const FrameMotion& vref)
    : Base(state, 0, 6), vref_(vref), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ConstraintModelFrameVelocityEqualityTpl<Scalar>::~ConstraintModelFrameVelocityEqualityTpl() {}

template <typename Scalar>
void ConstraintModelFrameVelocityEqualityTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                           const Eigen::Ref<const VectorXs>&,
                                                           const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  d->h = (pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, vref_.id, vref_.reference) - vref_.motion)
             .toVector();
  data->h = d->h;  // this is needed because we overwrite it
}

template <typename Scalar>
void ConstraintModelFrameVelocityEqualityTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                               const Eigen::Ref<const VectorXs>&,
                                                               const Eigen::Ref<const VectorXs>&) {
  // Get the partial derivatives of the local frame velocity
  Data* d = static_cast<Data*>(data.get());
  const std::size_t& nv = state_->get_nv();
  pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio, vref_.id, vref_.reference,
                                         data->Hx.leftCols(nv), data->Hx.rightCols(nv));
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> > ConstraintModelFrameVelocityEqualityTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ConstraintModelFrameVelocityEqualityTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameMotion)) {
    vref_ = *static_cast<const FrameMotion*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

template <typename Scalar>
void ConstraintModelFrameVelocityEqualityTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameMotion)) {
    FrameMotion& ref_map = *static_cast<FrameMotion*>(pv);
    ref_map = vref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

}  // namespace crocoddyl
