///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/constraints/frame-placement-equality.hpp"
#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
ConstraintModelFramePlacementEqualityTpl<Scalar>::ConstraintModelFramePlacementEqualityTpl(
    boost::shared_ptr<StateMultibody> state, const FramePlacement& Mref, const std::size_t& nu)
    : Base(state, nu, 0, 6), Mref_(Mref), oMf_inv_(Mref.placement.inverse()), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ConstraintModelFramePlacementEqualityTpl<Scalar>::ConstraintModelFramePlacementEqualityTpl(
    boost::shared_ptr<StateMultibody> state, const FramePlacement& Mref)
    : Base(state, 0, 6), Mref_(Mref), oMf_inv_(Mref.placement.inverse()), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ConstraintModelFramePlacementEqualityTpl<Scalar>::~ConstraintModelFramePlacementEqualityTpl() {}

template <typename Scalar>
void ConstraintModelFramePlacementEqualityTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                            const Eigen::Ref<const VectorXs>&,
                                                            const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, Mref_.id);
  d->rMf = oMf_inv_ * d->pinocchio->oMf[Mref_.id];
  d->h = pinocchio::log6(d->rMf);
  data->h = d->h;  // this is needed because we overwrite it
}

template <typename Scalar>
void ConstraintModelFramePlacementEqualityTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                                const Eigen::Ref<const VectorXs>&,
                                                                const Eigen::Ref<const VectorXs>&) {
  // Update the frame placements
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, Mref_.id, pinocchio::LOCAL, d->fJf);
  d->J.noalias() = d->rJf * d->fJf;

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  data->Hx.leftCols(nv) = d->J;
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> > ConstraintModelFramePlacementEqualityTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ConstraintModelFramePlacementEqualityTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FramePlacement)) {
    Mref_ = *static_cast<const FramePlacement*>(pv);
    oMf_inv_ = Mref_.placement.inverse();
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FramePlacement)");
  }
}

template <typename Scalar>
void ConstraintModelFramePlacementEqualityTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FramePlacement)) {
    FramePlacement& ref_map = *static_cast<FramePlacement*>(pv);
    ref_map = Mref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FramePlacement)");
  }
}

}  // namespace crocoddyl
