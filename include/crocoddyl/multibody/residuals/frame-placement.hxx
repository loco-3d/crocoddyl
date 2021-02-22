///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
ResidualModelFramePlacementTpl<Scalar>::ResidualModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const pinocchio::FrameIndex id, const SE3& pref,
                                                                       const std::size_t nu)
    : Base(state, 6, nu), id_(id), pref_(pref), oMf_inv_(pref.inverse()), pin_model_(state->get_pinocchio()) {
  v_dependent_ = false;
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelFramePlacementTpl<Scalar>::ResidualModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const pinocchio::FrameIndex id, const SE3& pref)
    : Base(state, 6), id_(id), pref_(pref), oMf_inv_(pref.inverse()), pin_model_(state->get_pinocchio()) {
  v_dependent_ = false;
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelFramePlacementTpl<Scalar>::~ResidualModelFramePlacementTpl() {}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rMf = oMf_inv_ * d->pinocchio->oMf[id_];
  d->r = pinocchio::log6(d->rMf);
  data->r = d->r;  // this is needed because we overwrite it
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  // Update the frame placements
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFramePlacementTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFramePlacementTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::SE3Tpl<Scalar>& ResidualModelFramePlacementTpl<Scalar>::get_reference() const {
  return pref_;
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::set_reference(const SE3& placement) {
  pref_ = placement;
  oMf_inv_ = placement.inverse();
}

}  // namespace crocoddyl
