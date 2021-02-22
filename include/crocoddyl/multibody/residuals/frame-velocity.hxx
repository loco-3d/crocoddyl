///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-velocity.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameVelocityTpl<Scalar>::ResidualModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Motion& velocity,
                                                                     const pinocchio::ReferenceFrame type,
                                                                     const std::size_t nu)
    : Base(state, 6, nu), id_(id), vref_(velocity), type_(type), pin_model_(state->get_pinocchio()) {
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelFrameVelocityTpl<Scalar>::ResidualModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Motion& velocity,
                                                                     const pinocchio::ReferenceFrame type)
    : Base(state, 6), id_(id), vref_(velocity), type_(type), pin_model_(state->get_pinocchio()) {
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelFrameVelocityTpl<Scalar>::~ResidualModelFrameVelocityTpl() {}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>&,
                                                 const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  data->r = (pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_, type_) - vref_).toVector();
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>&,
                                                     const Eigen::Ref<const VectorXs>&) {
  // Get the partial derivatives of the local frame velocity
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio, id_, type_, data->Rx.leftCols(nv),
                                         data->Rx.rightCols(nv));
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFrameVelocityTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameVelocityTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::MotionTpl<Scalar>& ResidualModelFrameVelocityTpl<Scalar>::get_reference() const {
  return vref_;
}

template <typename Scalar>
pinocchio::ReferenceFrame ResidualModelFrameVelocityTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::set_reference(const Motion& velocity) {
  vref_ = velocity;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::set_type(const pinocchio::ReferenceFrame type) {
  type_ = type;
}

}  // namespace crocoddyl
