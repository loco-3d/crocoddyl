///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2023-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/joint-acceleration.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelJointAccelerationTpl<Scalar>::ResidualModelJointAccelerationTpl(
    std::shared_ptr<StateAbstract> state, const VectorXs& aref,
    const std::size_t nu)
    : Base(state, state->get_nv(), nu, true, true, true), aref_(aref) {
  if (static_cast<std::size_t>(aref_.size()) != state->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "aref has wrong dimension (it should be " +
                                    std::to_string(state->get_nv()) + ")");
  }
}

template <typename Scalar>
ResidualModelJointAccelerationTpl<Scalar>::ResidualModelJointAccelerationTpl(
    std::shared_ptr<StateAbstract> state, const VectorXs& aref)
    : Base(state, state->get_nv(), state->get_nv(), true, true, true),
      aref_(aref) {
  if (static_cast<std::size_t>(aref_.size()) != state->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "aref has wrong dimension (it should be " +
                                    std::to_string(state->get_nv()) + ")");
  }
}

template <typename Scalar>
ResidualModelJointAccelerationTpl<Scalar>::ResidualModelJointAccelerationTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nu)
    : Base(state, state->get_nv(), nu, true, true, true),
      aref_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
ResidualModelJointAccelerationTpl<Scalar>::ResidualModelJointAccelerationTpl(
    std::shared_ptr<StateAbstract> state)
    : Base(state, state->get_nv(), state->get_nv(), true, true, true),
      aref_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
void ResidualModelJointAccelerationTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  data->r = d->joint->a - aref_;
}

template <typename Scalar>
void ResidualModelJointAccelerationTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelJointAccelerationTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  data->Rx = d->joint->da_dx;
  data->Ru = d->joint->da_du;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelJointAccelerationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  return d;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelJointAccelerationTpl<NewScalar>
ResidualModelJointAccelerationTpl<Scalar>::cast() const {
  typedef ResidualModelJointAccelerationTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 aref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelJointAccelerationTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelJointAcceleration";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ResidualModelJointAccelerationTpl<Scalar>::get_reference() const {
  return aref_;
}

template <typename Scalar>
void ResidualModelJointAccelerationTpl<Scalar>::set_reference(
    const VectorXs& reference) {
  if (static_cast<std::size_t>(reference.size()) != nr_) {
    throw_pretty(
        "Invalid argument: "
        << "the generalized-acceleration reference has wrong dimension ("
        << reference.size()
        << " provided - it should be " + std::to_string(nr_) + ")")
  }
  aref_ = reference;
}

}  // namespace crocoddyl
