///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2025, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/joint-effort.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelJointEffortTpl<Scalar>::ResidualModelJointEffortTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActuationModelAbstract> actuation, const VectorXs& uref,
    const std::size_t nu, const bool fwddyn)
    : Base(state, actuation->get_nu(), nu, fwddyn ? false : true,
           fwddyn ? false : true, true),
      actuation_(actuation),
      uref_(uref),
      fwddyn_(fwddyn) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this residual function");
  }
}

template <typename Scalar>
ResidualModelJointEffortTpl<Scalar>::ResidualModelJointEffortTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActuationModelAbstract> actuation, const VectorXs& uref)
    : Base(state, actuation->get_nu(), state->get_nv(), true, true, true),
      actuation_(actuation),
      uref_(uref),
      fwddyn_(false) {}

template <typename Scalar>
ResidualModelJointEffortTpl<Scalar>::ResidualModelJointEffortTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActuationModelAbstract> actuation, const std::size_t nu)
    : Base(state, actuation->get_nu(), nu, true, true, true),
      actuation_(actuation),
      uref_(VectorXs::Zero(actuation->get_nu())),
      fwddyn_(false) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this residual function");
  }
}

template <typename Scalar>
ResidualModelJointEffortTpl<Scalar>::ResidualModelJointEffortTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ActuationModelAbstract> actuation)
    : Base(state, actuation->get_nu(), state->get_nv(), true, true, true),
      actuation_(actuation),
      uref_(VectorXs::Zero(actuation->get_nu())) {}

template <typename Scalar>
void ResidualModelJointEffortTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  data->r = d->joint->tau - uref_;
}

template <typename Scalar>
void ResidualModelJointEffortTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_) {
    data->r.setZero();
  } else {
    Data* d = static_cast<Data*>(data.get());
    data->r = d->joint->tau - uref_;
  }
}

template <typename Scalar>
void ResidualModelJointEffortTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  if (q_dependent_ || v_dependent_) {
    data->Rx = d->joint->dtau_dx;
  }
  data->Ru = d->joint->dtau_du;
}

template <typename Scalar>
void ResidualModelJointEffortTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_) {
    data->Rx.setZero();
  } else {
    Data* d = static_cast<Data*>(data.get());
    data->Rx = d->joint->dtau_dx;
    data->Ru = d->joint->dtau_du;
  }
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelJointEffortTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  return d;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelJointEffortTpl<NewScalar>
ResidualModelJointEffortTpl<Scalar>::cast() const {
  typedef ResidualModelJointEffortTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 actuation_->template cast<NewScalar>(),
                 uref_.template cast<NewScalar>(), nu_, fwddyn_);
  return ret;
}

template <typename Scalar>
void ResidualModelJointEffortTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelJointEffort";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ResidualModelJointEffortTpl<Scalar>::get_reference() const {
  return uref_;
}

template <typename Scalar>
void ResidualModelJointEffortTpl<Scalar>::set_reference(
    const VectorXs& reference) {
  if (static_cast<std::size_t>(reference.size()) != nr_) {
    throw_pretty("Invalid argument: "
                 << "the joint-effort reference has wrong dimension ("
                 << reference.size()
                 << " provided - it should be " + std::to_string(nr_) + ")")
  }
  uref_ = reference;
}

}  // namespace crocoddyl
