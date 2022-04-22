///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/joint-torque.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelJointTorqueTpl<Scalar>::ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                                                                 boost::shared_ptr<ActuationModelAbstract> actuation,
                                                                 const VectorXs& uref, const std::size_t nu,
                                                                 const bool fwddyn)
    : Base(state, actuation->get_nu(), nu, fwddyn ? false : true, fwddyn ? false : true, true),
      uref_(uref),
      fwddyn_(fwddyn) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this residual function");
  }
}

template <typename Scalar>
ResidualModelJointTorqueTpl<Scalar>::ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                                                                 boost::shared_ptr<ActuationModelAbstract> actuation,
                                                                 const VectorXs& uref)
    : Base(state, actuation->get_nu(), state->get_nv(), true, true, true), uref_(uref), fwddyn_(false) {}

template <typename Scalar>
ResidualModelJointTorqueTpl<Scalar>::ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                                                                 boost::shared_ptr<ActuationModelAbstract> actuation,
                                                                 const std::size_t nu)
    : Base(state, actuation->get_nu(), nu, true, true, true),
      uref_(VectorXs::Zero(actuation->get_nu())),
      fwddyn_(false) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this residual function");
  }
}

template <typename Scalar>
ResidualModelJointTorqueTpl<Scalar>::ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                                                                 boost::shared_ptr<ActuationModelAbstract> actuation)
    : Base(state, actuation->get_nu(), state->get_nv(), true, true, true),
      uref_(VectorXs::Zero(actuation->get_nu())) {}

template <typename Scalar>
ResidualModelJointTorqueTpl<Scalar>::~ResidualModelJointTorqueTpl() {}

template <typename Scalar>
void ResidualModelJointTorqueTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  data->r = d->joint->tau - uref_;
}

template <typename Scalar>
void ResidualModelJointTorqueTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_) {
    data->r.setZero();
  } else {
    Data* d = static_cast<Data*>(data.get());
    data->r = d->joint->tau - uref_;
  }
}

template <typename Scalar>
void ResidualModelJointTorqueTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>&,
                                                   const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  data->Rx = d->joint->dtau_dx;
  data->Ru = d->joint->dtau_du;
}

template <typename Scalar>
void ResidualModelJointTorqueTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
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
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelJointTorqueTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  boost::shared_ptr<ResidualDataAbstract> d =
      boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  return d;
}

template <typename Scalar>
void ResidualModelJointTorqueTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelJointTorque";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ResidualModelJointTorqueTpl<Scalar>::get_reference() const {
  return uref_;
}

template <typename Scalar>
void ResidualModelJointTorqueTpl<Scalar>::set_reference(const VectorXs& reference) {
  uref_ = reference;
}

}  // namespace crocoddyl
