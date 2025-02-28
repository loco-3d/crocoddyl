///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

template <typename Scalar>
ImpulseModelAbstractTpl<Scalar>::ImpulseModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::ReferenceFrame type,
    const std::size_t nc)
    : state_(state), nc_(nc), id_(0), type_(type) {}

template <typename Scalar>
ImpulseModelAbstractTpl<Scalar>::ImpulseModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nc)
    : state_(state), nc_(nc), id_(0), type_(pinocchio::ReferenceFrame::LOCAL) {
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ImpulseDataAbstract>& data,
    const MatrixXs& df_dx) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx())
    throw_pretty("df_dq has wrong dimension");

  data->df_dx = df_dx;
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::setZeroForce(
    const std::shared_ptr<ImpulseDataAbstract>& data) const {
  data->f.setZero();
  data->fext.setZero();
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::setZeroForceDiff(
    const std::shared_ptr<ImpulseDataAbstract>& data) const {
  data->df_dx.setZero();
}

template <typename Scalar>
std::shared_ptr<ImpulseDataAbstractTpl<Scalar> >
ImpulseModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<ImpulseDataAbstract>(
      Eigen::aligned_allocator<ImpulseDataAbstract>(), this, data);
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
ImpulseModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ImpulseModelAbstractTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImpulseModelAbstractTpl<Scalar>::get_ni() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImpulseModelAbstractTpl<Scalar>::get_nu() const {
  return 0;
}

template <typename Scalar>
pinocchio::FrameIndex ImpulseModelAbstractTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
pinocchio::ReferenceFrame ImpulseModelAbstractTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  type_ = type;
}

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ImpulseModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
