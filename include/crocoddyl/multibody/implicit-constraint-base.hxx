///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ImplicitConstraintModelAbstractTpl<Scalar>::ImplicitConstraintModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::ReferenceFrame type,
    const std::size_t nc, const std::size_t nu)
    : state_(state), nc_(nc), nu_(nu), id_(0), type_(type) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state cannot be null");
  }
}

template <typename Scalar>
ImplicitConstraintModelAbstractTpl<Scalar>::ImplicitConstraintModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::ReferenceFrame type,
    const std::size_t nc)
    : state_(state),
      nc_(nc),
      nu_(state != nullptr ? state->get_nv() : 0),
      id_(0),
      type_(type) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state cannot be null");
  }
}

template <typename Scalar>
void ImplicitConstraintModelAbstractTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
    const Eigen::Ref<const MatrixXs>& df_dx,
    const Eigen::Ref<const MatrixXs>& df_du) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx()) {
    throw_pretty("df_dx has wrong dimension");
  }
  if (static_cast<std::size_t>(df_du.rows()) != nc_ ||
      static_cast<std::size_t>(df_du.cols()) != nu_) {
    throw_pretty("df_du has wrong dimension");
  }

  data->df_dx = df_dx;
  data->df_du = df_du;
}

template <typename Scalar>
void ImplicitConstraintModelAbstractTpl<Scalar>::setZeroForce(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data) const {
  data->f.setZero();
  data->fext.setZero();
}

template <typename Scalar>
void ImplicitConstraintModelAbstractTpl<Scalar>::setZeroForceDiff(
    const std::shared_ptr<ImplicitConstraintDataAbstract>& data) const {
  data->df_dx.setZero();
  data->df_du.setZero();
}

template <typename Scalar>
std::shared_ptr<ImplicitConstraintDataAbstractTpl<Scalar> >
ImplicitConstraintModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  if (data == nullptr) {
    throw_pretty("Invalid argument: Pinocchio data cannot be null");
  }
  return std::allocate_shared<ImplicitConstraintDataAbstract>(
      Eigen::aligned_allocator<ImplicitConstraintDataAbstract>(), this, data);
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
ImplicitConstraintModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ImplicitConstraintModelAbstractTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImplicitConstraintModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
pinocchio::FrameIndex ImplicitConstraintModelAbstractTpl<Scalar>::get_id()
    const {
  return id_;
}

template <typename Scalar>
void ImplicitConstraintModelAbstractTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ImplicitConstraintModelAbstractTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  type_ = type;
}

template <typename Scalar>
pinocchio::ReferenceFrame ImplicitConstraintModelAbstractTpl<Scalar>::get_type()
    const {
  return type_;
}

template <typename Scalar>
void ImplicitConstraintModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <class Scalar>
std::ostream& operator<<(
    std::ostream& os, const ImplicitConstraintModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
