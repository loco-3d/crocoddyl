///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::ReferenceFrame type,
    const std::size_t nc, const std::size_t nu)
    : state_(state), nc_(nc), nu_(nu), id_(0), type_(type) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::ReferenceFrame type,
    const std::size_t nc)
    : state_(state), nc_(nc), nu_(state->get_nv()), id_(0), type_(type) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nc,
    const std::size_t nu)
    : state_(state),
      nc_(nc),
      nu_(nu),
      id_(0),
      type_(pinocchio::ReferenceFrame::LOCAL) {
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nc)
    : state_(state),
      nc_(nc),
      nu_(state->get_nv()),
      id_(0),
      type_(pinocchio::ReferenceFrame::LOCAL) {
  std::cerr << "Deprecated: Use constructor that passes the type of contact, "
               "this assumes is pinocchio::LOCAL."
            << std::endl;
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ContactDataAbstract>& data, const MatrixXs& df_dx,
    const MatrixXs& df_du) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx())
    throw_pretty("df_dx has wrong dimension");

  if (static_cast<std::size_t>(df_du.rows()) != nc_ ||
      static_cast<std::size_t>(df_du.cols()) != nu_)
    throw_pretty("df_du has wrong dimension");

  data->df_dx = df_dx;
  data->df_du = df_du;
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::setZeroForce(
    const std::shared_ptr<ContactDataAbstract>& data) const {
  data->f.setZero();
  data->fext.setZero();
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::setZeroForceDiff(
    const std::shared_ptr<ContactDataAbstract>& data) const {
  data->df_dx.setZero();
  data->df_du.setZero();
}

template <typename Scalar>
std::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<ContactDataAbstract>(
      Eigen::aligned_allocator<ContactDataAbstract>(), this, data);
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
ContactModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ContactModelAbstractTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ContactModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
pinocchio::FrameIndex ContactModelAbstractTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
pinocchio::ReferenceFrame ContactModelAbstractTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  type_ = type;
}

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ContactModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
