///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    boost::shared_ptr<StateMultibody> state,
    const pinocchio::ReferenceFrame type, const std::size_t nf,
    const std::size_t nc, const std::size_t nu)
    : state_(state),
      nc_(nc),
      nu_(nu),
      nf_(nf),
      id_(nf, 0),
      placements_(nf, pinocchio::SE3::Identity()),
      type_(type) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    boost::shared_ptr<StateMultibody> state,
    const pinocchio::ReferenceFrame type, const std::size_t nf,
    const std::size_t nc)
    : state_(state),
      nc_(nc),
      nu_(state->get_nv()),
      nf_(nf),
      id_(nf, 0),
      placements_(nf, pinocchio::SE3::Identity()),
      type_(type) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(
    boost::shared_ptr<StateMultibody> state, const std::size_t nc,
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
    boost::shared_ptr<StateMultibody> state, const std::size_t nc)
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
ContactModelAbstractTpl<Scalar>::~ContactModelAbstractTpl() {}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::updateForceDiff(
    const boost::shared_ptr<ContactDataAbstract>& data, const MatrixXs& df_dx,
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
    const boost::shared_ptr<ContactDataAbstract>& data) const {
  for (size_t i = 0; i < nf_; ++i) {
    data->force_datas[i].f.setZero();
    data->force_datas[i].fext.setZero();
  }
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::setZeroForceDiff(
    const boost::shared_ptr<ContactDataAbstract>& data) const {
  data->df_dx.setZero();
  data->df_du.setZero();
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<ContactDataAbstract>(
      Eigen::aligned_allocator<ContactDataAbstract>(), this, data, nf_);
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >&
ContactModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ContactModelAbstractTpl<Scalar>::get_nf() const {
  return nf_;
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
pinocchio::FrameIndex ContactModelAbstractTpl<Scalar>::get_id(
    const int force_index) const {
  return id_[force_index];
}

template <typename Scalar>
pinocchio::ReferenceFrame ContactModelAbstractTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::set_id(const int force_index,
                                             const pinocchio::FrameIndex id) {
  id_[force_index] = id;
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
