///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nc,
                                                         const std::size_t nu)
    : state_(state), nc_(nc), nu_(nu) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nc)
    : state_(state), nc_(nc), nu_(state->get_nv()) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::~ContactModelAbstractTpl() {}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::updateForceDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                                      const MatrixXs& df_dx, const MatrixXs& df_du) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ || static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx())
    throw_pretty("df_dx has wrong dimension");

  if (static_cast<std::size_t>(df_du.rows()) != nc_ || static_cast<std::size_t>(df_du.cols()) != nu_)
    throw_pretty("df_du has wrong dimension");

  data->df_dx = df_dx;
  data->df_du = df_du;
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::setZeroForce(const boost::shared_ptr<ContactDataAbstract>& data) const {
  data->f.setZero();
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::setZeroForceDiff(const boost::shared_ptr<ContactDataAbstract>& data) const {
  data->df_dx.setZero();
  data->df_du.setZero();
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<ContactDataAbstract>(Eigen::aligned_allocator<ContactDataAbstract>(), this, data);
}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& ContactModelAbstractTpl<Scalar>::get_state() const {
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

template <class Scalar>
std::ostream& operator<<(std::ostream& os, const ContactModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
