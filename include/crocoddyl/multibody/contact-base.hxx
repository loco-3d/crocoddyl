///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state,
                                                         const std::size_t& nc, const std::size_t& nu)
    : state_(state), nc_(nc), nu_(nu) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state,
                                                         const std::size_t& nc)
    : state_(state), nc_(nc), nu_(state->get_nv()) {}

template <typename Scalar>
ContactModelAbstractTpl<Scalar>::~ContactModelAbstractTpl() {}

template <typename Scalar>
void ContactModelAbstractTpl<Scalar>::updateForceDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                                      const MatrixXs& df_dx, const MatrixXs& df_du) const {
  assert_pretty(
      (static_cast<std::size_t>(df_dx.rows()) == nc_ || static_cast<std::size_t>(df_dx.cols()) == state_->get_nx()),
      "df_dx has wrong dimension");
  assert_pretty((static_cast<std::size_t>(df_du.rows()) == nc_ || static_cast<std::size_t>(df_du.cols()) == nu_),
                "df_du has wrong dimension");
  data->df_dx = df_dx;
  data->df_du = df_du;
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::make_shared<ContactDataAbstract>(this, data);
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& ContactModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const std::size_t& ContactModelAbstractTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
const std::size_t& ContactModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

}  // namespace crocoddyl
