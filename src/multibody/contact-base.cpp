///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

ContactModelAbstract::ContactModelAbstract(StateMultibody& state, const std::size_t& nc, const std::size_t& nu)
    : state_(state), nc_(nc), nu_(nu) {}

ContactModelAbstract::ContactModelAbstract(StateMultibody& state, const std::size_t& nc)
    : state_(state), nc_(nc), nu_(state.get_nv()) {}

ContactModelAbstract::~ContactModelAbstract() {}

void ContactModelAbstract::updateForceDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                           const Eigen::MatrixXd& df_dx, const Eigen::MatrixXd& df_du) const {
  assert((df_dx.rows() == nc_ || df_dx.cols() == state_.get_nx()) && "df_dx has wrong dimension");
  assert((df_du.rows() == nc_ || df_du.cols() == nu_) && "df_du has wrong dimension");
  data->df_dx = df_dx;
  data->df_du = df_du;
}

boost::shared_ptr<ContactDataAbstract> ContactModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactDataAbstract>(this, data);
}

StateMultibody& ContactModelAbstract::get_state() const { return state_; }

const std::size_t& ContactModelAbstract::get_nc() const { return nc_; }

const std::size_t& ContactModelAbstract::get_nu() const { return nu_; }

}  // namespace crocoddyl
