///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

ContactModelAbstract::ContactModelAbstract(StateMultibody& state, unsigned int const& nc, unsigned int const& nu)
    : state_(state), nc_(nc), nu_(nu) {}

ContactModelAbstract::ContactModelAbstract(StateMultibody& state, unsigned int const& nc)
    : state_(state), nc_(nc), nu_(state.get_nv()) {}

ContactModelAbstract::~ContactModelAbstract() {}

void ContactModelAbstract::updateLagrangianDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                                const Eigen::MatrixXd& Gx, const Eigen::MatrixXd& Gu) {
  assert((Gx.rows() == nc_ || Gx.cols() == state_.get_nx()) && "Gx has wrong dimension");
  assert((Gu.rows() == nc_ || Gu.cols() == nu_) && "Gu has wrong dimension");
  data->Gx = Gx;
  data->Gu = Gu;
}

boost::shared_ptr<ContactDataAbstract> ContactModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactDataAbstract>(this, data);
}

StateMultibody& ContactModelAbstract::get_state() const { return state_; }

unsigned int const& ContactModelAbstract::get_nc() const { return nc_; }

unsigned int const& ContactModelAbstract::get_nu() const { return nu_; }

}  // namespace crocoddyl
