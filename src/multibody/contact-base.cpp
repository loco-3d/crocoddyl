///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

ContactModelAbstract::ContactModelAbstract(StateMultibody& state, const unsigned int& nc) : state_(state), nc_(nc) {}

ContactModelAbstract::~ContactModelAbstract() {}

boost::shared_ptr<ContactDataAbstract> ContactModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactDataAbstract>(this, data);
}

StateMultibody& ContactModelAbstract::get_state() const { return state_; }

const unsigned int& ContactModelAbstract::get_nc() const { return nc_; }

}  // namespace crocoddyl
