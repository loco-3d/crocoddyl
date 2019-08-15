///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

ActuationModelAbstract::ActuationModelAbstract(StateAbstract& state, unsigned int const& nu) : nu_(nu), state_(state) {
  assert(nu_ != 0 && "ActuationModelAbstract: nu cannot be zero");
}

ActuationModelAbstract::~ActuationModelAbstract() {}

boost::shared_ptr<ActuationDataAbstract> ActuationModelAbstract::createData() {
  return boost::make_shared<ActuationDataAbstract>(this);
}

const unsigned int& ActuationModelAbstract::get_nu() const { return nu_; }

StateAbstract& ActuationModelAbstract::get_state() const { return state_; }

}  // namespace crocoddyl
