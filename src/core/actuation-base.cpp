///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

ActuationModelAbstract::ActuationModelAbstract(boost::shared_ptr<StateAbstract> state, const std::size_t& nu)
    : nu_(nu), state_(state) {
  if (nu_ == 0) {
    throw CrocoddylException("nu cannot be zero");
  }
}

ActuationModelAbstract::~ActuationModelAbstract() {}

boost::shared_ptr<ActuationDataAbstract> ActuationModelAbstract::createData() {
  return boost::make_shared<ActuationDataAbstract>(this);
}

const std::size_t& ActuationModelAbstract::get_nu() const { return nu_; }

const boost::shared_ptr<StateAbstract>& ActuationModelAbstract::get_state() const { return state_; }

}  // namespace crocoddyl
