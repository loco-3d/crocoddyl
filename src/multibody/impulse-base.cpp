///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

ImpulseModelAbstract::ImpulseModelAbstract(StateMultibody& state, unsigned int const& ni)
    : state_(state), ni_(ni) {}

ImpulseModelAbstract::~ImpulseModelAbstract() {}

boost::shared_ptr<ImpulseDataAbstract> ImpulseModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseDataAbstract>(this, data);
}

StateMultibody& ImpulseModelAbstract::get_state() const { return state_; }

unsigned int const& ImpulseModelAbstract::get_ni() const { return ni_; }

}  // namespace crocoddyl
