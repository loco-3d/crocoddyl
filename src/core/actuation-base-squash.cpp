///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actuation-base-squash.hpp"

namespace crocoddyl {

ActuationModelSquashingAbstract::ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state,
                                                                 const std::size_t& nu)
    : ActuationModelAbstract(state, nu) {}

ActuationModelSquashingAbstract::~ActuationModelSquashingAbstract() {}

}  // namespace crocoddyl
