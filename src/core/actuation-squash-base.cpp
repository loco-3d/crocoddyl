///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh,
//                          Universitat Politecnica de Catalunya
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actuation-squash-base.hpp"

namespace crocoddyl {

ActuationModelSquashingAbstract::ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state,
                                                                 const std::size_t& nu)
    : ActuationModelAbstract(state, nu) {}

ActuationModelSquashingAbstract::~ActuationModelSquashingAbstract() {}

}  // namespace crocoddyl
