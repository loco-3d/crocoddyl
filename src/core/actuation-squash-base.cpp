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

ActuationModelSquashingAbstract::ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state, boost::shared_ptr<SquashingModelAbstract> squashing, const std::size_t& nu)
    : ActuationModelAbstract(state, nu), squashing_(squashing) {}

ActuationModelSquashingAbstract::~ActuationModelSquashingAbstract() {}

boost::shared_ptr<ActuationDataAbstract> ActuationModelSquashingAbstract::createData() {
  return boost::make_shared<ActuationDataSquashing>(this);
}

const boost::shared_ptr<SquashingModelAbstract>& ActuationModelSquashingAbstract::get_squashing() const {return squashing_; }

}  // namespace crocoddyl
