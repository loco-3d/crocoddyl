///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "actuation.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActuationModelTypes::Type> ActuationModelTypes::all(ActuationModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActuationModelTypes::Type type) {
  switch (type) {
    case ActuationModelTypes::ActuationModelFull:
      os << "ActuationModelFull";
      break;
    case ActuationModelTypes::ActuationModelFloatingBase:
      os << "ActuationModelFloatingBase";
      break;
    case ActuationModelTypes::NbActuationModelTypes:
      os << "NbActuationModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ActuationModelFactory::ActuationModelFactory() {}
ActuationModelFactory::~ActuationModelFactory() {}

boost::shared_ptr<crocoddyl::ActuationModelAbstract> ActuationModelFactory::create(
    ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  StateModelFactory factory;
  boost::shared_ptr<crocoddyl::StateAbstract> state = factory.create(state_type);
  boost::shared_ptr<crocoddyl::StateMultibody> state_multibody;
  switch (actuation_type) {
    case ActuationModelTypes::ActuationModelFull:
      state_multibody = boost::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation = boost::make_shared<crocoddyl::ActuationModelFull>(state_multibody);
      break;
    case ActuationModelTypes::ActuationModelFloatingBase:
      state_multibody = boost::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state_multibody);
      break;
    default:
      throw_pretty(__FILE__ ":\n Construct wrong ActuationModelTypes::Type");
      break;
  }
  return actuation;
}

}  // namespace unittest
}  // namespace crocoddyl
