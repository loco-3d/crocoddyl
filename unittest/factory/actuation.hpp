///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "state.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"

#ifndef CROCODDYL_ACTUATION_FACTORY_HPP_
#define CROCODDYL_ACTUATION_FACTORY_HPP_

namespace crocoddyl {
namespace unittest {

struct ActuationModelTypes {
  enum Type { ActuationModelFull, ActuationModelFloatingBase, NbActuationModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbActuationModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
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

class ActuationModelFactory {
 public:
  explicit ActuationModelFactory() {}
  ~ActuationModelFactory() {}

  boost::shared_ptr<crocoddyl::ActuationModelAbstract> create(ActuationModelTypes::Type actuation_type,
                                                              StateModelTypes::Type state_type) {
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
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTUATION_FACTORY_HPP_
