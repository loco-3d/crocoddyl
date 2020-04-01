///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ACTUATION_FACTORY_HPP_
#define CROCODDYL_ACTUATION_FACTORY_HPP_

#include "state.hpp"
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {
namespace unittest {

struct ActuationModelTypes {
  enum Type {
    ActuationModelFull,
    ActuationModelFloatingBase,
    ActuationModelMultiCopterBase,
    ActuationModelSquashingFull,
    NbActuationModelTypes
  };
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

std::ostream& operator<<(std::ostream& os, ActuationModelTypes::Type type);

class ActuationModelFactory {
 public:
  explicit ActuationModelFactory();
  ~ActuationModelFactory();

  boost::shared_ptr<crocoddyl::ActuationModelAbstract> create(ActuationModelTypes::Type actuation_type,
                                                              StateModelTypes::Type state_type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTUATION_FACTORY_HPP_
