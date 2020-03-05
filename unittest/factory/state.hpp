///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <example-robot-data/path.hpp>

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include "pinocchio_model.hpp"

#ifndef CROCODDYL_STATE_FACTORY_HPP_
#define CROCODDYL_STATE_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct StateTypes {
  enum Type { StateVector, StateMultibody, NbStateTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbStateTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<StateTypes::Type> StateTypes::all(StateTypes::init_all());

std::ostream& operator<<(std::ostream& os, StateTypes::Type type) {
  switch (type) {
    case StateTypes::StateVector:
      os << "StateVector";
      break;
    case StateTypes::StateMultibody:
      os << "StateMultibody";
      break;
      os << "NbStateTypes";
      break;
    default:
      break;
  }
  return os;
}

class StateFactory {
 public:
  StateFactory(StateTypes::Type state_type,
               PinocchioModelTypes::Type model_type = PinocchioModelTypes::NbPinocchioModelTypes) {
    nx_ = 0;
    num_diff_modifier_ = 1e4;
    PinocchioModelFactory factory(model_type);
    boost::shared_ptr<pinocchio::Model> model;

    switch (state_type) {
      case StateTypes::StateVector:
        nx_ = 80;
        state_ = boost::make_shared<crocoddyl::StateVector>(nx_);
        break;
      case StateTypes::StateMultibody:
        model = factory.create();
        nx_ = model->nq + model->nv;
        state_ = boost::make_shared<crocoddyl::StateMultibody>(model);
        break;
      default:
        throw_pretty(__FILE__ ": Wrong StateTypes::Type given");
        break;
    }
  }

  ~StateFactory() {}

  boost::shared_ptr<crocoddyl::StateAbstract> create() { return state_; }
  const std::size_t& get_nx() { return nx_; }
  double get_num_diff_modifier() { return num_diff_modifier_; }

 private:
  boost::shared_ptr<crocoddyl::StateAbstract> state_;  //!< The pointer to the state in testing
  std::size_t nx_;                                     //!< The size of the StateVector to test.
  double num_diff_modifier_;                           //!< Multiplier of the precision during the tests.
};

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_STATE_FACTORY_HPP_
