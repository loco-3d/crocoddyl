///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/states/unicycle.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "example-robot-data/path.hpp"

#ifndef CROCODDYL_STATE_FACTORY_HPP_
#define CROCODDYL_STATE_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct StateTypes {
  enum Type {
    StateVector,
    StateMultibodyTalosArm,
    StateMultibodyHyQ,
    StateMultibodyTalos,
    StateMultibodyRandomHumanoid,
    NbStateTypes
  };
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

class StateFactory {
 public:
  StateFactory(StateTypes::Type type) {
    // default initialization
    nx_ = 0;
    num_diff_modifier_ = 1e4;
    state_type_ = type;

    switch (state_type_) {
      case StateTypes::StateVector:
        nx_ = 80;
        state_ = boost::make_shared<crocoddyl::StateVector>(nx_);
        break;
      case StateTypes::StateMultibodyTalosArm:
        construct_state_multibody(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", false);
        break;
      case StateTypes::StateMultibodyHyQ:
        construct_state_multibody(EXAMPLE_ROBOT_DATA_MODEL_DIR "/hyq_description/robots/hyq_no_sensors.urdf");
        break;
      case StateTypes::StateMultibodyTalos:
        construct_state_multibody(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/urdf/talos_reduced.urdf");
        break;
      case StateTypes::StateMultibodyRandomHumanoid:
        construct_state_multibody();
        break;
      default:
        throw_pretty(__FILE__ ": Wrong StateTypes::Type given");
        break;
    }
  }

  ~StateFactory() {}

  void construct_state_multibody(const std::string& urdf_file = "", bool free_flyer = true) {
    if (urdf_file.size() != 0) {
      if (free_flyer) {
        pinocchio::urdf::buildModel(urdf_file, free_flyer_joint_, pinocchio_model_);
        pinocchio_model_.lowerPositionLimit.head<3>().fill(-1.0);
        pinocchio_model_.upperPositionLimit.head<3>().fill(1.0);
      } else {
        pinocchio::urdf::buildModel(urdf_file, pinocchio_model_);
      }
    } else {
      pinocchio::buildModels::humanoidRandom(pinocchio_model_, free_flyer);
    }
    state_ = boost::make_shared<crocoddyl::StateMultibody>(boost::ref(pinocchio_model_));
    nx_ = pinocchio_model_.nq + pinocchio_model_.nv;
  }

  boost::shared_ptr<crocoddyl::StateAbstract> get_state() { return state_; }
  const std::size_t& get_nx() { return nx_; }
  double get_num_diff_modifier() { return num_diff_modifier_; }
  const pinocchio::Model& get_pinocchio_model() { return pinocchio_model_; }

 private:
  StateTypes::Type state_type_;                        //!< The current type to test
  boost::shared_ptr<crocoddyl::StateAbstract> state_;  //!< The pointer to the state in testing
  std::size_t nx_;                                     //!< The size of the StateVector to test.
  double num_diff_modifier_;                           //!< Multiplier of the precision during the tests.
  pinocchio::JointModelFreeFlyer free_flyer_joint_;    //!< The free flyer joint to build the pinocchio model.
  pinocchio::Model pinocchio_model_;                   //!< The pinocchio_model to build the StateMultibody.
};

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_STATE_FACTORY_HPP_
