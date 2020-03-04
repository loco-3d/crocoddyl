///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <example-robot-data/path.hpp>

#ifndef CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_
#define CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct PinocchioModelTypes {
  enum Type {
    TalosArm,
    HyQ,
    Talos,
    RandomHumanoid,
    NbPinocchioModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbPinocchioModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<PinocchioModelTypes::Type> PinocchioModelTypes::all(PinocchioModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, PinocchioModelTypes::Type type) {
  switch (type) {
    case PinocchioModelTypes::TalosArm:
      os << "TalosArm";
      break;
    case PinocchioModelTypes::HyQ:
      os << "HyQ";
      break;
    case PinocchioModelTypes::Talos:
      os << "Talos";
      break;
    case PinocchioModelTypes::RandomHumanoid:
      os << "RandomHumanoid";
      break;
    case PinocchioModelTypes::NbPinocchioModelTypes:
      os << "NbPinocchioModelTypes";
      break;
    default:
      break;
  }
  return os;
}

class PinocchioModelFactory {
 public:
  PinocchioModelFactory(PinocchioModelTypes::Type type) {
    switch (type) {
      case PinocchioModelTypes::TalosArm:
        construct_model(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", false);
        break;
      case PinocchioModelTypes::HyQ:
        construct_model(EXAMPLE_ROBOT_DATA_MODEL_DIR "/hyq_description/robots/hyq_no_sensors.urdf", false);
        break;
      case PinocchioModelTypes::Talos:
        construct_model(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf");
        break;
      case PinocchioModelTypes::RandomHumanoid:
        construct_model();
        break;
      case PinocchioModelTypes::NbPinocchioModelTypes:
        break;
      default:
        throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
        break;
    }
  }

  ~PinocchioModelFactory() {}

  void construct_model(const std::string& urdf_file = "", bool free_flyer = true) {
    pinocchio::Model pinocchio_model;
    if (urdf_file.size() != 0) {
      if (free_flyer) {
        pinocchio::urdf::buildModel(urdf_file, pinocchio::JointModelFreeFlyer(), pinocchio_model);
        pinocchio_model.lowerPositionLimit.segment<7>(0).fill(-1.);
        pinocchio_model.upperPositionLimit.segment<7>(0).fill(1.);
      } else {
        pinocchio::urdf::buildModel(urdf_file, pinocchio_model);
      }
    } else {
      pinocchio::buildModels::humanoidRandom(pinocchio_model, free_flyer);
      pinocchio_model.lowerPositionLimit.segment<7>(0).fill(-1.);
      pinocchio_model.upperPositionLimit.segment<7>(0).fill(1.);
    }
    model_ = boost::make_shared<pinocchio::Model>(pinocchio_model);
  }

  boost::shared_ptr<pinocchio::Model> create() { return model_; }

 private:
  boost::shared_ptr<pinocchio::Model> model_;  //!< The pointer to the state in testing
};


}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_
