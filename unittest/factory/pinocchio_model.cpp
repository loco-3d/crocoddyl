///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "pinocchio_model.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<PinocchioModelTypes::Type> PinocchioModelTypes::all(
    PinocchioModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, PinocchioModelTypes::Type type) {
  switch (type) {
    case PinocchioModelTypes::Hector:
      os << "Hector";
      break;
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

PinocchioModelFactory::PinocchioModelFactory(PinocchioModelTypes::Type type) {
  frame_name_.clear();
  frame_id_.clear();
  switch (type) {
    case PinocchioModelTypes::Hector:
      construct_model(EXAMPLE_ROBOT_DATA_MODEL_DIR
                      "/hector_description/robots/quadrotor_base.urdf");
      frame_name_.resize(1);
      frame_id_.resize(1);
      frame_name_[0] = "base_link";
      frame_id_[0] = model_->getFrameId(frame_name_[0]);
      contact_nc_ = 6;
      break;
    case PinocchioModelTypes::TalosArm:
      construct_model(
          EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf",
          EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf", false);
      frame_name_.resize(1);
      frame_id_.resize(1);
      frame_name_[0] = "gripper_left_fingertip_1_link";
      frame_id_[0] = model_->getFrameId(frame_name_[0]);
      contact_nc_ = 6;
      break;
    case PinocchioModelTypes::HyQ:
      construct_model(EXAMPLE_ROBOT_DATA_MODEL_DIR
                      "/hyq_description/robots/hyq_no_sensors.urdf",
                      EXAMPLE_ROBOT_DATA_MODEL_DIR
                      "/hyq_description/srdf/hyq.srdf");
      frame_name_.resize(4);
      frame_id_.resize(4);
      frame_name_[0] = "lf_foot";
      frame_name_[1] = "rf_foot";
      frame_name_[2] = "lh_foot";
      frame_name_[3] = "rh_foot";
      for (std::size_t i = 0; i < frame_name_.size(); ++i) {
        frame_id_[i] = model_->getFrameId(frame_name_[i]);
      }
      contact_nc_ = 3;
      break;
    case PinocchioModelTypes::Talos:
      construct_model(
          EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf",
          EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf");
      frame_name_.resize(2);
      frame_id_.resize(2);
      frame_name_[0] = "left_sole_link";
      frame_name_[1] = "right_sole_link";
      for (std::size_t i = 0; i < frame_name_.size(); ++i) {
        frame_id_[i] = model_->getFrameId(frame_name_[i]);
      }
      contact_nc_ = 6;
      break;
    case PinocchioModelTypes::RandomHumanoid:
      construct_model();
      frame_name_.resize(2);
      frame_id_.resize(2);
      frame_name_[0] = "rleg6_body";
      frame_name_[1] = "lleg6_body";
      for (std::size_t i = 0; i < frame_name_.size(); ++i) {
        frame_id_[i] = model_->getFrameId(frame_name_[i]);
      }
      contact_nc_ = 6;
      break;
    case PinocchioModelTypes::NbPinocchioModelTypes:
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }
}

PinocchioModelFactory::~PinocchioModelFactory() {}

void PinocchioModelFactory::construct_model(const std::string& urdf_file,
                                            const std::string& srdf_file,
                                            bool free_flyer) {
  model_ = std::make_shared<pinocchio::Model>();
  if (!urdf_file.empty()) {
    if (free_flyer) {
      pinocchio::urdf::buildModel(urdf_file, pinocchio::JointModelFreeFlyer(),
                                  *model_.get());
      model_->lowerPositionLimit.segment<7>(0).fill(-1.);
      model_->upperPositionLimit.segment<7>(0).fill(1.);
      if (!srdf_file.empty()) {
        pinocchio::srdf::loadReferenceConfigurations(*model_.get(), srdf_file,
                                                     false);
      }
    } else {
      pinocchio::urdf::buildModel(urdf_file, *model_.get());
      if (!srdf_file.empty()) {
        pinocchio::srdf::loadReferenceConfigurations(*model_.get(), srdf_file,
                                                     false);
      }
    }
  } else {
    pinocchio::buildModels::humanoidRandom(*model_.get(), free_flyer);
    model_->lowerPositionLimit.segment<7>(0).fill(-1.);
    model_->upperPositionLimit.segment<7>(0).fill(1.);
  }
}

std::shared_ptr<pinocchio::Model> PinocchioModelFactory::create() const {
  return model_;
}
std::vector<std::string> PinocchioModelFactory::get_frame_names() const {
  return frame_name_;
}
std::vector<std::size_t> PinocchioModelFactory::get_frame_ids() const {
  return frame_id_;
}
std::size_t PinocchioModelFactory::get_contact_nc() const {
  return contact_nc_;
}

}  // namespace unittest
}  // namespace crocoddyl
