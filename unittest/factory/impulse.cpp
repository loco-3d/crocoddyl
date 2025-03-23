///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "impulse.hpp"

#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ImpulseModelTypes::Type> ImpulseModelTypes::all(
    ImpulseModelTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         const ImpulseModelTypes::Type& type) {
  switch (type) {
    case ImpulseModelTypes::ImpulseModel3D_LOCAL:
      os << "ImpulseModel3D_LOCAL";
      break;
    case ImpulseModelTypes::ImpulseModel3D_WORLD:
      os << "ImpulseModel3D_WORLD";
      break;
    case ImpulseModelTypes::ImpulseModel3D_LWA:
      os << "ImpulseModel3D_LWA";
      break;
    case ImpulseModelTypes::ImpulseModel6D_LOCAL:
      os << "ImpulseModel6D_LOCAL";
      break;
    case ImpulseModelTypes::ImpulseModel6D_WORLD:
      os << "ImpulseModel6D_WORLD";
      break;
    case ImpulseModelTypes::ImpulseModel6D_LWA:
      os << "ImpulseModel6D_LWA";
      break;
    case ImpulseModelTypes::NbImpulseModelTypes:
      os << "NbImpulseModelTypes";
      break;
    default:
      os << "Unknown type";
      break;
  }
  return os;
}

ImpulseModelFactory::ImpulseModelFactory() {}
ImpulseModelFactory::~ImpulseModelFactory() {}

std::shared_ptr<crocoddyl::ImpulseModelAbstract> ImpulseModelFactory::create(
    ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type,
    const std::string frame_name) const {
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> impulse;
  PinocchioModelFactory model_factory(model_type);
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::make_shared<crocoddyl::StateMultibody>(model_factory.create());
  std::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  std::size_t frame_id = 0;
  if (frame_name == "") {
    frame_id = model_factory.get_frame_ids()[0];
  } else {
    frame_id = state->get_pinocchio()->getFrameId(frame_name);
  }
  switch (impulse_type) {
    case ImpulseModelTypes::ImpulseModel3D_LOCAL:
      impulse = std::make_shared<crocoddyl::ImpulseModel3D>(
          state, frame_id, pinocchio::ReferenceFrame::LOCAL);
      break;
    case ImpulseModelTypes::ImpulseModel3D_WORLD:
      impulse = std::make_shared<crocoddyl::ImpulseModel3D>(
          state, frame_id, pinocchio::ReferenceFrame::WORLD);
      break;
    case ImpulseModelTypes::ImpulseModel3D_LWA:
      impulse = std::make_shared<crocoddyl::ImpulseModel3D>(
          state, frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      break;
    case ImpulseModelTypes::ImpulseModel6D_LOCAL:
      impulse = std::make_shared<crocoddyl::ImpulseModel6D>(
          state, frame_id, pinocchio::ReferenceFrame::LOCAL);
      break;
    case ImpulseModelTypes::ImpulseModel6D_WORLD:
      impulse = std::make_shared<crocoddyl::ImpulseModel6D>(
          state, frame_id, pinocchio::ReferenceFrame::WORLD);
      break;
    case ImpulseModelTypes::ImpulseModel6D_LWA:
      impulse = std::make_shared<crocoddyl::ImpulseModel6D>(
          state, frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ImpulseModelTypes::Type given");
      break;
  }
  return impulse;
}

std::shared_ptr<crocoddyl::ImpulseModelAbstract> create_random_impulse() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> impulse;
  ImpulseModelFactory factory;
  if (rand() % 2 == 0) {
    impulse = factory.create(ImpulseModelTypes::ImpulseModel3D_LOCAL,
                             PinocchioModelTypes::RandomHumanoid);
  } else {
    impulse = factory.create(ImpulseModelTypes::ImpulseModel6D_LOCAL,
                             PinocchioModelTypes::RandomHumanoid);
  }
  return impulse;
}

}  // namespace unittest
}  // namespace crocoddyl
