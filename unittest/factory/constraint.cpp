///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "constraint.hpp"
#include "crocoddyl/multibody/constraints/frame-placement-equality.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ConstraintModelTypes::Type> ConstraintModelTypes::all(ConstraintModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ConstraintModelTypes::Type type) {
  switch (type) {
    case ConstraintModelTypes::ConstraintModelFramePlacementEquality:
      os << "ConstraintModelFramePlacementEquality";
      break;
    case ConstraintModelTypes::NbConstraintModelTypes:
      os << "NbConstraintModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ConstraintModelFactory::ConstraintModelFactory() {}
ConstraintModelFactory::~ConstraintModelFactory() {}

boost::shared_ptr<crocoddyl::ConstraintModelAbstract> ConstraintModelFactory::create(
    ConstraintModelTypes::Type constraint_type, StateModelTypes::Type state_type, std::size_t nu) const {
  StateModelFactory state_factory;
  boost::shared_ptr<crocoddyl::ConstraintModelAbstract> constraint;
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(state_type));
  crocoddyl::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (constraint_type) {
    case ConstraintModelTypes::ConstraintModelFramePlacementEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelFramePlacementEquality>(
          state, crocoddyl::FramePlacement(frame_index, frame_SE3), nu);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ConstraintModelType::Type given");
      break;
  }
  return constraint;
}

boost::shared_ptr<crocoddyl::ConstraintModelAbstract> create_random_constraint(StateModelTypes::Type state_type) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  ConstraintModelFactory factory;
  ConstraintModelTypes::Type rand_type =
      static_cast<ConstraintModelTypes::Type>(rand() % ConstraintModelTypes::NbConstraintModelTypes);
  return factory.create(rand_type, state_type);
}

}  // namespace unittest
}  // namespace crocoddyl
