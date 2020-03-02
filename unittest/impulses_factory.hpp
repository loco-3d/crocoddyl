///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "state_factory.hpp"

#ifndef CROCODDYL_IMPULSES_FACTORY_HPP_
#define CROCODDYL_IMPULSES_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct ImpulseModelTypes {
  enum Type {
    ImpulseModel3DTalosArm,
    ImpulseModel3DHyQ,
    ImpulseModel3DTalos,
    ImpulseModel3DRandomHumanoid,
    ImpulseModel6DTalosArm,
    ImpulseModel6DHyQ,
    ImpulseModel6DTalos,
    ImpulseModel6DRandomHumanoid,
    NbImpulseModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbImpulseModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<ImpulseModelTypes::Type> ImpulseModelTypes::all(ImpulseModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, const ImpulseModelTypes::Type& type) {
  switch (type) {
    case ImpulseModelTypes::ImpulseModel3DTalosArm:
      os << "ImpulseModel3DTalosArm";
      break;
    case ImpulseModelTypes::ImpulseModel3DHyQ:
      os << "ImpulseModel3DHyQ";
      break;
    case ImpulseModelTypes::ImpulseModel3DTalos:
      os << "ImpulseModel3DTalos";
      break;
    case ImpulseModelTypes::ImpulseModel3DRandomHumanoid:
      os << "ImpulseModel3DRandomHumanoid";
      break;
    case ImpulseModelTypes::ImpulseModel6DTalosArm:
      os << "ImpulseModel6DTalosArm";
      break;
    case ImpulseModelTypes::ImpulseModel6DHyQ:
      os << "ImpulseModel6DHyQ";
      break;
    case ImpulseModelTypes::ImpulseModel6DTalos:
      os << "ImpulseModel6DTalos";
      break;
    case ImpulseModelTypes::ImpulseModel6DRandomHumanoid:
      os << "ImpulseModel6DRandomHumanoid";
      break;
    default:
      os << "Unkown type";
      break;
  }
  return os;
}

class ImpulseModelFactory {
 public:
  ImpulseModelFactory(ImpulseModelTypes::Type type) {
    test_type_ = type;

    size_t frame = 0;
    boost::shared_ptr<crocoddyl::StateMultibody> state;

    switch (test_type_) {
      case ImpulseModelTypes::ImpulseModel3DTalosArm:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalosArm);
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_fingertip_1_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel3DHyQ:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyHyQ);
        frame = state_factory_->get_pinocchio_model().getFrameId("lf_foot");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel3DTalos:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalos);
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_fingertip_1_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel3DRandomHumanoid:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyRandomHumanoid);
        frame = state_factory_->get_pinocchio_model().getFrameId("rleg6_body");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;

      case ImpulseModelTypes::ImpulseModel6DTalosArm:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalosArm);
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_fingertip_1_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel6DHyQ:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyHyQ);
        frame = state_factory_->get_pinocchio_model().getFrameId("lf_foot");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel6DTalos:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalos);
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_fingertip_1_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel6DRandomHumanoid:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyRandomHumanoid);
        frame = state_factory_->get_pinocchio_model().getFrameId("rleg6_body");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        impulse_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;

      default:
        throw_pretty(__FILE__ ": Wrong ImpulseModelTypes::Type given");
        break;
    }
  }

  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> get_impulse() { return impulse_; }
  boost::shared_ptr<StateFactory> get_state_factory() { return state_factory_; }
  double num_diff_modifier_;

 private:
  ImpulseModelTypes::Type test_type_;                           //!< The type of impulse to test
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> impulse_;  //!< The pointer to the impulse model
  boost::shared_ptr<StateFactory> state_factory_;               //!< The pointer to the multibody state factory
};

boost::shared_ptr<ImpulseModelFactory> create_random_factory() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<ImpulseModelFactory> ptr;
  if (rand() % 2 == 0) {
    ptr = boost::make_shared<ImpulseModelFactory>(ImpulseModelTypes::ImpulseModel3DRandomHumanoid);
  } else {
    ptr = boost::make_shared<ImpulseModelFactory>(ImpulseModelTypes::ImpulseModel6DRandomHumanoid);
  }
  return ptr;
}

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_IMPULSES_FACTORY_HPP_
