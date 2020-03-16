///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "state.hpp"

#ifndef CROCODDYL_IMPULSES_FACTORY_HPP_
#define CROCODDYL_IMPULSES_FACTORY_HPP_

namespace crocoddyl {
namespace unittest {

struct ImpulseModelTypes {
  enum Type { ImpulseModel3D, ImpulseModel6D, NbImpulseModelTypes };
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
    case ImpulseModelTypes::ImpulseModel3D:
      os << "ImpulseModel3D";
      break;
    case ImpulseModelTypes::ImpulseModel6D:
      os << "ImpulseModel6D";
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

class ImpulseModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImpulseModelFactory(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
    PinocchioModelFactory model_factory(model_type);
    StateModelFactory state_factory(StateModelTypes::StateMultibody, model_type);
    boost::shared_ptr<crocoddyl::StateMultibody> state =
        boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create());
    pinocchio_model_ = state->get_pinocchio();

    switch (impulse_type) {
      case ImpulseModelTypes::ImpulseModel3D:
        impulse_ = boost::make_shared<crocoddyl::ImpulseModel3D>(state, model_factory.get_frame_id());
        break;
      case ImpulseModelTypes::ImpulseModel6D:
        impulse_ = boost::make_shared<crocoddyl::ImpulseModel6D>(state, model_factory.get_frame_id());
        break;
      default:
        throw_pretty(__FILE__ ": Wrong ImpulseModelTypes::Type given");
        break;
    }
  }

  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> create() const { return impulse_; }
  boost::shared_ptr<pinocchio::Model> get_pinocchio_model() const { return pinocchio_model_; }

 private:
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> impulse_;  //!< The pointer to the impulse model
  boost::shared_ptr<pinocchio::Model> pinocchio_model_;         //!< Pinocchio model
};

boost::shared_ptr<ImpulseModelFactory> create_random_factory() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<ImpulseModelFactory> ptr;
  if (rand() % 2 == 0) {
    ptr = boost::make_shared<ImpulseModelFactory>(ImpulseModelTypes::ImpulseModel3D,
                                                  PinocchioModelTypes::RandomHumanoid);
  } else {
    ptr = boost::make_shared<ImpulseModelFactory>(ImpulseModelTypes::ImpulseModel6D,
                                                  PinocchioModelTypes::RandomHumanoid);
  }
  return ptr;
}

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_IMPULSES_FACTORY_HPP_
