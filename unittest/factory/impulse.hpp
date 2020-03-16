///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "state.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"

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

  explicit ImpulseModelFactory() {}
  ~ImpulseModelFactory() {}

  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> create(ImpulseModelTypes::Type impulse_type,
                                                            PinocchioModelTypes::Type model_type) {
    boost::shared_ptr<crocoddyl::ImpulseModelAbstract> impulse;
    PinocchioModelFactory model_factory(model_type);
    boost::shared_ptr<crocoddyl::StateMultibody> state =
        boost::make_shared<crocoddyl::StateMultibody>(model_factory.create());
    boost::shared_ptr<crocoddyl::ContactModelAbstract> contact;
    switch (impulse_type) {
      case ImpulseModelTypes::ImpulseModel3D:
        impulse = boost::make_shared<crocoddyl::ImpulseModel3D>(state, model_factory.get_frame_id());
        break;
      case ImpulseModelTypes::ImpulseModel6D:
        impulse = boost::make_shared<crocoddyl::ImpulseModel6D>(state, model_factory.get_frame_id());
        break;
      default:
        throw_pretty(__FILE__ ": Wrong ImpulseModelTypes::Type given");
        break;
    }
    return impulse;
  }
};

boost::shared_ptr<crocoddyl::ImpulseModelAbstract> create_random_impulse() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> impulse;
  ImpulseModelFactory factory;
  if (rand() % 2 == 0) {
    impulse = factory.create(ImpulseModelTypes::ImpulseModel3D, PinocchioModelTypes::RandomHumanoid);
  } else {
    impulse = factory.create(ImpulseModelTypes::ImpulseModel6D, PinocchioModelTypes::RandomHumanoid);
  }
  return impulse;
}

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_IMPULSES_FACTORY_HPP_
