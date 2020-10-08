///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActionModelTypes::Type> ActionModelTypes::all(ActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActionModelTypes::Type type) {
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      os << "ActionModelUnicycle";
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      os << "ActionModelLQRDriftFree";
      break;
    case ActionModelTypes::ActionModelLQR:
      os << "ActionModelLQR";
      break;
    case ActionModelTypes::NbActionModelTypes:
      os << "NbActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ActionModelFactory::ActionModelFactory() {}
ActionModelFactory::~ActionModelFactory() {}

  boost::shared_ptr<crocoddyl::ActionModelAbstract> ActionModelFactory::create(ActionModelTypes::Type type, bool secondInstance) const {
  boost::shared_ptr<crocoddyl::ActionModelAbstract> action;
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      action = boost::make_shared<crocoddyl::ActionModelUnicycle>();
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      if(secondInstance) {
          action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 40, true);
        } else {
        action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 20, true);
      }
      break;
    case ActionModelTypes::ActionModelLQR:
      if(secondInstance) {
          action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 40, false);
        } else {
        action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 20, false);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ActionModelTypes::Type given");
      break;
  }
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
