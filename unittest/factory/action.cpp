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

ActionModelFactory::ActionModelFactory(ActionModelTypes::Type type) {
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      nx_ = 3;
      nu_ = 2;
      action_ = boost::make_shared<crocoddyl::ActionModelUnicycle>();
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      nx_ = 80;
      nu_ = 40;
      action_ = boost::make_shared<crocoddyl::ActionModelLQR>(nx_, nu_, true);
      break;
    case ActionModelTypes::ActionModelLQR:
      nx_ = 80;
      nu_ = 40;
      action_ = boost::make_shared<crocoddyl::ActionModelLQR>(nx_, nu_, false);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ActionModelTypes::Type given");
      break;
  }
}

ActionModelFactory::~ActionModelFactory() {}

boost::shared_ptr<crocoddyl::ActionModelAbstract> ActionModelFactory::create() { return action_; }

const std::size_t& ActionModelFactory::get_nx() { return nx_; }

}  // namespace unittest
}  // namespace crocoddyl
