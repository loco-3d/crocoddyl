///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iterator>
#include <pinocchio/fwd.hpp>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#ifndef CROCODDYL_ACTION_FACTORY_HPP_
#define CROCODDYL_ACTION_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct ActionModelTypes {
  enum Type { ActionModelUnicycle, ActionModelLQRDriftFree, ActionModelLQR, NbActionModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
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

class ActionModelFactory {
 public:
  ActionModelFactory(ActionModelTypes::Type type) {
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

  ~ActionModelFactory() {}

  boost::shared_ptr<crocoddyl::ActionModelAbstract> create() { return action_; }

  const std::size_t& get_nx() { return nx_; }

 private:
  std::size_t nx_;
  std::size_t nu_;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> action_;
};

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_ACTION_FACTORY_HPP_
