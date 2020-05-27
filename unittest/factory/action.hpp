///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ACTION_FACTORY_HPP_
#define CROCODDYL_ACTION_FACTORY_HPP_

#include <iterator>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace unittest {

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

std::ostream& operator<<(std::ostream& os, ActionModelTypes::Type type);

class ActionModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ActionModelFactory();
  ~ActionModelFactory();

  boost::shared_ptr<crocoddyl::ActionModelAbstract> create(ActionModelTypes::Type type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTION_FACTORY_HPP_
