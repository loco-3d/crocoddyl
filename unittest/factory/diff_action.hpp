///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, CTU, INRIA,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_DIFF_ACTION_FACTORY_HPP_
#define CROCODDYL_DIFF_ACTION_FACTORY_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {
namespace unittest {

struct DifferentialActionModelTypes {
  enum Type {
    DifferentialActionModelLQR,
    DifferentialActionModelLQRDriftFree,
    DifferentialActionModelRandomLQR,
    NbDifferentialActionModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbDifferentialActionModelTypes);
    for (int i = 0; i < NbDifferentialActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os,
                         DifferentialActionModelTypes::Type type);

class DifferentialActionModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit DifferentialActionModelFactory();
  ~DifferentialActionModelFactory();

  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(
      DifferentialActionModelTypes::Type type,
      bool with_baumgarte = true) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_DIFF_ACTION_FACTORY_HPP_
