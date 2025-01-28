///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, University of Edinburgh, LAAS-CNRS,
//                          New York University, Max Planck Gesellschaft,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_SOLVER_FACTORY_HPP_
#define CROCODDYL_SOLVER_FACTORY_HPP_

#include "action.hpp"
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"

namespace crocoddyl {
namespace unittest {

struct SolverTypes {
  enum Type {
    SolverKKT,
    SolverDDP,
    SolverFDDP,
    SolverBoxDDP,
    SolverBoxFDDP,
    SolverIpopt,
    NbSolverTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbSolverTypes);
    for (int i = 0; i < NbSolverTypes; ++i) {
#ifndef CROCODDYL_WITH_IPOPT
      if ((Type)i == SolverIpopt) {
        continue;
      }
#endif
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, SolverTypes::Type type);

class SolverFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverFactory();
  ~SolverFactory();

  std::shared_ptr<crocoddyl::SolverAbstract> create(
      SolverTypes::Type solver_type,
      std::shared_ptr<crocoddyl::ActionModelAbstract> model,
      std::shared_ptr<crocoddyl::ActionModelAbstract> model2,
      std::shared_ptr<crocoddyl::ActionModelAbstract> modelT, size_t T) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_SOLVER_FACTORY_HPP_
