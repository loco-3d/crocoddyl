///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_STATE_FACTORY_HPP_
#define CROCODDYL_STATE_FACTORY_HPP_

#include "action.hpp"
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"

namespace crocoddyl {
namespace unittest {

struct SolverTypes {
  enum Type { SolverKKT, SolverDDP, SolverFDDP, SolverBoxDDP, SolverBoxFDDP, NbSolverTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbSolverTypes; ++i) {
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

  SolverFactory(SolverTypes::Type solver_type, ActionModelTypes::Type action_type, size_t nb_running_models);
  ~SolverFactory();

  boost::shared_ptr<crocoddyl::SolverAbstract> create();

 private:
  size_t nb_running_models_;                              //!< This is the number of models in the shooting problem.
  SolverTypes::Type solver_type_;                         //!< The current type to test
  boost::shared_ptr<crocoddyl::SolverAbstract> solver_;   //!< The pointer to the solver in testing
  boost::shared_ptr<ActionModelFactory> action_factory_;  //!< The pointer to the action_model in testing
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
      running_models_;                                     //!< The list of models in the shooting problem
  boost::shared_ptr<crocoddyl::ShootingProblem> problem_;  //!< The pointer to the shooting problem in testing
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_STATE_FACTORY_HPP_
