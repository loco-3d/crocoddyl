///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

#include "action_factory.hpp"

#ifndef CROCODDYL_STATE_FACTORY_HPP_
#define CROCODDYL_STATE_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct SolverTypes {
  enum Type { SolverKKT, SolverDDP, SolverFDDP, SolverBoxDDP, NbSolverTypes };
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
const std::vector<SolverTypes::Type> SolverTypes::all(SolverTypes::init_all());

class SolverFactory {
 public:
  SolverFactory(SolverTypes::Type solver_type, ActionModelTypes::Type action_type, size_t nb_running_models) {
    // default initialization
    solver_type_ = solver_type;
    action_factory_ = boost::make_shared<ActionModelFactory>(action_type);
    nb_running_models_ = nb_running_models;

    running_models_.resize(nb_running_models_, action_factory_->get_action());
    problem_ = boost::make_shared<crocoddyl::ShootingProblem>(action_factory_->get_action()->get_state()->zero(),
                                                              running_models_, action_factory_->get_action());

    switch (solver_type_) {
      case SolverTypes::SolverKKT:
        solver_ = boost::make_shared<crocoddyl::SolverKKT>(problem_);
        break;
      case SolverTypes::SolverDDP:
        solver_ = boost::make_shared<crocoddyl::SolverDDP>(problem_);
        break;
      case SolverTypes::SolverFDDP:
        solver_ = boost::make_shared<crocoddyl::SolverFDDP>(problem_);
        break;
      case SolverTypes::SolverBoxDDP:
        solver_ = boost::make_shared<crocoddyl::SolverFDDP>(problem_);
        break;
      default:
        throw_pretty(__FILE__ ": Wrong SolverTypes::Type given");
        break;
    }
  }

  ~SolverFactory() {}

  boost::shared_ptr<crocoddyl::SolverAbstract> get_solver() { return solver_; }

 private:
  size_t nb_running_models_;                              //!< This is the number of models in the shooting problem.
  SolverTypes::Type solver_type_;                         //!< The current type to test
  boost::shared_ptr<crocoddyl::SolverAbstract> solver_;   //!< The pointer to the solver in testing
  boost::shared_ptr<ActionModelFactory> action_factory_;  //!< The pointer to the action_model in testing
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
      running_models_;                                     //!< The list of models in the shooting problem
  boost::shared_ptr<crocoddyl::ShootingProblem> problem_;  //!< The pointer to the shooting problem in testing
};

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_STATE_FACTORY_HPP_
