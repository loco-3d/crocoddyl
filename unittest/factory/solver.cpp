///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "solver.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/box-ddp.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<SolverTypes::Type> SolverTypes::all(SolverTypes::init_all());

std::ostream& operator<<(std::ostream& os, SolverTypes::Type type) {
  switch (type) {
    case SolverTypes::SolverKKT:
      os << "SolverKKT";
      break;
    case SolverTypes::SolverDDP:
      os << "SolverDDP";
      break;
    case SolverTypes::SolverFDDP:
      os << "SolverFDDP";
      break;
    case SolverTypes::SolverBoxDDP:
      os << "SolverBoxDDP";
      break;
    case SolverTypes::SolverBoxFDDP:
      os << "SolverBoxFDDP";
      break;
    case SolverTypes::NbSolverTypes:
      os << "NbSolverTypes";
      break;
    default:
      break;
  }
  return os;
}

SolverFactory::SolverFactory(SolverTypes::Type solver_type, ActionModelTypes::Type action_type,
                             size_t nb_running_models) {
  // default initialization
  solver_type_ = solver_type;
  action_factory_ = boost::make_shared<ActionModelFactory>(action_type);
  nb_running_models_ = nb_running_models;

  running_models_.resize(nb_running_models_, action_factory_->create());
  problem_ = boost::make_shared<crocoddyl::ShootingProblem>(action_factory_->create()->get_state()->zero(),
                                                            running_models_, action_factory_->create());

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
      solver_ = boost::make_shared<crocoddyl::SolverBoxDDP>(problem_);
      break;
    case SolverTypes::SolverBoxFDDP:
      solver_ = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem_);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong SolverTypes::Type given");
      break;
  }
}

SolverFactory::~SolverFactory() {}

boost::shared_ptr<crocoddyl::SolverAbstract> SolverFactory::create() const { return solver_; }

}  // namespace unittest
}  // namespace crocoddyl
