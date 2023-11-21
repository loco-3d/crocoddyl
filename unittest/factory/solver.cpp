///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "solver.hpp"

#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#ifdef CROCODDYL_WITH_IPOPT
#include "crocoddyl/core/solvers/ipopt.hpp"
#endif

namespace crocoddyl {
namespace unittest {

const std::vector<SolverTypes::Type> SolverTypes::all(SolverTypes::init_all());

std::ostream& operator<<(std::ostream& os, SolverTypes::Type type) {
  switch (type) {
    case SolverTypes::SolverKKT:
      os << "SolverKKT";
      break;
    case SolverTypes::SolverFDDP_SingleShoot:
      os << "SolverFDDP_SingleShoot";
      break;
    case SolverTypes::SolverFDDP_FeasShoot:
      os << "SolverFDDP_FeasShoot";
      break;
    case SolverTypes::SolverFDDP_MultiShoot:
      os << "SolverFDDP_MultiShoot";
      break;
    case SolverTypes::SolverFDDP_HybridShoot:
      os << "SolverFDDP_HybridShoot";
      break;
    case SolverTypes::SolverBoxFDDP_SingleShoot:
      os << "SolverBoxFDDP_SingleShoot";
      break;
    case SolverTypes::SolverBoxFDDP_FeasShoot:
      os << "SolverBoxFDDP_FeasShoot";
      break;
    case SolverTypes::SolverBoxFDDP_MultiShoot:
      os << "SolverBoxFDDP_MultiShoot";
      break;
    case SolverTypes::SolverBoxFDDP_HybridShoot:
      os << "SolverBoxFDDP_HybridShoot";
      break;
#ifdef CROCODDYL_WITH_IPOPT
    case SolverTypes::SolverIpopt:
      os << "SolverIpopt";
      break;
#endif
    case SolverTypes::NbSolverTypes:
      os << "NbSolverTypes";
      break;
    default:
      break;
  }
  return os;
}

SolverFactory::SolverFactory() {}

SolverFactory::~SolverFactory() {}

std::shared_ptr<crocoddyl::SolverAbstract> SolverFactory::create(
    SolverTypes::Type solver_type,
    std::shared_ptr<crocoddyl::ActionModelAbstract> model,
    std::shared_ptr<crocoddyl::ActionModelAbstract> model2,
    std::shared_ptr<crocoddyl::ActionModelAbstract> modelT, size_t T) const {
  std::shared_ptr<crocoddyl::SolverAbstract> solver;
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> > running_models;
  const size_t halfway = T / 2;
  for (size_t i = 0; i < halfway; ++i) {
    running_models.push_back(model);
  }
  for (size_t i = 0; i < T - halfway; ++i) {
    running_models.push_back(model2);
  }

  std::shared_ptr<crocoddyl::ShootingProblem> problem =
      std::make_shared<crocoddyl::ShootingProblem>(model->get_state()->zero(),
                                                   running_models, modelT);

  switch (solver_type) {
    case SolverTypes::SolverKKT:
      solver = std::make_shared<crocoddyl::SolverKKT>(problem);
      break;
    case SolverTypes::SolverFDDP_SingleShoot:
      solver = boost::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot);
      break;
    case SolverTypes::SolverFDDP_FeasShoot:
      solver = boost::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot);
      break;
    case SolverTypes::SolverFDDP_MultiShoot:
      solver = boost::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot);
      break;
    case SolverTypes::SolverFDDP_HybridShoot:
      solver = boost::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot);
      break;
    case SolverTypes::SolverBoxFDDP_SingleShoot:
      solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot);
      break;
    case SolverTypes::SolverBoxFDDP_FeasShoot:
      solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot);
      break;
    case SolverTypes::SolverBoxFDDP_MultiShoot:
      solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot);
      break;
    case SolverTypes::SolverBoxFDDP_HybridShoot:
      solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot);
      break;
#ifdef CROCODDYL_WITH_IPOPT
    case SolverTypes::SolverIpopt:
      solver = std::make_shared<crocoddyl::SolverIpopt>(problem);
      break;
#endif
    default:
      throw_pretty(__FILE__ ": Wrong SolverTypes::Type given");
      break;
  }
  return solver;
}

}  // namespace unittest
}  // namespace crocoddyl
