///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "solver.hpp"

#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/intro.hpp"
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
    case SolverTypes::SolverFDDP_SingleShoot_LuNullTerm:
      os << "SolverFDDP_SingleShoot_LuNullTerm";
      break;
    case SolverTypes::SolverFDDP_SingleShoot_QrNullTerm:
      os << "SolverFDDP_SingleShoot_QrNullTerm";
      break;
    case SolverTypes::SolverFDDP_SingleShoot_SchurTerm:
      os << "SolverFDDP_SingleShoot_SchurTerm";
      break;
    case SolverTypes::SolverFDDP_FeasShoot_LuNullTerm:
      os << "SolverFDDP_FeasShoot_LuNullTerm";
      break;
    case SolverTypes::SolverFDDP_FeasShoot_QrNullTerm:
      os << "SolverFDDP_FeasShoot_QrNullTerm";
      break;
    case SolverTypes::SolverFDDP_FeasShoot_SchurTerm:
      os << "SolverFDDP_FeasShoot_SchurTerm";
      break;
    case SolverTypes::SolverFDDP_MultiShoot_LuNullTerm:
      os << "SolverFDDP_MultiShoot_LuNullTerm";
      break;
    case SolverTypes::SolverFDDP_MultiShoot_QrNullTerm:
      os << "SolverFDDP_MultiShoot_QrNullTerm";
      break;
    case SolverTypes::SolverFDDP_MultiShoot_SchurTerm:
      os << "SolverFDDP_MultiShoot_SchurTerm";
      break;
    case SolverTypes::SolverFDDP_HybridShoot_LuNullTerm:
      os << "SolverFDDP_HybridShoot_LuNullTerm";
      break;
    case SolverTypes::SolverFDDP_HybridShoot_QrNullTerm:
      os << "SolverFDDP_HybridShoot_QrNullTerm";
      break;
    case SolverTypes::SolverFDDP_HybridShoot_SchurTerm:
      os << "SolverFDDP_HybridShoot_SchurTerm";
      break;
    case SolverTypes::SolverIntro_SingleShoot_LuNullTerm:
      os << "SolverIntro_SingleShoot_LuNullTerm";
      break;
    case SolverTypes::SolverIntro_SingleShoot_QrNullTerm:
      os << "SolverIntro_SingleShoot_QrNullTerm";
      break;
    case SolverTypes::SolverIntro_SingleShoot_SchurTerm:
      os << "SolverIntro_SingleShoot_SchurTerm";
      break;
    case SolverTypes::SolverIntro_FeasShoot_LuNullTerm:
      os << "SolverIntro_FeasShoot_LuNullTerm";
      break;
    case SolverTypes::SolverIntro_FeasShoot_QrNullTerm:
      os << "SolverIntro_FeasShoot_QrNullTerm";
      break;
    case SolverTypes::SolverIntro_FeasShoot_SchurTerm:
      os << "SolverIntro_FeasShoot_SchurTerm";
      break;
    case SolverTypes::SolverIntro_MultiShoot_LuNullTerm:
      os << "SolverIntro_MultiShoot_LuNullTerm";
      break;
    case SolverTypes::SolverIntro_MultiShoot_QrNullTerm:
      os << "SolverIntro_MultiShoot_QrNullTerm";
      break;
    case SolverTypes::SolverIntro_MultiShoot_SchurTerm:
      os << "SolverIntro_MultiShoot_SchurTerm";
      break;
    case SolverTypes::SolverIntro_HybridShoot_LuNullTerm:
      os << "SolverIntro_HybridShoot_LuNullTerm";
      break;
    case SolverTypes::SolverIntro_HybridShoot_QrNullTerm:
      os << "SolverIntro_HybridShoot_QrNullTerm";
      break;
    case SolverTypes::SolverIntro_HybridShoot_SchurTerm:
      os << "SolverIntro_HybridShoot_SchurTerm";
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
#ifdef CROCODDYL_WITH_ODYN
    case SolverTypes::SolverOdynSQP:
      os << "SolverOdynSQP";
      break;
#endif
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
  problem->set_nthreads(1);

  switch (solver_type) {
    case SolverTypes::SolverKKT:
      solver = std::make_shared<crocoddyl::SolverKKT>(problem);
      break;
    case SolverTypes::SolverFDDP_SingleShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverFDDP_SingleShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverFDDP_SingleShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverFDDP_FeasShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverFDDP_FeasShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverFDDP_FeasShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverFDDP_MultiShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverFDDP_MultiShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverFDDP_MultiShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverFDDP_HybridShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverFDDP_HybridShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverFDDP_HybridShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverFDDP>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverIntro_SingleShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverIntro_SingleShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverIntro_SingleShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverIntro_FeasShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverIntro_FeasShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverIntro_FeasShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverIntro_MultiShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverIntro_MultiShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverIntro_MultiShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverIntro_HybridShoot_LuNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot,
          crocoddyl::EqualitySolverType::LuNull);
      break;
    case SolverTypes::SolverIntro_HybridShoot_QrNullTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot,
          crocoddyl::EqualitySolverType::QrNull);
      break;
    case SolverTypes::SolverIntro_HybridShoot_SchurTerm:
      solver = std::make_shared<crocoddyl::SolverIntro>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot,
          crocoddyl::EqualitySolverType::Schur);
      break;
    case SolverTypes::SolverBoxFDDP_SingleShoot:
      solver = std::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::SingleShoot);
      break;
    case SolverTypes::SolverBoxFDDP_FeasShoot:
      solver = std::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::FeasShoot);
      break;
    case SolverTypes::SolverBoxFDDP_MultiShoot:
      solver = std::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::MultiShoot);
      break;
    case SolverTypes::SolverBoxFDDP_HybridShoot:
      solver = std::make_shared<crocoddyl::SolverBoxFDDP>(
          problem, crocoddyl::DynamicsSolverType::HybridShoot);
      break;
#ifdef CROCODDYL_WITH_ODYN
    case SolverTypes::SolverOdynSQP:
      solver = std::make_shared<crocoddyl::SolverOdynSQP>(problem);
      break;
#endif
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

const std::vector<Eigen::MatrixXd>& SolverFactory::get_Vxx(
    SolverTypes::Type solver_type,
    const std::shared_ptr<crocoddyl::SolverAbstract>& solver) const {
  if (SolverTypes::isSolverFDDP(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverFDDP>(solver)->get_Vxx();
  } else if (SolverTypes::isSolverIntro(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverIntro>(solver)->get_Vxx();
  } else if (SolverTypes::isSolverBoxFDDP(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverBoxFDDP>(solver)
        ->get_Vxx();
  } else {
    throw_pretty("Invalid argument: this solver has Vxx");
  }
}

const std::vector<Eigen::VectorXd>& SolverFactory::get_Vx(
    SolverTypes::Type solver_type,
    const std::shared_ptr<crocoddyl::SolverAbstract>& solver) const {
  if (SolverTypes::isSolverFDDP(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverFDDP>(solver)->get_Vx();
  } else if (SolverTypes::isSolverIntro(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverIntro>(solver)->get_Vx();
  } else if (SolverTypes::isSolverBoxFDDP(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverBoxFDDP>(solver)->get_Vx();
  } else {
    throw_pretty("Invalid argument: this solver has Vx");
  }
}

const Eigen::MatrixXd& SolverFactory::get_dHc(
    SolverTypes::Type solver_type,
    const std::shared_ptr<crocoddyl::SolverAbstract>& solver) const {
  if (SolverTypes::isSolverFDDP(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverFDDP>(solver)->get_dHc();
  } else if (SolverTypes::isSolverIntro(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverIntro>(solver)->get_dHc();
  } else if (SolverTypes::isSolverBoxFDDP(solver_type)) {
    return std::static_pointer_cast<crocoddyl::SolverBoxFDDP>(solver)
        ->get_dHc();
  } else {
    throw_pretty("Invalid argument: this solver has dHc");
  }
}

}  // namespace unittest
}  // namespace crocoddyl
