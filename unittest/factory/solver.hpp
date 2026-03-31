///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, LAAS-CNRS,
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
#ifdef CROCODDYL_WITH_ODYN
#include "crocoddyl/core/solvers/odyn-sqp.hpp"
#endif

namespace crocoddyl {
namespace unittest {

struct SolverTypes {
  enum Type {
    SolverKKT,
    SolverFDDP_SingleShoot_LuNullTerm,
    SolverFDDP_SingleShoot_QrNullTerm,
    SolverFDDP_SingleShoot_SchurTerm,
    SolverFDDP_FeasShoot_LuNullTerm,
    SolverFDDP_FeasShoot_QrNullTerm,
    SolverFDDP_FeasShoot_SchurTerm,
    SolverFDDP_MultiShoot_LuNullTerm,
    SolverFDDP_MultiShoot_QrNullTerm,
    SolverFDDP_MultiShoot_SchurTerm,
    SolverFDDP_HybridShoot_LuNullTerm,
    SolverFDDP_HybridShoot_QrNullTerm,
    SolverFDDP_HybridShoot_SchurTerm,
    SolverIntro_SingleShoot_LuNullTerm,
    SolverIntro_SingleShoot_QrNullTerm,
    SolverIntro_SingleShoot_SchurTerm,
    SolverIntro_FeasShoot_LuNullTerm,
    SolverIntro_FeasShoot_QrNullTerm,
    SolverIntro_FeasShoot_SchurTerm,
    SolverIntro_MultiShoot_LuNullTerm,
    SolverIntro_MultiShoot_QrNullTerm,
    SolverIntro_MultiShoot_SchurTerm,
    SolverIntro_HybridShoot_LuNullTerm,
    SolverIntro_HybridShoot_QrNullTerm,
    SolverIntro_HybridShoot_SchurTerm,
    SolverBoxFDDP_SingleShoot,
    SolverBoxFDDP_FeasShoot,
    SolverBoxFDDP_MultiShoot,
    SolverBoxFDDP_HybridShoot,
#ifdef CROCODDYL_WITH_ODYN
    SolverOdynSQP,
#endif
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

  static Type getReferenceSolverType() {
#ifdef CROCODDYL_WITH_ODYN
    return SolverTypes::SolverOdynSQP;
#else
    return SolverTypes::SolverKKT;
#endif
  }

  static std::string getReferenceSolverName() {
#ifdef CROCODDYL_WITH_ODYN
    return "SolverOdynSQP";
#else
    return "SolverKKT";
#endif
  }

  static bool isSolverFDDP(Type type) {
    switch (type) {
      case SolverFDDP_SingleShoot_LuNullTerm:
      case SolverFDDP_SingleShoot_QrNullTerm:
      case SolverFDDP_SingleShoot_SchurTerm:
      case SolverFDDP_FeasShoot_LuNullTerm:
      case SolverFDDP_FeasShoot_QrNullTerm:
      case SolverFDDP_FeasShoot_SchurTerm:
      case SolverFDDP_MultiShoot_LuNullTerm:
      case SolverFDDP_MultiShoot_QrNullTerm:
      case SolverFDDP_MultiShoot_SchurTerm:
      case SolverFDDP_HybridShoot_LuNullTerm:
      case SolverFDDP_HybridShoot_QrNullTerm:
      case SolverFDDP_HybridShoot_SchurTerm:
        return true;
        break;
      default:
        return false;
        break;
    }
  }

  static bool isSolverIntro(Type type) {
    switch (type) {
      case SolverIntro_SingleShoot_LuNullTerm:
      case SolverIntro_SingleShoot_QrNullTerm:
      case SolverIntro_SingleShoot_SchurTerm:
      case SolverIntro_FeasShoot_LuNullTerm:
      case SolverIntro_FeasShoot_QrNullTerm:
      case SolverIntro_FeasShoot_SchurTerm:
      case SolverIntro_MultiShoot_LuNullTerm:
      case SolverIntro_MultiShoot_QrNullTerm:
      case SolverIntro_MultiShoot_SchurTerm:
      case SolverIntro_HybridShoot_LuNullTerm:
      case SolverIntro_HybridShoot_QrNullTerm:
      case SolverIntro_HybridShoot_SchurTerm:
        return true;
        break;
      default:
        return false;
        break;
    }
  }

  static bool isSolverBoxFDDP(Type type) {
    switch (type) {
      case SolverBoxFDDP_SingleShoot:
      case SolverBoxFDDP_FeasShoot:
      case SolverBoxFDDP_MultiShoot:
      case SolverBoxFDDP_HybridShoot:
        return true;
        break;
      default:
        return false;
        break;
    }
  }

  static bool isSingleShoot(Type type) {
    switch (type) {
      case SolverFDDP_SingleShoot_LuNullTerm:
      case SolverFDDP_SingleShoot_QrNullTerm:
      case SolverFDDP_SingleShoot_SchurTerm:
      case SolverIntro_SingleShoot_LuNullTerm:
      case SolverIntro_SingleShoot_QrNullTerm:
      case SolverIntro_SingleShoot_SchurTerm:
      case SolverBoxFDDP_SingleShoot:
        return true;
        break;
      default:
        return false;
        break;
    }
  }
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

  const std::vector<Eigen::MatrixXd>& get_Vxx(
      SolverTypes::Type solver_type,
      const std::shared_ptr<crocoddyl::SolverAbstract>& solver) const;
  const std::vector<Eigen::VectorXd>& get_Vx(
      SolverTypes::Type solver_type,
      const std::shared_ptr<crocoddyl::SolverAbstract>& solver) const;
  const Eigen::MatrixXd& get_dHc(
      SolverTypes::Type solver_type,
      const std::shared_ptr<crocoddyl::SolverAbstract>& solver) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_SOLVER_FACTORY_HPP_
