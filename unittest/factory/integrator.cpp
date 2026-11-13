///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, LAAS-CNRS,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "integrator.hpp"

#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<IntegratorTypes::Type> IntegratorTypes::all(
    IntegratorTypes::init_all());

std::ostream& operator<<(std::ostream& os, IntegratorTypes::Type type) {
  switch (type) {
    case IntegratorTypes::IntegratorEuler:
      os << "IntegratorEuler";
      break;
    case IntegratorTypes::IntegratorRK2:
      os << "IntegratorRK2";
      break;
    case IntegratorTypes::IntegratorRK3:
      os << "IntegratorRK3";
      break;
    case IntegratorTypes::IntegratorRK4:
      os << "IntegratorRK4";
      break;
    case IntegratorTypes::NbIntegratorTypes:
      os << "NbIntegratorTypes";
      break;
    default:
      break;
  }
  return os;
}

IntegratorFactory::IntegratorFactory() {}
IntegratorFactory::~IntegratorFactory() {}

std::shared_ptr<crocoddyl::IntegratedActionModelAbstract>
IntegratorFactory::create(
    IntegratorTypes::Type type, std::shared_ptr<DynamicsModelAbstract> dynamics,
    std::shared_ptr<CostModelSum> costs,
    std::shared_ptr<ConstraintModelManager> constraints,
    std::shared_ptr<ControlParametrizationModelAbstract> control) const {
  std::shared_ptr<crocoddyl::IntegratedActionModelAbstract> action;
  switch (type) {
    case IntegratorTypes::IntegratorEuler:
      action = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
          dynamics, costs, constraints, control);
      break;
    case IntegratorTypes::IntegratorRK2:
      action = std::make_shared<crocoddyl::IntegratedActionModelRK>(
          dynamics, costs, constraints, control, nullptr, RKType::two);
      break;
    case IntegratorTypes::IntegratorRK3:
      action = std::make_shared<crocoddyl::IntegratedActionModelRK>(
          dynamics, costs, constraints, control, nullptr, RKType::three);
      break;
    case IntegratorTypes::IntegratorRK4:
      action = std::make_shared<crocoddyl::IntegratedActionModelRK>(
          dynamics, costs, constraints, control, nullptr, RKType::four);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong IntegratorTypes::Type given");
  }
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
