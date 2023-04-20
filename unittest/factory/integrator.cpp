///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "integrator.hpp"

#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/utils/exception.hpp"

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

boost::shared_ptr<crocoddyl::IntegratedActionModelAbstract>
IntegratorFactory::create(
    IntegratorTypes::Type type,
    boost::shared_ptr<DifferentialActionModelAbstract> model) const {
  boost::shared_ptr<crocoddyl::IntegratedActionModelAbstract> action;
  switch (type) {
    case IntegratorTypes::IntegratorEuler:
      action = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(model);
      break;
    case IntegratorTypes::IntegratorRK2:
      action = boost::make_shared<crocoddyl::IntegratedActionModelRK>(
          model, RKType::two);
      break;
    case IntegratorTypes::IntegratorRK3:
      action = boost::make_shared<crocoddyl::IntegratedActionModelRK>(
          model, RKType::three);
      break;
    case IntegratorTypes::IntegratorRK4:
      action = boost::make_shared<crocoddyl::IntegratedActionModelRK>(
          model, RKType::four);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong IntegratorTypes::Type given");
      break;
  }
  return action;
}

boost::shared_ptr<crocoddyl::IntegratedActionModelAbstract>
IntegratorFactory::create(
    IntegratorTypes::Type type,
    boost::shared_ptr<DifferentialActionModelAbstract> model,
    boost::shared_ptr<ControlParametrizationModelAbstract> control) const {
  boost::shared_ptr<crocoddyl::IntegratedActionModelAbstract> action;
  switch (type) {
    case IntegratorTypes::IntegratorEuler:
      action = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
          model, control);
      break;
    case IntegratorTypes::IntegratorRK2:
      action = boost::make_shared<crocoddyl::IntegratedActionModelRK>(
          model, control, RKType::two);
      break;
    case IntegratorTypes::IntegratorRK3:
      action = boost::make_shared<crocoddyl::IntegratedActionModelRK>(
          model, control, RKType::three);
      break;
    case IntegratorTypes::IntegratorRK4:
      action = boost::make_shared<crocoddyl::IntegratedActionModelRK>(
          model, control, RKType::four);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong IntegratorTypes::Type given");
      break;
  }
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
