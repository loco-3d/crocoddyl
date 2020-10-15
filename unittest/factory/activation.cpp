///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "activation.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/smooth-abs.hpp"
#include "crocoddyl/core/activations/smooth-2norm.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActivationModelTypes::Type> ActivationModelTypes::all(ActivationModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActivationModelTypes::Type type) {
  switch (type) {
    case ActivationModelTypes::ActivationModelQuad:
      os << "ActivationModelQuad";
      break;
    case ActivationModelTypes::ActivationModelSmoothAbs:
      os << "ActivationModelSmoothAbs";
      break;
    case ActivationModelTypes::ActivationModelSmooth2Norm:
      os << "ActivationModelSmooth2Norm";
      break;
    case ActivationModelTypes::ActivationModelWeightedQuad:
      os << "ActivationModelWeightedQuad";
      break;
    case ActivationModelTypes::ActivationModelQuadraticBarrier:
      os << "ActivationModelQuadraticBarrier";
      break;
    case ActivationModelTypes::ActivationModelWeightedQuadraticBarrier:
      os << "ActivationModelWeightedQuadraticBarrier";
      break;
    case ActivationModelTypes::NbActivationModelTypes:
      os << "NbActivationModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ActivationModelFactory::ActivationModelFactory() {}
ActivationModelFactory::~ActivationModelFactory() {}

boost::shared_ptr<crocoddyl::ActivationModelAbstract> ActivationModelFactory::create(
    ActivationModelTypes::Type activation_type, std::size_t nr) const {
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation;
  Eigen::VectorXd lb = Eigen::VectorXd::Random(nr);
  Eigen::VectorXd ub = lb + Eigen::VectorXd::Ones(nr) + Eigen::VectorXd::Random(nr);
  Eigen::VectorXd weights = 0.1 * Eigen::VectorXd::Random(nr);
  double eps = abs(Eigen::VectorXd::Random(1)[0]);

  switch (activation_type) {
    case ActivationModelTypes::ActivationModelQuad:
      activation = boost::make_shared<crocoddyl::ActivationModelQuad>(nr);
      break;
    case ActivationModelTypes::ActivationModelSmoothAbs:
      activation = boost::make_shared<crocoddyl::ActivationModelSmoothAbs>(nr, eps);
      break;
    case ActivationModelTypes::ActivationModelSmooth2Norm:
      activation = boost::make_shared<crocoddyl::ActivationModelSmooth2Norm>(nr, eps);
      break;
    case ActivationModelTypes::ActivationModelWeightedQuad:
      activation = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);
      break;
    case ActivationModelTypes::ActivationModelQuadraticBarrier:
      activation = boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(lb, ub));
      break;
    case ActivationModelTypes::ActivationModelWeightedQuadraticBarrier:
      activation = boost::make_shared<crocoddyl::ActivationModelWeightedQuadraticBarrier>(
          crocoddyl::ActivationBounds(lb, ub), weights);
      break;
    default:
      throw_pretty(__FILE__ ":\n Construct wrong ActivationModelTypes::Type");
      break;
  }
  return activation;
}

}  // namespace unittest
}  // namespace crocoddyl
