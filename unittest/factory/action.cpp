///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action.hpp"

#include "../random_generator.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"

namespace crocoddyl {
namespace unittest {

namespace {

std::shared_ptr<crocoddyl::ActionModelAbstract> create_deterministic_random_lqr(
    const std::size_t nx, const std::size_t nu, const std::size_t ng = 0,
    const std::size_t nh = 0) {
  Eigen::MatrixXd A = random_matrix<double>(nx, nx);
  Eigen::MatrixXd B = random_matrix<double>(nx, nu);
  Eigen::MatrixXd L_tmp = random_matrix<double>(nx + nu, nx + nu);
  Eigen::MatrixXd L = L_tmp.transpose() * L_tmp;
  Eigen::MatrixXd Q = L.topLeftCorner(nx, nx);
  Eigen::MatrixXd R = L.bottomRightCorner(nu, nu);
  Eigen::MatrixXd N = L.topRightCorner(nx, nu);
  Eigen::MatrixXd G = random_matrix<double>(ng, nx + nu);
  Eigen::MatrixXd H = random_matrix<double>(nh, nx + nu);
  Eigen::VectorXd f = random_vector<double>(nx);
  Eigen::VectorXd q = random_vector<double>(nx);
  Eigen::VectorXd r = random_vector<double>(nu);
  Eigen::VectorXd g = random_vector<double>(ng);
  Eigen::VectorXd h = random_vector<double>(nh);
  return std::make_shared<crocoddyl::ActionModelLQR>(A, B, Q, R, N, G, H, f, q,
                                                     r, g, h);
}

}  // namespace

const std::vector<ActionModelTypes::Type> ActionModelTypes::all(
    ActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActionModelTypes::Type type) {
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      os << "ActionModelUnicycle";
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      os << "ActionModelLQRDriftFree";
      break;
    case ActionModelTypes::ActionModelLQR:
      os << "ActionModelLQR";
      break;
    case ActionModelTypes::ActionModelRandomLQR:
      os << "ActionModelRandomLQR";
      break;
    case ActionModelTypes::ActionModelRandomLQRwithTerminalConstraint:
      os << "ActionModelRandomLQRwithTerminalConstraint";
      break;
    case ActionModelTypes::NbActionModelTypes:
      os << "NbActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ActionModelFactory::ActionModelFactory() {}
ActionModelFactory::~ActionModelFactory() {}

std::shared_ptr<crocoddyl::ActionModelAbstract> ActionModelFactory::create(
    ActionModelTypes::Type type, Instance instance) const {
  std::shared_ptr<crocoddyl::ActionModelAbstract> action;
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      action = std::make_shared<crocoddyl::ActionModelUnicycle>();
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      switch (instance) {
        case First:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 2, true);
          break;
        case Second:
        case Terminal:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 4, true);
          break;
      }
    case ActionModelTypes::ActionModelLQR:
      switch (instance) {
        case First:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 2, false);
          break;
        case Second:
        case Terminal:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 4, false);
          break;
      }
      break;
    case ActionModelTypes::ActionModelRandomLQR:
      switch (instance) {
        case First:
          action = create_deterministic_random_lqr(8, 2);
          break;
        case Second:
        case Terminal:
          action = create_deterministic_random_lqr(8, 4);
          break;
      }
      break;
    case ActionModelTypes::ActionModelRandomLQRwithTerminalConstraint:
      switch (instance) {
        case First:
          action = create_deterministic_random_lqr(8, 2);
          break;
        case Second:
          action = create_deterministic_random_lqr(8, 4);
          break;
        case Terminal:
          action = create_deterministic_random_lqr(8, 4, 0, 2);
          break;
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ActionModelTypes::Type given");
  }
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
