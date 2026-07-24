///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, CTU, INRIA,
//                          Heriot-Watt University, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff_action.hpp"

#include "../random_generator.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"

namespace crocoddyl {
namespace unittest {

namespace {

std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>
create_deterministic_random_diff_lqr(const std::size_t nq, const std::size_t nu,
                                     const std::size_t ng = 0,
                                     const std::size_t nh = 0) {
  const std::size_t nv = nq;
  const std::size_t nx = nq + nv;
  Eigen::MatrixXd Aq = random_matrix<double>(nq, nq);
  Eigen::MatrixXd Av = random_matrix<double>(nv, nv);
  Eigen::MatrixXd B = random_matrix<double>(nv, nu);
  Eigen::MatrixXd L_tmp = random_matrix<double>(nx + nu, nx + nu);
  Eigen::MatrixXd L = L_tmp.transpose() * L_tmp;
  Eigen::MatrixXd Q = L.topLeftCorner(nx, nx);
  Eigen::MatrixXd R = L.bottomRightCorner(nu, nu);
  Eigen::MatrixXd N = L.topRightCorner(nx, nu);
  Eigen::MatrixXd G = random_matrix<double>(ng, nx + nu);
  Eigen::MatrixXd H = random_matrix<double>(nh, nx + nu);
  Eigen::VectorXd f = random_vector<double>(nv);
  Eigen::VectorXd q = random_vector<double>(nx);
  Eigen::VectorXd r = random_vector<double>(nu);
  Eigen::VectorXd g = random_vector<double>(ng);
  Eigen::VectorXd h = random_vector<double>(nh);
  return std::make_shared<crocoddyl::DifferentialActionModelLQR>(
      Aq, Av, B, Q, R, N, G, H, f, q, r, g, h);
}

}  // namespace

const std::vector<DifferentialActionModelTypes::Type>
    DifferentialActionModelTypes::all(DifferentialActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         DifferentialActionModelTypes::Type type) {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      os << "DifferentialActionModelLQR";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      os << "DifferentialActionModelLQRDriftFree";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelRandomLQR:
      os << "DifferentialActionModelRandomLQR";
      break;
    case DifferentialActionModelTypes::NbDifferentialActionModelTypes:
      os << "NbDifferentialActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

DifferentialActionModelFactory::DifferentialActionModelFactory() {}
DifferentialActionModelFactory::~DifferentialActionModelFactory() {}

std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>
DifferentialActionModelFactory::create(DifferentialActionModelTypes::Type type,
                                       bool) const {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      return std::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40,
                                                                     false);
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      return std::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40,
                                                                     true);
    case DifferentialActionModelTypes::DifferentialActionModelRandomLQR:
      return create_deterministic_random_diff_lqr(40, 40);
    default:
      throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
  }
}

}  // namespace unittest
}  // namespace crocoddyl
