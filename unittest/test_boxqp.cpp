///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <boost/random.hpp>
#include "crocoddyl/core/solvers/box-qp.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

void test_constructor() {
  // Setup the test
  std::size_t nx = random_int_in_range(1, 100);
  crocoddyl::BoxQP boxqp(nx);

  // Test dimension of the decision vector
  BOOST_CHECK(boxqp.get_nx() == nx);
}

void test_unconstrained_qp_with_identity_hessian() {
  std::size_t nx = random_int_in_range(2, 5);
  crocoddyl::BoxQP boxqp(nx);
  boxqp.set_reg(0.);

  Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(nx, nx);
  Eigen::VectorXd gradient = Eigen::VectorXd::Random(nx);
  Eigen::VectorXd lb = -std::numeric_limits<double>::infinity() * Eigen::VectorXd::Ones(nx);
  Eigen::VectorXd ub = std::numeric_limits<double>::infinity() * Eigen::VectorXd::Ones(nx);
  Eigen::VectorXd xinit = Eigen::VectorXd::Random(nx);
  crocoddyl::BoxQPSolution sol = boxqp.solve(hessian, gradient, lb, ub, xinit);

  // Checking the solution of the problem. Note that it the negative of the gradient since Hessian
  // is identity matrix
  BOOST_CHECK((sol.x + gradient).isMuchSmallerThan(1.0, 1e-9));

  // Checking the solution against a regularized case
  double reg = random_real_in_range(1e-9, 1e2);
  boxqp.set_reg(reg);
  crocoddyl::BoxQPSolution sol_reg = boxqp.solve(hessian, gradient, lb, ub, xinit);
  BOOST_CHECK((sol_reg.x + gradient / (1 + reg)).isMuchSmallerThan(1.0, 1e-9));

  // Checking the all bounds are free and zero clamped
  BOOST_CHECK(sol.free_idx.size() == nx);
  BOOST_CHECK(sol.clamped_idx.size() == 0);
  BOOST_CHECK(sol_reg.free_idx.size() == nx);
  BOOST_CHECK(sol_reg.clamped_idx.size() == 0);
}

void test_unconstrained_qp() {
  std::size_t nx = random_int_in_range(2, 5);
  crocoddyl::BoxQP boxqp(nx);
  boxqp.set_reg(0.);

  Eigen::MatrixXd H = Eigen::MatrixXd::Random(nx, nx);
  Eigen::MatrixXd hessian = H.transpose() * H + nx * Eigen::MatrixXd::Identity(nx, nx);
  hessian = 0.5 * (hessian + hessian.transpose()).eval();
  Eigen::VectorXd gradient = Eigen::VectorXd::Random(nx);
  Eigen::VectorXd lb = -std::numeric_limits<double>::infinity() * Eigen::VectorXd::Ones(nx);
  Eigen::VectorXd ub = std::numeric_limits<double>::infinity() * Eigen::VectorXd::Ones(nx);
  Eigen::VectorXd xinit = Eigen::VectorXd::Random(nx);
  crocoddyl::BoxQPSolution sol = boxqp.solve(hessian, gradient, lb, ub, xinit);

  // Checking the solution against the KKT solution
  Eigen::VectorXd xkkt = -hessian.inverse() * gradient;
  BOOST_CHECK((sol.x - xkkt).isMuchSmallerThan(1.0, 1e-9));

  // Checking the solution against a regularized KKT problem
  double reg = random_real_in_range(1e-9, 1e2);
  boxqp.set_reg(reg);
  crocoddyl::BoxQPSolution sol_reg = boxqp.solve(hessian, gradient, lb, ub, xinit);
  Eigen::VectorXd xkkt_reg = -(hessian + reg * Eigen::MatrixXd::Identity(nx, nx)).inverse() * gradient;
  BOOST_CHECK((sol_reg.x - xkkt_reg).isMuchSmallerThan(1.0, 1e-9));

  // Checking the all bounds are free and zero clamped
  BOOST_CHECK(sol.free_idx.size() == nx);
  BOOST_CHECK(sol.clamped_idx.size() == 0);
  BOOST_CHECK(sol_reg.free_idx.size() == nx);
  BOOST_CHECK(sol_reg.clamped_idx.size() == 0);
}

void test_box_qp_with_identity_hessian() {
  std::size_t nx = random_int_in_range(2, 5);
  crocoddyl::BoxQP boxqp(nx);
  boxqp.set_reg(0.);

  Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(nx, nx);
  Eigen::VectorXd gradient = Eigen::VectorXd::Ones(nx);
  for (std::size_t i = 0; i < nx; ++i) {
    gradient(i) *= random_real_in_range(-1., 1.);
  }
  Eigen::VectorXd lb = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd ub = Eigen::VectorXd::Ones(nx);
  Eigen::VectorXd xinit = Eigen::VectorXd::Random(nx);
  crocoddyl::BoxQPSolution sol = boxqp.solve(hessian, gradient, lb, ub, xinit);

  // The analytical solution is the a bounded, and negative, gradient
  Eigen::VectorXd negbounded_gradient(nx), negbounded_gradient_reg(nx);
  std::size_t nf = nx, nc = 0, nf_reg = nx, nc_reg = 0;
  double reg = random_real_in_range(1e-9, 1e2);
  for (std::size_t i = 0; i < nx; ++i) {
    negbounded_gradient(i) = std::max(std::min(-gradient(i), ub(i)), lb(i));
    negbounded_gradient_reg(i) = std::max(std::min(-gradient(i) / (1 + reg), ub(i)), lb(i));
    if (negbounded_gradient(i) != -gradient(i)) {
      nc += 1;
      nf -= 1;
    }
    if (negbounded_gradient_reg(i) != -gradient(i) / (1 + reg)) {
      nc_reg += 1;
      nf_reg -= 1;
    }
  }

  // Checking the solution of the problem. Note that it the negative of the gradient since Hessian
  // is identity matrix
  BOOST_CHECK((sol.x - negbounded_gradient).isMuchSmallerThan(1.0, 1e-9));

  // Checking the solution against a regularized case
  boxqp.set_reg(reg);
  crocoddyl::BoxQPSolution sol_reg = boxqp.solve(hessian, gradient, lb, ub, xinit);
  BOOST_CHECK((sol_reg.x - negbounded_gradient / (1 + reg)).isMuchSmallerThan(1.0, 1e-9));

  // Checking the all bounds are free and zero clamped
  BOOST_CHECK(sol.free_idx.size() == nf);
  BOOST_CHECK(sol.clamped_idx.size() == nc);
  BOOST_CHECK(sol_reg.free_idx.size() == nf_reg);
  BOOST_CHECK(sol_reg.clamped_idx.size() == nc_reg);
}

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_unconstrained_qp_with_identity_hessian)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_unconstrained_qp)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_box_qp_with_identity_hessian)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
