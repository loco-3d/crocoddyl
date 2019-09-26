///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>

#include "crocoddyl/core/solvers/ddp.hpp"

using namespace boost::unit_test;

class SolverDPPFactory {
 public:
  SolverDPPFactory() {}

  std::unique_ptr<crocoddyl::ShootingProblem> shooting_problem_;
  std::unique_ptr<crocoddyl::SolverDDP> ddp_solver_;
};

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_1(crocoddyl::SolverAbstract& solver) {}

//____________________________________________________________________________//

void test_2(crocoddyl::SolverAbstract& solver) {}

//____________________________________________________________________________//
//____________________________________________________________________________//

void register_action_model_lqr_unit_tests() {
  int nx = 80;
  int nu = 40;
  bool driftfree = true;
  double num_diff_modifier = 1e4;

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_1, crocoddyl::SolverDDP(crocoddyl::ShootingProblem()))));
}

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_action_model_lqr_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
