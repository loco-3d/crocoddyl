///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/states/unicycle.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include <Eigen/Dense>


using namespace boost::unit_test;

//____________________________________________________________________________//

void test_kkt_constructor(crocoddyl::ShootingProblem& problem) {
  // construct the solver 
  // crocoddyl::SolverAbstract* solver = crocoddyl::SolverKKT(problem); 
  // check if the dimensions are stored correctly 
  // this is only a sanity check to see if everything compiles 
  long unsigned int& T = problem->get_T();  
  crocoddyl::ActionModelAbstract* model_zero = problem->get_runningModels()[0];
  const unsigned int nx_ac = model_zero->get_nx();
  const unsigned int ndx_ac = model_zero->get_ndx();
  const unsigned int nu_ac = model_zero->get_nu();

  for (long unsigned int t = 0; t < T; ++t) {
    crocoddyl::ActionModelAbstract* model_i = problem->get_runningModels()[t];
    BOOST_CHECK_EQUAL(model_i->get_nx(), nx_ac);
    BOOST_CHECK_EQUAL(model_i->get_ndx(), ndx_ac);
    BOOST_CHECK_EQUAL(model_i->get_nu(), nu_ac);
}
}



//____________________________________________________________________________//

void register_state_vector_unit_tests() {
  unsigned int N = 200;  // number of nodes

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<crocoddyl::ActionModelAbstract*> runningModels;
  crocoddyl::ActionModelAbstract* terminalModel;
  x0 = Eigen::Vector3d(1., 0., 0.);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < N; ++i) {
    crocoddyl::ActionModelAbstract* model_i = new crocoddyl::ActionModelUnicycle();
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);
  crocoddyl::ActionModelAbstract* terminalModel = new crocoddyl::ActionModelUnicycle();

  // Formulating the optimal control problem
  crocoddyl::ShootingProblem problem(x0, runningModels, terminalModel);

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_kkt_constructor, problem)));

    }

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_state_vector_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
