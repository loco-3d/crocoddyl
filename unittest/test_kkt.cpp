///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include <Eigen/Dense>

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"


using namespace boost::unit_test;

//____________________________________________________________________________//


void test_solution(crocoddyl::ShootingProblem &problem)
{
  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());

  const long unsigned int T = problem.get_T();
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < T; ++i)
  {
    crocoddyl::ActionModelAbstract *model = problem.running_models_[i];
    xs.push_back(problem.get_x0());
    us.push_back(Eigen::VectorXd::Zero(model->get_nu()));
  }
  xs.push_back(problem.get_x0());

  crocoddyl::SolverKKT kkt(problem);
  BOOST_CHECK_EQUAL(kkt.get_kkt().rows(), 2 * kkt.get_ndx_() + kkt.get_nu_());
  BOOST_CHECK_EQUAL(kkt.get_kkt().cols(), 2 * kkt.get_ndx_() + kkt.get_nu_());
  BOOST_CHECK_EQUAL(kkt.get_kktref().size(), 2 * kkt.get_ndx_() + kkt.get_nu_());
  BOOST_CHECK_EQUAL(kkt.get_primaldual().size(), 2 * kkt.get_ndx_() + kkt.get_nu_());

  kkt.setCallbacks(cbs);
  kkt.solve(xs, us, 100);
  // check trajectory dimensions
  BOOST_CHECK_EQUAL(kkt.get_us().size(), T);
  BOOST_CHECK_EQUAL(kkt.get_xs().size(), T + 1);
  // initial state
  BOOST_CHECK((kkt.get_xs()[0] - problem.get_x0()).isMuchSmallerThan(1.0, 1e-9));

  crocoddyl::SolverDDP ddp(problem);
  ddp.setCallbacks(cbs);
  ddp.solve(xs, us, 100);
  // check trajectory dimensions
  BOOST_CHECK_EQUAL(ddp.get_us().size(), T);
  BOOST_CHECK_EQUAL(ddp.get_xs().size(), T + 1);
  // initial state
  BOOST_CHECK((ddp.get_xs()[0] - problem.get_x0()).isMuchSmallerThan(1.0, 1e-9));
  // check solutions against each other  
  for (unsigned int t = 0; t < T; ++t)
  {
    BOOST_CHECK((ddp.get_xs()[t] - kkt.get_xs()[t]).isMuchSmallerThan(1.0, 1e-9));
    BOOST_CHECK((ddp.get_us()[t] - kkt.get_us()[t]).isMuchSmallerThan(1.0, 1e-9));
  }
  // terminal state 
  BOOST_CHECK((ddp.get_xs()[T] - kkt.get_xs()[T]).isMuchSmallerThan(1.0, 1e-9));
}


//____________________________________________________________________________//

void register_state_vector_unit_tests() {
  // lqr model
  unsigned int NX = 6;
  unsigned int NU = 3;
  unsigned int T = 10;
  Eigen::VectorXd lqr_x0 = Eigen::VectorXd::Zero(NX);
  std::vector<crocoddyl::ActionModelAbstract *> lqr_runningModels;
  crocoddyl::ActionModelAbstract *lqr_terminalModel;
  // Creating the action models and warm point for the LQR system
  for (unsigned int i = 0; i < T; ++i)
  {
    crocoddyl::ActionModelAbstract *model_i = new crocoddyl::ActionModelLQR(NX, NU);
    lqr_runningModels.push_back(model_i);
  }
  lqr_terminalModel = new crocoddyl::ActionModelLQR(NX, NU);
  crocoddyl::ShootingProblem lqr_problem(lqr_x0, lqr_runningModels, lqr_terminalModel);

  // unicycle model
  unsigned int N = 20;  
  Eigen::VectorXd unicycle_x0 = Eigen::Vector3d(1., 0., 0.);
  std::vector<crocoddyl::ActionModelAbstract *> unicycle_runningModels;
  crocoddyl::ActionModelAbstract *unicycle_terminalModel;
  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < N; ++i) {
    crocoddyl::ActionModelAbstract* model_i = new crocoddyl::ActionModelUnicycle();
    unicycle_runningModels.push_back(model_i);
  }
  unicycle_terminalModel = new crocoddyl::ActionModelUnicycle();
  crocoddyl::ShootingProblem unicycle_problem(unicycle_x0, unicycle_runningModels, unicycle_terminalModel);

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_solution, lqr_problem)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_solution, unicycle_problem)));
}

//____________________________________________________________________________//

//____________________________________________________________________________//

bool init_function() {
  register_state_vector_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
