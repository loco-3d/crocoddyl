///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/utils/callbacks.hpp"
#include "factory/solver.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//____________________________________________________________________________//

void test_kkt_dimension(ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the kkt solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverKKT> kkt =
      std::static_pointer_cast<crocoddyl::SolverKKT>(
          factory.create(SolverTypes::SolverKKT, model, model2, modelT, T));

  // define some aliases
  const std::size_t ndx = kkt->get_ndx();
  const std::size_t nu = kkt->get_nu();

  // Test the different matrix sizes
  BOOST_CHECK_EQUAL(kkt->get_kkt().rows(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_kkt().cols(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_kktref().size(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_primaldual().size(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_us().size(), T);
  BOOST_CHECK_EQUAL(kkt->get_xs().size(), T + 1);
}

//____________________________________________________________________________//

void test_kkt_search_direction(ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the kkt solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverKKT> kkt =
      std::static_pointer_cast<crocoddyl::SolverKKT>(
          factory.create(SolverTypes::SolverKKT, model, model2, modelT, T));

  // Generate the different state along the trajectory
  const std::shared_ptr<crocoddyl::ShootingProblem>& problem =
      kkt->get_problem();
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      problem->get_runningModels()[0]->get_state();
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        problem->get_runningModels()[i];
    xs.push_back(state->rand());
    us.push_back(Eigen::VectorXd::Random(model->get_nu()));
  }
  xs.push_back(state->rand());

  // Compute the search direction
  kkt->setCandidate(xs, us);
  kkt->computeDirection();

  // define some aliases
  const std::size_t ndx = kkt->get_ndx();
  const std::size_t nu = kkt->get_nu();
  Eigen::MatrixXd kkt_mat = kkt->get_kkt();
  Eigen::Block<Eigen::MatrixXd> hess = kkt_mat.block(0, 0, ndx + nu, ndx + nu);

  // Checking the symmetricity of the Hessian
  BOOST_CHECK((hess - hess.transpose()).isZero(1e-9));

  // Check initial state
  BOOST_CHECK((state->diff_dx(state->integrate_x(xs[0], kkt->get_dxs()[0]),
                              kkt->get_problem()->get_x0()))
                  .isZero(1e-9));
}

//____________________________________________________________________________//

void test_solver_against_kkt_solver(SolverTypes::Type solver_type,
                                    ActionModelTypes::Type action_type,
                                    size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the testing and KKT solvers
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);
  std::shared_ptr<crocoddyl::SolverAbstract> kkt =
      solver_factory.create(SolverTypes::SolverKKT, model, model2, modelT, T);

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const std::shared_ptr<crocoddyl::ShootingProblem>& problem =
      solver->get_problem();

  // Generate the different state along the trajectory
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      problem->get_runningModels()[0]->get_state();
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        problem->get_runningModels()[i];
    xs.push_back(state->rand());
    us.push_back(Eigen::VectorXd::Random(model->get_nu()));
  }
  xs.push_back(state->rand());

  // Define the callback function
  std::vector<std::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
  cbs.push_back(std::make_shared<crocoddyl::CallbackVerbose>());
  kkt->setCallbacks(cbs);
  solver->setCallbacks(cbs);

  // Print the name of the action model for introspection
  std::cout << ActionModelTypes::all[action_type] << std::endl;

  // Solve the problem using the KKT solver
  kkt->solve(xs, us, 100);

  // Solve the problem using the solver in testing
  solver->solve(xs, us, 100);

  // check trajectory dimensions
  BOOST_CHECK_EQUAL(solver->get_us().size(), T);
  BOOST_CHECK_EQUAL(solver->get_xs().size(), T + 1);

  // initial state
  BOOST_CHECK((solver->get_xs()[0] - problem->get_x0()).isZero(1e-9));

  // check solutions against each other
  for (unsigned int t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        solver->get_problem()->get_runningModels()[t];
    std::size_t nu = model->get_nu();
    BOOST_CHECK(
        (state->diff_dx(solver->get_xs()[t], kkt->get_xs()[t])).isZero(1e-9));
    BOOST_CHECK((solver->get_us()[t].head(nu) - kkt->get_us()[t]).isZero(1e-9));
  }
  BOOST_CHECK(
      (state->diff_dx(solver->get_xs()[T], kkt->get_xs()[T])).isZero(1e-9));
}

//____________________________________________________________________________//

void register_kkt_solver_unit_tests(ActionModelTypes::Type action_type,
                                    const std::size_t T) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_SolverKKT_" << action_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_kkt_dimension, action_type, T)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_kkt_search_direction, action_type, T)));
  framework::master_test_suite().add(ts);
}

void register_solvers_againt_kkt_unit_tests(SolverTypes::Type solver_type,
                                            ActionModelTypes::Type action_type,
                                            const std::size_t T) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << solver_type << "_vs_SolverKKT_" << action_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_against_kkt_solver,
                                      solver_type, action_type, T)));
  framework::master_test_suite().add(ts);
}

//____________________________________________________________________________//

bool init_function() {
  std::size_t T = 10;

  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    register_kkt_solver_unit_tests(ActionModelTypes::all[i], T);
  }

  // We start from 1 as 0 is the kkt solver
  for (size_t s = 1; s < SolverTypes::all.size(); ++s) {
    for (size_t i = 0; i < ActionModelTypes::ActionModelImpulseFwdDynamics_HyQ;
         ++i) {
      register_solvers_againt_kkt_unit_tests(SolverTypes::all[s],
                                             ActionModelTypes::all[i], T);
    }
  }
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
