///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/callbacks.hpp"
#include "factory/solver.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//____________________________________________________________________________//

void test_kkt_dimension(ActionModelTypes::Type action_type, size_t T) {
  // Create the kkt solver
  SolverFactory factory;
  boost::shared_ptr<crocoddyl::SolverKKT> kkt =
      boost::static_pointer_cast<crocoddyl::SolverKKT>(factory.create(SolverTypes::SolverKKT, action_type, T));

  // define some aliases
  const std::size_t& ndx = kkt->get_ndx();
  const std::size_t& nu = kkt->get_nu();

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
  // Create the kkt solver
  SolverFactory factory;
  boost::shared_ptr<crocoddyl::SolverKKT> kkt =
      boost::static_pointer_cast<crocoddyl::SolverKKT>(factory.create(SolverTypes::SolverKKT, action_type, T));

  // Compute the search direction
  kkt->computeDirection();

  // define some aliases
  const std::size_t& ndx = kkt->get_ndx();
  const std::size_t& nu = kkt->get_nu();
  Eigen::MatrixXd kkt_mat = kkt->get_kkt();
  Eigen::Block<Eigen::MatrixXd> hess = kkt_mat.block(0, 0, ndx + nu, ndx + nu);

  // Checking the symmetricity of the Hessian
  BOOST_CHECK((hess - hess.transpose()).isMuchSmallerThan(1.0, 1e-9));

  // Check initial state
  BOOST_CHECK((kkt->get_dxs()[0] - kkt->get_problem()->get_x0()).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_solver_against_kkt_solver(SolverTypes::Type solver_type, ActionModelTypes::Type action_type, size_t T) {
  // Create the solver
  SolverFactory solver_factory;
  boost::shared_ptr<crocoddyl::SolverAbstract> solver =
      boost::static_pointer_cast<crocoddyl::SolverKKT>(solver_factory.create(solver_type, action_type, T));

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const boost::shared_ptr<crocoddyl::ShootingProblem>& problem = solver->get_problem();

  // Define the callback function
  std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
  cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());

  // Genreate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->get_runningModels()[i];
    xs.push_back(problem->get_x0());
    us.push_back(Eigen::VectorXd::Zero(model->get_nu()));
  }
  xs.push_back(problem->get_x0());

  // Solve the problem using the KKT solver
  crocoddyl::SolverKKT kkt(problem);
  kkt.setCallbacks(cbs);
  kkt.solve(xs, us, 100);

  // Solve the problem using the solver in testing
  solver->setCallbacks(cbs);
  solver->solve(xs, us, 100);

  // check trajectory dimensions
  BOOST_CHECK_EQUAL(solver->get_us().size(), T);
  BOOST_CHECK_EQUAL(solver->get_xs().size(), T + 1);

  // initial state
  BOOST_CHECK((solver->get_xs()[0] - problem->get_x0()).isMuchSmallerThan(1.0, 1e-9));

  // check solutions against each other
  for (unsigned int t = 0; t < T; ++t) {
    BOOST_CHECK((solver->get_xs()[t] - kkt.get_xs()[t]).isMuchSmallerThan(1.0, 1e-9));
    BOOST_CHECK((solver->get_us()[t] - kkt.get_us()[t]).isMuchSmallerThan(1.0, 1e-9));
  }
  BOOST_CHECK((solver->get_xs()[T] - kkt.get_xs()[T]).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

bool init_function() {
  size_t T = 10;

  for (size_t action_type = 0; action_type < ActionModelTypes::all.size(); ++action_type) {
    const std::string test_name = "test_action_" + std::to_string(action_type);
    test_suite* ts = BOOST_TEST_SUITE(test_name);
    ts->add(BOOST_TEST_CASE(boost::bind(&test_kkt_dimension, ActionModelTypes::all[action_type], T)));
    ts->add(BOOST_TEST_CASE(boost::bind(&test_kkt_search_direction, ActionModelTypes::all[action_type], T)));
    framework::master_test_suite().add(ts);
  }

  // We start from 1 as 0 is the kkt solver
  for (size_t solver_type = 1; solver_type < SolverTypes::all.size(); ++solver_type) {
    for (size_t action_type = 0; action_type < ActionModelTypes::all.size(); ++action_type) {
      std::ostringstream test_name;
      test_name << "test_" << SolverTypes::all[solver_type] << "_" << ActionModelTypes::all[action_type];
      test_suite* ts = BOOST_TEST_SUITE(test_name.str());
      std::cout << "Running " << test_name.str() << std::endl;
      ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_against_kkt_solver, SolverTypes::all[solver_type],
                                          ActionModelTypes::all[action_type], T)));
      framework::master_test_suite().add(ts);
    }
  }
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
