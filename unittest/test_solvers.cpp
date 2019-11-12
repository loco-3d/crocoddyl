///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, New York University,
//                          Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "action_factory.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

using namespace boost::unit_test;
using namespace crocoddyl_unit_test;

//____________________________________________________________________________//

void test_kkt_dimension(const boost::shared_ptr<crocoddyl::ShootingProblem>& problem) {
  crocoddyl::SolverKKT kkt(problem);
  const std::size_t& T = problem->get_T();
  const std::size_t& ndx = kkt.get_ndx();
  const std::size_t& nu = kkt.get_nu();
  BOOST_CHECK_EQUAL(kkt.get_kkt().rows(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt.get_kkt().cols(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt.get_kktref().size(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt.get_primaldual().size(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt.get_us().size(), T);
  BOOST_CHECK_EQUAL(kkt.get_xs().size(), T + 1);
}

//____________________________________________________________________________//

void test_kkt_search_direction(const boost::shared_ptr<crocoddyl::ShootingProblem>& problem) {
  crocoddyl::SolverKKT kkt(problem);
  kkt.computeDirection();

  const std::size_t& ndx = kkt.get_ndx();
  const std::size_t& nu = kkt.get_nu();
  Eigen::MatrixXd kkt_mat = kkt.get_kkt();
  Eigen::Block<Eigen::MatrixXd> hess = kkt_mat.block(0, 0, ndx + nu, ndx + nu);

  // Checking the symmetricity of the Hessian
  BOOST_CHECK((hess - hess.transpose()).isMuchSmallerThan(1.0, 1e-9));

  // Check initial state
  BOOST_CHECK((kkt.get_dxs()[0] - problem->get_x0()).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_solver_against_kkt_solver(const boost::shared_ptr<crocoddyl::SolverAbstract>& solver) {
  const boost::shared_ptr<crocoddyl::ShootingProblem>& problem = solver->get_problem();

  std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
  cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());

  const std::size_t& T = problem->get_T();
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->running_models_[i];
    xs.push_back(problem->get_x0());
    us.push_back(Eigen::VectorXd::Zero(model->get_nu()));
  }
  xs.push_back(problem->get_x0());

  crocoddyl::SolverKKT kkt(problem);
  kkt.setCallbacks(cbs);
  kkt.solve(xs, us, 100);

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

void register_kkt_unit_tests(std::size_t T, ActionModelTypes::Type action_type) {
  ActionModelFactory factory(action_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.get_action_model();

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(T, model);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(model->get_state()->zero(), runningModels, model);

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_kkt_dimension, problem)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_kkt_search_direction, problem)));
}

//____________________________________________________________________________//

void register_ddp_unit_tests(std::size_t T, ActionModelTypes::Type action_type) {
  ActionModelFactory factory(action_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.get_action_model();

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(T, model);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(model->get_state()->zero(), runningModels, model);

  boost::shared_ptr<crocoddyl::SolverAbstract> ddp = boost::make_shared<crocoddyl::SolverDDP>(problem);
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_solver_against_kkt_solver, ddp)));
}

//____________________________________________________________________________//

void register_fddp_unit_tests(std::size_t T, ActionModelTypes::Type action_type) {
  ActionModelFactory factory(action_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.get_action_model();

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(T, model);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(model->get_state()->zero(), runningModels, model);

  boost::shared_ptr<crocoddyl::SolverAbstract> ddp = boost::make_shared<crocoddyl::SolverFDDP>(problem);
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_solver_against_kkt_solver, ddp)));
}

//____________________________________________________________________________//

bool init_function() {
  register_kkt_unit_tests(10, ActionModelTypes::ActionModelLQR);
  register_kkt_unit_tests(10, ActionModelTypes::ActionModelUnicycle);

  register_ddp_unit_tests(10, ActionModelTypes::ActionModelLQR);
  register_ddp_unit_tests(10, ActionModelTypes::ActionModelUnicycle);

  register_fddp_unit_tests(10, ActionModelTypes::ActionModelLQR);
  register_fddp_unit_tests(10, ActionModelTypes::ActionModelUnicycle);
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
