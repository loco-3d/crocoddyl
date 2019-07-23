///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actions/action-lqr.hpp"
#include "crocoddyl/core/actions/action-unicycle.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include <Eigen/Dense>

using namespace boost::unit_test;

void test_construct_data(crocoddyl::ActionModelAbstract& model) {
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();
}

//____________________________________________________________________________//

void test_calc_returns_state(crocoddyl::ActionModelAbstract& model) {
  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();

  // Generating random state and control vectors
  Eigen::VectorXd x = model.get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model.get_nu());

  // Getting the state dimension from calc() call
  model.calc(data, x, u);

  BOOST_CHECK(data->get_xnext().size() == model.get_nx());
}

//____________________________________________________________________________//

void test_calc_returns_a_cost(crocoddyl::ActionModelAbstract& model) {
  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  Eigen::VectorXd x = model.get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model.get_nu());
  model.calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!isnan(data->cost));
}

//____________________________________________________________________________//

void test_partial_derivatives_against_numdiff(crocoddyl::ActionModelAbstract& model, double num_diff_modifier) {
  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();

  // create the num diff model and data
  bool with_gauss_approx = model.get_nr() > 1;

  crocoddyl::ActionModelNumDiff model_num_diff(model, with_gauss_approx);
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model.get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model.get_nu());

  // Computing the action derivatives
  model.calcDiff(data, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = num_diff_modifier * model_num_diff.get_disturbance();
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isMuchSmallerThan(1.0, tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  }
}

//____________________________________________________________________________//

void register_action_model_lqr_unit_tests() {
  int nx = 80;
  int nu = 40;
  bool driftfree = true;
  double num_diff_modifier = 1e4;

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_construct_data, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_partial_derivatives_against_numdiff, crocoddyl::ActionModelLQR(nx, nu, driftfree), num_diff_modifier)));
}

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_action_model_lqr_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
