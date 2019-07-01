///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include <Eigen/Dense>

using namespace boost::unit_test;

void test_construct_data(crocoddyl::ActionModelAbstract& model) {
  std::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();
}

//____________________________________________________________________________//

void test_calc_returns_state(crocoddyl::ActionModelAbstract& model) {
  // create the corresponding data object
  std::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();

  // Generating random state and control vectors
  Eigen::VectorXd x = model.get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model.get_nu());

  // Getting the state dimension from calc() call
  model.calc(data, x, u);
  long int nx = data->get_xnext().size();
  BOOST_CHECK(nx == model.get_nx());
}

//____________________________________________________________________________//

void test_calc_returns_a_cost(crocoddyl::ActionModelAbstract& model) {
  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();
  data->cost = std::nan("");

  // Getting the cost value computed by calc()
  Eigen::VectorXd x = model.get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model.get_nu());
  model.calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

//____________________________________________________________________________//

void test_partial_derivatives_against_numdiff(crocoddyl::ActionModelAbstract& model) {
  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();

  // create the num diff model and data
  // crocoddyl::ActionModelNumDiff model_num_diff (model);
  // std::shared_ptr<crocoddyl::ActionDataNumDiff> data_num_diff =
  //   model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model.get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model.get_nu());

  // Computing the action derivatives
  model.calcDiff(data, x, u);
  // model_num_diff.calcDiff(data_num_diff, x, u)

  // Checking the partial derivatives against NumDiff
  // double tol = self.NUMDIFF_MOD * self.MODEL_NUMDIFF.disturbance;
  // BOOST_CHECK((data->Fx - data_num_diff->Fx).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK((data->Fu - data_num_diff->Fu).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK((data->Lx - data_num_diff->Lx).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK((data->Lu - data_num_diff->Lu).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK(data->Lxx - data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK(data->Lxu - data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK(data->Luu - data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  BOOST_MESSAGE("test_partial_derivatives_against_numdiff is Disabled\n");
}

//____________________________________________________________________________//

void register_action_model_lqr_unit_tests() {
  int nx = 80;
  int nu = 40;
  bool driftfree = true;

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_construct_data, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, crocoddyl::ActionModelLQR(nx, nu, driftfree))));
}

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_action_model_lqr_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
