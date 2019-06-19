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

void test_construct_data(crocoddyl::ActionModelAbstract& model)
{
  std::shared_ptr<crocoddyl::ActionDataAbstract> data = model.createData();
}

//____________________________________________________________________________//

void test_calc_returns_state(crocoddyl::ActionModelAbstract& model)
{
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

void test_calc_returns_a_cost(crocoddyl::ActionModelAbstract& )
{
}

//____________________________________________________________________________//

void test_partial_derivatives_against_numdiff(crocoddyl::ActionModelAbstract& )
{
}

//____________________________________________________________________________//

void register_action_model_lqr_unit_tests()
{
  int nx = 80 ;
  int nu = 40 ;
  bool driftfree = true;

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
    &test_construct_data, crocoddyl::ActionModelLQR(nx, nu, driftfree)
  )));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
    &test_calc_returns_state, crocoddyl::ActionModelLQR(nx, nu, driftfree)
  )));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
    &test_calc_returns_a_cost, crocoddyl::ActionModelLQR(nx, nu, driftfree)
  )));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
    &test_partial_derivatives_against_numdiff, crocoddyl::ActionModelLQR(nx, nu, driftfree)
  )));
}

//____________________________________________________________________________//

bool init_function()
{
  // Here we test the state_vector
  register_action_model_lqr_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main( int argc, char* argv[] )
{
    return ::boost::unit_test::unit_test_main( &init_function, argc, argv );
}
