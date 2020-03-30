///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft
//                          University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/action.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_construct_data(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
}

void test_calc_returns_state(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(static_cast<std::size_t>(data->xnext.size()) == model->get_state()->get_nx());
}

void test_calc_returns_a_cost(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_partial_derivatives_against_numdiff(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();

  crocoddyl::ActionModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = NUMDIFF_MODIFIER * model_num_diff.get_disturbance();
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isMuchSmallerThan(1.0, tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  }
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(ActionModelTypes::Type action_model_type, test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, action_model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, action_model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, action_model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, action_model_type)));
}

bool init_function() {
  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    boost::test_tools::output_test_stream test_name;
    test_name << "test_" << ActionModelTypes::all[i];
    test_suite* ts = BOOST_TEST_SUITE(test_name.str());
    std::cout << "Running " << test_name.str() << std::endl;
    register_action_model_unit_tests(ActionModelTypes::all[i], *ts);
    framework::master_test_suite().add(ts);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
