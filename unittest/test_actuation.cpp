///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/actuation.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_construct_data(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& model = factory.create(actuation_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data = model->createData();
}

void test_calc_returns_tau(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& model = factory.create(actuation_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data = model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(static_cast<std::size_t>(data->tau.size()) == model->get_state()->get_nv());
}

void test_partial_derivatives_against_numdiff(ActuationModelTypes::Type actuation_type,
                                              StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& model = factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data = model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the actuation derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = NUMDIFF_MODIFIER * model_num_diff.get_disturbance();
  BOOST_CHECK((data->dtau_dx - data_num_diff->dtau_dx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->dtau_du - data_num_diff->dtau_du).isMuchSmallerThan(1.0, tol));
}

//----------------------------------------------------------------------------//

void register_actuation_model_unit_tests(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << actuation_type << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_construct_data, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_tau, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, actuation_type, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFull, StateModelTypes::StateMultibody_TalosArm);
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFloatingBase, StateModelTypes::StateMultibody_HyQ);
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFloatingBase, StateModelTypes::StateMultibody_Talos);
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFloatingBase, StateModelTypes::StateMultibody_RandomHumanoid);
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }