///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
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
  if (!data) throw std::runtime_error("[test_construct_data] Data pointer is dead.");
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

void test_actuationSet(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& model = factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data = model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the selection matrix
  model->calc(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  const std::size_t nv = model->get_state()->get_nv();
  Eigen::MatrixXd S = data_num_diff->dtau_du * pseudoInverse(data_num_diff->dtau_du);
  for (std::size_t k = 0; k < nv; ++k) {
    if (fabs(S(k, k)) < std::numeric_limits<double>::epsilon()) {
      BOOST_CHECK(data->tau_set[k] == false);
    } else {
      BOOST_CHECK(data->tau_set[k] == true);
    }
  }
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
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the actuation derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->dtau_dx - data_num_diff->dtau_dx).isZero(tol));
  BOOST_CHECK((data->dtau_du - data_num_diff->dtau_du).isZero(tol));

  // Computing the actuation derivatives
  x = model->get_state()->rand();
  model->calc(data, x);
  model->calcDiff(data, x);

  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);

  // Checking the partial derivatives against numdiff
  BOOST_CHECK((data->dtau_dx - data_num_diff->dtau_dx).isZero(tol));
}

void test_commands(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& model = factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data = model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd& tau = Eigen::VectorXd::Random(model->get_state()->get_nv());

  // Computing the actuation commands
  model->commands(data, x, tau);
  model_num_diff.commands(data_num_diff, x, tau);

  // Checking the joint torques
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->u - data_num_diff->u).isZero(tol));
}

void test_torqueTransform(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& model = factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data = model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the torque transform
  model->torqueTransform(data, x, u);
  model_num_diff.torqueTransform(data_num_diff, x, u);

  // Checking the torque transform
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->Mtau - data_num_diff->Mtau).isZero(tol));
}

//----------------------------------------------------------------------------//

void register_actuation_model_unit_tests(ActuationModelTypes::Type actuation_type, StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << actuation_type << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_construct_data, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_tau, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_actuationSet, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_commands, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_torqueTransform, actuation_type, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < StateModelTypes::all.size(); ++i) {
    register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFull, StateModelTypes::all[i]);
  }
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFloatingBase,
                                      StateModelTypes::StateMultibody_HyQ);
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFloatingBase,
                                      StateModelTypes::StateMultibody_Talos);
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFloatingBase,
                                      StateModelTypes::StateMultibody_RandomHumanoid);
  register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelSquashingFull,
                                      StateModelTypes::StateMultibody_TalosArm);
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
