///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
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

void test_construct_data(ActuationModelTypes::Type actuation_type,
                         StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model =
      factory.create(actuation_type, state_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data =
      model->createData();
  if (!data)
    throw std::runtime_error("[test_construct_data] Data pointer is dead.");
}

void test_calc_returns_tau(ActuationModelTypes::Type actuation_type,
                           StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model =
      factory.create(actuation_type, state_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data =
      model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);
  BOOST_CHECK(static_cast<std::size_t>(data->tau.size()) ==
              model->get_state()->get_nv());

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  float tol_f = std::sqrt(float(2.0) * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->tau.cast<float>() - casted_data->tau).isZero(tol_f));
}

void test_actuationSet(ActuationModelTypes::Type actuation_type,
                       StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model =
      factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data =
      model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the selection matrix
  model->calc(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  const std::size_t nv = model->get_state()->get_nv();
  Eigen::MatrixXd S =
      data_num_diff->dtau_du * crocoddyl::pseudoInverse(data_num_diff->dtau_du);
  for (std::size_t k = 0; k < nv; ++k) {
    if (fabs(S(k, k)) < std::numeric_limits<double>::epsilon()) {
      BOOST_CHECK(data->tau_set[k] == false);
    } else {
      BOOST_CHECK(data->tau_set[k] == true);
    }
  }

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  for (std::size_t k = 0; k < nv; ++k) {
    BOOST_CHECK(data->tau_set[k] == casted_data->tau_set[k]);
  }
}

void test_partial_derivatives_against_numdiff(
    ActuationModelTypes::Type actuation_type,
    StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model =
      factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data =
      model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the actuation derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
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

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(
      (data->dtau_dx.cast<float>() - casted_data->dtau_dx).isZero(tol_f));
  BOOST_CHECK(
      (data->dtau_du.cast<float>() - casted_data->dtau_du).isZero(tol_f));

  casted_model->calc(casted_data, x_f);
  casted_model->calcDiff(casted_data, x_f);
  BOOST_CHECK(
      (data->dtau_dx.cast<float>() - casted_data->dtau_dx).isZero(tol_f));
}

void test_commands(ActuationModelTypes::Type actuation_type,
                   StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model =
      factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data =
      model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd tau =
      Eigen::VectorXd::Random(model->get_state()->get_nv());

  // Computing the actuation commands
  model->commands(data, x, tau);
  model_num_diff.commands(data_num_diff, x, tau);

  // Checking the joint torques
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->u - data_num_diff->u).isZero(tol));

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf tau_f = tau.cast<float>();
  casted_model->commands(casted_data, x_f, tau_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->u.cast<float>() - casted_data->u).isZero(tol_f));
}

void test_torqueTransform(ActuationModelTypes::Type actuation_type,
                          StateModelTypes::Type state_type) {
  // create the model
  ActuationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model =
      factory.create(actuation_type, state_type);

  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data =
      model->createData();

  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the torque transform
  model->torqueTransform(data, x, u);
  model_num_diff.torqueTransform(data_num_diff, x, u);

  // Checking the torque transform
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->Mtau - data_num_diff->Mtau).isZero(tol));

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->torqueTransform(casted_data, x_f, u_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->Mtau.cast<float>() - casted_data->Mtau).isZero(tol_f));
}

//----------------------------------------------------------------------------//

void register_actuation_model_unit_tests(
    ActuationModelTypes::Type actuation_type,
    StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << actuation_type << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_construct_data, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_returns_tau, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_actuationSet, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                      actuation_type, state_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_commands, actuation_type, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_torqueTransform, actuation_type, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < StateModelTypes::all.size(); ++i) {
    register_actuation_model_unit_tests(ActuationModelTypes::ActuationModelFull,
                                        StateModelTypes::all[i]);
  }
  for (size_t i = 0; i < StateModelTypes::all.size(); ++i) {
    if (StateModelTypes::all[i] != StateModelTypes::StateVector &&
        StateModelTypes::all[i] != StateModelTypes::StateMultibody_Hector) {
      register_actuation_model_unit_tests(
          ActuationModelTypes::ActuationModelFloatingBase,
          StateModelTypes::all[i]);
      register_actuation_model_unit_tests(
          ActuationModelTypes::ActuationModelSquashingFull,
          StateModelTypes::all[i]);
    }
  }

  register_actuation_model_unit_tests(
      ActuationModelTypes::ActuationModelFloatingBaseThrusters,
      StateModelTypes::StateMultibody_Hector);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
