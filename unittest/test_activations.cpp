///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft
//                          University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "factory/activation.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_construct_data(ActivationModelTypes::Type test_type) {
  // create the model
  ActivationModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model = factory.create();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();
}

void test_calc_returns_a_value(ActivationModelTypes::Type test_type) {
  // create the model
  ActivationModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model = factory.create();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();

  // Generating random input vector
  const Eigen::VectorXd& r = Eigen::VectorXd::Random(model->get_nr());
  data->a_value = nan("");

  // Getting the state dimension from calc() call
  model->calc(data, r);

  // Checking that calc returns a value
  BOOST_CHECK(!std::isnan(data->a_value));
}

void test_partial_derivatives_against_numdiff(ActivationModelTypes::Type test_type) {
  // create the model
  ActivationModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model = factory.create();

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();

  crocoddyl::ActivationModelNumDiff model_num_diff(model);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd& r = Eigen::VectorXd::Random(model->get_nr());

  // Computing the activation derivatives
  model->calc(data, r);
  model->calcDiff(data, r);

  model_num_diff.calc(data_num_diff, r);
  model_num_diff.calcDiff(data_num_diff, r);

  // Checking the partial derivatives against NumDiff
  double tol = NUMDIFF_MODIFIER * model_num_diff.get_disturbance();
  BOOST_CHECK(std::abs(data->a_value - data_num_diff->a_value) < tol);
  BOOST_CHECK((data->Ar - data_num_diff->Ar).isMuchSmallerThan(1.0, tol));

  // numerical differentiation of the Hessian is not good enough to be tested.
  // BOOST_CHECK((data->Arr - data_num_diff->Arr).isMuchSmallerThan(1.0, tol));
}

//----------------------------------------------------------------------------//

void register_unit_tests(ActivationModelTypes::Type type, test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_value, type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, type)));
}

bool init_function() {
  for (size_t i = 0; i < ActivationModelTypes::all.size(); ++i) {
    std::ostringstream test_name;
    test_name << "test_" << ActivationModelTypes::all[i];
    test_suite* ts = BOOST_TEST_SUITE(test_name.str());
    std::cout << "Running " << test_name.str() << std::endl;
    register_unit_tests(ActivationModelTypes::all[i], *ts);
    framework::master_test_suite().add(ts);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
