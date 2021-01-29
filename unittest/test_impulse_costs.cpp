///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/impulse_cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_partial_derivatives_against_impulse_numdiff(
    ImpulseCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type) {
  // create the model
  const boost::shared_ptr<crocoddyl::ActionModelAbstract> &model =
      ImpulseCostModelFactory().create(cost_type, model_type, activation_type);

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::ActionDataAbstract> &data =
      model->createData();

  crocoddyl::ActionModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ActionDataAbstract> &data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd &x = model->get_state()->rand();
  const Eigen::VectorXd &u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = NUMDIFF_MODIFIER * model_num_diff.get_disturbance();
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

void register_impulse_cost_model_unit_tests(
    ImpulseCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << cost_type << "_" << activation_type << "_"
            << model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite *ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_impulse_numdiff, cost_type,
                  model_type, activation_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all the impulse cost model. Note that we can do it only with humanoids
  // as it needs to test the contact wrench cone
  for (size_t cost_type = 0; cost_type < ImpulseCostModelTypes::all.size();
       ++cost_type) {
    for (size_t activation_type = 0;
         activation_type < ActivationModelTypes::all.size();
         ++activation_type) {
      register_impulse_cost_model_unit_tests(
          ImpulseCostModelTypes::all[cost_type], PinocchioModelTypes::Talos,
          ActivationModelTypes::all[activation_type]);
    }
  }

  return true;
}

int main(int argc, char **argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
