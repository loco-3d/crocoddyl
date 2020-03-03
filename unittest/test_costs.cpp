///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft
//                          University of Edinburgh, INRIA,
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "cost_factory.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl_unit_test;

//----------------------------------------------------------------------------//

void test_construct_data(CostModelTypes::Type cost_type, ActivationModelTypes::Type activation_type,
                         StateTypes::Type state_multibody_type) {
  // create the model
  CostModelFactory factory(cost_type, activation_type, state_multibody_type);
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(model->get_state()->get_pinocchio());
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data = model->createData(&shared_data);
}

void test_calc_returns_a_cost(CostModelTypes::Type cost_type, ActivationModelTypes::Type activation_type,
                              StateTypes::Type state_multibody_type) {
  // create the model
  CostModelFactory factory(cost_type, activation_type, state_multibody_type);
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(model->get_state()->get_pinocchio());
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data = model->createData(&shared_data);
  data->cost = nan("");

  // Getting the cost value computed by calc()
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_calc_against_numdiff(CostModelTypes::Type cost_type, ActivationModelTypes::Type activation_type,
                               StateTypes::Type state_multibody_type) {
  // create the model
  CostModelFactory factory(cost_type, activation_type, state_multibody_type);
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(model->get_state()->get_pinocchio());
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calc(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->cost == data_num_diff->cost);
}

void test_partial_derivatives_against_numdiff(CostModelTypes::Type cost_type,
                                              ActivationModelTypes::Type activation_type,
                                              StateTypes::Type state_multibody_type) {
  // create the model
  CostModelFactory factory(cost_type, activation_type, state_multibody_type);
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model = factory.create();

  // create the corresponding data object
  pinocchio::Model& pinocchio_model = model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl_unit_test::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::CostModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl_unit_test::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1));
  model_num_diff.set_reevals(reevals);

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the action derivatives via numerical differentiation
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = factory.get_num_diff_modifier() * model_num_diff.get_disturbance();

  BOOST_CHECK((data->Lx - data_num_diff->Lx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isMuchSmallerThan(1.0, tol));
  if (model_num_diff.get_with_gauss_approx()) {
    // The num diff is not precise enough to be tested here.
    // BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    // BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    // BOOST_CHECK((data->Luu - data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  }
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(CostModelTypes::Type cost_type, ActivationModelTypes::Type activation_type,
                                      StateTypes::Type state_multibody_type, test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, cost_type, activation_type, state_multibody_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, cost_type, activation_type, state_multibody_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, cost_type, activation_type, state_multibody_type)));
  ts.add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, cost_type, activation_type, state_multibody_type)));
}

bool init_function() {
  // Test all costs available with all the activation types with all available states types.
  for (size_t cost_type = 0; cost_type < CostModelTypes::all.size(); ++cost_type) {
    for (size_t activation_type = 0; activation_type < ActivationModelTypes::all.size(); ++activation_type) {
      for (size_t state_type = 0; state_type < StateTypes::all_multibody.size(); ++state_type) {
        std::ostringstream test_name;
        test_name << "test_" << CostModelTypes::all[cost_type] << "_" << ActivationModelTypes::all[activation_type]
                  << "_" << StateTypes::all_multibody[state_type];
        test_suite* ts = BOOST_TEST_SUITE(test_name.str());
        std::cout << "Running " << test_name.str() << std::endl;
        register_action_model_unit_tests(CostModelTypes::all[cost_type], ActivationModelTypes::all[activation_type],
                                         StateTypes::all_multibody[state_type], *ts);
        framework::master_test_suite().add(ts);
      }
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
