///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/data/multibody.hpp"

#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_returns_a_cost(CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
                              ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data = model->createData(&shared_data);
  data->cost = nan("");

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Getting the cost value computed by calc()
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_calc_against_numdiff(CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
                               ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
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
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->cost == data_num_diff->cost);
}

void test_partial_derivatives_against_numdiff(CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
                                              ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
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
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::CostModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1, _2));
  model_num_diff.set_reevals(reevals);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the cost derivatives via numerical differentiation
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    // The num diff is not precise enough to be tested here.
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isZero(tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data_num_diff->Luu).isZero(tol));
  }
}

void test_dimensions_in_cost_sum(CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
                                 ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = state->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  BOOST_CHECK(model->get_state()->get_nx() == cost_sum.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() == cost_sum.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == cost_sum.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() == cost_sum.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() == cost_sum.get_state()->get_nv());
  BOOST_CHECK(model->get_activation()->get_nr() == cost_sum.get_nr());
}

void test_partial_derivatives_in_cost_sum(CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
                                          ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const boost::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::CostDataAbstract>& data = model->createData(&shared_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);
  const boost::shared_ptr<crocoddyl::CostDataSum>& data_sum = cost_sum.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the cost-sum derivatives
  cost_sum.calc(data_sum, x, u);
  cost_sum.calcDiff(data_sum, x, u);

  BOOST_CHECK((data->Lx - data_sum->Lx).isZero());
  BOOST_CHECK((data->Lu - data_sum->Lu).isZero());
  BOOST_CHECK((data->Lxx - data_sum->Lxx).isZero());
  BOOST_CHECK((data->Lxu - data_sum->Lxu).isZero());
  BOOST_CHECK((data->Luu - data_sum->Luu).isZero());
}

//----------------------------------------------------------------------------//

void register_cost_model_unit_tests(CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
                                    ActivationModelTypes::Type activation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << cost_type << "_" << activation_type << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, cost_type, state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, cost_type, state_type, activation_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, cost_type, state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_dimensions_in_cost_sum, cost_type, state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_in_cost_sum, cost_type, state_type, activation_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all costs available with all the activation types with all available states types.
  for (size_t cost_type = 0; cost_type < CostModelTypes::all.size(); ++cost_type) {
    for (size_t state_type = StateModelTypes::all[StateModelTypes::StateMultibody_TalosArm];
         state_type < StateModelTypes::all.size(); ++state_type) {
      for (size_t activation_type = 0; activation_type < ActivationModelTypes::all.size(); ++activation_type) {
        register_cost_model_unit_tests(CostModelTypes::all[cost_type], StateModelTypes::all[state_type],
                                       ActivationModelTypes::all[activation_type]);
      }
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
