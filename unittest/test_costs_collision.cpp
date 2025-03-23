///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION

#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_returns_a_cost(CostModelCollisionTypes::Type cost_type,
                              StateModelTypes::Type state_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);
  data->cost = nan("");

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Getting the cost value computed by calc()
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_calc_against_numdiff(CostModelCollisionTypes::Type cost_type,
                               StateModelTypes::Type state_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff =
      model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->cost == data_num_diff->cost);
}

void test_partial_derivatives_against_numdiff(
    CostModelCollisionTypes::Type cost_type, StateModelTypes::Type state_type) {
  using namespace boost::placeholders;

  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff =
      model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::CostModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      double, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model, &pinocchio_data, _1, _2));
  model_num_diff.set_reevals(reevals);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
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

void test_dimensions_in_cost_sum(CostModelCollisionTypes::Type cost_type,
                                 StateModelTypes::Type state_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);

  // Generating random values for the state and control
  const Eigen::VectorXd x = state->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  BOOST_CHECK(model->get_state()->get_nx() == cost_sum.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() == cost_sum.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == cost_sum.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() == cost_sum.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() == cost_sum.get_state()->get_nv());
  BOOST_CHECK(model->get_activation()->get_nr() == cost_sum.get_nr());
}

void test_partial_derivatives_in_cost_sum(
    CostModelCollisionTypes::Type cost_type, StateModelTypes::Type state_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);
  const std::shared_ptr<crocoddyl::CostDataSum>& data_sum =
      cost_sum.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  cost_sum.calc(data_sum, x, u);
  cost_sum.calcDiff(data_sum, x, u);

  BOOST_CHECK((data->Lx - data_sum->Lx).isZero());
  BOOST_CHECK((data->Lu - data_sum->Lu).isZero());
  BOOST_CHECK((data->Lxx - data_sum->Lxx).isZero());
  BOOST_CHECK((data->Lxu - data_sum->Lxu).isZero());
  BOOST_CHECK((data->Luu - data_sum->Luu).isZero());
}

//----------------------------------------------------------------------------//

void register_cost_model_unit_tests(CostModelCollisionTypes::Type cost_type,
                                    StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << cost_type << "_2norm_barrier_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_returns_a_cost, cost_type, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_against_numdiff, cost_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                      cost_type, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_dimensions_in_cost_sum, cost_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_in_cost_sum,
                                      cost_type, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all costs available with all the activation types with all available
  // states types.
  for (size_t cost_type = 0; cost_type < CostModelCollisionTypes::all.size();
       ++cost_type) {
    register_cost_model_unit_tests(CostModelCollisionTypes::all[cost_type],
                                   StateModelTypes::StateMultibody_HyQ);
    register_cost_model_unit_tests(
        CostModelCollisionTypes::all[cost_type],
        StateModelTypes::StateMultibody_RandomHumanoid);
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else

int main(int, char**) {}

#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL
