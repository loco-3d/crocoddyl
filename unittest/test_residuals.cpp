///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/data/multibody.hpp"

#include "factory/residual.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_returns_a_residual(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type) {
  // create the model
  ResidualModelFactory factory;
  const boost::shared_ptr<crocoddyl::ResidualModelAbstract>& model = factory.create(residual_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data = model->createData(&shared_data);
  data->r *= nan("");

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Getting the residual value computed by calc()
  model->calc(data, x, u);

  // Checking that calc returns a residual value
  for (std::size_t i = 0; i < model->get_nr(); ++i) BOOST_CHECK(!std::isnan(data->r(i)));
}

void test_calc_against_numdiff(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type) {
  // create the model
  ResidualModelFactory factory;
  const boost::shared_ptr<crocoddyl::ResidualModelAbstract>& model = factory.create(residual_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ResidualModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the residual derivatives
  model->calc(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->r == data_num_diff->r);
}

void test_partial_derivatives_against_numdiff(ResidualModelTypes::Type residual_type,
                                              StateModelTypes::Type state_type) {
  // create the model
  ResidualModelFactory factory;
  const boost::shared_ptr<crocoddyl::ResidualModelAbstract>& model = factory.create(residual_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ResidualModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::ResidualModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1));
  model_num_diff.set_reevals(reevals);

  // Computing the residual derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the residual derivatives via numerical differentiation
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->Rx - data_num_diff->Rx).isZero(tol));
  BOOST_CHECK((data->Ru - data_num_diff->Ru).isZero(tol));
}

//----------------------------------------------------------------------------//

void register_residual_model_unit_tests(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << residual_type << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_residual, residual_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, residual_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, residual_type, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all residuals available with all the activation types with all available states types.
  for (size_t residual_type = 0; residual_type < ResidualModelTypes::all.size(); ++residual_type) {
    for (size_t state_type = StateModelTypes::all[StateModelTypes::StateMultibody_TalosArm];
         state_type < StateModelTypes::all.size(); ++state_type) {
      register_residual_model_unit_tests(ResidualModelTypes::all[residual_type], StateModelTypes::all[state_type]);
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
