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
#include "factory/actuation.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_returns_a_residual(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
                                  ActuationModelTypes::Type actuation_type) {
  // Create the model
  ResidualModelFactory residual_factory;
  ActuationModelFactory actuation_factory;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      actuation_factory.create(actuation_type, state_type);
  const boost::shared_ptr<crocoddyl::ResidualModelAbstract>& model =
      residual_factory.create(residual_type, state_type, actuation_model->get_nu());

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // Create the corresponding shared data
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data = actuation_model->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data, actuation_data);

  // create the residual data
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data = model->createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateActuation(actuation_model, actuation_data, x, u);

  // Getting the residual value computed by calc()
  data->r *= nan("");
  model->calc(data, x, u);

  // Checking that calc returns a residual value
  for (std::size_t i = 0; i < model->get_nr(); ++i) BOOST_CHECK(!std::isnan(data->r(i)));
}

void test_calc_against_numdiff(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
                               ActuationModelTypes::Type actuation_type) {
  // Create the model
  ResidualModelFactory residual_factory;
  ActuationModelFactory actuation_factory;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      actuation_factory.create(actuation_type, state_type);
  const boost::shared_ptr<crocoddyl::ResidualModelAbstract>& model =
      residual_factory.create(residual_type, state_type, actuation_model->get_nu());

  // Create the corresponding shared data
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data = actuation_model->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data, actuation_data);

  // Create the residual data
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ResidualModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the residual
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  actuation_model->calc(actuation_data, x, u);
  model->calc(data, x, u);

  // Computing the residual from num diff
  std::vector<crocoddyl::ResidualModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1, _2));
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateActuation, actuation_model, actuation_data, _1, _2));
  model_num_diff.set_reevals(reevals);
  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->r == data_num_diff->r);
}

void test_partial_derivatives_against_numdiff(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
                                              ActuationModelTypes::Type actuation_type) {
  // Create the model
  ResidualModelFactory residual_factory;
  ActuationModelFactory actuation_factory;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      actuation_factory.create(actuation_type, state_type);
  const boost::shared_ptr<crocoddyl::ResidualModelAbstract>& model =
      residual_factory.create(residual_type, state_type, actuation_model->get_nu());

  // Create the corresponding shared data
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  const boost::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data = actuation_model->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data, actuation_data);

  // Create the residual data
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ResidualModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ResidualDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the residual derivatives
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  actuation_model->calc(actuation_data, x, u);
  actuation_model->calcDiff(actuation_data, x, u);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the residual derivatives via numerical differentiation
  std::vector<crocoddyl::ResidualModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1, _2));
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateActuation, actuation_model, actuation_data, _1, _2));
  model_num_diff.set_reevals(reevals);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = sqrt(model_num_diff.get_disturbance());
  BOOST_CHECK((data->Rx - data_num_diff->Rx).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((data->Ru - data_num_diff->Ru).isZero(tol));
}

//----------------------------------------------------------------------------//

void register_residual_model_unit_tests(ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
                                        ActuationModelTypes::Type actuation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << residual_type << "_" << state_type << "_" << actuation_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_residual, residual_type, state_type, actuation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, residual_type, state_type, actuation_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, residual_type, state_type, actuation_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all residuals available with all the activation types with all available states types.
  for (size_t residual_type = 0; residual_type < ResidualModelTypes::all.size(); ++residual_type) {
    for (size_t state_type = StateModelTypes::all[StateModelTypes::StateMultibody_TalosArm];
         state_type < StateModelTypes::all.size(); ++state_type) {
      for (size_t actuation_type = 0; actuation_type < ActuationModelTypes::all.size(); ++actuation_type) {
        if (ActuationModelTypes::all[actuation_type] != ActuationModelTypes::ActuationModelMultiCopterBase) {
          register_residual_model_unit_tests(ResidualModelTypes::all[residual_type], StateModelTypes::all[state_type],
                                             ActuationModelTypes::all[actuation_type]);
        } else if (StateModelTypes::all[state_type] != StateModelTypes::StateMultibody_TalosArm) {
          register_residual_model_unit_tests(ResidualModelTypes::all[residual_type], StateModelTypes::all[state_type],
                                             ActuationModelTypes::all[actuation_type]);
        }
      }
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
