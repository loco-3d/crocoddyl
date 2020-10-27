///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/data/multibody.hpp"

#include "factory/constraint.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_returns_a_residual(ConstraintModelTypes::Type constraint_type, StateModelTypes::Type state_type) {
  // create the model
  ConstraintModelFactory factory;
  const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& model = factory.create(constraint_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ConstraintDataAbstract>& data = model->createData(&shared_data);
  data->g *= nan("");
  data->h *= nan("");

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Getting the constraint residual computed by calc()
  model->calc(data, x, u);

  // Checking that calc returns a residual vector
  BOOST_CHECK(!data->g.hasNaN());
  BOOST_CHECK(!data->h.hasNaN());
}

void test_calc_against_numdiff(ConstraintModelTypes::Type constraint_type, StateModelTypes::Type state_type) {
  // create the model
  ConstraintModelFactory factory;
  const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& model = factory.create(constraint_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ConstraintDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ConstraintModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ConstraintDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK((data->g - data_num_diff->g).isMuchSmallerThan(1.));
  BOOST_CHECK((data->h - data_num_diff->h).isMuchSmallerThan(1.));
}

void test_partial_derivatives_against_numdiff(ConstraintModelTypes::Type constraint_type,
                                              StateModelTypes::Type state_type) {
  // create the model
  ConstraintModelFactory factory;
  const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& model = factory.create(constraint_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ConstraintDataAbstract>& data = model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ConstraintModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ConstraintDataAbstract>& data_num_diff = model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::ConstraintModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1));
  model_num_diff.set_reevals(reevals);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the cost derivatives via numerical differentiation
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = NUMDIFF_MODIFIER * model_num_diff.get_disturbance();

  BOOST_CHECK((data->Gx - data_num_diff->Gx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Gu - data_num_diff->Gu).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Hu - data_num_diff->Hu).isMuchSmallerThan(1.0, tol));
}

void test_dimensions_in_constraint_manager(ConstraintModelTypes::Type constraint_type,
                                           StateModelTypes::Type state_type) {
  // create the model
  ConstraintModelFactory factory;
  const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& model = factory.create(constraint_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create the constraint manager model
  crocoddyl::ConstraintModelManager constraint_man(state, model->get_nu());
  constraint_man.addConstraint("myConstraint", model);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = state->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  BOOST_CHECK(model->get_state()->get_nx() == constraint_man.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() == constraint_man.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == constraint_man.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() == constraint_man.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() == constraint_man.get_state()->get_nv());
  BOOST_CHECK(model->get_ng() == constraint_man.get_ng());
  BOOST_CHECK(model->get_nh() == constraint_man.get_nh());
}

void test_partial_derivatives_in_constraint_manager(ConstraintModelTypes::Type constraint_type,
                                                    StateModelTypes::Type state_type) {
  // create the model
  ConstraintModelFactory factory;
  const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& model = factory.create(constraint_type, state_type);

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const boost::shared_ptr<crocoddyl::ConstraintDataAbstract>& data = model->createData(&shared_data);

  // create the constraint manager model
  crocoddyl::ConstraintModelManager constraint_man(state, model->get_nu());
  constraint_man.addConstraint("myConstraint", model, 1.);
  const boost::shared_ptr<crocoddyl::ConstraintDataManager>& data_man = constraint_man.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the constraint derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the constraint-manager derivatives
  constraint_man.calc(data_man, x, u);
  constraint_man.calcDiff(data_man, x, u);

  BOOST_CHECK((data->Hx - data_man->Hx).isMuchSmallerThan(1.0));
  BOOST_CHECK((data->Hu - data_man->Hu).isMuchSmallerThan(1.0));
}

//----------------------------------------------------------------------------//

void register_constraint_model_unit_tests(ConstraintModelTypes::Type constraint_type,
                                          StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << constraint_type << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_residual, constraint_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, constraint_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, constraint_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_dimensions_in_constraint_manager, constraint_type, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_in_constraint_manager, constraint_type, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all constraints available with all available states types.
  for (size_t constraint_type = 0; constraint_type < ConstraintModelTypes::all.size(); ++constraint_type) {
    for (size_t state_type = StateModelTypes::all[StateModelTypes::StateMultibody_TalosArm];
         state_type < StateModelTypes::all.size(); ++state_type) {
      register_constraint_model_unit_tests(ConstraintModelTypes::all[constraint_type],
                                           StateModelTypes::all[state_type]);
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }