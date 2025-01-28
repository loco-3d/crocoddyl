///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/contact_constraint.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_partial_derivatives_against_contact_numdiff(
    ContactConstraintModelTypes::Type constraint_type,
    PinocchioModelTypes::Type model_type,
    ActuationModelTypes::Type actuation_type) {
  // create the model
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> &model =
      ContactConstraintModelFactory().create(constraint_type, model_type,
                                             actuation_type);

  // create the corresponding data object and set the constraint to nan
  const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> &data =
      model->createData();

  crocoddyl::DifferentialActionModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>
      &data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->Gx - data_num_diff->Gx).isZero(tol));
  BOOST_CHECK((data->Gu - data_num_diff->Gu).isZero(tol));
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
  BOOST_CHECK((data->Hu - data_num_diff->Hu).isZero(tol));

  // Computing the action derivatives
  x = model->get_state()->rand();
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);
  BOOST_CHECK((data->Gx - data_num_diff->Gx).isZero(tol));
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
}

//----------------------------------------------------------------------------//

void register_contact_constraint_model_unit_tests(
    ContactConstraintModelTypes::Type constraint_type,
    PinocchioModelTypes::Type model_type,
    ActuationModelTypes::Type actuation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << constraint_type << "_" << actuation_type << "_"
            << model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite *ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_contact_numdiff,
                  constraint_type, model_type, actuation_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all the contact constraint model. Note that we can do it only with
  // humanoids as it needs to test the contact wrench cone
  for (size_t constraint_type = 0;
       constraint_type < ContactConstraintModelTypes::all.size();
       ++constraint_type) {
    register_contact_constraint_model_unit_tests(
        ContactConstraintModelTypes::all[constraint_type],
        PinocchioModelTypes::Talos,
        ActuationModelTypes::ActuationModelFloatingBase);
    register_contact_constraint_model_unit_tests(
        ContactConstraintModelTypes::all[constraint_type],
        PinocchioModelTypes::RandomHumanoid,
        ActuationModelTypes::ActuationModelFloatingBase);
    if (ContactConstraintModelTypes::all[constraint_type] ==
            ContactConstraintModelTypes::
                ConstraintModelResidualContactForceEquality ||
        ContactConstraintModelTypes::all[constraint_type] ==
            ContactConstraintModelTypes::
                ConstraintModelResidualContactFrictionConeInequality ||
        ContactConstraintModelTypes::all[constraint_type] ==
            ContactConstraintModelTypes::
                ConstraintModelResidualContactControlGravInequality) {
      register_contact_constraint_model_unit_tests(
          ContactConstraintModelTypes::all[constraint_type],
          PinocchioModelTypes::HyQ,
          ActuationModelTypes::ActuationModelFloatingBase);
    }
  }
  return true;
}

int main(int argc, char **argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
