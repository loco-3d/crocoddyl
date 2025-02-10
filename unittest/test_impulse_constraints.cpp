///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/impulse_constraint.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_partial_derivatives_against_impulse_numdiff(
    ImpulseConstraintModelTypes::Type constraint_type,
    PinocchioModelTypes::Type model_type) {
  // create the model
  const std::shared_ptr<crocoddyl::ActionModelAbstract> &model =
      ImpulseConstraintModelFactory().create(constraint_type, model_type);

  // create the corresponding data object and set the constraint to nan
  const std::shared_ptr<crocoddyl::ActionDataAbstract> &data =
      model->createData();

  crocoddyl::ActionModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> &data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
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
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
}

//----------------------------------------------------------------------------//

void register_impulse_constraint_model_unit_tests(
    ImpulseConstraintModelTypes::Type constraint_type,
    PinocchioModelTypes::Type model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << constraint_type << "_" << model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite *ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_impulse_numdiff,
                  constraint_type, model_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all the impulse constraint model. Note that we can do it only with
  // humanoids as it needs to test the impulse wrench cone
  for (size_t constraint_type = 0;
       constraint_type < ImpulseConstraintModelTypes::all.size();
       ++constraint_type) {
    register_impulse_constraint_model_unit_tests(
        ImpulseConstraintModelTypes::all[constraint_type],
        PinocchioModelTypes::Talos);
    register_impulse_constraint_model_unit_tests(
        ImpulseConstraintModelTypes::all[constraint_type],
        PinocchioModelTypes::RandomHumanoid);
    if (ImpulseConstraintModelTypes::all[constraint_type] ==
            ImpulseConstraintModelTypes::
                ConstraintModelResidualImpulseForceEquality ||
        ImpulseConstraintModelTypes::all[constraint_type] ==
            ImpulseConstraintModelTypes::
                ConstraintModelResidualImpulseFrictionConeInequality) {
      register_impulse_constraint_model_unit_tests(
          ImpulseConstraintModelTypes::all[constraint_type],
          PinocchioModelTypes::HyQ);
    }
  }

  return true;
}

int main(int argc, char **argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
