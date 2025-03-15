///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/contact_cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_partial_derivatives_against_contact_numdiff(
    ContactCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type,
    ActuationModelTypes::Type actuation_type) {
  // create the model
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& model =
      ContactCostModelFactory().create(cost_type, model_type, activation_type,
                                       actuation_type);

  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data =
      model->createData();

  crocoddyl::DifferentialActionModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&
      data_num_diff = model_num_diff.createData();

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
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isZero(tol));
  }

  // Computing the action derivatives
  x = model->get_state()->rand();
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  float tol_f = 10.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->Lx.cast<float>() - casted_data->Lx).isZero(tol_f));
  BOOST_CHECK((data->Lu.cast<float>() - casted_data->Lu).isZero(tol_f));
  BOOST_CHECK((data->Lxx.cast<float>() - casted_data->Lxx).isZero(tol_f));
  BOOST_CHECK((data->Lxu.cast<float>() - casted_data->Lxu).isZero(tol_f));
  BOOST_CHECK((data->Luu.cast<float>() - casted_data->Luu).isZero(tol_f));
  model->calc(data, x);
  model->calcDiff(data, x);
  casted_model->calc(casted_data, x_f);
  casted_model->calcDiff(casted_data, x_f);
  BOOST_CHECK((data->Lx.cast<float>() - casted_data->Lx).isZero(tol_f));
  BOOST_CHECK((data->Lxx.cast<float>() - casted_data->Lxx).isZero(tol_f));
#endif
}

//----------------------------------------------------------------------------//

void register_contact_cost_model_unit_tests(
    ContactCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type,
    ActuationModelTypes::Type actuation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << cost_type << "_" << activation_type << "_"
            << actuation_type << "_" << model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_contact_numdiff, cost_type,
                  model_type, activation_type, actuation_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all the contact cost model. Note that we can do it only with humanoids
  // as it needs to test the contact wrench cone
  for (std::size_t cost_type = 0; cost_type < ContactCostModelTypes::all.size();
       ++cost_type) {
    for (std::size_t activation_type = 0;
         activation_type <
         ActivationModelTypes::ActivationModelQuadraticBarrier;
         ++activation_type) {
      register_contact_cost_model_unit_tests(
          ContactCostModelTypes::all[cost_type], PinocchioModelTypes::Talos,
          ActivationModelTypes::all[activation_type],
          ActuationModelTypes::ActuationModelFloatingBase);
      if (ContactCostModelTypes::all[cost_type] ==
              ContactCostModelTypes::CostModelResidualContactForce ||
          ContactCostModelTypes::all[cost_type] ==
              ContactCostModelTypes::CostModelResidualContactFrictionCone ||
          ContactCostModelTypes::all[cost_type] ==
              ContactCostModelTypes::CostModelResidualContactControlGrav) {
        register_contact_cost_model_unit_tests(
            ContactCostModelTypes::all[cost_type], PinocchioModelTypes::HyQ,
            ActivationModelTypes::all[activation_type],
            ActuationModelTypes::ActuationModelFloatingBase);
      }
    }
  }

  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
