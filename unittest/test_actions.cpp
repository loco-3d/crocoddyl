///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/action.hpp"
#include "factory/control.hpp"
#include "factory/diff_action.hpp"
#include "factory/integrator.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_check_data(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  BOOST_CHECK(model->checkData(data));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>> casted_data =
      casted_model->createData();
  BOOST_CHECK(casted_model->checkData(casted_data));
#endif
}

void test_calc(const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  data->cost = nan("");

  // Generating random state and control vectors
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);
  BOOST_CHECK(static_cast<std::size_t>(data->xnext.size()) ==
              model->get_state()->get_nx());

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));

  // Checking the termninal state
  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  model->calc(data, x);
  BOOST_CHECK((data->xnext - x).head(model->get_state()->get_nq()).isZero(tol));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>> casted_data =
      casted_model->createData();
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  model->calc(data, x, u);
  casted_model->calc(casted_data, x_f, u_f);
  BOOST_CHECK(static_cast<std::size_t>(casted_data->xnext.size()) ==
              casted_model->get_state()->get_nx());
  float tol_f = 10.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
#endif
}

void test_partial_derivatives_against_numdiff(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();

  crocoddyl::ActionModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data_num_diff =
      model_num_diff.createData();

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
  BOOST_CHECK((data->h - data_num_diff->h).isZero(tol));
  BOOST_CHECK((data->g - data_num_diff->g).isZero(tol));
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isZero(tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isZero(tol));
  }
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
  BOOST_CHECK((data->Hu - data_num_diff->Hu).isZero(tol));
  BOOST_CHECK((data->Gx - data_num_diff->Gx).isZero(tol));
  BOOST_CHECK((data->Gu - data_num_diff->Gu).isZero(tol));

  // Computing the action derivatives
  x = model->get_state()->rand();
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);
  BOOST_CHECK((data->h - data_num_diff->h).isZero(tol));
  BOOST_CHECK((data->g - data_num_diff->g).isZero(tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
  }
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
  BOOST_CHECK((data->Gx - data_num_diff->Gx).isZero(tol));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>> casted_data =
      casted_model->createData();
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->h.cast<float>() - casted_data->h).isZero(tol_f));
  BOOST_CHECK((data->g.cast<float>() - casted_data->g).isZero(tol_f));
  BOOST_CHECK((data->Fx.cast<float>() - casted_data->Fx).isZero(tol_f));
  BOOST_CHECK((data->Fu.cast<float>() - casted_data->Fu).isZero(tol_f));
  BOOST_CHECK((data->Lx.cast<float>() - casted_data->Lx).isZero(tol_f));
  BOOST_CHECK((data->Lu.cast<float>() - casted_data->Lu).isZero(tol_f));
  BOOST_CHECK((data->Gx.cast<float>() - casted_data->Gx).isZero(tol_f));
  BOOST_CHECK((data->Gu.cast<float>() - casted_data->Gu).isZero(tol_f));
  BOOST_CHECK((data->Hx.cast<float>() - casted_data->Hx).isZero(tol_f));
  BOOST_CHECK((data->Hu.cast<float>() - casted_data->Hu).isZero(tol_f));
  crocoddyl::ActionModelNumDiffTpl<float> casted_model_num_diff =
      model_num_diff.cast<float>();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>>
      casted_data_num_diff = casted_model_num_diff.createData();
  casted_model_num_diff.calc(casted_data_num_diff, x_f, u_f);
  casted_model_num_diff.calcDiff(casted_data_num_diff, x_f, u_f);
  tol_f = 80.0f * sqrt(casted_model_num_diff.get_disturbance());
  BOOST_CHECK((casted_data->Gx - casted_data_num_diff->Gx).isZero(tol_f));
  BOOST_CHECK((casted_data->Gu - casted_data_num_diff->Gu).isZero(tol_f));
  BOOST_CHECK((casted_data->Hx - casted_data_num_diff->Hx).isZero(tol_f));
  BOOST_CHECK((casted_data->Hu - casted_data_num_diff->Hu).isZero(tol_f));
#endif
}

void test_check_action_data(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);
  test_check_data(model);
}

void test_check_integrated_action_data(
    DifferentialActionModelTypes::Type dam_type,
    IntegratorTypes::Type integrator_type, ControlTypes::Type control_type) {
  // create the differential action model
  DifferentialActionModelFactory factory_dam;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& dam =
      factory_dam.create(dam_type);
  // create the control discretization
  ControlFactory factory_ctrl;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>& ctrl =
      factory_ctrl.create(control_type, dam->get_nu());
  // create the integrator
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::IntegratedActionModelAbstract>& model =
      factory_int.create(integrator_type, dam, ctrl);
  test_check_data(model);
}

void test_calc_action_model(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);
  test_calc(model);
}

void test_calc_integrated_action_model(
    DifferentialActionModelTypes::Type dam_type,
    IntegratorTypes::Type integrator_type, ControlTypes::Type control_type) {
  // create the differential action model
  DifferentialActionModelFactory factory_dam;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& dam =
      factory_dam.create(dam_type);
  // create the control discretization
  ControlFactory factory_ctrl;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>& ctrl =
      factory_ctrl.create(control_type, dam->get_nu());
  // create the integrator
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::IntegratedActionModelAbstract>& model =
      factory_int.create(integrator_type, dam, ctrl);
  test_calc(model);
}

void test_partial_derivatives_action_model(
    ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);
  test_partial_derivatives_against_numdiff(model);
}

void test_partial_derivatives_integrated_action_model(
    DifferentialActionModelTypes::Type dam_type,
    IntegratorTypes::Type integrator_type, ControlTypes::Type control_type) {
  // create the differential action model
  DifferentialActionModelFactory factory_dam;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& dam =
      factory_dam.create(dam_type);
  // create the control discretization
  ControlFactory factory_ctrl;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>& ctrl =
      factory_ctrl.create(control_type, dam->get_nu());
  // create the integrator
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::IntegratedActionModelAbstract>& model =
      factory_int.create(integrator_type, dam, ctrl);
  test_partial_derivatives_against_numdiff(model);
}

/**
 * Test two action models that should provide the same result when calling calc
 * if the first part of the control input u of model2 is equal to the control
 * input of model1. A typical case would be an integrated action model using an
 * Euler integration scheme, which can be coupled either with a constant control
 * parametrization (model1) or a linear control parametrization (model2), and
 * should thus provide the same result as long as the control input at the
 * beginning of the step has the same value.
 */
void test_calc_against_calc(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model1,
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model2) {
  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data1 =
      model1->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data2 =
      model2->createData();

  // Generating random values for the state and control
  const Eigen::VectorXd x = model1->get_state()->rand();
  Eigen::VectorXd u1 = Eigen::VectorXd::Random(model1->get_nu());
  Eigen::VectorXd u2 = Eigen::VectorXd::Random(model2->get_nu());
  // copy u1 to the first part of u2 (assuming u2 is larger than u1)
  u2.head(u1.size()) = u1;

  // Computing the action
  model1->calc(data1, x, u1);
  model2->calc(data2, x, u2);

  // Checking the state and cost integration
  BOOST_CHECK((data1->xnext - data2->xnext).isZero(1e-9));
  BOOST_CHECK(abs(data1->cost - data2->cost) < 1e-9);
}

void register_test_calc_integrated_action_model(
    DifferentialActionModelTypes::Type dam_type,
    IntegratorTypes::Type integrator_type, ControlTypes::Type control_type1,
    ControlTypes::Type control_type2) {
  // create the differential action model
  DifferentialActionModelFactory factory_dam;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& dam =
      factory_dam.create(dam_type);
  // create the control discretization
  ControlFactory factory_ctrl;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>& ctrl1 =
      factory_ctrl.create(control_type1, dam->get_nu());
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>& ctrl2 =
      factory_ctrl.create(control_type2, dam->get_nu());
  // create the integrator
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::IntegratedActionModelAbstract>& model1 =
      factory_int.create(integrator_type, dam, ctrl1);
  const std::shared_ptr<crocoddyl::IntegratedActionModelAbstract>& model2 =
      factory_int.create(integrator_type, dam, ctrl2);

  boost::test_tools::output_test_stream test_name;
  test_name << "test_calc_integrated_action_model_" << dam_type << "_"
            << integrator_type << "_" << control_type1 << "_" << control_type2;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_calc_against_calc, model1, model2)));
  framework::master_test_suite().add(ts);
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(
    ActionModelTypes::Type action_model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_check_action_data, action_model_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_calc_action_model, action_model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_action_model, action_model_type)));
  framework::master_test_suite().add(ts);
}

void register_integrated_action_model_unit_tests(
    DifferentialActionModelTypes::Type dam_type,
    IntegratorTypes::Type integrator_type, ControlTypes::Type control_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << dam_type << "_" << integrator_type << "_"
            << control_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_check_integrated_action_data, dam_type,
                                  integrator_type, control_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_calc_integrated_action_model, dam_type,
                                  integrator_type, control_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_integrated_action_model, dam_type,
                  integrator_type, control_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    register_action_model_unit_tests(ActionModelTypes::all[i]);
  }

  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorEuler,
        ControlTypes::PolyZero);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK2,
        ControlTypes::PolyZero);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK2,
        ControlTypes::PolyOne);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK3,
        ControlTypes::PolyZero);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK3,
        ControlTypes::PolyOne);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK3,
        ControlTypes::PolyTwoRK3);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK4,
        ControlTypes::PolyZero);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK4,
        ControlTypes::PolyOne);
    register_integrated_action_model_unit_tests(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorRK4,
        ControlTypes::PolyTwoRK4);
  }

  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    register_test_calc_integrated_action_model(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorEuler,
        ControlTypes::PolyZero, ControlTypes::PolyOne);
    register_test_calc_integrated_action_model(
        DifferentialActionModelTypes::all[i], IntegratorTypes::IntegratorEuler,
        ControlTypes::PolyOne, ControlTypes::PolyTwoRK4);
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
