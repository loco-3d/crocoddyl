///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/smooth-1norm.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/activations/weighted-smooth-1norm.hpp"
#include "factory/activation.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_construct_data(ActivationModelTypes::Type activation_type) {
  // create the model
  ActivationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActivationModelAbstract>& model =
      factory.create(activation_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::ActivationDataAbstract>& data =
      model->createData();
  const std::shared_ptr<crocoddyl::ActivationDataAbstractTpl<float>>&
      casted_data = model->cast<float>()->createData();
}

void test_calc_returns_a_value(ActivationModelTypes::Type activation_type) {
  // create the model
  ActivationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActivationModelAbstract>& model =
      factory.create(activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::ActivationDataAbstract>& data =
      model->createData();

  // Generating random input vector
  const Eigen::VectorXd r = Eigen::VectorXd::Random(model->get_nr());
  data->a_value = nan("");

  // Getting the state dimension from calc() call
  model->calc(data, r);

  // Checking that calc returns a value
  BOOST_CHECK(!std::isnan(data->a_value));

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActivationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActivationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  const Eigen::VectorXf r_f = r.cast<float>();
  casted_data->a_value = float(nan(""));
  casted_model->calc(casted_data, r_f);
  BOOST_CHECK(!std::isnan(casted_data->a_value));
  BOOST_CHECK(std::abs(data->a_value - casted_data->a_value) < 1e-6);
}

void test_partial_derivatives_against_numdiff(
    ActivationModelTypes::Type activation_type) {
  // create the model
  ActivationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActivationModelAbstract>& model =
      factory.create(activation_type);

  // create the corresponding data object and set the cost to nan
  const std::shared_ptr<crocoddyl::ActivationDataAbstract>& data =
      model->createData();

  crocoddyl::ActivationModelNumDiff model_num_diff(model);
  std::shared_ptr<crocoddyl::ActivationDataAbstract> data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd r = Eigen::VectorXd::Random(model->get_nr());

  // Computing the activation derivatives
  model->calc(data, r);
  model->calcDiff(data, r);
  model_num_diff.calc(data_num_diff, r);
  model_num_diff.calcDiff(data_num_diff, r);

  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK(std::abs(data->a_value - data_num_diff->a_value) < tol);
  BOOST_CHECK(isCloseAbsRel(data->Ar, data_num_diff->Ar, tol, tol));
  BOOST_CHECK(
      (data->Arr.diagonal() - data_num_diff->Arr.diagonal()).isZero(tol));

  // Checking that casted computation is the same
  const std::shared_ptr<crocoddyl::ActivationModelAbstractTpl<float>>&
      casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::ActivationDataAbstractTpl<float>>&
      casted_data = casted_model->createData();
  const Eigen::VectorXf r_f = r.cast<float>();
  casted_model->calc(casted_data, r_f);
  casted_model->calcDiff(casted_data, r_f);
  float tol_f =
      std::pow(model_num_diff.cast<float>().get_disturbance(), float(1. / 3.));
  BOOST_CHECK(std::abs(data->a_value - casted_data->a_value) < tol_f);
  BOOST_CHECK(
      isCloseAbsRel(data->Ar.cast<float>(), casted_data->Ar, tol_f, tol_f));
}

void test_activation_is_zero_at_zero(
    ActivationModelTypes::Type activation_type) {
  ActivationModelFactory factory;
  const std::shared_ptr<crocoddyl::ActivationModelAbstract>& model =
      factory.create(activation_type);
  const Eigen::VectorXd r =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model->get_nr()));
  const std::shared_ptr<crocoddyl::ActivationDataAbstract>& data =
      model->createData();
  if (activation_type != ActivationModelTypes::ActivationModelSmooth2Norm &
      activation_type !=
          ActivationModelTypes::ActivationModelWeightedQuadraticBarrier &
      activation_type != ActivationModelTypes::ActivationModelQuadraticBarrier &
      activation_type != ActivationModelTypes::ActivationModel2NormBarrier) {
    model->calc(data, r);
    BOOST_CHECK_SMALL(data->a_value, std::numeric_limits<double>::epsilon());
  }
}

void check_activation_local_quadratic_response(
    crocoddyl::ActivationModelAbstract& model,
    crocoddyl::ActivationModelAbstract& quadratic_model) {
  const Eigen::VectorXd r =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.get_nr()));
  const std::shared_ptr<crocoddyl::ActivationDataAbstract>& data =
      model.createData();
  const std::shared_ptr<crocoddyl::ActivationDataAbstract>& quadratic_data =
      quadratic_model.createData();
  model.calc(data, r);
  model.calcDiff(data, r);
  quadratic_model.calc(quadratic_data, r);
  quadratic_model.calcDiff(quadratic_data, r);
  const Eigen::MatrixXd hessian =
      crocoddyl::ActivationDataAbstract::getHessianMatrix(*data);
  const Eigen::MatrixXd quadratic_hessian =
      crocoddyl::ActivationDataAbstract::getHessianMatrix(*quadratic_data);
  BOOST_CHECK(hessian.isApprox(quadratic_hessian));
}

void test_smooth_1norm_properties() {
  const double delta = 0.25;
  crocoddyl::ActivationModelSmooth1Norm model(3, delta);
  crocoddyl::ActivationModelSmooth1Norm large_delta_model(3, 100.);
  crocoddyl::ActivationModelQuad quadratic_model(3);
  BOOST_CHECK_EQUAL(model.get_delta(), delta);
  check_activation_local_quadratic_response(model, quadratic_model);
  check_activation_local_quadratic_response(large_delta_model, quadratic_model);
}

void test_weighted_smooth_1norm_properties() {
  Eigen::VectorXd weights(3);
  weights << 0.5, 2., 4.;
  const double delta = 0.25;
  crocoddyl::ActivationModelWeightedSmooth1Norm model(weights, delta);
  crocoddyl::ActivationModelWeightedSmooth1Norm large_delta_model(weights,
                                                                  100.);
  crocoddyl::ActivationModelWeightedQuad quadratic_model(weights);
  BOOST_CHECK_EQUAL(model.get_delta(), delta);
  check_activation_local_quadratic_response(model, quadratic_model);
  check_activation_local_quadratic_response(large_delta_model, quadratic_model);
}

void test_activation_bounds_with_infinity() {
  Eigen::VectorXd lb(1);
  Eigen::VectorXd ub(1);
  double beta;
  beta = 0.1;
  lb[0] = 0;
  ub[0] = std::numeric_limits<double>::infinity();

  Eigen::VectorXd m =
      0.5 * (lb + Eigen::VectorXd::Constant(
                      lb.size(), std::numeric_limits<double>::max()));
  Eigen::VectorXd d =
      0.5 * (Eigen::VectorXd::Constant(lb.size(),
                                       std::numeric_limits<double>::max()) -
             lb);
  crocoddyl::ActivationBounds bounds(lb, ub, beta);
  BOOST_CHECK(bounds.lb != m - beta * d);

  // Checking that casted computation is the same
  crocoddyl::ActivationBoundsTpl<float> casted_bounds = bounds.cast<float>();
  BOOST_CHECK(bounds.lb.cast<float>() == casted_bounds.lb);
}

//----------------------------------------------------------------------------//

void register_unit_tests(ActivationModelTypes::Type activation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << activation_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_construct_data, activation_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_returns_a_value, activation_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, activation_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_activation_is_zero_at_zero, activation_type)));
  framework::master_test_suite().add(ts);
}

bool register_bounds_unit_test() {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_"
            << "ActivationBoundsInfinity";
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_activation_bounds_with_infinity)));
  framework::master_test_suite().add(ts);
  return true;
}

bool register_smooth_1norm_unit_tests() {
  test_suite* smooth_ts =
      BOOST_TEST_SUITE("test_ActivationModelSmooth1NormProperties");
  smooth_ts->add(BOOST_TEST_CASE(&test_smooth_1norm_properties));
  framework::master_test_suite().add(smooth_ts);

  test_suite* ts =
      BOOST_TEST_SUITE("test_ActivationModelWeightedSmooth1NormProperties");
  ts->add(BOOST_TEST_CASE(&test_weighted_smooth_1norm_properties));
  framework::master_test_suite().add(ts);
  return true;
}

bool init_function() {
  for (size_t i = 0; i < ActivationModelTypes::all.size(); ++i) {
    register_unit_tests(ActivationModelTypes::all[i]);
  }
  register_bounds_unit_test();
  register_smooth_1norm_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
