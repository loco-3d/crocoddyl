///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>

#include "factory/guidance.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

Eigen::VectorXd evaluate_rate(const crocoddyl::GuidanceModelAbstract& model,
                              const Eigen::VectorXd& error) {
  const std::shared_ptr<crocoddyl::GuidanceDataAbstract>& data =
      model.createData();
  model.calc(data, error);
  return data->g;
}

void check_casted_guidance_results(
    const std::shared_ptr<crocoddyl::GuidanceModelAbstract>& model,
    const std::shared_ptr<crocoddyl::GuidanceDataAbstract>& data,
    const Eigen::VectorXd& error) {
#ifdef NDEBUG
  const auto casted_model = model->cast<float>();
  const auto casted_data = casted_model->createData();
  const Eigen::VectorXf error_f = error.cast<float>();
  casted_model->calc(casted_data, error_f);
  casted_model->calcDiff(casted_data, error_f);

  const float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(isCloseAbsRel(data->g, casted_data->g, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Ge, casted_data->Ge, tol_f, tol_f));
#else
  (void)model;
  (void)data;
  (void)error;
#endif
}

void test_construct_data(const GuidanceModelTypes::Type guidance_type) {
  GuidanceModelFactory factory;
  const std::shared_ptr<crocoddyl::GuidanceModelAbstract>& model =
      factory.create(guidance_type);

  const std::shared_ptr<crocoddyl::GuidanceDataAbstract>& data =
      model->createData();
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->g.size()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Ge.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Ge.cols()), model->get_nr());
  BOOST_CHECK(data->g.isZero());
  BOOST_CHECK(data->Ge.isZero());
}

void test_calc_returns_a_value(const GuidanceModelTypes::Type guidance_type) {
  GuidanceModelFactory factory;
  const std::shared_ptr<crocoddyl::GuidanceModelAbstract>& model =
      factory.create(guidance_type);
  const std::shared_ptr<crocoddyl::GuidanceDataAbstract>& data =
      model->createData();

  const Eigen::VectorXd error =
      Eigen::VectorXd::Random(static_cast<Eigen::Index>(model->get_nr()));
  model->calc(data, error);
  model->calcDiff(data, error);

  BOOST_CHECK(data->g.allFinite());
  BOOST_CHECK(data->Ge.allFinite());
  check_casted_guidance_results(model, data, error);
}

void test_partial_derivatives_against_numdiff(
    const GuidanceModelTypes::Type guidance_type) {
  GuidanceModelFactory factory;
  const std::shared_ptr<crocoddyl::GuidanceModelAbstract>& model =
      factory.create(guidance_type);
  const std::shared_ptr<crocoddyl::GuidanceDataAbstract>& data =
      model->createData();

  const Eigen::VectorXd error =
      Eigen::VectorXd::Random(static_cast<Eigen::Index>(model->get_nr()));
  model->calc(data, error);
  model->calcDiff(data, error);

  Eigen::MatrixXd jacobian_fd =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(model->get_nr()),
                            static_cast<Eigen::Index>(model->get_nr()));
  for (Eigen::Index i = 0; i < error.size(); ++i) {
    const double step = 1e-7 * (1.0 + std::abs(error[i]));
    Eigen::VectorXd error_plus = error;
    Eigen::VectorXd error_minus = error;
    error_plus[i] += step;
    error_minus[i] -= step;
    jacobian_fd.col(i) = (evaluate_rate(*model, error_plus) -
                          evaluate_rate(*model, error_minus)) /
                         (2.0 * step);
  }

  const double max_error = (data->Ge - jacobian_fd).cwiseAbs().maxCoeff();
  BOOST_CHECK_LT(max_error, 5e-6);
  check_casted_guidance_results(model, data, error);
}

void register_unit_tests(const GuidanceModelTypes::Type guidance_type) {
  std::ostringstream test_name;
  test_name << "test_" << guidance_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_construct_data, guidance_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_value, guidance_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, guidance_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (const auto guidance_type : GuidanceModelTypes::all) {
    register_unit_tests(guidance_type);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
