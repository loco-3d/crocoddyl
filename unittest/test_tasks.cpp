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
#include <memory>
#include <sstream>

#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/task.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

const std::vector<StateModelTypes::Type> multibody_state_types = {
    StateModelTypes::StateMultibody_Hector,
    StateModelTypes::StateMultibody_TalosArm,
    StateModelTypes::StateMultibodyContact2D_TalosArm,
    StateModelTypes::StateMultibody_HyQ,
    StateModelTypes::StateMultibody_Talos,
    StateModelTypes::StateMultibody_RandomHumanoid};

std::string make_task_test_case_name(const char* test_kind,
                                     const TaskModelTypes::Type task_type,
                                     const StateModelTypes::Type state_type) {
  std::ostringstream oss;
  oss << test_kind << "/" << task_type << "/" << state_type;
  return oss.str();
}

void check_task_partial_derivatives_against_numdiff(
    const std::shared_ptr<crocoddyl::TaskModelAbstract>& model,
    const std::shared_ptr<crocoddyl::TaskDataAbstract>& data,
    const std::shared_ptr<crocoddyl::StateMultibody>& state,
    pinocchio::Model& pinocchio_model, pinocchio::Data& pinocchio_data,
    const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x,
                                          u);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  Eigen::MatrixXd Yx_fd =
      Eigen::MatrixXd::Zero(model->get_nr(), state->get_ndx());
  Eigen::MatrixXd Vx_fd =
      Eigen::MatrixXd::Zero(model->get_nr(), state->get_ndx());
  Eigen::MatrixXd Ax_fd =
      Eigen::MatrixXd::Zero(model->get_nr(), state->get_ndx());
  Eigen::MatrixXd Yu_fd =
      Eigen::MatrixXd::Zero(model->get_nr(), model->get_nu());
  Eigen::MatrixXd Vu_fd =
      Eigen::MatrixXd::Zero(model->get_nr(), model->get_nu());
  Eigen::MatrixXd Au_fd =
      Eigen::MatrixXd::Zero(model->get_nr(), model->get_nu());

  const Eigen::Index ndx = static_cast<Eigen::Index>(state->get_ndx());
  for (Eigen::Index i = 0; i < ndx; ++i) {
    const double step = 1e-7;
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(state->get_ndx());
    dx[i] = step;
    Eigen::VectorXd x_plus = state->zero();
    Eigen::VectorXd x_minus = state->zero();
    state->integrate(x, dx, x_plus);
    state->integrate(x, -dx, x_minus);
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x_plus, u);
    model->calc(data, x_plus, u);
    const Eigen::VectorXd y_plus = data->y;
    const Eigen::VectorXd v_plus = data->v;
    const Eigen::VectorXd a_plus = data->a;
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x_minus, u);
    model->calc(data, x_minus, u);
    const Eigen::VectorXd y_minus = data->y;
    const Eigen::VectorXd v_minus = data->v;
    const Eigen::VectorXd a_minus = data->a;
    Yx_fd.col(i) = (y_plus - y_minus) / (2.0 * step);
    Vx_fd.col(i) = (v_plus - v_minus) / (2.0 * step);
    Ax_fd.col(i) = (a_plus - a_minus) / (2.0 * step);
  }

  const Eigen::Index nu = static_cast<Eigen::Index>(model->get_nu());
  for (Eigen::Index i = 0; i < nu; ++i) {
    const double step = 1e-7 * (1.0 + std::abs(u[i]));
    Eigen::VectorXd u_plus = u;
    Eigen::VectorXd u_minus = u;
    u_plus[i] += step;
    u_minus[i] -= step;
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x, u_plus);
    model->calc(data, x, u_plus);
    const Eigen::VectorXd y_plus = data->y;
    const Eigen::VectorXd v_plus = data->v;
    const Eigen::VectorXd a_plus = data->a;
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x, u_minus);
    model->calc(data, x, u_minus);
    const Eigen::VectorXd y_minus = data->y;
    const Eigen::VectorXd v_minus = data->v;
    const Eigen::VectorXd a_minus = data->a;
    Yu_fd.col(i) = (y_plus - y_minus) / (2.0 * step);
    Vu_fd.col(i) = (v_plus - v_minus) / (2.0 * step);
    Au_fd.col(i) = (a_plus - a_minus) / (2.0 * step);
  }

  const double abs_tol = 5e-6;
  const double rel_tol = 5e-5;
  BOOST_CHECK(isCloseAbsRel(data->Yx, Yx_fd, abs_tol, rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Yu, Yu_fd, abs_tol, rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Vx, Vx_fd, abs_tol, rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Vu, Vu_fd, abs_tol, rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Ax, Ax_fd, abs_tol, rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Au, Au_fd, abs_tol, rel_tol));
}

void test_construct_data(const TaskModelTypes::Type task_type,
                         const StateModelTypes::Type state_type) {
  seedUnitTestRandomGenerators(
      getUnitTestSeed() + 1009u * (static_cast<unsigned int>(task_type) + 1u) +
      2003u * (static_cast<unsigned int>(state_type) + 1u) + 1u);
  TaskModelFactory factory;
  const std::shared_ptr<crocoddyl::TaskModelAbstract> model =
      factory.create(task_type, state_type);
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::TaskDataAbstract>& data =
      model->createData(&shared_data);

  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->y.size()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->v.size()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->a.size()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Yx.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Yx.cols()),
                    state->get_ndx());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Yu.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Yu.cols()), model->get_nu());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Vx.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Vx.cols()),
                    state->get_ndx());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Vu.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Vu.cols()), model->get_nu());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Ax.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Ax.cols()),
                    state->get_ndx());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Au.rows()), model->get_nr());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Au.cols()), model->get_nu());
  BOOST_CHECK(data->y.isZero());
  BOOST_CHECK(data->v.isZero());
  BOOST_CHECK(data->a.isZero());
  BOOST_CHECK(data->Yx.isZero());
  BOOST_CHECK(data->Yu.isZero());
  BOOST_CHECK(data->Vx.isZero());
  BOOST_CHECK(data->Vu.isZero());
  BOOST_CHECK(data->Ax.isZero());
  BOOST_CHECK(data->Au.isZero());
}

void test_calc_returns_a_value(const TaskModelTypes::Type task_type,
                               const StateModelTypes::Type state_type) {
  seedUnitTestRandomGenerators(
      getUnitTestSeed() + 1009u * (static_cast<unsigned int>(task_type) + 1u) +
      2003u * (static_cast<unsigned int>(state_type) + 1u) + 2u);
  TaskModelFactory factory;
  const std::shared_ptr<crocoddyl::TaskModelAbstract> model =
      factory.create(task_type, state_type);
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::TaskDataAbstract>& data =
      model->createData(&shared_data);

  const Eigen::VectorXd x = sampleUnitTestState(state, 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x,
                                          u);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  BOOST_CHECK(data->y.allFinite());
  BOOST_CHECK(data->v.allFinite());
  BOOST_CHECK(data->a.allFinite());
  BOOST_CHECK(data->Yx.allFinite());
  BOOST_CHECK(data->Yu.allFinite());
  BOOST_CHECK(data->Vx.allFinite());
  BOOST_CHECK(data->Vu.allFinite());
  BOOST_CHECK(data->Ax.allFinite());
  BOOST_CHECK(data->Au.allFinite());
}

void test_partial_derivatives_against_numdiff(
    const TaskModelTypes::Type task_type,
    const StateModelTypes::Type state_type) {
  seedUnitTestRandomGenerators(
      getUnitTestSeed() + 1009u * (static_cast<unsigned int>(task_type) + 1u) +
      2003u * (static_cast<unsigned int>(state_type) + 1u) + 3u);
  TaskModelFactory factory;
  const std::shared_ptr<crocoddyl::TaskModelAbstract> model =
      factory.create(task_type, state_type);
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::TaskDataAbstract>& data =
      model->createData(&shared_data);

  const Eigen::VectorXd x = sampleUnitTestState(state, 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  check_task_partial_derivatives_against_numdiff(
      model, data, state, pinocchio_model, pinocchio_data, x, u);
}

void register_unit_tests(const TaskModelTypes::Type task_type) {
  std::ostringstream test_name;
  test_name << "test_" << task_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  for (const auto state_type : multibody_state_types) {
    const std::string construct_name =
        make_task_test_case_name("construct_data", task_type, state_type);
    const std::string calc_name =
        make_task_test_case_name("calc_returns_a_value", task_type, state_type);
    const std::string numdiff_name = make_task_test_case_name(
        "partial_derivatives_against_numdiff", task_type, state_type);
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_construct_data, task_type, state_type),
        construct_name));
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_calc_returns_a_value, task_type, state_type),
        calc_name));
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_partial_derivatives_against_numdiff, task_type,
                    state_type),
        numdiff_name));
  }
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (const auto task_type : TaskModelTypes::all) {
    register_unit_tests(task_type);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
