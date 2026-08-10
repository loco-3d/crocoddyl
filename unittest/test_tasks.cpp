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

#include "crocoddyl/multibody/actuations/full.hpp"
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

template <typename Scalar>
struct TaskDataCollectorTpl : crocoddyl::DataCollectorMultibodyTpl<Scalar>,
                              crocoddyl::DataCollectorJointTpl<Scalar> {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ActuationModelFullTpl<Scalar> Actuation;
  typedef crocoddyl::JointDataAbstractTpl<Scalar> JointData;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

  TaskDataCollectorTpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                       const std::shared_ptr<State>& state,
                       const std::size_t nu)
      : crocoddyl::DataCollectorMultibodyTpl<Scalar>(pinocchio),
        crocoddyl::DataCollectorJointTpl<Scalar>(std::make_shared<JointData>(
            state, std::make_shared<Actuation>(state), nu)) {
    if (nu == state->get_nv()) {
      this->joint->da_du.diagonal().setOnes();
    }
  }

  const VectorXs& set_acceleration(const VectorXs& a) {
    this->joint->a.setZero();
    if (a.size() == this->joint->a.size()) {
      this->joint->a = a;
    }
    return this->joint->a;
  }
};

typedef TaskDataCollectorTpl<double> TaskDataCollector;

std::string make_task_test_case_name(const char* test_kind,
                                     const TaskModelTypes::Type task_type,
                                     const StateModelTypes::Type state_type) {
  std::ostringstream oss;
  oss << test_kind << "/" << task_type << "/" << state_type;
  return oss.str();
}

void check_casted_task_results(
    const std::shared_ptr<crocoddyl::TaskModelAbstract>& model,
    const std::shared_ptr<crocoddyl::TaskDataAbstract>& data,
    const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
#ifdef NDEBUG
  const auto casted_model = model->cast<float>();
  const auto casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& pinocchio_model_f =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> pinocchio_data_f(pinocchio_model_f);
  TaskDataCollectorTpl<float> casted_shared_data(
      &pinocchio_data_f, casted_state, casted_model->get_nu());
  const auto casted_data = casted_model->createData(&casted_shared_data);
  casted_data->compute_acceleration = data->compute_acceleration;

  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(
      &pinocchio_model_f, &pinocchio_data_f, x_f, u_f,
      casted_shared_data.set_acceleration(u_f));
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);

  const float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(isCloseAbsRel(data->y, casted_data->y, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->v, casted_data->v, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->a, casted_data->a, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Yx, casted_data->Yx, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Yu, casted_data->Yu, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Vx, casted_data->Vx, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Vu, casted_data->Vu, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Ax, casted_data->Ax, tol_f, tol_f));
  BOOST_CHECK(isCloseAbsRel(data->Au, casted_data->Au, tol_f, tol_f));
#else
  (void)model;
  (void)data;
  (void)x;
  (void)u;
#endif
}

void check_task_partial_derivatives_against_numdiff(
    const std::shared_ptr<crocoddyl::TaskModelAbstract>& model,
    const std::shared_ptr<crocoddyl::TaskDataAbstract>& data,
    const std::shared_ptr<crocoddyl::StateMultibody>& state,
    pinocchio::Model& pinocchio_model, pinocchio::Data& pinocchio_data,
    TaskDataCollector* const shared_data, const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x,
                                          u, shared_data->set_acceleration(u));
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  check_casted_task_results(model, data, x, u);

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
                                            x_plus, u,
                                            shared_data->set_acceleration(u));
    model->calc(data, x_plus, u);
    const Eigen::VectorXd y_plus = data->y;
    const Eigen::VectorXd v_plus = data->v;
    const Eigen::VectorXd a_plus = data->a;
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x_minus, u,
                                            shared_data->set_acceleration(u));
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
    crocoddyl::unittest::updateAllPinocchio(
        &pinocchio_model, &pinocchio_data, x, u_plus,
        shared_data->set_acceleration(u_plus));
    model->calc(data, x, u_plus);
    const Eigen::VectorXd y_plus = data->y;
    const Eigen::VectorXd v_plus = data->v;
    const Eigen::VectorXd a_plus = data->a;
    crocoddyl::unittest::updateAllPinocchio(
        &pinocchio_model, &pinocchio_data, x, u_minus,
        shared_data->set_acceleration(u_minus));
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
  TaskDataCollector shared_data(&pinocchio_data, state, model->get_nu());
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
  BOOST_CHECK_EQUAL(data->compute_acceleration, model->get_has_acceleration());
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
  TaskDataCollector shared_data(&pinocchio_data, state, model->get_nu());
  const std::shared_ptr<crocoddyl::TaskDataAbstract>& data =
      model->createData(&shared_data);

  const Eigen::VectorXd x = sampleUnitTestState(state, 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x,
                                          u, shared_data.set_acceleration(u));
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
  check_casted_task_results(model, data, x, u);
}

void test_calc_without_acceleration(const TaskModelTypes::Type task_type,
                                    const StateModelTypes::Type state_type) {
  seedUnitTestRandomGenerators(
      getUnitTestSeed() + 1009u * (static_cast<unsigned int>(task_type) + 1u) +
      2003u * (static_cast<unsigned int>(state_type) + 1u) + 4u);
  TaskModelFactory factory;
  const std::shared_ptr<crocoddyl::TaskModelAbstract> model =
      factory.create(task_type, state_type);
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  TaskDataCollector shared_data(&pinocchio_data, state, model->get_nu());
  const std::shared_ptr<crocoddyl::TaskDataAbstract>& data =
      model->createData(&shared_data);

  data->compute_acceleration = false;

  const Eigen::VectorXd x = sampleUnitTestState(state, 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x,
                                          u, shared_data.set_acceleration(u));
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  BOOST_CHECK(!data->compute_acceleration);
  BOOST_CHECK(data->a.isZero());
  BOOST_CHECK(data->Ax.isZero());
  BOOST_CHECK(data->Au.isZero());

  check_task_partial_derivatives_against_numdiff(
      model, data, state, pinocchio_model, pinocchio_data, &shared_data, x, u);
}

void test_calc_without_joint_data(const TaskModelTypes::Type task_type,
                                  const StateModelTypes::Type state_type) {
  seedUnitTestRandomGenerators(
      getUnitTestSeed() + 1009u * (static_cast<unsigned int>(task_type) + 1u) +
      2003u * (static_cast<unsigned int>(state_type) + 1u) + 5u);
  TaskModelFactory factory;
  const std::shared_ptr<crocoddyl::TaskModelAbstract> model =
      factory.create(task_type, state_type);
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  TaskDataCollector shared_data_with_joint(&pinocchio_data, state,
                                           model->get_nu());
  crocoddyl::DataCollectorMultibody shared_data_without_joint(&pinocchio_data);
  const std::shared_ptr<crocoddyl::TaskDataAbstract> data_with_joint =
      model->createData(&shared_data_with_joint);
  const std::shared_ptr<crocoddyl::TaskDataAbstract> data_without_joint =
      model->createData(&shared_data_without_joint);

  const Eigen::VectorXd x = sampleUnitTestState(state, 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  const Eigen::VectorXd a = random_vector<double>(state->get_nv()) * 2.5e-1;

  // Compare fixed-acceleration partial derivatives. The action-model chain
  // rule is intentionally disabled in the collector used as the reference.
  shared_data_with_joint.joint->da_dx.setZero();
  shared_data_with_joint.joint->da_du.setZero();
  crocoddyl::unittest::updateAllPinocchio(
      &pinocchio_model, &pinocchio_data, x, u,
      shared_data_with_joint.set_acceleration(a));
  model->calc(data_with_joint, x, u);
  model->calcDiff(data_with_joint, x, u);

  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x,
                                          u, a);
  model->calc(data_without_joint, x, u);
  model->calcDiff(data_without_joint, x, u);

  const double tol = 10. * std::numeric_limits<double>::epsilon();
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->y, data_with_joint->y, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->v, data_with_joint->v, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->a, data_with_joint->a, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->Yx, data_with_joint->Yx, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->Yu, data_with_joint->Yu, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->Vx, data_with_joint->Vx, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->Vu, data_with_joint->Vu, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->Ax, data_with_joint->Ax, tol, tol));
  BOOST_CHECK(
      isCloseAbsRel(data_without_joint->Au, data_with_joint->Au, tol, tol));
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
  TaskDataCollector shared_data(&pinocchio_data, state, model->get_nu());
  const std::shared_ptr<crocoddyl::TaskDataAbstract>& data =
      model->createData(&shared_data);

  const Eigen::VectorXd x = sampleUnitTestState(state, 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  check_task_partial_derivatives_against_numdiff(
      model, data, state, pinocchio_model, pinocchio_data, &shared_data, x, u);
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
    const std::string noaccel_name = make_task_test_case_name(
        "calc_without_acceleration", task_type, state_type);
    const std::string nojoint_name = make_task_test_case_name(
        "calc_without_joint_data", task_type, state_type);
    const std::string numdiff_name = make_task_test_case_name(
        "partial_derivatives_against_numdiff", task_type, state_type);
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_construct_data, task_type, state_type),
        construct_name));
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_calc_returns_a_value, task_type, state_type),
        calc_name));
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_calc_without_acceleration, task_type, state_type),
        noaccel_name));
    ts->add(BOOST_TEST_CASE_NAME(
        boost::bind(&test_calc_without_joint_data, task_type, state_type),
        nojoint_name));
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
