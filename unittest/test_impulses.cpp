///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "impulses_factory.hpp"
#include "unittest_common.hpp"

using namespace crocoddyl_unit_test;
using namespace boost::unit_test;

//----------------------------------------------------------------------------//

void test_construct_data(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(factory.get_state_factory()->get_pinocchio_model());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);
}

void test_calc_no_computation(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(factory.get_state_factory()->get_pinocchio_model());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_fetch_jacobians(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const pinocchio::Model& pinocchio_model = factory.get_state_factory()->get_pinocchio_model();
  pinocchio::Data pinocchio_data(pinocchio_model);
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd q = model->get_state()->rand().segment(0, model->get_state()->get_nq());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_no_computation(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(factory.get_state_factory()->get_pinocchio_model());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);
  model->calcDiff(data, dx);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.hasNaN() || data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_fetch_derivatives(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const pinocchio::Model& pinocchio_model = factory.get_state_factory()->get_pinocchio_model();
  pinocchio::Data pinocchio_data(pinocchio_model);
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd q = model->get_state()->rand().segment(0, model->get_state()->get_nq());
  Eigen::VectorXd v = Eigen::VectorXd::Random(model->get_state()->get_nv());
  Eigen::VectorXd a = Eigen::VectorXd::Random(model->get_state()->get_nv());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);
  pinocchio::computeForwardKinematicsDerivatives(pinocchio_model, pinocchio_data, q, v, a);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);
  model->calcDiff(data, dx);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_update_force(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const pinocchio::Model& pinocchio_model = factory.get_state_factory()->get_pinocchio_model();
  pinocchio::Data pinocchio_data(pinocchio_model);
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::VectorXd f = Eigen::VectorXd::Random(data->Jc.rows());
  model->updateForce(data, f);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(!data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_update_force_diff(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const pinocchio::Model& pinocchio_model = factory.get_state_factory()->get_pinocchio_model();
  pinocchio::Data pinocchio_data(pinocchio_model);
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::MatrixXd df_dq = Eigen::MatrixXd::Random(data->df_dq.rows(), data->df_dq.cols());
  model->updateForceDiff(data, df_dq);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(!data->df_dq.isZero());
}

//----------------------------------------------------------------------------//

void register_unit_tests(ImpulseModelTypes::Type test_type, test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, test_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_no_computation, test_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_fetch_jacobians, test_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_computation, test_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_fetch_derivatives, test_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force, test_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force_diff, test_type)));
}

bool init_function() {
  for (size_t i = 0; i < ImpulseModelTypes::all.size(); ++i) {
    const std::string test_name = "test_" + std::to_string(i);
    test_suite* ts = BOOST_TEST_SUITE(test_name);
    register_unit_tests(ImpulseModelTypes::all[i], *ts);
    framework::master_test_suite().add(ts);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
