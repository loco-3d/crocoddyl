///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          INRIA, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "factory/impulse.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_construct_data(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*factory.get_pinocchio_model().get());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);
}

void test_calc_no_computation(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*factory.get_pinocchio_model().get());
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

void test_calc_fetch_jacobians(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(), &pinocchio_data, x);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_no_computation(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*factory.get_pinocchio_model().get());
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

void test_calc_diff_fetch_derivatives(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(), &pinocchio_data, x);

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

void test_update_force(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::VectorXd f = Eigen::VectorXd::Random(data->Jc.rows());
  model->updateForce(data, f);
  boost::shared_ptr<crocoddyl::ImpulseModel3D> m = boost::static_pointer_cast<crocoddyl::ImpulseModel3D>(model);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(!data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_update_force_diff(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory(impulse_type, model_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
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

void register_impulse_model_unit_tests(ImpulseModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type,
                                       test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, impulse_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_no_computation, impulse_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_fetch_jacobians, impulse_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_computation, impulse_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_fetch_derivatives, impulse_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force, impulse_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force_diff, impulse_type, model_type)));
}

bool init_function() {
  for (size_t impulse_type = 0; impulse_type < ImpulseModelTypes::all.size(); ++impulse_type) {
    for (size_t model_type = 0; model_type < PinocchioModelTypes::all.size(); ++model_type) {
      std::ostringstream test_name;
      test_name << "test_" << ImpulseModelTypes::all[impulse_type] << "_" << PinocchioModelTypes::all[model_type];
      test_suite* ts = BOOST_TEST_SUITE(test_name.str());
      std::cout << "Running " << test_name.str() << std::endl;
      register_impulse_model_unit_tests(ImpulseModelTypes::all[impulse_type], PinocchioModelTypes::all[model_type],
                                        *ts);
      framework::master_test_suite().add(ts);
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
