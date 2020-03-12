///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "factory/contact.hpp"
#include "unittest_common.hpp"

using namespace crocoddyl::unittest;
using namespace boost::unit_test;

//----------------------------------------------------------------------------//

void test_construct_data(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*factory.get_pinocchio_model().get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);
}

void test_calc_no_computation(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*factory.get_pinocchio_model().get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Getting the jacobian from the model
  Eigen::VectorXd x;
  model->calc(data, x);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());
}

void test_calc_fetch_jacobians(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(), &pinocchio_data, x);

  // Getting the jacobian from the model
  model->calc(data, x);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());
}

void test_calc_diff_no_computation(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*factory.get_pinocchio_model().get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Getting the jacobian from the model
  Eigen::VectorXd x;
  model->calc(data, x);
  model->calcDiff(data, x);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());
}

void test_calc_diff_fetch_derivatives(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(), &pinocchio_data, x);

  // Getting the jacobian from the model
  model->calc(data, x);
  model->calcDiff(data, x);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->a0.isZero());
  BOOST_CHECK(!data->da0_dx.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());
}

void test_update_force(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::VectorXd f = Eigen::VectorXd::Random(data->Jc.rows());
  model->updateForce(data, f);
  boost::shared_ptr<crocoddyl::ContactModel3D> m = boost::static_pointer_cast<crocoddyl::ContactModel3D>(model);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(!data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());
}

void test_update_force_diff(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory(contact_type, model_type);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create();

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::MatrixXd df_dx = Eigen::MatrixXd::Random(data->df_dx.rows(), data->df_dx.cols());
  Eigen::MatrixXd df_du = Eigen::MatrixXd::Random(data->df_du.rows(), data->df_du.cols());
  model->updateForceDiff(data, df_dx, df_du);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(!data->df_dx.isZero());
  BOOST_CHECK(!data->df_du.isZero());
}

//----------------------------------------------------------------------------//

void register_contact_model_unit_tests(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type,
                                       test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_no_computation, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_fetch_jacobians, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_computation, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_fetch_derivatives, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force_diff, contact_type, model_type)));
}

bool init_function() {
  for (size_t contact_type = 0; contact_type < ContactModelTypes::all.size(); ++contact_type) {
    for (size_t model_type = 0; model_type < PinocchioModelTypes::all.size(); ++model_type) {
      std::ostringstream test_name;
      test_name << "test_" << ContactModelTypes::all[contact_type] << "_" << PinocchioModelTypes::all[model_type];
      test_suite* ts = BOOST_TEST_SUITE(test_name.str());
      std::cout << "Running " << test_name.str() << std::endl;
      register_contact_model_unit_tests(ContactModelTypes::all[contact_type], PinocchioModelTypes::all[model_type],
                                        *ts);
      framework::master_test_suite().add(ts);
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
