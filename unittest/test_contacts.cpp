///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

#include "factory/contact.hpp"
#include "unittest_common.hpp"

using namespace crocoddyl::unittest;
using namespace boost::unit_test;

//----------------------------------------------------------------------------//

void test_construct_data(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create(contact_type, model_type);

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);
}

void test_calc_fetch_jacobians(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create(contact_type, model_type);

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = model->get_state()->get_pinocchio();
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

void test_calc_diff_fetch_derivatives(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create(contact_type, model_type);

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = model->get_state()->get_pinocchio();
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
  ContactModelFactory factory;
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create(contact_type, model_type);

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = model->get_state()->get_pinocchio();
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
  ContactModelFactory factory;
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create(contact_type, model_type);

  // create the corresponding data object
  const boost::shared_ptr<pinocchio::Model>& pinocchio_model = model->get_state()->get_pinocchio();
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

void test_partial_derivatives_against_numdiff(ContactModelTypes::Type contact_type,
                                              PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  boost::shared_ptr<crocoddyl::ContactModelAbstract> model = factory.create(contact_type, model_type);

  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model->get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  boost::shared_ptr<crocoddyl::ContactDataAbstract> data = model->createData(&pinocchio_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ContactModelNumDiff model_num_diff(model);
  const boost::shared_ptr<crocoddyl::ContactDataAbstract>& data_num_diff = model_num_diff.createData(&pinocchio_data);

  // Generating random values for the state
  const Eigen::VectorXd& x = model->get_state()->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::ContactModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateAllPinocchio, &pinocchio_model, &pinocchio_data, _1));
  model_num_diff.set_reevals(reevals);

  // Computing the contact derivatives
  model->calc(data, x);
  model->calcDiff(data, x);

  // Computing the contact derivatives via numerical differentiation
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);

  // Checking the partial derivatives against NumDiff
  double tol = NUMDIFF_MODIFIER * model_num_diff.get_disturbance();
  BOOST_CHECK((data->da0_dx - data_num_diff->da0_dx).isMuchSmallerThan(1.0, tol));
}

//----------------------------------------------------------------------------//

void register_contact_model_unit_tests(ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type,
                                       test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_fetch_jacobians, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_fetch_derivatives, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_update_force_diff, contact_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, contact_type, model_type)));
}

bool init_function() {
  for (size_t contact_type = 0; contact_type < ContactModelTypes::all.size(); ++contact_type) {
    for (size_t model_type = 0; model_type < PinocchioModelTypes::all.size(); ++model_type) {
      boost::test_tools::output_test_stream test_name;
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
