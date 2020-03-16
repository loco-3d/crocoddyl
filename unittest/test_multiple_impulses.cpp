///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "factory/impulse.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

/**
 * These methods modify the return type of the model function in
 * order to use the boost::execution_monitor::execute method which catch the
 * assert signal
 */
int calc(crocoddyl::ImpulseModelMultiple& model, boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
         Eigen::VectorXd& dx) {
  model.calc(data, dx);
  return 0;
}

int calcDiff(crocoddyl::ImpulseModelMultiple& model, boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
             Eigen::VectorXd& dx) {
  model.calcDiff(data, dx);
  return 0;
}

int updateForce(crocoddyl::ImpulseModelMultiple& model, boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                Eigen::VectorXd& dx) {
  model.updateForce(data, dx);
  return 0;
}

int updateVelocityDiff(crocoddyl::ImpulseModelMultiple& model, boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                       const Eigen::MatrixXd& dvnext_dx) {
  model.updateVelocityDiff(data, dvnext_dx);
  return 0;
}

int updateForceDiff(crocoddyl::ImpulseModelMultiple& model, boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                    const Eigen::MatrixXd& df_dq) {
  model.updateForceDiff(data, df_dq);
  return 0;
}

//----------------------------------------------------------------------------//

void test_constructor() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));

  // Test the initial size of the map
  BOOST_CHECK(model.get_impulses().size() == 0);
}

void test_addImpulse() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));

  // create and impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // add an impulse object to the container
  model.addImpulse("random_impulse", impulse_factories[0]->create());

  // Test the final size of the map
  BOOST_CHECK(model.get_impulses().size() == 1);
}

void test_addImpulse_error_message() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));

  // create an impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // add twice the same impulse object to the container
  model.addImpulse("random_impulse", impulse_factories[0]->create());

  // Expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addImpulse("random_impulse", impulse_factories[0]->create());
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this impulse item already existed, we cannot add it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeImpulse() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));

  // create and impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // add an impulse object to the container
  model.addImpulse("random_impulse", impulse_factories[0]->create());

  // add an impulse object to the container
  model.removeImpulse("random_impulse");

  // Test the final size of the map
  BOOST_CHECK(model.get_impulses().size() == 0);
}

void test_removeImpulse_error_message() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));

  // create and impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // remove a none existing impulse form the container, we expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeImpulse("random_impulse");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this impulse item doesn't exist, we cannot remove it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse models fetch it.
  Eigen::VectorXd x = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calc(data, x);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_no_computation() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calc(data, dx);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.hasNaN() || data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse models fetch it.
  Eigen::VectorXd x = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // pinocchio data have been filled so the results of this operation are
  // none null matrices
  model.calc(data, x);
  model.calcDiff(data, x);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_no_recalc() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse models fetch it.
  Eigen::VectorXd x = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calcDiff(data, x);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_no_computation() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calc(data, dx);
  model.calcDiff(data, dx);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.hasNaN() || data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_updateForce() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // create random forces
  Eigen::VectorXd forces = Eigen::VectorXd::Random(model.get_ni());

  // update forces
  model.updateForce(data, forces);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  crocoddyl::ImpulseModelMultiple::ImpulseDataContainer::iterator it_d, end_d;
  for (it_d = data->impulses.begin(), end_d = data->impulses.end(); it_d != end_d; ++it_d) {
    BOOST_CHECK(!it_d->second->f.toVector().isZero());
  }
  BOOST_CHECK(data->df_dq.isZero());
}

void test_updateVelocityDiff() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // create the velocity diff
  Eigen::MatrixXd dvnext_dx =
      Eigen::MatrixXd::Random(state_factory.create()->get_nv(), state_factory.create()->get_ndx());

  // call the update
  model.updateVelocityDiff(data, dvnext_dx);

  // Test
  BOOST_CHECK((data->dvnext_dx - dvnext_dx).isMuchSmallerThan(1.0, 1e-9));
}

void test_updateForceDiff() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data = model.createData(&pinocchio_data);

  // create force diff
  Eigen::MatrixXd df_dq = Eigen::MatrixXd::Random(model.get_ni(), state_factory.create()->get_nv());

  // call update force diff
  model.updateForceDiff(data, df_dq);

  // Test
  crocoddyl::ImpulseModelMultiple::ImpulseDataContainer::iterator it_d, end_d;
  for (it_d = data->impulses.begin(), end_d = data->impulses.end(); it_d != end_d; ++it_d) {
    BOOST_CHECK(!it_d->second->df_dq.isZero());
  }
}

void test_assert_updateForceDiff_assert_mismatch_model_data() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model1(
      boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  crocoddyl::ImpulseModelMultiple model2(
      boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model1.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    {
      std::ostringstream os;
      os << "random_impulse1_" << i;
      model1.addImpulse(os.str(), impulse_factories.back()->create());
    }
    {
      std::ostringstream os;
      os << "random_impulse2_" << i;
      model2.addImpulse(os.str(), impulse_factories.back()->create());
    }
  }

  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data1 = model1.createData(&pinocchio_data);
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data2 = model2.createData(&pinocchio_data);

  // create force diff
  Eigen::MatrixXd df_dq = Eigen::MatrixXd::Random(model1.get_ni(), state_factory.create()->get_nv());

  // call that trigger assert
  std::string error_message = GetErrorMessages(boost::bind(&updateForceDiff, model1, data2, df_dq));

  // expected error message content
  std::string function_name =
      "void crocoddyl::ImpulseModelMultiple::updateForceDiff("
      "const boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&,"
      " const MatrixXd&) const";
  std::string assert_argument =
      "it_m->first == it_d->first && \"it doesn't match"
      " the impulse name between data and model\"";

  // Perform the checks
#ifndef __APPLE__
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
#endif
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_create() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));

  // Test
  BOOST_CHECK(state_factory.create() == model.get_state());
}

void test_get_impulses() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // get the impulses
  const crocoddyl::ImpulseModelMultiple::ImpulseModelContainer& impulses = model.get_impulses();

  // test
  crocoddyl::ImpulseModelMultiple::ImpulseModelContainer::const_iterator it_m, end_m;
  unsigned i;
  for (i = 0, it_m = impulses.begin(), end_m = impulses.end(); it_m != end_m; ++it_m, ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_get_ni() {
  // Setup the test
  StateModelFactory state_factory(StateModelTypes::StateMultibody, PinocchioModelTypes::RandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->create());
  }

  // compute ni
  std::size_t ni = 0;
  crocoddyl::ImpulseModelMultiple::ImpulseModelContainer::const_iterator it_m, end_m;
  for (it_m = model.get_impulses().begin(), end_m = model.get_impulses().end(); it_m != end_m; ++it_m) {
    ni += it_m->second->impulse->get_ni();
  }

  BOOST_CHECK(ni == model.get_ni());
}

//----------------------------------------------------------------------------//

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addImpulse)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addImpulse_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeImpulse)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeImpulse_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_no_computation)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_computation)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateForce)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateVelocityDiff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_create)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_impulses)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_ni)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
