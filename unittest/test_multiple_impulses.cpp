///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "impulses_factory.hpp"
#include "unittest_common.hpp"

using namespace crocoddyl_unit_test;
using namespace boost::unit_test;

//----------------------------------------------------------------------------//

void test_constructor(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  
  // Test the initial size of the map
  BOOST_CHECK(model.get_impulses().size() == 0);
}

void test_addImpulse() {
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  
  // create and impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // add an impulse object to the container
  model.addImpulse("random_impulse", impulse_factories[0]->get_model());

  // Test the final size of the map
  BOOST_CHECK(model.get_impulses().size() == 1);
}

void test_addImpulse_error_message(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  
  // create an impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // add twice the same impulse object to the container
  model.addImpulse("random_impulse", impulse_factories[0]->get_model());

  // Expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addImpulse("random_impulse", impulse_factories[0]->get_model());
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this impulse item already existed, we cannot add it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeImpulse() {
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));

  // create and impulse object
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  impulse_factories.push_back(create_random_factory());

  // add an impulse object to the container
  model.addImpulse("random_impulse", impulse_factories[0]->get_model());

  // add an impulse object to the container
  model.removeImpulse("random_impulse");

  // Test the final size of the map
  BOOST_CHECK(model.get_impulses().size() == 0);
}

void test_removeImpulse_error_message() {
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));

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

void test_calc(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Model pinocchio_model = state_factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse models fetch it.
  Eigen::VectorXd q = model.get_state()->rand().segment(0, model.get_state()->get_nq());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calc(data, dx);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_no_computation() {

  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

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

/**
 * @brief This method basically modify the return type of the calc function in
 * order to use the boost::execution_monitor::execute method which catch the
 * assert signal
 */
int calc(crocoddyl::ImpulseModelMultiple& model,
                         boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                         Eigen::VectorXd& dx){
  model.calc(data, dx);
  return 0;
}

void test_calc_wrong_data_size() {

  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // Empty the vector of impusle data.
  data->impulses.clear();

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // run the code monitoring the errors and grabing the iostreams.
  std::string error_message = GetErrorMessages(
    boost::bind(&calc, model, data, dx));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::calc(const"
                              " boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const Eigen::Ref<const Eigen::Matrix<double, -1, 1> >&)";
  std::string assert_argument = "static_cast<std::size_t>(data->impulses.size()) == "
                                "impulses_.size() && \"it doesn't match the number of "
                                "impulse datas and models\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_calc_mismatch_model_data() {

  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model1(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  crocoddyl::ImpulseModelMultiple model2(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    {
      std::ostringstream os;
      os << "random_impulse1_" << i;
      model1.addImpulse(os.str(), impulse_factories.back()->get_model());
    }
    {
      std::ostringstream os;
      os << "random_impulse2_" << i;
      model2.addImpulse(os.str(), impulse_factories.back()->get_model());
    }
  }
  
  // create the data of the multiple-impulses with the second model
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data2 =
    model2.createData(&pinocchio_data);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // run the code monitoring the errors and grabing the iostreams.
  std::string error_message = GetErrorMessages(
    boost::bind(&calc, model1, data2, dx));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::calc(const"
                              " boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const Eigen::Ref<const Eigen::Matrix<double, -1, 1> >&)";
  std::string assert_argument = "it_m->first == it_d->first && \"it doesn't match "
                                "the impulse name between data and model\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_calc_diff(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Model pinocchio_model = state_factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse models fetch it.
  Eigen::VectorXd q = model.get_state()->rand().segment(0, model.get_state()->get_nq());
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.get_state()->get_nv());
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.get_state()->get_nv());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);
  pinocchio::computeForwardKinematicsDerivatives(pinocchio_model, pinocchio_data, q, v, a);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // pinocchio data have been filled so the results of this operation are
  // none null matrices
  model.calcDiff(data, dx, true);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_no_recalc(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Model pinocchio_model = state_factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse models fetch it.
  Eigen::VectorXd q = model.get_state()->rand().segment(0, model.get_state()->get_nq());
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.get_state()->get_nv());
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.get_state()->get_nv());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);
  pinocchio::computeForwardKinematicsDerivatives(pinocchio_model, pinocchio_data, q, v, a);


  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calcDiff(data, dx, false);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

void test_calc_diff_no_computation() {

  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // pinocchio data have not been filled so the results of this operation are
  // null matrices
  model.calcDiff(data, dx, true);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.hasNaN() || data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.hasNaN() || data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dq.isZero());
}

/**
 * @brief This method basically modify the return type of the calc_diff function in
 * order to use the boost::execution_monitor::execute method which catch the
 * assert signal
 */
int calcDiff(crocoddyl::ImpulseModelMultiple& model,
                         boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                         Eigen::VectorXd& dx, bool recalc){
  model.calcDiff(data, dx, recalc);
  return 0;
}

void test_calc_diff_wrong_data_size() {

  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // Empty the vector of impusle data.
  data->impulses.clear();

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // run the code monitoring the errors and grabing the iostreams.
  std::string error_message = GetErrorMessages(
    boost::bind(&calcDiff, model, data, dx, false));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::calcDiff(const"
                              " boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const Eigen::Ref<const Eigen::Matrix<double, -1, 1> >&,"
                              " const bool&)";
  std::string assert_argument = "static_cast<std::size_t>(data->impulses.size()) == "
                                "impulses_.size() && \"it doesn't match the number of "
                                "impulse datas and models\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_calc_diff_mismatch_model_data() {

  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model1(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  crocoddyl::ImpulseModelMultiple model2(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    {
      std::ostringstream os;
      os << "random_impulse1_" << i;
      model1.addImpulse(os.str(), impulse_factories.back()->get_model());
    }
    {
      std::ostringstream os;
      os << "random_impulse2_" << i;
      model2.addImpulse(os.str(), impulse_factories.back()->get_model());
    }
  }
  
  // create the data of the multiple-impulses with the second model
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data2 =
    model2.createData(&pinocchio_data);

  // create a dummy state vector (not used for the impulses)
  Eigen::VectorXd dx;

  // run the code monitoring the errors and grabing the iostreams.
  std::string error_message = GetErrorMessages(
    boost::bind(&calcDiff, model1, data2, dx, false));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::calcDiff(const"
                              " boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const Eigen::Ref<const Eigen::Matrix<double, -1, 1> >&,"
                              " const bool&)";
  std::string assert_argument = "it_m->first == it_d->first && \"it doesn't match "
                                "the impulse name between data and model\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}


void test_updateForce() {
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  const pinocchio::Model& pinocchio_model = state_factory.get_pinocchio_model();
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd q = model.get_state()->rand().segment(0, model.get_state()->get_nq());
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.get_state()->get_nv());
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.get_state()->get_nv());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);
  pinocchio::computeForwardKinematicsDerivatives(pinocchio_model, pinocchio_data, q, v, a);

  // create random forces
  Eigen::VectorXd forces = Eigen::VectorXd::Random(model.get_ni());

  // update forces
  model.updateForce(data, forces);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  crocoddyl::ImpulseModelMultiple::ImpulseDataContainer::iterator it_d, end_d;
  for (it_d = data->impulses.begin(), end_d = data->impulses.end() ; it_d != end_d ; ++it_d) {
    BOOST_CHECK(!it_d->second->f.toVector().isZero());
  }
  BOOST_CHECK(data->df_dq.isZero());
}

/**
 * @brief This method basically modify the return type of the updateForce 
 * function in order to use the boost::execution_monitor::execute method which 
 * catch the assert signal
 */
int updateForce(crocoddyl::ImpulseModelMultiple& model,
                boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                Eigen::VectorXd& dx){
  model.updateForce(data, dx);
  return 0;
}


void test_updateForce_assert_force_size(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // create empty force vector
  Eigen::VectorXd forces;

  // update forces
  std::string error_message = GetErrorMessages(
    boost::bind(&updateForce, model, data, forces));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::updateForce"
                              "(const boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const VectorXd&";
  std::string assert_argument = "static_cast<std::size_t>(force.size()) == ni_ && "
                                "\"force has wrong dimension, it should be ni vector\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_updateForce_assert_wrong_data_size(){
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), impulse_factories.back()->get_model());
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
    model.createData(&pinocchio_data);

  // create random forces
  Eigen::VectorXd forces = Eigen::VectorXd::Random(model.get_ni());

  // clear the data container
  data->impulses.clear();

  // update forces
  std::string error_message = GetErrorMessages(
    boost::bind(&updateForce, model, data, forces));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::updateForce"
                              "(const boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const VectorXd&";
  std::string assert_argument = "static_cast<std::size_t>(data->impulses.size()) == impulses_.size() && "
                                "\"it doesn't match the number of impulse datas and models\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_updateFroce_mismatch_model_data() {
  // Setup the test
  StateFactory state_factory(StateTypes::StateMultibodyRandomHumanoid);
  crocoddyl::ImpulseModelMultiple model1(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  crocoddyl::ImpulseModelMultiple model2(
    boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.get_state()));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(state_factory.get_pinocchio_model());
  
  // create and add some impulse objects
  std::vector<boost::shared_ptr<ImpulseModelFactory> > impulse_factories;
  for(unsigned i=0 ; i<5 ; ++i){
    impulse_factories.push_back(create_random_factory());
    {
      std::ostringstream os;
      os << "random_impulse1_" << i;
      model1.addImpulse(os.str(), impulse_factories.back()->get_model());
    }
    {
      std::ostringstream os;
      os << "random_impulse2_" << i;
      model2.addImpulse(os.str(), impulse_factories.back()->get_model());
    }
  }
  
  // create the data of the multiple-impulses
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> data2 =
    model2.createData(&pinocchio_data);

  // create random forces
  Eigen::VectorXd forces = Eigen::VectorXd::Random(model1.get_ni());

  // update forces
  std::string error_message = GetErrorMessages(
    boost::bind(&updateForce, model1, data2, forces));

  // expected error message content
  std::string function_name = "void crocoddyl::ImpulseModelMultiple::updateForce"
                              "(const boost::shared_ptr<crocoddyl::ImpulseDataMultiple>&, "
                              "const VectorXd&";
  std::string assert_argument = "it_m->first == it_d->first && \"it doesn't match "
                                "the impulse name between data and model\"";

  // Perform the checks
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_updateVelocityDiff() {

}

void test_updateForceDiff() {

}

void test_get_state() {

}

void test_get_impulses() {

}

void test_get_ni(){

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
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_wrong_data_size)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_mismatch_model_data)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_computation)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_wrong_data_size)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_mismatch_model_data)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateForce)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateForce_assert_force_size)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateForce_assert_wrong_data_size)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateFroce_mismatch_model_data)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateVelocityDiff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_state)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_impulses)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_ni)));
}

bool init_function() {
  register_unit_tests();  
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
