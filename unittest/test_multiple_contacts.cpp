///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "factory/contact.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

/**
 * These methods modify the return type of the model function in
 * order to use the boost::execution_monitor::execute method which catch the
 * assert signal
 */
int calc(crocoddyl::ContactModelMultiple& model, boost::shared_ptr<crocoddyl::ContactDataMultiple> data,
         Eigen::VectorXd& dx) {
  model.calc(data, dx);
  return 0;
}

int calcDiff(crocoddyl::ContactModelMultiple& model, boost::shared_ptr<crocoddyl::ContactDataMultiple> data,
             Eigen::VectorXd& dx) {
  model.calcDiff(data, dx);
  return 0;
}

int updateForce(crocoddyl::ContactModelMultiple& model, boost::shared_ptr<crocoddyl::ContactDataMultiple> data,
                Eigen::VectorXd& dx) {
  model.updateForce(data, dx);
  return 0;
}

int updateAccelerationDiff(crocoddyl::ContactModelMultiple& model,
                           boost::shared_ptr<crocoddyl::ContactDataMultiple> data, const Eigen::MatrixXd& ddv_dx) {
  model.updateAccelerationDiff(data, ddv_dx);
  return 0;
}

int updateForceDiff(crocoddyl::ContactModelMultiple& model, boost::shared_ptr<crocoddyl::ContactDataMultiple> data,
                    const Eigen::MatrixXd& df_dx, const Eigen::MatrixXd& df_du) {
  model.updateForceDiff(data, df_dx, df_du);
  return 0;
}

//----------------------------------------------------------------------------//

void test_constructor() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // Test the initial size of the map
  BOOST_CHECK(model.get_contacts().size() == 0);
}

void test_addContact() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // add an active contact
  boost::shared_ptr<crocoddyl::ContactModelAbstract> rand_contact_1 = create_random_contact();
  model.addContact("random_contact_1", rand_contact_1);
  BOOST_CHECK(model.get_nc() == rand_contact_1->get_nc());
  BOOST_CHECK(model.get_nc_total() == rand_contact_1->get_nc());

  // add an inactive contact
  boost::shared_ptr<crocoddyl::ContactModelAbstract> rand_contact_2 = create_random_contact();
  model.addContact("random_contact_2", rand_contact_2, false);
  BOOST_CHECK(model.get_nc() == rand_contact_1->get_nc());
  BOOST_CHECK(model.get_nc_total() == rand_contact_1->get_nc() + rand_contact_2->get_nc());

  // change the random contact 2 status
  model.changeContactStatus("random_contact_2", true);
  BOOST_CHECK(model.get_nc() == rand_contact_1->get_nc() + rand_contact_2->get_nc());
  BOOST_CHECK(model.get_nc_total() == rand_contact_1->get_nc() + rand_contact_2->get_nc());

  // change the random contact 1 status
  model.changeContactStatus("random_contact_1", false);
  BOOST_CHECK(model.get_nc() == rand_contact_2->get_nc());
  BOOST_CHECK(model.get_nc_total() == rand_contact_1->get_nc() + rand_contact_2->get_nc());
}

void test_addContact_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create an contact object
  boost::shared_ptr<crocoddyl::ContactModelAbstract> rand_contact = create_random_contact();

  // add twice the same contact object to the container
  model.addContact("random_contact", rand_contact);

  // test error message when we add a duplicate contact
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addContact("random_contact", rand_contact);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this contact item already existed, we cannot add it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the contact status of an inexistent contact
  capture_ios.beginCapture();
  model.changeContactStatus("no_exist_contact", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: this contact item doesn't exist, we cannot change its status" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeContact() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // add an active contact
  boost::shared_ptr<crocoddyl::ContactModelAbstract> rand_contact = create_random_contact();
  model.addContact("random_contact", rand_contact);
  BOOST_CHECK(model.get_nc() == rand_contact->get_nc());
  BOOST_CHECK(model.get_nc_total() == rand_contact->get_nc());

  // remove the contact
  model.removeContact("random_contact");
  BOOST_CHECK(model.get_nc() == 0);
  BOOST_CHECK(model.get_nc_total() == 0);
}

void test_removeContact_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // remove a none existing contact form the container, we expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeContact("random_contact");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this contact item doesn't exist, we cannot remove it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  std::vector<boost::shared_ptr<crocoddyl::ContactModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::ContactDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    const boost::shared_ptr<crocoddyl::ContactModelAbstract>& m = create_random_contact();
    model.addContact(os.str(), m);
    models.push_back(m);
    datas.push_back(m->createData(&pinocchio_data));
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data = model.createData(&pinocchio_data);

  // compute the multiple contact data for the case when all contacts are defined as active
  Eigen::VectorXd x1 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1);

  // check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());

  // check Jc and a0 against single contact computations
  std::size_t nc = 0;
  const std::size_t& nv = model.get_state()->get_nv();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t& nc_i = models[i]->get_nc();
    models[i]->calc(datas[i], x1);
    BOOST_CHECK(data->Jc.block(nc, 0, nc_i, nv) == datas[i]->Jc);
    BOOST_CHECK(data->a0.segment(nc, nc_i) == datas[i]->a0);
    nc += nc_i;
  }
  nc = 0;

  // compute the multiple contact data for the case when the first three contacts are defined as active
  model.changeContactStatus("random_contact_3", false);
  model.changeContactStatus("random_contact_4", false);
  Eigen::VectorXd x2 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x2);
  model.calc(data, x2);
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t& nc_i = models[i]->get_nc();
    if (i < 3) {  // we need to update data because this contacts are active
      models[i]->calc(datas[i], x2);
    }
    BOOST_CHECK(data->Jc.block(nc, 0, nc_i, nv) == datas[i]->Jc);
    BOOST_CHECK(data->a0.segment(nc, nc_i) == datas[i]->a0);
    nc += nc_i;
  }
}

void test_calc_diff() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  std::vector<boost::shared_ptr<crocoddyl::ContactModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::ContactDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    const boost::shared_ptr<crocoddyl::ContactModelAbstract>& m = create_random_contact();
    model.addContact(os.str(), m);
    models.push_back(m);
    datas.push_back(m->createData(&pinocchio_data));
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data = model.createData(&pinocchio_data);

  // compute the multiple contact data for the case when all contacts are defined as active
  Eigen::VectorXd x1 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1);
  model.calcDiff(data, x1);

  // check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->a0.isZero());
  BOOST_CHECK(!data->da0_dx.isZero());

  // check Jc and a0 against single contact computations
  std::size_t nc = 0;
  const std::size_t& nv = model.get_state()->get_nv();
  const std::size_t& ndx = model.get_state()->get_ndx();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t& nc_i = models[i]->get_nc();
    models[i]->calc(datas[i], x1);
    models[i]->calcDiff(datas[i], x1);
    BOOST_CHECK(data->Jc.block(nc, 0, nc_i, nv) == datas[i]->Jc);
    BOOST_CHECK(data->a0.segment(nc, nc_i) == datas[i]->a0);
    BOOST_CHECK(data->da0_dx.block(nc, 0, nc_i, ndx) == datas[i]->da0_dx);
    nc += nc_i;
  }
  nc = 0;

  // compute the multiple contact data for the case when the first three contacts are defined as active
  model.changeContactStatus("random_contact_3", false);
  model.changeContactStatus("random_contact_4", false);
  Eigen::VectorXd x2 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x2);
  model.calc(data, x2);
  model.calcDiff(data, x2);
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t& nc_i = models[i]->get_nc();
    if (i < 3) {  // we need to update data because this contacts are active
      models[i]->calc(datas[i], x2);
      models[i]->calcDiff(datas[i], x2);
    }
    BOOST_CHECK(data->Jc.block(nc, 0, nc_i, nv) == datas[i]->Jc);
    BOOST_CHECK(data->a0.segment(nc, nc_i) == datas[i]->a0);
    BOOST_CHECK(data->da0_dx.block(nc, 0, nc_i, ndx) == datas[i]->da0_dx);
    nc += nc_i;
  }
}

void test_calc_diff_no_recalc() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  std::vector<boost::shared_ptr<crocoddyl::ContactModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::ContactDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    const boost::shared_ptr<crocoddyl::ContactModelAbstract>& m = create_random_contact();
    model.addContact(os.str(), m);
    models.push_back(m);
    datas.push_back(m->createData(&pinocchio_data));
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data = model.createData(&pinocchio_data);

  // compute the multiple contact data for the case when all contacts are defined as active
  Eigen::VectorXd x1 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calcDiff(data, x1);

  // check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->a0.isZero());
  BOOST_CHECK(!data->da0_dx.isZero());

  // check Jc and a0 against single contact computations
  std::size_t nc = 0;
  const std::size_t& nv = model.get_state()->get_nv();
  const std::size_t& ndx = model.get_state()->get_ndx();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t& nc_i = models[i]->get_nc();
    models[i]->calcDiff(datas[i], x1);
    BOOST_CHECK(data->Jc.block(nc, 0, nc_i, nv).isZero());
    BOOST_CHECK(data->a0.segment(nc, nc_i).isZero());
    BOOST_CHECK(data->da0_dx.block(nc, 0, nc_i, ndx) == datas[i]->da0_dx);
    nc += nc_i;
  }
  nc = 0;

  // compute the multiple contact data for the case when the first three contacts are defined as active
  model.changeContactStatus("random_contact_3", false);
  model.changeContactStatus("random_contact_4", false);
  Eigen::VectorXd x2 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x2);
  model.calcDiff(data, x2);
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t& nc_i = models[i]->get_nc();
    if (i < 3) {  // we need to update data because this contacts are active
      models[i]->calcDiff(datas[i], x2);
    }
    BOOST_CHECK(data->Jc.block(nc, 0, nc_i, nv).isZero());
    BOOST_CHECK(data->a0.segment(nc, nc_i).isZero());
    BOOST_CHECK(data->da0_dx.block(nc, 0, nc_i, ndx) == datas[i]->da0_dx);
    nc += nc_i;
  }
}

void test_updateForce() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  const pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    model.addContact(os.str(), create_random_contact());
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data = model.createData(&pinocchio_data);

  // Compute the jacobian and check that the contact model fetch it.
  Eigen::VectorXd q = model.get_state()->rand().segment(0, model.get_state()->get_nq());
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.get_state()->get_nv());
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.get_state()->get_nv());
  pinocchio::computeJointJacobians(pinocchio_model, pinocchio_data, q);
  pinocchio::updateFramePlacements(pinocchio_model, pinocchio_data);
  pinocchio::computeForwardKinematicsDerivatives(pinocchio_model, pinocchio_data, q, v, a);

  // create random forces
  Eigen::VectorXd forces = Eigen::VectorXd::Random(model.get_nc());

  // update forces
  model.updateForce(data, forces);

  // Check that nothing has been computed and that all value are initialized to 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  crocoddyl::ContactModelMultiple::ContactDataContainer::iterator it_d, end_d;
  for (it_d = data->contacts.begin(), end_d = data->contacts.end(); it_d != end_d; ++it_d) {
    BOOST_CHECK(!it_d->second->f.toVector().isZero());
  }
}

void test_updateAccelerationDiff() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    model.addContact(os.str(), create_random_contact());
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data = model.createData(&pinocchio_data);

  // create the velocity diff
  Eigen::MatrixXd ddv_dx = Eigen::MatrixXd::Random(model.get_state()->get_nv(), model.get_state()->get_ndx());

  // call the update
  model.updateAccelerationDiff(data, ddv_dx);

  // Test
  BOOST_CHECK((data->ddv_dx - ddv_dx).isMuchSmallerThan(1.0, 1e-9));
}

void test_updateForceDiff() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    model.addContact(os.str(), create_random_contact());
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data = model.createData(&pinocchio_data);

  // create force diff
  Eigen::MatrixXd df_dx = Eigen::MatrixXd::Random(model.get_nc(), model.get_state()->get_nv());
  Eigen::MatrixXd df_du = Eigen::MatrixXd::Random(model.get_nc(), model.get_state()->get_nv());

  // call update force diff
  model.updateForceDiff(data, df_dx, df_du);

  // Test
  crocoddyl::ContactModelMultiple::ContactDataContainer::iterator it_d, end_d;
  for (it_d = data->contacts.begin(), end_d = data->contacts.end(); it_d != end_d; ++it_d) {
    BOOST_CHECK(!it_d->second->df_dx.isZero());
  }
}

void test_assert_updateForceDiff_assert_mismatch_model_data() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model1(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  crocoddyl::ContactModelMultiple model2(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model1.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    boost::shared_ptr<crocoddyl::ContactModelAbstract> rand_contact = create_random_contact();
    {
      std::ostringstream os;
      os << "random_contact1_" << i;
      model1.addContact(os.str(), rand_contact);
    }
    {
      std::ostringstream os;
      os << "random_contact2_" << i;
      model2.addContact(os.str(), rand_contact);
    }
  }

  // create the data of the multiple-contacts
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data1 = model1.createData(&pinocchio_data);
  boost::shared_ptr<crocoddyl::ContactDataMultiple> data2 = model2.createData(&pinocchio_data);

  // create force diff
  Eigen::MatrixXd df_dx = Eigen::MatrixXd::Random(model1.get_nc(), model1.get_state()->get_nv());
  Eigen::MatrixXd df_du = Eigen::MatrixXd::Random(model1.get_nc(), model1.get_state()->get_nv());

  // call that trigger assert
  std::string error_message = GetErrorMessages(boost::bind(&updateForceDiff, model1, data2, df_dx, df_du));

  // expected error message content
  std::string function_name =
      "void crocoddyl::ContactModelMultiple::updateForceDiff("
      "const boost::shared_ptr<crocoddyl::ContactDataMultiple>&,"
      " const MatrixXd&) const";
  std::string assert_argument =
      "it_m->first == it_d->first && \"it doesn't match"
      " the contact name between data and model\"";

  // Perform the checks
#ifndef __APPLE__
  BOOST_CHECK(error_message.find(function_name) != std::string::npos);
#endif
  BOOST_CHECK(error_message.find(assert_argument) != std::string::npos);
}

void test_get_contacts() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    model.addContact(os.str(), create_random_contact());
  }

  // get the contacts
  const crocoddyl::ContactModelMultiple::ContactModelContainer& contacts = model.get_contacts();

  // test
  crocoddyl::ContactModelMultiple::ContactModelContainer::const_iterator it_m, end_m;
  unsigned i;
  for (i = 0, it_m = contacts.begin(), end_m = contacts.end(); it_m != end_m; ++it_m, ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_get_nc() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ContactModelMultiple model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_contact_" << i;
    model.addContact(os.str(), create_random_contact());
  }

  // compute ni
  std::size_t ni = 0;
  crocoddyl::ContactModelMultiple::ContactModelContainer::const_iterator it_m, end_m;
  for (it_m = model.get_contacts().begin(), end_m = model.get_contacts().end(); it_m != end_m; ++it_m) {
    ni += it_m->second->contact->get_nc();
  }

  BOOST_CHECK(ni == model.get_nc());
}

//----------------------------------------------------------------------------//

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addContact)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addContact_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeContact)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeContact_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_recalc)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateForce)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_updateAccelerationDiff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_contacts)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_nc)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
