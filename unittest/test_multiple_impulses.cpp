///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, INRIA,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

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
int calc(crocoddyl::ImpulseModelMultiple& model,
         std::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
         Eigen::VectorXd& dx) {
  model.calc(data, dx);
  return 0;
}

int calcDiff(crocoddyl::ImpulseModelMultiple& model,
             std::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
             Eigen::VectorXd& dx) {
  model.calcDiff(data, dx);
  return 0;
}

int updateForce(crocoddyl::ImpulseModelMultiple& model,
                std::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                Eigen::VectorXd& dx) {
  model.updateForce(data, dx);
  return 0;
}

int updateVelocityDiff(crocoddyl::ImpulseModelMultiple& model,
                       std::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                       const Eigen::MatrixXd& dvnext_dx) {
  model.updateVelocityDiff(data, dvnext_dx);
  return 0;
}

int updateForceDiff(crocoddyl::ImpulseModelMultiple& model,
                    std::shared_ptr<crocoddyl::ImpulseDataMultiple> data,
                    const Eigen::MatrixXd& df_dx) {
  model.updateForceDiff(data, df_dx);
  return 0;
}

//----------------------------------------------------------------------------//

void test_constructor() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // Run the print function
  std::ostringstream tmp;
  tmp << model;

  // Test the initial size of the map
  BOOST_CHECK(model.get_impulses().size() == 0);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();
  BOOST_CHECK(casted_model.get_impulses().size() == 0);
#endif
}

void test_addImpulse() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();

  // add an active impulse
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> rand_impulse_1 =
      create_random_impulse();
  model.addImpulse("random_impulse_1", rand_impulse_1);
  BOOST_CHECK(model.get_nc() == rand_impulse_1->get_nc());
  BOOST_CHECK(model.get_nc_total() == rand_impulse_1->get_nc());

  // add an inactive impulse
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> rand_impulse_2 =
      create_random_impulse();
  model.addImpulse("random_impulse_2", rand_impulse_2, false);
  BOOST_CHECK(model.get_nc() == rand_impulse_1->get_nc());
  BOOST_CHECK(model.get_nc_total() ==
              rand_impulse_1->get_nc() + rand_impulse_2->get_nc());

  // change the random impulse 2 status
  model.changeImpulseStatus("random_impulse_2", true);
  BOOST_CHECK(model.get_nc() ==
              rand_impulse_1->get_nc() + rand_impulse_2->get_nc());
  BOOST_CHECK(model.get_nc_total() ==
              rand_impulse_1->get_nc() + rand_impulse_2->get_nc());

  // change the random impulse 1 status
  model.changeImpulseStatus("random_impulse_1", false);
  BOOST_CHECK(model.get_nc() == rand_impulse_2->get_nc());
  BOOST_CHECK(model.get_nc_total() ==
              rand_impulse_1->get_nc() + rand_impulse_2->get_nc());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>>
      casted_rand_impulse_1 = rand_impulse_1->cast<float>();
  casted_model.addImpulse("random_impulse_1", casted_rand_impulse_1);
  BOOST_CHECK(casted_model.get_nc() == casted_rand_impulse_1->get_nc());
  BOOST_CHECK(casted_model.get_nc_total() == casted_rand_impulse_1->get_nc());
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>>
      casted_rand_impulse_2 = rand_impulse_2->cast<float>();
  casted_model.addImpulse("random_impulse_2", casted_rand_impulse_2, false);
  BOOST_CHECK(casted_model.get_nc() == casted_rand_impulse_1->get_nc());
  BOOST_CHECK(casted_model.get_nc_total() ==
              casted_rand_impulse_1->get_nc() +
                  casted_rand_impulse_2->get_nc());
  casted_model.changeImpulseStatus("random_impulse_2", true);
  BOOST_CHECK(casted_model.get_nc() == casted_rand_impulse_1->get_nc() +
                                           casted_rand_impulse_2->get_nc());
  BOOST_CHECK(casted_model.get_nc_total() ==
              casted_rand_impulse_1->get_nc() +
                  casted_rand_impulse_2->get_nc());
  casted_model.changeImpulseStatus("random_impulse_1", false);
  BOOST_CHECK(casted_model.get_nc() == casted_rand_impulse_2->get_nc());
  BOOST_CHECK(casted_model.get_nc_total() ==
              casted_rand_impulse_1->get_nc() +
                  casted_rand_impulse_2->get_nc());
#endif
}

void test_addImpulse_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create an impulse object
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> rand_impulse =
      create_random_impulse();

  // add twice the same impulse object to the container
  model.addImpulse("random_impulse", rand_impulse);

  // test error message when we add a duplicate impulse
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addImpulse("random_impulse", rand_impulse);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't add the random_impulse impulse "
                     "item, it already existed."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the impulse status of an inexistent
  // impulse
  capture_ios.beginCapture();
  model.changeImpulseStatus("no_exist_impulse", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: we couldn't change the status of the "
                     "no_exist_impulse impulse item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeImpulse() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();

  // add an active impulse
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> rand_impulse =
      create_random_impulse();
  model.addImpulse("random_impulse", rand_impulse);
  BOOST_CHECK(model.get_nc() == rand_impulse->get_nc());

  // remove the impulse
  model.removeImpulse("random_impulse");
  BOOST_CHECK(model.get_nc() == 0);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>>
      casted_rand_impulse = rand_impulse->cast<float>();
  casted_model.addImpulse("random_impulse", casted_rand_impulse);
  BOOST_CHECK(casted_model.get_nc() == casted_rand_impulse->get_nc());
  BOOST_CHECK(casted_model.get_nc_total() == casted_rand_impulse->get_nc());
  casted_model.removeImpulse("random_impulse");
  BOOST_CHECK(casted_model.get_nc() == 0);
  BOOST_CHECK(casted_model.get_nc_total() == 0);
#endif
}

void test_removeImpulse_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // remove a none existing impulse form the container, we expect a cout message
  // here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeImpulse("random_impulse");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't remove the random_impulse impulse "
                     "item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<std::shared_ptr<crocoddyl::ImpulseModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::ImpulseDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    const std::shared_ptr<crocoddyl::ImpulseModelAbstract>& m =
        create_random_impulse();
    model.addImpulse(os.str(), m);
    models.push_back(m);
    datas.push_back(m->createData(&pinocchio_data));
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
      model.createData(&pinocchio_data);

  // compute the multiple contact data for the case when all impulses are
  // defined as active
  Eigen::VectorXd x1 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1);

  // check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());

  // check Jc against single impulse computations
  std::size_t ni = 0;
  std::size_t nv = model.get_state()->get_nv();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = models[i]->get_nc();
    models[i]->calc(datas[i], x1);
    BOOST_CHECK(data->Jc.block(ni, 0, ni_i, nv) == datas[i]->Jc);
    ni += ni_i;
  }
  ni = 0;

  // compute the multiple impulse data for the case when the first three
  // impulses are defined as active
  model.changeImpulseStatus("random_impulse_3", false);
  model.changeImpulseStatus("random_impulse_4", false);
  Eigen::VectorXd x2 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x2);
  model.calc(data, x2);
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = models[i]->get_nc();
    if (i < 3) {  // we need to update data because this impulses are active
      models[i]->calc(datas[i], x2);
    }
    BOOST_CHECK(data->Jc.block(ni, 0, ni_i, nv) == datas[i]->Jc);
    ni += ni_i;
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  model.changeImpulseStatus("random_impulse_3", true);
  model.changeImpulseStatus("random_impulse_4", true);
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_model.get_state()->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(
      *casted_model.get_state()->get_pinocchio().get());
  std::vector<std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>>>
      casted_models;
  std::vector<std::shared_ptr<crocoddyl::ImpulseDataAbstractTpl<float>>>
      casted_datas;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models.push_back(models[i]->cast<float>());
    casted_datas.push_back(
        casted_models[i]->createData(&casted_pinocchio_data));
  }
  std::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<float>> casted_data =
      casted_model.createData(&casted_pinocchio_data);
  const Eigen::VectorXf x1_f = x1.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x1_f);
  model.calc(data, x1);
  casted_model.calc(casted_data, x1_f);
  BOOST_CHECK(!casted_data->Jc.isZero());
  BOOST_CHECK(casted_data->dv0_dq.isZero());
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->Jc.cast<float>() - casted_data->Jc).isZero(tol_f));
  BOOST_CHECK((data->dv0_dq.cast<float>() - casted_data->dv0_dq).isZero(tol_f));
  BOOST_CHECK(casted_data->dv0_dq.isZero());
  ni = 0;
  nv = casted_model.get_state()->get_nv();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = casted_models[i]->get_nc();
    casted_models[i]->calc(casted_datas[i], x1_f);
    BOOST_CHECK(casted_data->Jc.block(ni, 0, ni_i, nv) == casted_datas[i]->Jc);
    ni += ni_i;
  }
  ni = 0;
#endif
}

void test_calc_diff() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<std::shared_ptr<crocoddyl::ImpulseModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::ImpulseDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    const std::shared_ptr<crocoddyl::ImpulseModelAbstract>& m =
        create_random_impulse();
    model.addImpulse(os.str(), m);
    models.push_back(m);
    datas.push_back(m->createData(&pinocchio_data));
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
      model.createData(&pinocchio_data);

  // compute the multiple contact data for the case when all impulses are
  // defined as active
  Eigen::VectorXd x1 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1);
  model.calcDiff(data, x1);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());

  // check Jc against single impulse computations
  std::size_t ni = 0;
  std::size_t nv = model.get_state()->get_nv();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = models[i]->get_nc();
    models[i]->calc(datas[i], x1);
    models[i]->calcDiff(datas[i], x1);
    BOOST_CHECK(data->Jc.block(ni, 0, ni_i, nv) == datas[i]->Jc);
    BOOST_CHECK(data->dv0_dq.block(ni, 0, ni_i, nv) == datas[i]->dv0_dq);
    ni += ni_i;
  }
  ni = 0;

  // compute the multiple impulse data for the case when the first three
  // impulses are defined as active
  model.changeImpulseStatus("random_impulse_3", false);
  model.changeImpulseStatus("random_impulse_4", false);
  Eigen::VectorXd x2 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x2);
  model.calc(data, x2);
  model.calcDiff(data, x2);
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = models[i]->get_nc();
    if (i < 3) {  // we need to update data because this impulses are active
      models[i]->calc(datas[i], x2);
      models[i]->calcDiff(datas[i], x2);
    }
    BOOST_CHECK(data->Jc.block(ni, 0, ni_i, nv) == datas[i]->Jc);
    BOOST_CHECK(data->dv0_dq.block(ni, 0, ni_i, nv) == datas[i]->dv0_dq);
    ni += ni_i;
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  model.changeImpulseStatus("random_impulse_3", true);
  model.changeImpulseStatus("random_impulse_4", true);
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_model.get_state()->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(
      *casted_model.get_state()->get_pinocchio().get());
  std::vector<std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>>>
      casted_models;
  std::vector<std::shared_ptr<crocoddyl::ImpulseDataAbstractTpl<float>>>
      casted_datas;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models.push_back(models[i]->cast<float>());
    casted_datas.push_back(
        casted_models[i]->createData(&casted_pinocchio_data));
  }
  std::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<float>> casted_data =
      casted_model.createData(&casted_pinocchio_data);
  const Eigen::VectorXf x1_f = x1.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x1_f);
  model.calc(data, x1);
  model.calcDiff(data, x1);
  casted_model.calc(casted_data, x1_f);
  casted_model.calcDiff(casted_data, x1_f);
  BOOST_CHECK(!casted_data->Jc.isZero());
  BOOST_CHECK(!casted_data->dv0_dq.isZero());
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->Jc.cast<float>() - casted_data->Jc).isZero(tol_f));
  BOOST_CHECK((data->dv0_dq.cast<float>() - casted_data->dv0_dq).isZero(tol_f));
  ni = 0;
  nv = casted_model.get_state()->get_nv();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = casted_models[i]->get_nc();
    casted_models[i]->calc(casted_datas[i], x1_f);
    casted_models[i]->calcDiff(casted_datas[i], x1_f);
    BOOST_CHECK(casted_data->Jc.block(ni, 0, ni_i, nv) == casted_datas[i]->Jc);
    BOOST_CHECK(casted_data->dv0_dq.block(ni, 0, ni_i, nv) ==
                casted_datas[i]->dv0_dq);
    ni += ni_i;
  }
  ni = 0;
#endif
}

void test_calc_diff_no_recalc() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<std::shared_ptr<crocoddyl::ImpulseModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::ImpulseDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    const std::shared_ptr<crocoddyl::ImpulseModelAbstract>& m =
        create_random_impulse();
    model.addImpulse(os.str(), m);
    models.push_back(m);
    datas.push_back(m->createData(&pinocchio_data));
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
      model.createData(&pinocchio_data);

  // compute the multiple contact data for the case when all impulses are
  // defined as active
  Eigen::VectorXd x1 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calcDiff(data, x1);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());

  // check Jc against single impulse computations
  std::size_t ni = 0;
  const std::size_t nv = model.get_state()->get_nv();
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = models[i]->get_nc();
    models[i]->calcDiff(datas[i], x1);
    BOOST_CHECK(data->Jc.block(ni, 0, ni_i, nv).isZero());
    BOOST_CHECK(data->dv0_dq.block(ni, 0, ni_i, nv) == datas[i]->dv0_dq);
    ni += ni_i;
  }
  ni = 0;

  // compute the multiple impulse data for the case when the first three
  // impulses are defined as active
  model.changeImpulseStatus("random_impulse_3", false);
  model.changeImpulseStatus("random_impulse_4", false);
  Eigen::VectorXd x2 = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x2);
  model.calcDiff(data, x2);
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t ni_i = models[i]->get_nc();
    if (i < 3) {  // we need to update data because this impulses are active
      models[i]->calcDiff(datas[i], x2);
    }
    BOOST_CHECK(data->Jc.block(ni, 0, ni_i, nv).isZero());
    BOOST_CHECK(data->dv0_dq.block(ni, 0, ni_i, nv) == datas[i]->dv0_dq);
    ni += ni_i;
  }
}

void test_updateForce() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Model& pinocchio_model = *model.get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), create_random_impulse());
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
      model.createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model.get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // create random forces
  Eigen::VectorXd forces = Eigen::VectorXd::Random(model.get_nc());

  // update forces
  model.updateForce(data, forces);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  crocoddyl::ImpulseModelMultiple::ImpulseDataContainer::iterator it_d, end_d;
  for (it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_d != end_d; ++it_d) {
    BOOST_CHECK(!it_d->second->f.toVector().isZero());
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_model.get_state()->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(
      *casted_model.get_state()->get_pinocchio().get());
  std::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<float>> casted_data =
      casted_model.createData(&casted_pinocchio_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  const Eigen::VectorXf forces_f = forces.cast<float>();
  casted_model.updateForce(casted_data, forces_f);
  BOOST_CHECK(casted_data->Jc.isZero());
  BOOST_CHECK(casted_data->dv0_dq.isZero());
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  crocoddyl::ImpulseModelMultipleTpl<float>::ImpulseDataContainer::iterator
      it_d_f,
      end_d_f;
  for (it_d_f = casted_data->impulses.begin(),
      end_d_f = casted_data->impulses.end(), it_d = data->impulses.begin(),
      end_d = data->impulses.end();
       it_d_f != end_d_f || it_d != end_d; ++it_d_f, ++it_d) {
    BOOST_CHECK(!it_d_f->second->f.toVector().isZero());
    BOOST_CHECK((it_d->second->f.toVector().cast<float>() -
                 it_d_f->second->f.toVector())
                    .isZero(tol_f));
  }
#endif
}

void test_updateVelocityDiff() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), create_random_impulse());
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
      model.createData(&pinocchio_data);

  // create the velocity diff
  Eigen::MatrixXd dvnext_dx = Eigen::MatrixXd::Random(
      model.get_state()->get_nv(), model.get_state()->get_ndx());

  // call the update
  model.updateVelocityDiff(data, dvnext_dx);

  // Test
  BOOST_CHECK((data->dvnext_dx - dvnext_dx).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();
  pinocchio::DataTpl<float> casted_pinocchio_data(
      *casted_model.get_state()->get_pinocchio().get());
  std::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<float>> casted_data =
      casted_model.createData(&casted_pinocchio_data);
  const Eigen::MatrixXf dvnext_dx_f = dvnext_dx.cast<float>();
  casted_model.updateVelocityDiff(casted_data, dvnext_dx_f);
  BOOST_CHECK((casted_data->dvnext_dx - dvnext_dx_f).isZero(1e-9f));
#endif
}

void test_updateForceDiff() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), create_random_impulse());
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data =
      model.createData(&pinocchio_data);

  // create force diff
  Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(model.get_nc(), model.get_state()->get_nv());

  // call update force diff
  model.updateForceDiff(data, df_dx);

  // Test
  crocoddyl::ImpulseModelMultiple::ImpulseDataContainer::iterator it_d, end_d;
  for (it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_d != end_d; ++it_d) {
    BOOST_CHECK(!it_d->second->df_dx.isZero());
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ImpulseModelMultipleTpl<float> casted_model = model.cast<float>();
  pinocchio::DataTpl<float> casted_pinocchio_data(
      *casted_model.get_state()->get_pinocchio().get());
  std::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<float>> casted_data =
      casted_model.createData(&casted_pinocchio_data);
  const Eigen::MatrixXf df_dx_f = df_dx.cast<float>();
  casted_model.updateForceDiff(casted_data, df_dx_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  crocoddyl::ImpulseModelMultipleTpl<float>::ImpulseDataContainer::iterator
      it_d_f,
      end_d_f;
  for (it_d_f = casted_data->impulses.begin(),
      end_d_f = casted_data->impulses.end(), it_d = data->impulses.begin(),
      end_d = data->impulses.end();
       it_d_f != end_d_f || it_d != end_d; ++it_d_f, ++it_d) {
    BOOST_CHECK(!it_d_f->second->df_dx.isZero());
    BOOST_CHECK((it_d->second->df_dx.cast<float>() - it_d_f->second->df_dx)
                    .isZero(tol_f));
  }
#endif
}

void test_assert_updateForceDiff_assert_mismatch_model_data() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model1(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));
  crocoddyl::ImpulseModelMultiple model2(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model1.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  std::vector<std::shared_ptr<ImpulseModelFactory>> impulse_factories;
  for (unsigned i = 0; i < 5; ++i) {
    std::shared_ptr<crocoddyl::ImpulseModelAbstract> rand_impulse =
        create_random_impulse();
    {
      std::ostringstream os;
      os << "random_impulse1_" << i;
      model1.addImpulse(os.str(), rand_impulse);
    }
    {
      std::ostringstream os;
      os << "random_impulse2_" << i;
      model2.addImpulse(os.str(), rand_impulse);
    }
  }

  // create the data of the multiple-impulses
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data1 =
      model1.createData(&pinocchio_data);
  std::shared_ptr<crocoddyl::ImpulseDataMultiple> data2 =
      model2.createData(&pinocchio_data);

  // create force diff
  Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(model1.get_nc(), model1.get_state()->get_nv());

  // call that trigger assert
  std::string error_message =
      GetErrorMessages(boost::bind(&updateForceDiff, model1, data2, df_dx));

  // expected error message content
  std::string function_name =
      "void crocoddyl::ImpulseModelMultiple::updateForceDiff("
      "const std::shared_ptr<crocoddyl::ImpulseDataMultiple>&,"
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

void test_get_impulses() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), create_random_impulse());
  }

  // get the impulses
  const crocoddyl::ImpulseModelMultiple::ImpulseModelContainer& impulses =
      model.get_impulses();

  // test
  crocoddyl::ImpulseModelMultiple::ImpulseModelContainer::const_iterator it_m,
      end_m;
  unsigned i;
  for (i = 0, it_m = impulses.begin(), end_m = impulses.end(); it_m != end_m;
       ++it_m, ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_get_nc() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ImpulseModelMultiple model(
      std::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(
          StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some impulse objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_impulse_" << i;
    model.addImpulse(os.str(), create_random_impulse());
  }

  // compute ni
  std::size_t ni = 0;
  crocoddyl::ImpulseModelMultiple::ImpulseModelContainer::const_iterator it_m,
      end_m;
  for (it_m = model.get_impulses().begin(), end_m = model.get_impulses().end();
       it_m != end_m; ++it_m) {
    ni += it_m->second->impulse->get_nc();
  }

  BOOST_CHECK(ni == model.get_nc());
}

//----------------------------------------------------------------------------//

void register_unit_tests() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_addImpulse)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_addImpulse_error_message)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_removeImpulse)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_removeImpulse_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_diff)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_diff_no_recalc)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_updateForce)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_updateVelocityDiff)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_get_impulses)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_get_nc)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
