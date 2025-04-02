///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft,
//                          INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "factory/impulse.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_construct_data(ImpulseModelTypes::Type impulse_type,
                         PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory;
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> model =
      factory.create(impulse_type, model_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model->get_state()->get_pinocchio().get());
  std::shared_ptr<crocoddyl::ImpulseDataAbstract> data =
      model->createData(&pinocchio_data);
}

void test_calc_fetch_jacobians(ImpulseModelTypes::Type impulse_type,
                               PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory;
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> model =
      factory.create(impulse_type, model_type);

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(),
                                          &pinocchio_data, x);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::VectorXf x_f = x.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model_f.get(),
                                          &pinocchio_data_f, x_f);
  Eigen::VectorXf dx_f;
  casted_model->calc(casted_data, dx_f);
  BOOST_CHECK(!casted_data->Jc.isZero());
  BOOST_CHECK((data->Jc.cast<float>() - casted_data->Jc).isZero());
  BOOST_CHECK(casted_data->dv0_dq.isZero());
  BOOST_CHECK(casted_data->f.toVector().isZero());
  BOOST_CHECK(casted_data->df_dx.isZero());
#endif
}

void test_calc_diff_fetch_derivatives(ImpulseModelTypes::Type impulse_type,
                                      PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory;
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> model =
      factory.create(impulse_type, model_type);

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(),
                                          &pinocchio_data, x);

  // Getting the jacobian from the model
  Eigen::VectorXd dx;
  model->calc(data, dx);
  model->calcDiff(data, dx);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(!data->Jc.isZero());
  if (model_type == PinocchioModelTypes::Hector &&
      (impulse_type == ImpulseModelTypes::ImpulseModel3D_LOCAL ||
       impulse_type ==
           ImpulseModelTypes::ImpulseModel6D_LOCAL)) {  // this is due to Hector
                                                        // is a single rigid
                                                        // body system.
    BOOST_CHECK(data->dv0_dq.isZero());
  } else {
    BOOST_CHECK(!data->dv0_dq.isZero());
  }
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::VectorXf x_f = x.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model_f.get(),
                                          &pinocchio_data_f, x_f);
  Eigen::VectorXf dx_f;
  casted_model->calc(casted_data, dx_f);
  casted_model->calcDiff(casted_data, dx_f);
  BOOST_CHECK(!casted_data->Jc.isZero());
  BOOST_CHECK((data->Jc.cast<float>() - casted_data->Jc).isZero());
  if (model_type == PinocchioModelTypes::Hector &&
      (impulse_type == ImpulseModelTypes::ImpulseModel3D_LOCAL ||
       impulse_type ==
           ImpulseModelTypes::ImpulseModel6D_LOCAL)) {  // this is due to Hector
                                                        // is a single rigid
                                                        // body system.
    BOOST_CHECK(casted_data->dv0_dq.isZero());
  } else {
    BOOST_CHECK(!casted_data->dv0_dq.isZero());
    BOOST_CHECK((data->dv0_dq.cast<float>() - casted_data->dv0_dq).isZero());
  }
  BOOST_CHECK(casted_data->f.toVector().isZero());
  BOOST_CHECK(casted_data->df_dx.isZero());
#endif
}

void test_update_force(ImpulseModelTypes::Type impulse_type,
                       PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory;
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> model =
      factory.create(impulse_type, model_type);

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::VectorXd f = Eigen::VectorXd::Random(data->Jc.rows());
  model->updateForce(data, f);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(!data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::VectorXf f_f = f.cast<float>();
  casted_model->updateForce(casted_data, f_f);
  BOOST_CHECK(casted_data->Jc.isZero());
  BOOST_CHECK(casted_data->dv0_dq.isZero());
  BOOST_CHECK(!casted_data->f.toVector().isZero());
  BOOST_CHECK(
      (data->f.toVector().cast<float>() - casted_data->f.toVector()).isZero());
  BOOST_CHECK(casted_data->df_dx.isZero());
#endif
}

void test_update_force_diff(ImpulseModelTypes::Type impulse_type,
                            PinocchioModelTypes::Type model_type) {
  // create the model
  ImpulseModelFactory factory;
  std::shared_ptr<crocoddyl::ImpulseModelAbstract> model =
      factory.create(impulse_type, model_type);

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(data->df_dx.rows(), data->df_dx.cols());
  model->updateForceDiff(data, df_dx);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->dv0_dq.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(!data->df_dx.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ImpulseModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ImpulseDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::MatrixXf df_dx_f = df_dx.cast<float>();
  casted_model->updateForceDiff(casted_data, df_dx_f);
  BOOST_CHECK(casted_data->Jc.isZero());
  BOOST_CHECK(casted_data->dv0_dq.isZero());
  BOOST_CHECK(casted_data->f.toVector().isZero());
  BOOST_CHECK(!casted_data->df_dx.isZero());
  BOOST_CHECK((data->df_dx.cast<float>() - casted_data->df_dx).isZero());
#endif
}

//----------------------------------------------------------------------------//

void register_impulse_model_unit_tests(ImpulseModelTypes::Type impulse_type,
                                       PinocchioModelTypes::Type model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << impulse_type << "_" << model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_construct_data, impulse_type, model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_fetch_jacobians, impulse_type, model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_fetch_derivatives,
                                      impulse_type, model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_update_force, impulse_type, model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_update_force_diff, impulse_type, model_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t impulse_type = 0; impulse_type < ImpulseModelTypes::all.size();
       ++impulse_type) {
    for (size_t model_type = 0; model_type < PinocchioModelTypes::all.size();
         ++model_type) {
      register_impulse_model_unit_tests(ImpulseModelTypes::all[impulse_type],
                                        PinocchioModelTypes::all[model_type]);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
