///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/contacts/contact-1d.hpp"
#include "crocoddyl/multibody/contacts/contact-2d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "factory/contact.hpp"
#include "unittest_common.hpp"

using namespace crocoddyl::unittest;
using namespace boost::unit_test;

//----------------------------------------------------------------------------//

void test_construct_data(ContactModelTypes::Type contact_type,
                         PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  std::shared_ptr<crocoddyl::ContactModelAbstract> model =
      factory.create(contact_type, model_type, Eigen::Vector2d::Random());

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  std::shared_ptr<crocoddyl::ContactDataAbstract> data =
      model->createData(&pinocchio_data);
}

void test_calc_fetch_jacobians(ContactModelTypes::Type contact_type,
                               PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  std::shared_ptr<crocoddyl::ContactModelAbstract> model =
      factory.create(contact_type, model_type, Eigen::Vector2d::Random());

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(),
                                          &pinocchio_data, x);

  // Getting the jacobian from the model
  model->calc(data, x);

  // Check that only the Jacobian has been filled
  BOOST_CHECK(!data->Jc.isZero());
  if (model_type !=
      PinocchioModelTypes::Hector) {  // this is due to Hector is a single rigid
                                      // body system.
    BOOST_CHECK(!data->a0.isZero());
  }
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ContactModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::VectorXf x_f = x.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model_f.get(),
                                          &pinocchio_data_f, x_f);
  casted_model->calc(casted_data, x_f);
  BOOST_CHECK(!casted_data->Jc.isZero());
  if (model_type !=
      PinocchioModelTypes::Hector) {  // this is due to Hector is a single rigid
                                      // body system.
    BOOST_CHECK(!casted_data->a0.isZero());
  }
  BOOST_CHECK(casted_data->da0_dx.isZero());
  BOOST_CHECK(casted_data->f.toVector().isZero());
  BOOST_CHECK(casted_data->df_dx.isZero());
  BOOST_CHECK(casted_data->df_du.isZero());
#endif
}

void test_calc_diff_fetch_derivatives(ContactModelTypes::Type contact_type,
                                      PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  std::shared_ptr<crocoddyl::ContactModelAbstract> model =
      factory.create(contact_type, model_type, Eigen::Vector2d::Random());

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Compute the jacobian and check that the impulse model fetch it.
  Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model.get(),
                                          &pinocchio_data, x);

  // Getting the jacobian from the model
  model->calc(data, x);
  model->calcDiff(data, x);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(!data->Jc.isZero());
  if (model_type !=
      PinocchioModelTypes::Hector) {  // this is due to Hector is a single rigid
                                      // body system.
    BOOST_CHECK(!data->a0.isZero());
    BOOST_CHECK(!data->da0_dx.isZero());
  }
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ContactModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::VectorXf x_f = x.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(pinocchio_model_f.get(),
                                          &pinocchio_data_f, x_f);
  casted_model->calc(casted_data, x_f);
  casted_model->calcDiff(casted_data, x_f);
  BOOST_CHECK(!casted_data->Jc.isZero());
  if (model_type !=
      PinocchioModelTypes::Hector) {  // this is due to Hector is a single rigid
                                      // body system.
    float tol_f =
        80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
    BOOST_CHECK(!casted_data->a0.isZero());
    BOOST_CHECK(!casted_data->da0_dx.isZero());
    BOOST_CHECK((data->a0.cast<float>() - casted_data->a0).isZero());
    BOOST_CHECK(
        (data->da0_dx.cast<float>() - casted_data->da0_dx).isZero(tol_f));
  }
  BOOST_CHECK(casted_data->f.toVector().isZero());
  BOOST_CHECK(casted_data->df_dx.isZero());
  BOOST_CHECK(casted_data->df_du.isZero());
#endif
}

void test_update_force(ContactModelTypes::Type contact_type,
                       PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  std::shared_ptr<crocoddyl::ContactModelAbstract> model =
      factory.create(contact_type, model_type, Eigen::Vector2d::Random());

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::VectorXd f = Eigen::VectorXd::Random(data->Jc.rows());
  model->updateForce(data, f);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(!data->f.toVector().isZero());
  BOOST_CHECK(!data->fext.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ContactModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::VectorXf f_f = f.cast<float>();
  casted_model->updateForce(casted_data, f_f);
  BOOST_CHECK(casted_data->Jc.isZero());
  BOOST_CHECK(casted_data->a0.isZero());
  BOOST_CHECK(casted_data->da0_dx.isZero());
  BOOST_CHECK(!casted_data->f.toVector().isZero());
  BOOST_CHECK(!casted_data->fext.toVector().isZero());
  BOOST_CHECK(
      (data->f.toVector().cast<float>() - casted_data->f.toVector()).isZero());
  BOOST_CHECK(
      (data->fext.toVector().cast<float>() - casted_data->fext.toVector())
          .isZero());
  BOOST_CHECK(casted_data->df_dx.isZero());
  BOOST_CHECK(casted_data->df_du.isZero());
#endif
}

void test_update_force_diff(ContactModelTypes::Type contact_type,
                            PinocchioModelTypes::Type model_type) {
  // create the model
  ContactModelFactory factory;
  std::shared_ptr<crocoddyl::ContactModelAbstract> model =
      factory.create(contact_type, model_type, Eigen::Vector2d::Random());

  // create the corresponding data object
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Create a random force and update it
  Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(data->df_dx.rows(), data->df_dx.cols());
  Eigen::MatrixXd df_du =
      Eigen::MatrixXd::Random(data->df_du.rows(), data->df_du.cols());
  model->updateForceDiff(data, df_dx, df_du);

  // Check that nothing has been computed and that all value are initialized to
  // 0
  BOOST_CHECK(data->Jc.isZero());
  BOOST_CHECK(data->a0.isZero());
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->fext.toVector().isZero());
  BOOST_CHECK(!data->df_dx.isZero());
  BOOST_CHECK(!data->df_du.isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ContactModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  const std::shared_ptr<pinocchio::ModelTpl<float>>& pinocchio_model_f =
      casted_model->get_state()->get_pinocchio();
  pinocchio::DataTpl<float> pinocchio_data_f(*pinocchio_model_f.get());
  const std::shared_ptr<crocoddyl::ContactDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  Eigen::MatrixXf df_dx_f = df_dx.cast<float>();
  Eigen::MatrixXf df_du_f = df_du.cast<float>();
  casted_model->updateForceDiff(casted_data, df_dx_f, df_du_f);
  BOOST_CHECK(casted_data->Jc.isZero());
  BOOST_CHECK(casted_data->a0.isZero());
  BOOST_CHECK(casted_data->da0_dx.isZero());
  BOOST_CHECK(casted_data->f.toVector().isZero());
  BOOST_CHECK(casted_data->fext.toVector().isZero());
  BOOST_CHECK(!casted_data->df_dx.isZero());
  BOOST_CHECK(!casted_data->df_du.isZero());
  BOOST_CHECK((data->df_dx.cast<float>() - casted_data->df_dx).isZero());
  BOOST_CHECK((data->df_du.cast<float>() - casted_data->df_du).isZero());
#endif
}

void test_partial_derivatives_against_numdiff(
    ContactModelTypes::Type contact_type,
    PinocchioModelTypes::Type model_type) {
  using namespace boost::placeholders;

  // create the model
  ContactModelFactory factory;
  std::shared_ptr<crocoddyl::ContactModelAbstract> model =
      factory.create(contact_type, model_type, Eigen::Vector2d::Random());

  // create the corresponding data object
  pinocchio::Model& pinocchio_model =
      *model->get_state()->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  const std::shared_ptr<crocoddyl::ContactDataAbstract>& data =
      model->createData(&pinocchio_data);

  // Create the equivalent num diff model and data.
  crocoddyl::ContactModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ContactDataAbstract>& data_num_diff =
      model_num_diff.createData(&pinocchio_data);

  // Generating random values for the state
  const Eigen::VectorXd x = model->get_state()->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::ContactModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      double, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model, &pinocchio_data, _1, _2));
  model_num_diff.set_reevals(reevals);

  // Computing the contact derivatives
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->da0_dx - data_num_diff->da0_dx).isZero(tol));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ContactModelAbstractTpl<float>> casted_model =
      model->cast<float>();
  pinocchio::ModelTpl<float>& pinocchio_model_f =
      *casted_model->get_state()->get_pinocchio().get();
  pinocchio::DataTpl<float> pinocchio_data_f(pinocchio_model_f);
  const std::shared_ptr<crocoddyl::ContactDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&pinocchio_data_f);
  const Eigen::VectorXf x_f = x.cast<float>();
  crocoddyl::ContactModelNumDiffTpl<float> casted_model_num_diff =
      model_num_diff.cast<float>();
  const std::shared_ptr<crocoddyl::ContactDataAbstractTpl<float>>&
      casted_data_num_diff =
          casted_model_num_diff.createData(&pinocchio_data_f);
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model_f, &pinocchio_data_f,
                                          x_f);
  std::vector<crocoddyl::ContactModelNumDiffTpl<float>::ReevaluationFunction>
      reevals_f;
  reevals_f.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      float, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model_f, &pinocchio_data_f, _1, _2));
  casted_model_num_diff.set_reevals(reevals_f);
  model->calc(data, x);
  model->calcDiff(data, x);
  casted_model->calc(casted_data, x_f);
  casted_model->calcDiff(casted_data, x_f);
  casted_model_num_diff.calc(casted_data_num_diff, x_f);
  casted_model_num_diff.calcDiff(casted_data_num_diff, x_f);
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->da0_dx.cast<float>() - casted_data->da0_dx).isZero(tol_f));
  BOOST_CHECK((casted_data->da0_dx - casted_data_num_diff->da0_dx)
                  .isZero(30.f * tol_f));
#endif
}

//----------------------------------------------------------------------------//

void register_contact_model_unit_tests(ContactModelTypes::Type contact_type,
                                       PinocchioModelTypes::Type model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << contact_type << "_" << model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_construct_data, contact_type, model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_fetch_jacobians, contact_type, model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_diff_fetch_derivatives,
                                      contact_type, model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_update_force, contact_type, model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_update_force_diff, contact_type, model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                      contact_type, model_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t contact_type = 0; contact_type < ContactModelTypes::all.size();
       ++contact_type) {
    for (size_t model_type = 0; model_type < PinocchioModelTypes::all.size();
         ++model_type) {
      register_contact_model_unit_tests(ContactModelTypes::all[contact_type],
                                        PinocchioModelTypes::all[model_type]);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
