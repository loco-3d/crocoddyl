///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/joint-acceleration.hpp"
#include "crocoddyl/core/residuals/joint-effort.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "factory/actuation.hpp"
#include "factory/residual.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

bool is_task_residual(ResidualModelTypes::Type residual_type) {
  return residual_type == ResidualModelTypes::ResidualModelTaskFirstOrder ||
         residual_type == ResidualModelTypes::ResidualModelTaskSecondOrder;
}

//----------------------------------------------------------------------------//

void test_calc_returns_a_residual(ResidualModelTypes::Type residual_type,
                                  StateModelTypes::Type state_type,
                                  ActuationModelTypes::Type actuation_type) {
  // Create the model
  ResidualModelFactory residual_factory;
  ActuationModelFactory actuation_factory;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      actuation_factory.create(actuation_type, state_type);
  const std::shared_ptr<crocoddyl::ResidualModelAbstract>& model =
      residual_factory.create(residual_type, state_type,
                              actuation_model->get_nu());
  const bool with_actuation = !is_task_residual(residual_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // Create the corresponding shared data
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data;
  std::shared_ptr<crocoddyl::DataCollectorAbstract> shared_data;
  if (with_actuation) {
    actuation_data = actuation_model->createData();
    shared_data = std::make_shared<crocoddyl::DataCollectorActMultibody>(
        &pinocchio_data, actuation_data);
  } else {
    shared_data =
        std::make_shared<crocoddyl::DataCollectorMultibody>(&pinocchio_data);
  }

  // create the residual data
  const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data =
      model->createData(shared_data.get());

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  if (with_actuation) {
    crocoddyl::unittest::updateActuation(actuation_model, actuation_data, x, u);
  }

  // Getting the residual value computed by calc()
  data->r *= nan("");
  model->calc(data, x, u);

  // Checking that calc returns a residual value
  for (std::size_t i = 0; i < model->get_nr(); ++i)
    BOOST_CHECK(!std::isnan(data->r(i)));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const auto casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& pinocchio_model_f =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> pinocchio_data_f(pinocchio_model_f);
  const Eigen::VectorXf x_f = x.cast<float>();
  const float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  if (with_actuation) {
    const auto casted_actuation_model = actuation_model->cast<float>();
    const auto casted_actuation_data = casted_actuation_model->createData();
    const Eigen::VectorXf u_f = u.cast<float>();
    crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
        &pinocchio_data_f, casted_actuation_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model_f,
                                            &pinocchio_data_f, x_f);
    crocoddyl::unittest::updateActuation(casted_actuation_model,
                                         casted_actuation_data, x_f, u_f);
    casted_data->r *= float(nan(""));
    casted_model->calc(casted_data, x_f, u_f);
    for (std::size_t i = 0; i < casted_model->get_nr(); ++i)
      BOOST_CHECK(!std::isnan(casted_data->r(i)));
    BOOST_CHECK(
        isCloseAbsRel(data->r.cast<float>(), casted_data->r, tol_f, tol_f));
  } else {
    crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
        &pinocchio_data_f);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model_f,
                                            &pinocchio_data_f, x_f);
    casted_data->r *= float(nan(""));
    casted_model->calc(casted_data, x_f, u.cast<float>());
    for (std::size_t i = 0; i < casted_model->get_nr(); ++i)
      BOOST_CHECK(!std::isnan(casted_data->r(i)));
    BOOST_CHECK(
        isCloseAbsRel(data->r.cast<float>(), casted_data->r, tol_f, tol_f));
  }
#endif
}

void test_calc_against_numdiff(ResidualModelTypes::Type residual_type,
                               StateModelTypes::Type state_type,
                               ActuationModelTypes::Type actuation_type) {
  using namespace boost::placeholders;

  // Create the model
  ResidualModelFactory residual_factory;
  ActuationModelFactory actuation_factory;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      actuation_factory.create(actuation_type, state_type);
  const std::shared_ptr<crocoddyl::ResidualModelAbstract>& model =
      residual_factory.create(residual_type, state_type,
                              actuation_model->get_nu());
  const bool with_actuation = !is_task_residual(residual_type);

  // Create the corresponding shared data
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data;
  std::shared_ptr<crocoddyl::DataCollectorAbstract> shared_data;
  if (with_actuation) {
    actuation_data = actuation_model->createData();
    shared_data = std::make_shared<crocoddyl::DataCollectorActMultibody>(
        &pinocchio_data, actuation_data);
  } else {
    shared_data =
        std::make_shared<crocoddyl::DataCollectorMultibody>(&pinocchio_data);
  }

  // Create the residual data
  const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data =
      model->createData(shared_data.get());

  // Create the equivalent num diff model and data.
  crocoddyl::ResidualModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data_num_diff =
      model_num_diff.createData(shared_data.get());

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the residual
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  if (with_actuation) {
    actuation_model->calc(actuation_data, x, u);
  }
  model->calc(data, x, u);

  // Computing the residual from num diff
  std::vector<crocoddyl::ResidualModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      double, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model, &pinocchio_data, _1, _2));
  if (with_actuation) {
    reevals.push_back(boost::bind(&crocoddyl::unittest::updateActuation<double>,
                                  actuation_model, actuation_data, _1, _2));
  }
  model_num_diff.set_reevals(reevals);
  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->r == data_num_diff->r);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const auto casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  const Eigen::VectorXf x_f = x.cast<float>();
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  if (with_actuation) {
    const auto casted_actuation_model = actuation_model->cast<float>();
    const auto casted_actuation_data = casted_actuation_model->createData();
    const Eigen::VectorXf u_f = u.cast<float>();
    crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
        &casted_pinocchio_data, casted_actuation_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                            &casted_pinocchio_data, x_f);
    casted_actuation_model->calc(casted_actuation_data, x_f, u_f);
    casted_model->calc(casted_data, x_f, u_f);
    BOOST_CHECK(
        isCloseAbsRel(data->r.cast<float>(), casted_data->r, tol_f, tol_f));
  } else {
    crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
        &casted_pinocchio_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                            &casted_pinocchio_data, x_f);
    casted_model->calc(casted_data, x_f, u.cast<float>());
    BOOST_CHECK(
        isCloseAbsRel(data->r.cast<float>(), casted_data->r, tol_f, tol_f));
  }
#endif
}

void test_partial_derivatives_against_numdiff(
    ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
    ActuationModelTypes::Type actuation_type) {
  using namespace boost::placeholders;

  // Create the model
  ResidualModelFactory residual_factory;
  ActuationModelFactory actuation_factory;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      actuation_factory.create(actuation_type, state_type);
  const std::shared_ptr<crocoddyl::ResidualModelAbstract>& model =
      residual_factory.create(residual_type, state_type,
                              actuation_model->get_nu());
  const bool with_actuation = !is_task_residual(residual_type);

  // Create the corresponding shared data
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data;
  std::shared_ptr<crocoddyl::DataCollectorAbstract> shared_data;
  if (with_actuation) {
    actuation_data = actuation_model->createData();
    shared_data = std::make_shared<crocoddyl::DataCollectorActMultibody>(
        &pinocchio_data, actuation_data);
  } else {
    shared_data =
        std::make_shared<crocoddyl::DataCollectorMultibody>(&pinocchio_data);
  }

  // Create the residual data
  const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data =
      model->createData(shared_data.get());

  // Create the equivalent num diff model and data.
  crocoddyl::ResidualModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data_num_diff =
      model_num_diff.createData(shared_data.get());

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the residual derivatives
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  if (with_actuation) {
    actuation_model->calc(actuation_data, x, u);
    actuation_model->calcDiff(actuation_data, x, u);
  }
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  // Computing the residual derivatives via numerical differentiation
  std::vector<crocoddyl::ResidualModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      double, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model, &pinocchio_data, _1, _2));
  if (with_actuation) {
    reevals.push_back(boost::bind(&crocoddyl::unittest::updateActuation<double>,
                                  actuation_model, actuation_data, _1, _2));
  }
  model_num_diff.set_reevals(reevals);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against numdiff
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK(isCloseAbsRel(data->Rx, data_num_diff->Rx, tol, tol));
  BOOST_CHECK(isCloseAbsRel(data->Ru, data_num_diff->Ru, tol, tol));

  // Computing the residual derivatives
  x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  if (with_actuation) {
    actuation_model->calc(actuation_data, x);
    actuation_model->calcDiff(actuation_data, x);
  }

  // Computing the residual derivatives via numerical differentiation
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);

  // Checking the partial derivatives against numdiff
  BOOST_CHECK(isCloseAbsRel(data->Rx, data_num_diff->Rx, tol, tol));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const Eigen::VectorXd x_u = x;
  const auto casted_model = model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  const Eigen::VectorXf x_u_f = x_u.cast<float>();
  const Eigen::VectorXf x_f = x.cast<float>();
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  if (with_actuation) {
    const auto casted_actuation_model = actuation_model->cast<float>();
    const auto casted_actuation_data = casted_actuation_model->createData();
    const Eigen::VectorXf u_f = u.cast<float>();
    crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
        &casted_pinocchio_data, casted_actuation_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x_u);
    crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                            &casted_pinocchio_data, x_u_f);
    actuation_model->calc(actuation_data, x_u, u);
    actuation_model->calcDiff(actuation_data, x_u, u);
    model->calc(data, x_u, u);
    model->calcDiff(data, x_u, u);
    casted_actuation_model->calc(casted_actuation_data, x_u_f, u_f);
    casted_actuation_model->calcDiff(casted_actuation_data, x_u_f, u_f);
    casted_model->calc(casted_data, x_u_f, u_f);
    casted_model->calcDiff(casted_data, x_u_f, u_f);
    BOOST_CHECK(
        isCloseAbsRel(data->Rx.cast<float>(), casted_data->Rx, tol_f, tol_f));
    BOOST_CHECK(
        isCloseAbsRel(data->Ru.cast<float>(), casted_data->Ru, tol_f, tol_f));
  } else {
    crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
        &casted_pinocchio_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                            x_u);
    crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                            &casted_pinocchio_data, x_u_f);
    model->calc(data, x_u, u);
    model->calcDiff(data, x_u, u);
    casted_model->calc(casted_data, x_u_f, u.cast<float>());
    casted_model->calcDiff(casted_data, x_u_f, u.cast<float>());
    BOOST_CHECK(
        isCloseAbsRel(data->Rx.cast<float>(), casted_data->Rx, tol_f, tol_f));
    BOOST_CHECK(
        isCloseAbsRel(data->Ru.cast<float>(), casted_data->Ru, tol_f, tol_f));
  }
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  if (with_actuation) {
    actuation_model->calc(actuation_data, x);
    actuation_model->calcDiff(actuation_data, x);
  }
  model->calc(data, x);
  model->calcDiff(data, x);
  if (with_actuation) {
    const auto casted_actuation_model = actuation_model->cast<float>();
    const auto casted_actuation_data = casted_actuation_model->createData();
    crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
        &casted_pinocchio_data, casted_actuation_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    casted_actuation_model->calc(casted_actuation_data, x_f);
    casted_actuation_model->calcDiff(casted_actuation_data, x_f);
    casted_model->calc(casted_data, x_f);
    casted_model->calcDiff(casted_data, x_f);
    BOOST_CHECK(
        isCloseAbsRel(data->Rx.cast<float>(), casted_data->Rx, tol_f, tol_f));
  } else {
    crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
        &casted_pinocchio_data);
    const auto casted_data = casted_model->createData(&casted_shared_data);
    casted_model->calc(casted_data, x_f);
    casted_model->calcDiff(casted_data, x_f);
    BOOST_CHECK(
        isCloseAbsRel(data->Rx.cast<float>(), casted_data->Rx, tol_f, tol_f));
  }
#endif
}

void test_reference() {
  ResidualModelFactory factory;
  StateModelTypes::Type state_type = StateModelTypes::StateMultibody_Talos;
  ActuationModelTypes::Type actuation_type =
      ActuationModelTypes::ActuationModelFloatingBase;
  StateModelFactory state_factory;
  ActuationModelFactory actuation_factory;
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(state_type));
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation =
      actuation_factory.create(actuation_type, state_type);

  const std::size_t nu = actuation->get_nu();
  const std::size_t nv = state->get_nv();

  // Test reference in state residual
  crocoddyl::ResidualModelState state_residual(state, state->rand(), nu);
  Eigen::VectorXd x_ref = state_residual.get_state()->rand();
  state_residual.set_reference(x_ref);
  BOOST_CHECK(isCloseAbsRel(x_ref, state_residual.get_reference(), 1e-9, 1e-9));
  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ResidualModelStateTpl<float> casted_state_residual =
      state_residual.cast<float>();
  Eigen::VectorXf x_ref_f = casted_state_residual.get_state()->rand();
  casted_state_residual.set_reference(x_ref_f);
  BOOST_CHECK(isCloseAbsRel(x_ref_f, casted_state_residual.get_reference(),
                            1e-6f, 1e-6f));
#endif
  // Test reference in control residual
  crocoddyl::ResidualModelControl control_residual(state, nu);
  Eigen::VectorXd u_ref = Eigen::VectorXd::Random(nu);
  control_residual.set_reference(u_ref);
  BOOST_CHECK(
      isCloseAbsRel(u_ref, control_residual.get_reference(), 1e-9, 1e-9));
  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ResidualModelControlTpl<float> casted_control_residual =
      control_residual.cast<float>();
  Eigen::VectorXf u_ref_f = Eigen::VectorXf::Random(nu);
  casted_control_residual.set_reference(u_ref_f);
  BOOST_CHECK(isCloseAbsRel(u_ref_f, casted_control_residual.get_reference(),
                            1e-6f, 1e-6f));
#endif
  // Test reference in joint-acceleration residual
  crocoddyl::ResidualModelJointAcceleration jacc_residual(state, nu);
  Eigen::VectorXd a_ref = Eigen::VectorXd::Random(nv);
  jacc_residual.set_reference(a_ref);
  BOOST_CHECK(isCloseAbsRel(a_ref, jacc_residual.get_reference(), 1e-9, 1e-9));
  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ResidualModelJointAccelerationTpl<float> casted_jacc_residual =
      jacc_residual.cast<float>();
  Eigen::VectorXf a_ref_f = Eigen::VectorXf::Random(nv);
  casted_jacc_residual.set_reference(a_ref_f);
  BOOST_CHECK(isCloseAbsRel(a_ref_f, casted_jacc_residual.get_reference(),
                            1e-6f, 1e-6f));
#endif
  // Test reference in joint-effort residual
  crocoddyl::ResidualModelJointEffort jeff_residual(state, actuation, nu);
  Eigen::VectorXd tau_ref = Eigen::VectorXd::Random(nu);
  jeff_residual.set_reference(tau_ref);
  BOOST_CHECK(
      isCloseAbsRel(tau_ref, jeff_residual.get_reference(), 1e-9, 1e-9));
  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ResidualModelJointEffortTpl<float> casted_jeff_residual =
      jeff_residual.cast<float>();
  Eigen::VectorXf tau_ref_f = Eigen::VectorXf::Random(nu);
  casted_jeff_residual.set_reference(tau_ref_f);
  BOOST_CHECK(isCloseAbsRel(tau_ref_f, casted_jeff_residual.get_reference(),
                            1e-6f, 1e-6f));
#endif
  // Test reference in centroidal-momentum residual
  crocoddyl::ResidualModelCentroidalMomentum cmon_residual(
      state, Eigen::Matrix<double, 6, 1>::Zero());
  Eigen::Matrix<double, 6, 1> h_ref = Eigen::Matrix<double, 6, 1>::Random();
  cmon_residual.set_reference(h_ref);
  BOOST_CHECK(isCloseAbsRel(h_ref, cmon_residual.get_reference(), 1e-9, 1e-9));
  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ResidualModelCentroidalMomentumTpl<float> casted_cmon_residual =
      cmon_residual.cast<float>();
  Eigen::Matrix<float, 6, 1> h_ref_f = Eigen::Matrix<float, 6, 1>::Random();
  casted_cmon_residual.set_reference(h_ref_f);
  BOOST_CHECK(isCloseAbsRel(h_ref_f, casted_cmon_residual.get_reference(),
                            1e-6f, 1e-6f));
#endif
  // Test reference in com-position residual
  crocoddyl::ResidualModelCoMPosition c_residual(state,
                                                 Eigen::Vector3d::Zero());
  Eigen::Vector3d c_ref = Eigen::Vector3d::Random();
  c_residual.set_reference(c_ref);
  BOOST_CHECK(isCloseAbsRel(c_ref, c_residual.get_reference(), 1e-9, 1e-9));
  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ResidualModelCoMPositionTpl<float> casted_c_residual =
      c_residual.cast<float>();
  Eigen::Vector3f c_ref_f = Eigen::Vector3f::Random();
  casted_c_residual.set_reference(c_ref_f);
  BOOST_CHECK(
      isCloseAbsRel(c_ref_f, casted_c_residual.get_reference(), 1e-6f, 1e-6f));
#endif
}

//----------------------------------------------------------------------------//

void register_residual_model_unit_tests(
    ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
    ActuationModelTypes::Type actuation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << residual_type << "_" << state_type << "_"
            << actuation_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_residual, residual_type,
                                  state_type, actuation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, residual_type,
                                      state_type, actuation_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                  residual_type, state_type, actuation_type)));
  framework::master_test_suite().add(ts);
}

void regiter_residual_reference_unit_tests() {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_reference";
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_reference)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all residuals available with all the activation types with all
  // available states types.
  for (size_t residual_type = 0; residual_type < ResidualModelTypes::all.size();
       ++residual_type) {
    // size_t residual_type = ResidualModelTypes::ResidualModelCoMPosition;
    for (size_t state_type =
             StateModelTypes::all[StateModelTypes::StateMultibody_TalosArm];
         state_type < StateModelTypes::all.size(); ++state_type) {
      for (size_t actuation_type = 0;
           actuation_type < ActuationModelTypes::all.size(); ++actuation_type) {
        if (ActuationModelTypes::all[actuation_type] !=
            ActuationModelTypes::ActuationModelFloatingBaseThrusters) {
          register_residual_model_unit_tests(
              ResidualModelTypes::all[residual_type],
              StateModelTypes::all[state_type],
              ActuationModelTypes::all[actuation_type]);
        } else if (StateModelTypes::all[state_type] !=
                       StateModelTypes::StateMultibody_TalosArm &&
                   StateModelTypes::all[state_type] !=
                       StateModelTypes::StateMultibodyContact2D_TalosArm) {
          register_residual_model_unit_tests(
              ResidualModelTypes::all[residual_type],
              StateModelTypes::all[state_type],
              ActuationModelTypes::all[actuation_type]);
        }
      }
    }
  }
  regiter_residual_reference_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
