///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/actuation.hpp"
#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_returns_a_cost(CostModelNoFFTypes::Type cost_type,
                              ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, activation_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);

  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation =
      std::make_shared<crocoddyl::ActuationModelFull>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data =
      actuation->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data,
                                                   actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);
  data->cost = nan("");

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Getting the cost value computed by calc()
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>
      casted_actuation =
          std::make_shared<crocoddyl::ActuationModelFullTpl<float>>(
              casted_state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_actuation_data = casted_actuation->createData();
  crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data, casted_actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  casted_data->cost = float(nan(""));
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  casted_model->calc(casted_data, x_f, u_f);
  BOOST_CHECK(!std::isnan(casted_data->cost));
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
#endif
}

void test_calc_against_numdiff(CostModelNoFFTypes::Type cost_type,
                               ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);

  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation =
      std::make_shared<crocoddyl::ActuationModelFull>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data =
      actuation->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data,
                                                   actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff =
      model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->cost == data_num_diff->cost);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>
      casted_actuation =
          std::make_shared<crocoddyl::ActuationModelFullTpl<float>>(
              casted_state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_actuation_data = casted_actuation->createData();
  crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data, casted_actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  casted_model->calc(casted_data, x_f, u_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
#endif
}

void test_partial_derivatives_against_numdiff(
    CostModelNoFFTypes::Type cost_type,
    ActivationModelTypes::Type activation_type) {
  using namespace boost::placeholders;

  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);

  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_model =
      std::make_shared<crocoddyl::ActuationModelFull>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data =
      actuation_model->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data,
                                                   actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff =
      model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::CostModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      double, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model, &pinocchio_data, _1, _2));
  reevals.push_back(boost::bind(&crocoddyl::unittest::updateActuation<double>,
                                actuation_model, actuation_data, _1, _2));
  model_num_diff.set_reevals(reevals);

  // Computing the cost derivatives
  actuation_model->calc(actuation_data, x, u);
  actuation_model->calcDiff(actuation_data, x, u);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isZero(tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data_num_diff->Luu).isZero(tol));
  }

  // Computing the cost derivatives
  x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  actuation_model->calc(actuation_data, x);
  actuation_model->calcDiff(actuation_data, x);
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);

  // Checking the partial derivatives against numdiff
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isZero(tol));
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>
      casted_actuation =
          std::make_shared<crocoddyl::ActuationModelFullTpl<float>>(
              casted_state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_actuation_data = casted_actuation->createData();
  crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data, casted_actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
  BOOST_CHECK((data->Lx.cast<float>() - casted_data->Lx).isZero(tol_f));
  BOOST_CHECK((data->Lu.cast<float>() - casted_data->Lu).isZero(tol_f));
  BOOST_CHECK((data->Lxx.cast<float>() - casted_data->Lxx).isZero(tol_f));
  BOOST_CHECK((data->Lxu.cast<float>() - casted_data->Lxu).isZero(tol_f));
  BOOST_CHECK((data->Luu.cast<float>() - casted_data->Luu).isZero(tol_f));
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  model->calc(data, x);
  model->calcDiff(data, x);
  casted_model->calc(casted_data, x_f);
  casted_model->calcDiff(casted_data, x_f);
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
  BOOST_CHECK((data->Lx.cast<float>() - casted_data->Lx).isZero(tol_f));
  BOOST_CHECK((data->Lxx.cast<float>() - casted_data->Lxx).isZero(tol_f));
#endif
}

void test_dimensions_in_cost_sum(CostModelNoFFTypes::Type cost_type,
                                 ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);

  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation =
      std::make_shared<crocoddyl::ActuationModelFull>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data =
      actuation->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data,
                                                   actuation_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = state->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  BOOST_CHECK(model->get_state()->get_nx() == cost_sum.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() == cost_sum.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == cost_sum.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() == cost_sum.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() == cost_sum.get_state()->get_nv());
  BOOST_CHECK(model->get_activation()->get_nr() == cost_sum.get_nr());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::CostModelSumTpl<float> casted_cost_sum = cost_sum.cast<float>();
  BOOST_CHECK(model->get_state()->get_nx() ==
              casted_cost_sum.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() ==
              casted_cost_sum.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == casted_cost_sum.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() ==
              casted_cost_sum.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() ==
              casted_cost_sum.get_state()->get_nv());
  BOOST_CHECK(model->get_activation()->get_nr() == casted_cost_sum.get_nr());
#endif
}

void test_partial_derivatives_in_cost_sum(
    CostModelNoFFTypes::Type cost_type,
    ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);

  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation =
      std::make_shared<crocoddyl::ActuationModelFull>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract>& actuation_data =
      actuation->createData();
  crocoddyl::DataCollectorActMultibody shared_data(&pinocchio_data,
                                                   actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);
  const std::shared_ptr<crocoddyl::CostDataSum>& data_sum =
      cost_sum.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  cost_sum.calc(data_sum, x, u);
  cost_sum.calcDiff(data_sum, x, u);

  BOOST_CHECK((data->Lx - data_sum->Lx).isZero());
  BOOST_CHECK((data->Lu - data_sum->Lu).isZero());
  BOOST_CHECK((data->Lxx - data_sum->Lxx).isZero());
  BOOST_CHECK((data->Lxu - data_sum->Lxu).isZero());
  BOOST_CHECK((data->Luu - data_sum->Luu).isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<float>>
      casted_actuation =
          std::make_shared<crocoddyl::ActuationModelFullTpl<float>>(
              casted_state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<float>>&
      casted_actuation_data = casted_actuation->createData();
  crocoddyl::DataCollectorActMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data, casted_actuation_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  crocoddyl::CostModelSumTpl<float> casted_cost_sum = cost_sum.cast<float>();
  const std::shared_ptr<crocoddyl::CostDataSumTpl<float>>& casted_data_sum =
      casted_cost_sum.createData(&casted_shared_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  casted_cost_sum.calc(casted_data_sum, x_f, u_f);
  casted_cost_sum.calcDiff(casted_data_sum, x_f, u_f);
  BOOST_CHECK((casted_data->Lx - casted_data_sum->Lx).isZero());
  BOOST_CHECK((casted_data->Lu - casted_data_sum->Lu).isZero());
  BOOST_CHECK((casted_data->Lxx - casted_data_sum->Lxx).isZero());
  BOOST_CHECK((casted_data->Lxu - casted_data_sum->Lxu).isZero());
  BOOST_CHECK((casted_data->Luu - casted_data_sum->Luu).isZero());
#endif
}

//----------------------------------------------------------------------------//

void register_cost_model_unit_tests(
    CostModelNoFFTypes::Type cost_type,
    ActivationModelTypes::Type activation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << cost_type << "_" << activation_type
            << "_StateMultibody_TalosArm";
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_returns_a_cost, cost_type, activation_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_against_numdiff, cost_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                      cost_type, activation_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_dimensions_in_cost_sum, cost_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_in_cost_sum,
                                      cost_type, activation_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all costs available with all the activation types with state type
  // TalosArm.
  for (size_t cost_type = 0; cost_type < CostModelNoFFTypes::all.size();
       ++cost_type) {
    for (size_t activation_type = 0;
         activation_type < ActivationModelTypes::all.size();
         ++activation_type) {
      register_cost_model_unit_tests(
          CostModelNoFFTypes::all[cost_type],
          ActivationModelTypes::all[activation_type]);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
