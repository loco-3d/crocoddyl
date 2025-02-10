///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, INRIA, University of
//                          Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/diff_action.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_check_data(DifferentialActionModelTypes::Type action_type) {
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  BOOST_CHECK(model->checkData(data));
}

void test_calc_returns_state(DifferentialActionModelTypes::Type action_type) {
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // create the corresponding data object
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(static_cast<std::size_t>(data->xout.size()) ==
              model->get_state()->get_nv());
}

void test_calc_returns_a_cost(DifferentialActionModelTypes::Type action_type) {
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_quasi_static(DifferentialActionModelTypes::Type action_type) {
  if (action_type ==
      DifferentialActionModelTypes::
          DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed)
    return;
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, false);

  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  // Getting the cost value computed by calc()
  Eigen::VectorXd x = model->get_state()->rand();
  x.tail(model->get_state()->get_nv()).setZero();
  Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
  model->quasiStatic(data, u, x);
  model->calc(data, x, u);

  // Check for inactive contacts
  if (action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactFwdDynamics_HyQ ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactFwdDynamicsWithFriction_HyQ ||
      action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactFwdDynamics_Talos ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactFwdDynamicsWithFriction_Talos ||
      action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactInvDynamics_HyQ ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactInvDynamicsWithFriction_HyQ ||
      action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactInvDynamics_Talos ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactInvDynamicsWithFriction_Talos) {
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> m =
        std::static_pointer_cast<
            crocoddyl::DifferentialActionModelContactFwdDynamics>(model);
    m->get_contacts()->changeContactStatus("lf", false);

    model->quasiStatic(data, u, x);
    model->calc(data, x, u);

    // Checking that the acceleration is zero as supposed to be in a quasi
    // static condition
    BOOST_CHECK(data->xout.norm() <= 1e-8);
  }
}

void test_partial_derivatives_against_numdiff(
    DifferentialActionModelTypes::Type action_type) {
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  crocoddyl::DifferentialActionModelNumDiff model_num_diff(model);
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->h - data_num_diff->h).isZero(tol));
  BOOST_CHECK((data->g - data_num_diff->g).isZero(tol));
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isZero(tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isZero(tol));
  }
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
  BOOST_CHECK((data->Hu - data_num_diff->Hu).isZero(tol));
  BOOST_CHECK((data->Gx - data_num_diff->Gx).isZero(tol));
  BOOST_CHECK((data->Gu - data_num_diff->Gu).isZero(tol));

  // Computing the action derivatives
  x = model->get_state()->rand();
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);
  BOOST_CHECK((data->h - data_num_diff->h).isZero(tol));
  BOOST_CHECK((data->g - data_num_diff->g).isZero(tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
  }
  BOOST_CHECK((data->Hx - data_num_diff->Hx).isZero(tol));
  BOOST_CHECK((data->Gx - data_num_diff->Gx).isZero(tol));
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(
    DifferentialActionModelTypes::Type action_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_check_data, action_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, action_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, action_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, action_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_quasi_static, action_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    register_action_model_unit_tests(DifferentialActionModelTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
