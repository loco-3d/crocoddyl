///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl_impulses_factory.hpp"
#include "crocoddyl_unittest_common.hpp"

using namespace crocoddyl_unit_test;
using namespace boost::unit_test;
    
//----------------------------------------------------------------------------//

void test_construct_data(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.get_model();

  // create the corresponding data object
  pinocchio::Data pinocchio_data (factory.get_state_factory()->get_pinocchio_model());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);
}

void test_calc_returns_jacobian(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.get_model();

  // create the corresponding data object
  pinocchio::Data pinocchio_data (factory.get_state_factory()->get_pinocchio_model());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // // Generating random state and control vectors
  // Eigen::VectorXd x = model->get_state().rand();
  // Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // // Getting the state dimension from calc() call
  // model->calc(data, x, u);

  // BOOST_CHECK(data->xout.size() == model->get_state().get_nv());
}

void test_partial_derivatives_against_numdiff(ImpulseModelTypes::Type test_type) {
  // create the model
  ImpulseModelFactory factory(test_type);
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model = factory.get_model();

  // create the corresponding data object
  pinocchio::Data pinocchio_data (factory.get_state_factory()->get_pinocchio_model());
  boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data = model->createData(&pinocchio_data);

  // crocoddyl::DifferentialActionModelNumDiff model_num_diff(*model);
  // boost::shared_ptr<crocoddyl::ImpulseDataAbstract> data_num_diff = model_num_diff.createData();

  // // Generating random values for the state and control
  // Eigen::VectorXd x = model->get_state().rand();
  // Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // // Computing the action derivatives
  // model->calcDiff(data, x, u);
  // model_num_diff.calcDiff(data_num_diff, x, u);

  // // Checking the partial derivatives against NumDiff
  // double tol = factory.num_diff_modifier_ * model_num_diff.get_disturbance();
  // BOOST_CHECK((data->Fx - data_num_diff->Fx).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK((data->Fu - data_num_diff->Fu).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK((data->Lx - data_num_diff->Lx).isMuchSmallerThan(1.0, tol));
  // BOOST_CHECK((data->Lu - data_num_diff->Lu).isMuchSmallerThan(1.0, tol));
  // if (model_num_diff.get_with_gauss_approx()) {
  //   BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
  //   BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
  //   BOOST_CHECK((data->Luu - data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  // } else {
  //   BOOST_CHECK((data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
  //   BOOST_CHECK((data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
  //   BOOST_CHECK((data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  // }
}

//----------------------------------------------------------------------------//

void register_unit_tests(ImpulseModelTypes::Type test_type) {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_construct_data, test_type)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_jacobian, test_type)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, test_type)));
}

bool init_function() {
  for (size_t i = 0; i < ImpulseModelTypes::all.size(); ++i) {
    register_unit_tests(ImpulseModelTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
