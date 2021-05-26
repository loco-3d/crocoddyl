///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                     University of Edinburgh, INRIA, University of Trento
// Copyright note valid unless otherwise controld in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/control.hpp"
#include "crocoddyl/core/numdiff/control.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_Dvalue_num_diff(ControlTypes::Type control_type) {
  ControlFactory factory;
  const boost::shared_ptr<crocoddyl::ControlAbstract>& control = factory.create(control_type, 10);
  // Generating random values for the control parameters
  const Eigen::VectorXd& p = Eigen::VectorXd::Random(control->get_np());
  double t = Eigen::VectorXd::Random(1)(0)*0.5 + 1.; // random in [0, 1]

  // Get the num diff control
  ControlNumDiff control_num_diff(control);

  // Computing the partial derivatives of the value function
  Eigen::MatrixXd J(Eigen::MatrixXd::Zero(control->get_nu(), control->get_np()));
  Eigen::MatrixXd J_num_diff(Eigen::MatrixXd::Zero(control->get_nu(), control->get_np()));
  control->dValue(t, p, J);
  control_num_diff.dValue(t, p, J_num_diff);

  BOOST_CHECK((J - J_num_diff).isZero(1e-9));
}

void test_multiplyByDValue_num_diff(ControlTypes::Type control_type) {
  ControlFactory factory;
  const boost::shared_ptr<crocoddyl::ControlAbstract>& control = factory.create(control_type, 10);
  // Generating random values for the control parameters, the time, and the matrix to multiply
  const Eigen::VectorXd& p = Eigen::VectorXd::Random(control->get_np());
  double t = Eigen::VectorXd::Random(1)(0)*0.5 + 1.; // random in [0, 1]
  const Eigen::MatrixXd& A = Eigen::MatrixXd::Random(5, control->get_nu());

  // Get the num diff control
  ControlNumDiff control_num_diff(control);

  // Computing 
  Eigen::MatrixXd A_J(Eigen::MatrixXd::Zero(A.rows(), control->get_np()));
  Eigen::MatrixXd A_J_num_diff(Eigen::MatrixXd::Zero(A.rows(), control->get_np()));
  control->multiplyByDValue(t, p, A, A_J);
  control_num_diff.multiplyByDValue(t, p, A, A_J_num_diff);

  BOOST_CHECK((A_J - A_J_num_diff).isZero(1e-9));
}

void test_multiplyDValueTransposeBy_num_diff(ControlTypes::Type control_type) {
  ControlFactory factory;
  const boost::shared_ptr<crocoddyl::ControlAbstract>& control = factory.create(control_type, 10);
  // Generating random values for the control parameters, the time, and the matrix to multiply
  const Eigen::VectorXd& p = Eigen::VectorXd::Random(control->get_np());
  double t = Eigen::VectorXd::Random(1)(0)*0.5 + 1.; // random in [0, 1]
  const Eigen::MatrixXd& A = Eigen::MatrixXd::Random(control->get_nu(), 5);

  // Get the num diff control
  ControlNumDiff control_num_diff(control);

  // Computing 
  Eigen::MatrixXd JT_A(Eigen::MatrixXd::Zero(control->get_np(), A.cols()));
  Eigen::MatrixXd JT_A_num_diff(Eigen::MatrixXd::Zero(control->get_np(), A.cols()));
  control->multiplyDValueTransposeBy(t, p, A, JT_A);
  control_num_diff.multiplyDValueTransposeBy(t, p, A, JT_A_num_diff);

  BOOST_CHECK((JT_A - JT_A_num_diff).isZero(1e-9));
}

//----------------------------------------------------------------------------//

void register_control_unit_tests(ControlTypes::Type control_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << control_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Dvalue_num_diff, control_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_multiplyByDValue_num_diff, control_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_multiplyDValueTransposeBy_num_diff, control_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < ControlTypes::all.size(); ++i) {
    register_control_unit_tests(ControlTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
