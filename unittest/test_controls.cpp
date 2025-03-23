///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft,
//                          University of Edinburgh, INRIA,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise controld in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/numdiff/control.hpp"
#include "factory/control.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calcDiff_num_diff(ControlTypes::Type control_type) {
  ControlFactory factory;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>&
      control = factory.create(control_type, 10);

  const std::shared_ptr<crocoddyl::ControlParametrizationDataAbstract>& data =
      control->createData();

  // Generating random values for the control parameters
  const Eigen::VectorXd p = Eigen::VectorXd::Random(control->get_nu());
  double t = Eigen::VectorXd::Random(1)(0) * 0.5 + 1.;  // random in [0, 1]

  // Get the num diff control
  crocoddyl::ControlParametrizationModelNumDiff control_num_diff(control);
  std::shared_ptr<crocoddyl::ControlParametrizationDataAbstract> data_num_diff =
      control_num_diff.createData();

  // Computing the partial derivatives of the value function
  control->calc(data, t, p);
  control_num_diff.calc(data_num_diff, t, p);
  control->calcDiff(data, t, p);
  control_num_diff.calcDiff(data_num_diff, t, p);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(control_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->dw_du - data_num_diff->dw_du).isZero(tol));

  // Checking that casted computation is the same
  const std::shared_ptr<
      crocoddyl::ControlParametrizationModelAbstractTpl<float>>&
      casted_control = control->cast<float>();
  const std::shared_ptr<
      crocoddyl::ControlParametrizationDataAbstractTpl<float>>& casted_data =
      casted_control->createData();
  const Eigen::VectorXf p_f = p.cast<float>();
  float t_f = float(t);
  casted_control->calc(casted_data, t_f, p_f);
  casted_control->calcDiff(casted_data, t_f, p_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->dw_du.cast<float>() - casted_data->dw_du).isZero(tol_f));
}

void test_multiplyByJacobian_num_diff(ControlTypes::Type control_type) {
  ControlFactory factory;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>&
      control = factory.create(control_type, 10);

  std::shared_ptr<crocoddyl::ControlParametrizationDataAbstract> data =
      control->createData();

  // Generating random values for the control parameters, the time, and the
  // matrix to multiply
  const Eigen::VectorXd p = Eigen::VectorXd::Random(control->get_nu());
  double t = Eigen::VectorXd::Random(1)(0) * 0.5 + 1.;  // random in [0, 1]
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(5, control->get_nw());

  // Get the num diff control and datas
  crocoddyl::ControlParametrizationModelNumDiff control_num_diff(control);
  std::shared_ptr<crocoddyl::ControlParametrizationDataAbstract> data_num_diff =
      control_num_diff.createData();

  // Checking the operator
  Eigen::MatrixXd A_J(Eigen::MatrixXd::Zero(A.rows(), control->get_nu()));
  Eigen::MatrixXd A_J_num_diff(
      Eigen::MatrixXd::Zero(A.rows(), control->get_nu()));
  control->calc(data, t, p);
  control->calcDiff(data, t, p);
  control_num_diff.calc(data_num_diff, t, p);
  control_num_diff.calcDiff(data_num_diff, t, p);
  control->multiplyByJacobian(data, A, A_J);
  control_num_diff.multiplyByJacobian(data_num_diff, A, A_J_num_diff);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(control_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((A_J - A_J_num_diff).isZero(tol));

  // Checking that casted computation is the same
  const std::shared_ptr<
      crocoddyl::ControlParametrizationModelAbstractTpl<float>>&
      casted_control = control->cast<float>();
  const std::shared_ptr<
      crocoddyl::ControlParametrizationDataAbstractTpl<float>>& casted_data =
      casted_control->createData();
  Eigen::MatrixXf A_J_f(
      Eigen::MatrixXf::Zero(A.rows(), casted_control->get_nu()));
  const Eigen::VectorXf p_f = p.cast<float>();
  float t_f = float(t);
  const Eigen::MatrixXf A_f = A.cast<float>();
  casted_control->calc(casted_data, t_f, p_f);
  casted_control->calcDiff(casted_data, t_f, p_f);
  casted_control->multiplyByJacobian(casted_data, A_f, A_J_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((data->dw_du.cast<float>() - casted_data->dw_du).isZero(tol_f));
  BOOST_CHECK((A_J.cast<float>() - A_J_f).isZero(tol_f));
}

void test_multiplyJacobianTransposeBy_num_diff(
    ControlTypes::Type control_type) {
  ControlFactory factory;
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>&
      control = factory.create(control_type, 10);

  std::shared_ptr<crocoddyl::ControlParametrizationDataAbstract> data =
      control->createData();

  // Generating random values for the control parameters, the time, and the
  // matrix to multiply
  const Eigen::VectorXd p = Eigen::VectorXd::Random(control->get_nu());
  double t = Eigen::VectorXd::Random(1)(0) * 0.5 + 1.;  // random in [0, 1]
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(control->get_nw(), 5);

  // Get the num diff control and datas
  crocoddyl::ControlParametrizationModelNumDiff control_num_diff(control);
  std::shared_ptr<crocoddyl::ControlParametrizationDataAbstract> data_num_diff =
      control_num_diff.createData();

  // Checking the operator
  Eigen::MatrixXd JT_A(Eigen::MatrixXd::Zero(control->get_nu(), A.cols()));
  Eigen::MatrixXd JT_A_num_diff(
      Eigen::MatrixXd::Zero(control->get_nu(), A.cols()));
  control->calc(data, t, p);
  control->calcDiff(data, t, p);
  control_num_diff.calc(data_num_diff, t, p);
  control_num_diff.calcDiff(data_num_diff, t, p);
  control->multiplyJacobianTransposeBy(data, A, JT_A);
  control_num_diff.multiplyJacobianTransposeBy(data_num_diff, A, JT_A_num_diff);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(control_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((JT_A - JT_A_num_diff).isZero(tol));

  // Checking that casted computation is the same
  const std::shared_ptr<
      crocoddyl::ControlParametrizationModelAbstractTpl<float>>&
      casted_control = control->cast<float>();
  const std::shared_ptr<
      crocoddyl::ControlParametrizationDataAbstractTpl<float>>& casted_data =
      casted_control->createData();
  const Eigen::VectorXf p_f = p.cast<float>();
  float t_f = float(t);
  const Eigen::MatrixXf A_f = A.cast<float>();
  Eigen::MatrixXf JT_A_f(
      Eigen::MatrixXf::Zero(casted_control->get_nu(), A_f.cols()));
  casted_control->calc(casted_data, t_f, p_f);
  casted_control->calcDiff(casted_data, t_f, p_f);
  casted_control->multiplyJacobianTransposeBy(casted_data, A_f, JT_A_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK((JT_A.cast<float>() - JT_A_f).isZero(tol_f));
}

//----------------------------------------------------------------------------//

void register_control_unit_tests(ControlTypes::Type control_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << control_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff_num_diff, control_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_multiplyByJacobian_num_diff, control_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_multiplyJacobianTransposeBy_num_diff, control_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < ControlTypes::all.size(); ++i) {
    register_control_unit_tests(ControlTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
