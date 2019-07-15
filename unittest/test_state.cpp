///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/states/state-euclidean.hpp"
#include "crocoddyl/core/states/state-unicycle.hpp"
#include "crocoddyl/core/states/state-num-diff.hpp"
#include <Eigen/Dense>

using namespace boost::unit_test;

//____________________________________________________________________________//

void test_state_dimension(crocoddyl::StateAbstract& state, int nx) {
  // Checking the dimension of zero and random states
  BOOST_CHECK(state.zero().size() == nx);
  BOOST_CHECK(state.rand().size() == nx);
}

//____________________________________________________________________________//

void test_integrate_against_difference(crocoddyl::StateAbstract& state) {
  // Generating random states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Computing x2 by integrating its difference
  Eigen::VectorXd dx(state.get_ndx());
  state.diff(x1, x2, dx);
  Eigen::VectorXd x2i(state.get_nx());
  state.integrate(x1, dx, x2i);

  Eigen::VectorXd dxi(state.get_ndx());
  state.diff(x2i, x2, dxi);

  // Checking that both states agree
  BOOST_CHECK(dxi.isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_difference_against_integrate(crocoddyl::StateAbstract& state) {
  // Generating random states
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Computing dx by differentiation of its integrate
  Eigen::VectorXd xidx(state.get_nx());
  ;
  state.integrate(x, dx, xidx);
  Eigen::VectorXd dxd(state.get_ndx());
  ;
  state.diff(x, xidx, dxd);

  // Checking that both states agree
  BOOST_CHECK((dxd - dx).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_Jdiff_firstsecond(crocoddyl::StateAbstract& state) {
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_second(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, Jdiff_first, Jdiff_tmp, crocoddyl::Jcomponent::first);
  state.Jdiff(x1, x2, Jdiff_tmp, Jdiff_second, crocoddyl::Jcomponent::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_both_second(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, Jdiff_both_first, Jdiff_both_second);

  BOOST_CHECK((Jdiff_first - Jdiff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jdiff_second - Jdiff_both_second).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_Jint_firstsecond(crocoddyl::StateAbstract& state) {
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_second(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x, dx, Jint_first, Jint_tmp, crocoddyl::Jcomponent::first);
  state.Jintegrate(x, dx, Jint_tmp, Jint_second, crocoddyl::Jcomponent::second);

  // Computing the partial derivatives of the interence function separately
  Eigen::MatrixXd Jint_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_both_second(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x, dx, Jint_both_first, Jint_both_second);

  BOOST_CHECK((Jint_first - Jint_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jint_second - Jint_both_second).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_Jdiff_num_diff_firstsecond(crocoddyl::StateAbstract& state) {
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_first, Jdiff_num_diff_tmp, crocoddyl::Jcomponent::first);
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_tmp, Jdiff_num_diff_second, crocoddyl::Jcomponent::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_both_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_both_first, Jdiff_num_diff_both_second);

  BOOST_CHECK((Jdiff_num_diff_first - Jdiff_num_diff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jdiff_num_diff_second - Jdiff_num_diff_both_second).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_Jint_num_diff_firstsecond(crocoddyl::StateAbstract& state) {
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_num_diff_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_diff_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_diff_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_first, Jint_num_diff_tmp, crocoddyl::Jcomponent::first);
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_tmp, Jint_num_diff_second, crocoddyl::Jcomponent::second);

  // Computing the partial derivatives of the interence function separately
  Eigen::MatrixXd Jint_num_diff_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_diff_both_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_both_first, Jint_num_diff_both_second);

  BOOST_CHECK((Jint_num_diff_first - Jint_num_diff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jint_num_diff_second - Jint_num_diff_both_second).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_Jdiff_against_numdiff(crocoddyl::StateAbstract& state, double num_diff_modifier) {
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jdiff_1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_2(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::Jcomponent::first);
  state.Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::Jcomponent::second);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jdiff_num_1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_2(state.get_ndx(), state.get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_1, Jdiff_num_2);

  // Checking the partial derivatives against NumDiff
  // The previous tolerance was 10*disturbance
  double tol = num_diff_modifier * state_num_diff.get_disturbance();
  BOOST_CHECK((Jdiff_1 - Jdiff_num_1).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((Jdiff_2 - Jdiff_num_2).isMuchSmallerThan(1.0, tol));
}

//____________________________________________________________________________//

void test_Jintegrate_against_numdiff(crocoddyl::StateAbstract& state, double num_diff_modifier) {
  // Generating random values for the initial state and its rate of change
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jint_1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_2(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x, dx, Jint_1, Jint_2);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jint_num_1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_2(state.get_ndx(), state.get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_1, Jint_num_2);

  // Checking the partial derivatives against NumDiff
  // The previous tolerance was 10*disturbance
  double tol = num_diff_modifier * state_num_diff.get_disturbance();
  BOOST_CHECK((Jint_1 - Jint_num_1).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((Jint_2 - Jint_num_2).isMuchSmallerThan(1.0, tol));
}

//____________________________________________________________________________//

void test_Jdiff_and_Jintegrate_are_inverses(crocoddyl::StateAbstract& state) {
  // Generating random states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());
  Eigen::VectorXd x2(state.get_nx());
  state.integrate(x1, dx, x2);

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdx(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd J2(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, J1, J2);

  // Checking that Jdiff and Jintegrate are inverses
  Eigen::MatrixXd dX_dDX = Jdx;
  Eigen::MatrixXd dDX_dX = J2;
  BOOST_CHECK((dX_dDX - dDX_dX.inverse()).isMuchSmallerThan(1.0, 1e-9));
}

//____________________________________________________________________________//

void test_velocity_from_Jintegrate_Jdiff(crocoddyl::StateAbstract& state) {
  // Generating random states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());
  Eigen::VectorXd x2(state.get_nx());
  state.integrate(x1, dx, x2);
  Eigen::VectorXd eps = Eigen::VectorXd::Random(state.get_ndx());
  double h = 1e-8;

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdx(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd J2(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, J1, J2);

  // Checking that computed velocity from Jintegrate
  Eigen::MatrixXd dX_dDX = Jdx;
  Eigen::VectorXd x2eps(state.get_nx());
  state.integrate(x1, dx + eps * h, x2eps);
  Eigen::VectorXd x2_eps(state.get_nx());
  state.diff(x2, x2eps, x2_eps);
  BOOST_CHECK((dX_dDX * eps - x2_eps / h).isMuchSmallerThan(1.0, 1e-3));

  // Checking the velocity computed from Jdiff
  Eigen::VectorXd x = state.rand();
  dx.setZero();
  state.diff(x1, x, dx);
  Eigen::VectorXd x2i(state.get_nx());
  state.integrate(x, eps * h, x2i);
  Eigen::VectorXd dxi(state.get_ndx());
  state.diff(x1, x2i, dxi);
  J1.setZero();
  J2.setZero();
  state.Jdiff(x1, x, J1, J2);
  BOOST_CHECK((J2 * eps - (-dx + dxi) / h).isMuchSmallerThan(1.0, 1e-3));
}

//____________________________________________________________________________//

void register_state_vector_unit_tests() {
  int nx = 10;
  double num_diff_modifier = 1e4;

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_state_dimension, crocoddyl::StateVector(nx), nx)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_integrate_against_difference, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_difference_against_integrate, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_firstsecond, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_Jint_firstsecond, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_num_diff_firstsecond, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jint_num_diff_firstsecond, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_against_numdiff, crocoddyl::StateVector(nx), num_diff_modifier)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jintegrate_against_numdiff, crocoddyl::StateVector(nx), num_diff_modifier)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_and_Jintegrate_are_inverses, crocoddyl::StateVector(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_velocity_from_Jintegrate_Jdiff, crocoddyl::StateVector(nx))));
}

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_state_vector_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
