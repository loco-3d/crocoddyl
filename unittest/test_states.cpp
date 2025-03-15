///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
//                          INRIA, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_state_dimension(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Checking the dimension of zero and random states
  BOOST_CHECK(static_cast<std::size_t>(state->zero().size()) ==
              state->get_nx());
  BOOST_CHECK(static_cast<std::size_t>(state->rand().size()) ==
              state->get_nx());
  BOOST_CHECK(state->get_nx() == (state->get_nq() + state->get_nv()));
  BOOST_CHECK(state->get_ndx() == (2 * state->get_nv()));
  BOOST_CHECK(static_cast<std::size_t>(state->get_lb().size()) ==
              state->get_nx());
  BOOST_CHECK(static_cast<std::size_t>(state->get_ub().size()) ==
              state->get_nx());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  BOOST_CHECK(static_cast<std::size_t>(casted_state->zero().size()) ==
              casted_state->get_nx());
  BOOST_CHECK(static_cast<std::size_t>(casted_state->rand().size()) ==
              casted_state->get_nx());
  BOOST_CHECK(casted_state->get_nx() ==
              (casted_state->get_nq() + casted_state->get_nv()));
  BOOST_CHECK(casted_state->get_ndx() == (2 * casted_state->get_nv()));
  BOOST_CHECK(static_cast<std::size_t>(casted_state->get_lb().size()) ==
              casted_state->get_nx());
  BOOST_CHECK(static_cast<std::size_t>(casted_state->get_ub().size()) ==
              casted_state->get_nx());
#endif
}

void test_integrate_against_difference(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random states
  const Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd x2 = state->rand();

  // Computing x2 by integrating its difference
  Eigen::VectorXd dx(state->get_ndx());
  Eigen::VectorXd x2i(state->get_nx());
  Eigen::VectorXd dxi(state->get_ndx());
  state->diff(x1, x2, dx);
  state->integrate(x1, dx, x2i);
  state->diff(x2i, x2, dxi);

  // Checking that both states agree
  BOOST_CHECK(dxi.isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  const Eigen::VectorXf x1_f = casted_state->rand();
  const Eigen::VectorXf x2_f = casted_state->rand();
  Eigen::VectorXf dx_f(casted_state->get_ndx());
  Eigen::VectorXf x2i_f(casted_state->get_nx());
  Eigen::VectorXf dxi_f(casted_state->get_ndx());
  casted_state->diff(x1_f, x2_f, dx_f);
  casted_state->integrate(x1_f, dx_f, x2i_f);
  casted_state->diff(x2i_f, x2_f, dxi_f);
  BOOST_CHECK(dxi_f.isZero(1e-6f));
#endif
}

void test_difference_against_integrate(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random states
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing dx by differentiation of its integrate
  Eigen::VectorXd xidx(state->get_nx());
  Eigen::VectorXd dxd(state->get_ndx());
  state->integrate(x, dx, xidx);
  state->diff(x, xidx, dxd);

  // Checking that both states agree
  BOOST_CHECK((dxd - dx).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  const Eigen::VectorXf x_f = casted_state->rand();
  const Eigen::VectorXf dx_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  Eigen::VectorXf xidx_f(casted_state->get_nx());
  Eigen::VectorXf dxd_f(casted_state->get_ndx());
  casted_state->integrate(x_f, dx_f, xidx_f);
  casted_state->diff(x_f, xidx_f, dxd_f);
  BOOST_CHECK((dxd_f - dx_f).isZero(1e-6f));
#endif
}

void test_Jdiff_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random values for the initial and terminal states
  const Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd x2 = state->rand();

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_tmp(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, Jdiff_first, Jdiff_tmp, crocoddyl::first);
  state->Jdiff(x1, x2, Jdiff_tmp, Jdiff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_both_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_both_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, Jdiff_both_first, Jdiff_both_second);

  BOOST_CHECK((Jdiff_first - Jdiff_both_first).isZero(1e-9));
  BOOST_CHECK((Jdiff_second - Jdiff_both_second).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  const Eigen::VectorXf x1_f = casted_state->rand();
  const Eigen::VectorXf x2_f = casted_state->rand();
  Eigen::MatrixXf Jdiff_tmp_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jdiff_first_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jdiff_second_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jdiff_both_first_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jdiff_both_second_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  casted_state->Jdiff(x1_f, x2_f, Jdiff_first_f, Jdiff_tmp_f, crocoddyl::first);
  casted_state->Jdiff(x1_f, x2_f, Jdiff_tmp_f, Jdiff_second_f,
                      crocoddyl::second);
  casted_state->Jdiff(x1_f, x2_f, Jdiff_both_first_f, Jdiff_both_second_f);
  BOOST_CHECK((Jdiff_first_f - Jdiff_both_first_f).isZero(1e-9f));
  BOOST_CHECK((Jdiff_second_f - Jdiff_both_second_f).isZero(1e-9f));
#endif
}

void test_Jint_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random values for the initial and terminal states
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_tmp(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_first, Jint_tmp, crocoddyl::first);
  state->Jintegrate(x, dx, Jint_tmp, Jint_second, crocoddyl::second);

  // Computing the partial derivatives of the integrate function separately
  Eigen::MatrixXd Jint_both_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_both_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_both_first, Jint_both_second);

  BOOST_CHECK((Jint_first - Jint_both_first).isZero(1e-9));
  BOOST_CHECK((Jint_second - Jint_both_second).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  const Eigen::VectorXf x_f = casted_state->rand();
  const Eigen::VectorXf dx_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  Eigen::MatrixXf Jint_tmp_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_first_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_second_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_both_first_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_both_second_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  casted_state->Jintegrate(x_f, dx_f, Jint_first_f, Jint_tmp_f,
                           crocoddyl::first);
  casted_state->Jintegrate(x_f, dx_f, Jint_tmp_f, Jint_second_f,
                           crocoddyl::second);
  casted_state->Jintegrate(x_f, dx_f, Jint_both_first_f, Jint_both_second_f);
  BOOST_CHECK((Jint_first_f - Jint_both_first_f).isZero(1e-9f));
  BOOST_CHECK((Jint_second_f - Jint_both_second_f).isZero(1e-9f));
#endif
}

void test_Jdiff_num_diff_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd x2 = state->rand();

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_tmp(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_diff_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_diff_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_first, Jdiff_num_diff_tmp,
                       crocoddyl::first);
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_tmp, Jdiff_num_diff_second,
                       crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_both_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_diff_both_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_both_first,
                       Jdiff_num_diff_both_second);

  BOOST_CHECK((Jdiff_num_diff_first - Jdiff_num_diff_both_first).isZero(1e-9));
  BOOST_CHECK(
      (Jdiff_num_diff_second - Jdiff_num_diff_both_second).isZero(1e-9));
}

void test_Jint_num_diff_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_num_diff_tmp(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_diff_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_diff_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_first, Jint_num_diff_tmp,
                            crocoddyl::first);
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_tmp, Jint_num_diff_second,
                            crocoddyl::second);

  // Computing the partial derivatives of the given function separately
  Eigen::MatrixXd Jint_num_diff_both_first(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_diff_both_second(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_both_first,
                            Jint_num_diff_both_second);

  BOOST_CHECK((Jint_num_diff_first - Jint_num_diff_both_first).isZero(1e-9));
  BOOST_CHECK((Jint_num_diff_second - Jint_num_diff_both_second).isZero(1e-9));
}

void test_Jdiff_against_numdiff(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd x2 = state->rand();

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jdiff_1(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_2(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::first);
  state->Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::second);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jdiff_num_1(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_2(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jdiff(x1, x2, Jdiff_num_1, Jdiff_num_2);

  // Checking the partial derivatives against numerical differentiation
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(std::sqrt(2.0 * std::numeric_limits<double>::epsilon()),
                        1. / 3.);
  BOOST_CHECK((Jdiff_1 - Jdiff_num_1).isZero(tol));
  BOOST_CHECK((Jdiff_2 - Jdiff_num_2).isZero(tol));
}

void test_Jintegrate_against_numdiff(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random values for the initial state and its rate of change
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jint_1(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_2(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_1, Jint_2);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jint_num_1(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_2(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jintegrate(x, dx, Jint_num_1, Jint_num_2);

  // Checking the partial derivatives against numerical differentiation
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(std::sqrt(2.0 * std::numeric_limits<double>::epsilon()),
                        1. / 3.);
  BOOST_CHECK((Jint_1 - Jint_num_1).isZero(tol));
  BOOST_CHECK((Jint_2 - Jint_num_2).isZero(tol));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  float tol_f = std::sqrt(float(2.0) * std::numeric_limits<float>::epsilon());
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  crocoddyl::StateNumDiffTpl<float> casted_state_num_diff =
      state_num_diff.cast<float>();
  const Eigen::VectorXf x_f = casted_state->rand();
  const Eigen::VectorXf dx_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  Eigen::MatrixXf Jint_1_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_2_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_num_1_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_num_2_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  casted_state_num_diff.Jintegrate(x_f, dx_f, Jint_num_1_f, Jint_num_2_f);
  casted_state->Jintegrate(x_f, dx_f, Jint_1_f, Jint_2_f);
  BOOST_CHECK((Jint_1_f - Jint_num_1_f).isZero(tol_f));
  BOOST_CHECK((Jint_2_f - Jint_num_2_f).isZero(tol_f));
#endif
}

void test_JintegrateTransport(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random values for the initial state and its rate of change
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jint_1(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_2(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_1, Jint_2);

  Eigen::MatrixXd Jref(
      Eigen::MatrixXd::Random(state->get_ndx(), 2 * state->get_ndx()));
  const Eigen::MatrixXd Jtest(Jref);

  state->JintegrateTransport(x, dx, Jref, crocoddyl::first);
  BOOST_CHECK((Jref - Jint_1 * Jtest).isZero(1e-10));

  Jref = Jtest;
  state->JintegrateTransport(x, dx, Jref, crocoddyl::second);
  BOOST_CHECK((Jref - Jint_2 * Jtest).isZero(1e-10));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  const Eigen::VectorXf x_f = casted_state->rand();
  const Eigen::VectorXf dx_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  Eigen::MatrixXf Jint_1_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jint_2_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jref_f(Eigen::MatrixXf::Random(casted_state->get_ndx(),
                                                 2 * casted_state->get_ndx()));
  const Eigen::MatrixXf Jtest_f(Jref_f);
  casted_state->Jintegrate(x_f, dx_f, Jint_1_f, Jint_2_f);
  Jref_f = Jtest_f;
  casted_state->JintegrateTransport(x_f, dx_f, Jref_f, crocoddyl::first);
  BOOST_CHECK((Jref_f - Jint_1_f * Jtest_f).isZero(1e-6f));
  casted_state->JintegrateTransport(x_f, dx_f, Jref_f, crocoddyl::second);
  Jref_f = Jtest_f;
  casted_state->JintegrateTransport(x_f, dx_f, Jref_f, crocoddyl::second);
  BOOST_CHECK((Jref_f - Jint_2_f * Jtest_f).isZero(1e-6f));
#endif
}

void test_Jdiff_and_Jintegrate_are_inverses(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random states
  const Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());
  Eigen::VectorXd x2(state->get_nx());
  state->integrate(x1, dx, x2);

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdx(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd J2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, J1, J2);

  // Checking that Jdiff and Jintegrate are inverses
  Eigen::MatrixXd dX_dDX = Jdx;
  Eigen::MatrixXd dDX_dX = J2;
  BOOST_CHECK((dX_dDX - dDX_dX.inverse()).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  const Eigen::VectorXf x1_f = casted_state->rand();
  const Eigen::VectorXf dx_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  Eigen::VectorXf x2_f(casted_state->get_nx());

  Eigen::MatrixXf Jx_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jdx_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf J1_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf J2_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf dX_dDX_f = Jdx_f;
  Eigen::MatrixXf dDX_dX_f = J2_f;
  casted_state->integrate(x1_f, dx_f, x2_f);
  casted_state->Jintegrate(x1_f, dx_f, Jx_f, Jdx_f);
  casted_state->Jdiff(x1_f, x2_f, J1_f, J2_f);
  dX_dDX_f = Jdx_f;
  dDX_dX_f = J2_f;
  BOOST_CHECK((dX_dDX_f - dDX_dX_f.inverse()).isZero(1e-4f));
#endif
}

void test_velocity_from_Jintegrate_Jdiff(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      factory.create(state_type);

  // Generating random states
  const Eigen::VectorXd x1 = state->rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());
  Eigen::VectorXd x2(state->get_nx());
  state->integrate(x1, dx, x2);
  Eigen::VectorXd eps = Eigen::VectorXd::Random(state->get_ndx());
  double h = 1e-8;

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdx(
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd J2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, J1, J2);

  // Checking that computed velocity from Jintegrate
  Eigen::MatrixXd dX_dDX = Jdx;
  Eigen::VectorXd x2eps(state->get_nx());
  state->integrate(x1, dx + eps * h, x2eps);
  Eigen::VectorXd x2_eps(state->get_ndx());
  state->diff(x2, x2eps, x2_eps);
  BOOST_CHECK((dX_dDX * eps - x2_eps / h).isZero(1e-3));

  // Checking the velocity computed from Jdiff
  const Eigen::VectorXd x = state->rand();
  dx.setZero();
  state->diff(x1, x, dx);
  Eigen::VectorXd x2i(state->get_nx());
  state->integrate(x, eps * h, x2i);
  Eigen::VectorXd dxi(state->get_ndx());
  state->diff(x1, x2i, dxi);
  J1.setZero();
  J2.setZero();
  state->Jdiff(x1, x, J1, J2);
  BOOST_CHECK((J2 * eps - (-dx + dxi) / h).isZero(1e-3));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  float h_f = std::sqrt(float(2.0) * std::numeric_limits<float>::epsilon());
  Eigen::VectorXf eps_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  const Eigen::VectorXf x1_f = casted_state->rand();
  Eigen::VectorXf dx_f = Eigen::VectorXf::Random(casted_state->get_ndx());
  Eigen::VectorXf x2_f(casted_state->get_nx());
  Eigen::MatrixXf Jx_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf Jdx_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf J1_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf J2_f(
      Eigen::MatrixXf::Zero(casted_state->get_ndx(), casted_state->get_ndx()));
  Eigen::MatrixXf dX_dDX_f = Jdx_f;
  Eigen::VectorXf x2eps_f(casted_state->get_nx());
  Eigen::VectorXf x2_eps_f(casted_state->get_ndx());
  const Eigen::VectorXf x_f = casted_state->rand();
  Eigen::VectorXf x2i_f(casted_state->get_nx());
  Eigen::VectorXf dxi_f(casted_state->get_ndx());
  casted_state->integrate(x1_f, dx_f, x2_f);
  casted_state->Jintegrate(x1_f, dx_f, Jx_f, Jdx_f);
  casted_state->Jdiff(x1_f, x2_f, J1_f, J2_f);
  dX_dDX_f = Jdx_f;
  casted_state->integrate(x1_f, dx_f + eps_f * h_f, x2eps_f);
  casted_state->diff(x2_f, x2eps_f, x2_eps_f);
  BOOST_CHECK((dX_dDX_f * eps_f - x2_eps_f / h_f).isZero(1e-3f));
  dx_f.setZero();
  casted_state->diff(x1_f, x_f, dx_f);
  casted_state->integrate(x_f, eps_f * h_f, x2i_f);
  casted_state->diff(x1_f, x2i_f, dxi_f);
  casted_state->Jdiff(x1_f, x_f, J1_f, J2_f);
  BOOST_CHECK((J2_f * eps_f - (dxi_f - dx_f) / h_f).isZero(1e-2f));
#endif
}

//----------------------------------------------------------------------------//

void register_state_unit_tests(StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_state_dimension, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_integrate_against_difference, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_difference_against_integrate, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jint_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_Jdiff_num_diff_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_Jint_num_diff_firstsecond, state_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_against_numdiff, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_Jintegrate_against_numdiff, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_JintegrateTransport, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_Jdiff_and_Jintegrate_are_inverses, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_velocity_from_Jintegrate_Jdiff, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < StateModelTypes::all.size(); ++i) {
    register_state_unit_tests(StateModelTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
