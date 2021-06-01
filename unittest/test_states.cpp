///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          University of Edinburgh, INRIA
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
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Checking the dimension of zero and random states
  BOOST_CHECK(static_cast<std::size_t>(state->zero().size()) == state->get_nx());
  BOOST_CHECK(static_cast<std::size_t>(state->rand().size()) == state->get_nx());
  BOOST_CHECK(state->get_nx() == (state->get_nq() + state->get_nv()));
  BOOST_CHECK(state->get_ndx() == (2 * state->get_nv()));
  BOOST_CHECK(static_cast<std::size_t>(state->get_lb().size()) == state->get_nx());
  BOOST_CHECK(static_cast<std::size_t>(state->get_ub().size()) == state->get_nx());
}

void test_integrate_against_difference(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Computing x2 by integrating its difference
  Eigen::VectorXd dx(state->get_ndx());
  state->diff(x1, x2, dx);
  Eigen::VectorXd x2i(state->get_nx());
  state->integrate(x1, dx, x2i);

  Eigen::VectorXd dxi(state->get_ndx());
  state->diff(x2i, x2, dxi);

  // Checking that both states agree
  BOOST_CHECK(dxi.isZero(1e-9));
}

void test_difference_against_integrate(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random states
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing dx by differentiation of its integrate
  Eigen::VectorXd xidx(state->get_nx());
  state->integrate(x, dx, xidx);
  Eigen::VectorXd dxd(state->get_ndx());
  state->diff(x, xidx, dxd);

  // Checking that both states agree
  BOOST_CHECK((dxd - dx).isZero(1e-9));
}

void test_Jdiff_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_tmp(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, Jdiff_first, Jdiff_tmp, crocoddyl::first);
  state->Jdiff(x1, x2, Jdiff_tmp, Jdiff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_both_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_both_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, Jdiff_both_first, Jdiff_both_second);

  BOOST_CHECK((Jdiff_first - Jdiff_both_first).isZero(1e-9));
  BOOST_CHECK((Jdiff_second - Jdiff_both_second).isZero(1e-9));
}

void test_Jint_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_tmp(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_first, Jint_tmp, crocoddyl::first);
  state->Jintegrate(x, dx, Jint_tmp, Jint_second, crocoddyl::second);

  // Computing the partial derivatives of the integrate function separately
  Eigen::MatrixXd Jint_both_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_both_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_both_first, Jint_both_second);

  BOOST_CHECK((Jint_first - Jint_both_first).isZero(1e-9));
  BOOST_CHECK((Jint_second - Jint_both_second).isZero(1e-9));
}

void test_Jdiff_num_diff_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_tmp(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_diff_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_diff_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_first, Jdiff_num_diff_tmp, crocoddyl::first);
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_tmp, Jdiff_num_diff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_both_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_diff_both_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_both_first, Jdiff_num_diff_both_second);

  BOOST_CHECK((Jdiff_num_diff_first - Jdiff_num_diff_both_first).isZero(1e-9));
  BOOST_CHECK((Jdiff_num_diff_second - Jdiff_num_diff_both_second).isZero(1e-9));
}

void test_Jint_num_diff_firstsecond(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_num_diff_tmp(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_diff_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_diff_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_first, Jint_num_diff_tmp, crocoddyl::first);
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_tmp, Jint_num_diff_second, crocoddyl::second);

  // Computing the partial derivatives of the given function separately
  Eigen::MatrixXd Jint_num_diff_both_first(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_diff_both_second(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_both_first, Jint_num_diff_both_second);

  BOOST_CHECK((Jint_num_diff_first - Jint_num_diff_both_first).isZero(1e-9));
  BOOST_CHECK((Jint_num_diff_second - Jint_num_diff_both_second).isZero(1e-9));
}

void test_Jdiff_against_numdiff(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jdiff_1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::first);
  state->Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::second);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jdiff_num_1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdiff_num_2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jdiff(x1, x2, Jdiff_num_1, Jdiff_num_2);

  // Checking the partial derivatives against NumDiff
  // The previous tolerance was 10*disturbance
  double tol = NUMDIFF_MODIFIER * sqrt(state_num_diff.get_disturbance());
  BOOST_CHECK((Jdiff_1 - Jdiff_num_1).isZero(tol));
  BOOST_CHECK((Jdiff_2 - Jdiff_num_2).isZero(tol));
}

void test_Jintegrate_against_numdiff(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial state and its rate of change
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jint_1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_1, Jint_2);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jint_num_1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_num_2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state_num_diff.Jintegrate(x, dx, Jint_num_1, Jint_num_2);

  // Checking the partial derivatives against NumDiff
  // The previous tolerance was 10*disturbance
  double tol = sqrt(state_num_diff.get_disturbance());
  BOOST_CHECK((Jint_1 - Jint_num_1).isZero(tol));
  BOOST_CHECK((Jint_2 - Jint_num_2).isZero(tol));
}

void test_JintegrateTransport(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random values for the initial state and its rate of change
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jint_1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jint_2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x, dx, Jint_1, Jint_2);

  Eigen::MatrixXd Jref(Eigen::MatrixXd::Random(state->get_ndx(), 2 * state->get_ndx()));
  const Eigen::MatrixXd Jtest(Jref);

  state->JintegrateTransport(x, dx, Jref, crocoddyl::first);
  BOOST_CHECK((Jref - Jint_1 * Jtest).isZero(1e-10));

  Jref = Jtest;
  state->JintegrateTransport(x, dx, Jref, crocoddyl::second);
  BOOST_CHECK((Jref - Jint_2 * Jtest).isZero(1e-10));
}

void test_Jdiff_and_Jintegrate_are_inverses(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());
  Eigen::VectorXd x2(state->get_nx());
  state->integrate(x1, dx, x2);

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdx(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd J2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, J1, J2);

  // Checking that Jdiff and Jintegrate are inverses
  Eigen::MatrixXd dX_dDX = Jdx;
  Eigen::MatrixXd dDX_dX = J2;
  BOOST_CHECK((dX_dDX - dDX_dX.inverse()).isZero(1e-9));
}

void test_velocity_from_Jintegrate_Jdiff(StateModelTypes::Type state_type) {
  StateModelFactory factory;
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create(state_type);
  // Generating random states
  const Eigen::VectorXd& x1 = state->rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());
  Eigen::VectorXd x2(state->get_nx());
  state->integrate(x1, dx, x2);
  Eigen::VectorXd eps = Eigen::VectorXd::Random(state->get_ndx());
  double h = 1e-8;

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd Jdx(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  Eigen::MatrixXd J2(Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx()));
  state->Jdiff(x1, x2, J1, J2);

  // Checking that computed velocity from Jintegrate
  const Eigen::MatrixXd& dX_dDX = Jdx;
  Eigen::VectorXd x2eps(state->get_nx());
  state->integrate(x1, dx + eps * h, x2eps);
  Eigen::VectorXd x2_eps(state->get_ndx());
  state->diff(x2, x2eps, x2_eps);
  BOOST_CHECK((dX_dDX * eps - x2_eps / h).isZero(1e-3));

  // Checking the velocity computed from Jdiff
  const Eigen::VectorXd& x = state->rand();
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
}

//----------------------------------------------------------------------------//

void register_state_unit_tests(StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_state_dimension, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_integrate_against_difference, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_difference_against_integrate, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jint_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_num_diff_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jint_num_diff_firstsecond, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_against_numdiff, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jintegrate_against_numdiff, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_JintegrateTransport, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_and_Jintegrate_are_inverses, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_velocity_from_Jintegrate_Jdiff, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < StateModelTypes::all.size(); ++i) {
    register_state_unit_tests(StateModelTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
