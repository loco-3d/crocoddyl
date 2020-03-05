///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl_unit_test;

//----------------------------------------------------------------------------//

void test_state_dimension(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Checking the dimension of zero and random states
  BOOST_CHECK(static_cast<std::size_t>(state->zero().size()) == factory.get_nx());
  BOOST_CHECK(static_cast<std::size_t>(state->rand().size()) == factory.get_nx());
  BOOST_CHECK(state->get_nx() == (state->get_nq() + state->get_nv()));
  BOOST_CHECK(state->get_ndx() == (2 * state->get_nv()));
  BOOST_CHECK(static_cast<std::size_t>(state->get_lb().size()) == factory.get_nx());
  BOOST_CHECK(static_cast<std::size_t>(state->get_ub().size()) == factory.get_nx());
}

void test_integrate_against_difference(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
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
  BOOST_CHECK(dxi.isMuchSmallerThan(1.0, 1e-9));
}

void test_difference_against_integrate(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random states
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing dx by differentiation of its integrate
  Eigen::VectorXd xidx(state->get_nx());
  state->integrate(x, dx, xidx);
  Eigen::VectorXd dxd(state->get_ndx());
  state->diff(x, xidx, dxd);

  // Checking that both states agree
  BOOST_CHECK((dxd - dx).isMuchSmallerThan(1.0, 1e-9));
}

void test_Jdiff_firstsecond(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_tmp(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_second(state->get_ndx(), state->get_ndx());
  state->Jdiff(x1, x2, Jdiff_first, Jdiff_tmp, crocoddyl::first);
  state->Jdiff(x1, x2, Jdiff_tmp, Jdiff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_both_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_both_second(state->get_ndx(), state->get_ndx());
  state->Jdiff(x1, x2, Jdiff_both_first, Jdiff_both_second);

  BOOST_CHECK((Jdiff_first - Jdiff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jdiff_second - Jdiff_both_second).isMuchSmallerThan(1.0, 1e-9));
}

void test_Jint_firstsecond(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_tmp(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_second(state->get_ndx(), state->get_ndx());
  state->Jintegrate(x, dx, Jint_first, Jint_tmp, crocoddyl::first);
  state->Jintegrate(x, dx, Jint_tmp, Jint_second, crocoddyl::second);

  // Computing the partial derivatives of the integrate function separately
  Eigen::MatrixXd Jint_both_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_both_second(state->get_ndx(), state->get_ndx());
  state->Jintegrate(x, dx, Jint_both_first, Jint_both_second);

  BOOST_CHECK((Jint_first - Jint_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jint_second - Jint_both_second).isMuchSmallerThan(1.0, 1e-9));
}

void test_Jdiff_num_diff_firstsecond(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_tmp(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_second(state->get_ndx(), state->get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_first, Jdiff_num_diff_tmp, crocoddyl::first);
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_tmp, Jdiff_num_diff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_both_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_both_second(state->get_ndx(), state->get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_both_first, Jdiff_num_diff_both_second);

  BOOST_CHECK((Jdiff_num_diff_first - Jdiff_num_diff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jdiff_num_diff_second - Jdiff_num_diff_both_second).isMuchSmallerThan(1.0, 1e-9));
}

void test_Jint_num_diff_firstsecond(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_num_diff_tmp(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_num_diff_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_num_diff_second(state->get_ndx(), state->get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_first, Jint_num_diff_tmp, crocoddyl::first);
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_tmp, Jint_num_diff_second, crocoddyl::second);

  // Computing the partial derivatives of the interence function separately
  Eigen::MatrixXd Jint_num_diff_both_first(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_num_diff_both_second(state->get_ndx(), state->get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_both_first, Jint_num_diff_both_second);

  BOOST_CHECK((Jint_num_diff_first - Jint_num_diff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jint_num_diff_second - Jint_num_diff_both_second).isMuchSmallerThan(1.0, 1e-9));
}

void test_Jdiff_against_numdiff(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random values for the initial and terminal states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& x2 = state->rand();

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jdiff_1(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_2(state->get_ndx(), state->get_ndx());
  state->Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::first);
  state->Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::second);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jdiff_num_1(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdiff_num_2(state->get_ndx(), state->get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_1, Jdiff_num_2);

  // Checking the partial derivatives against NumDiff
  // The previous tolerance was 10*disturbance
  double tol = NUMDIFF_MODIFIER * state_num_diff.get_disturbance();
  BOOST_CHECK((Jdiff_1 - Jdiff_num_1).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((Jdiff_2 - Jdiff_num_2).isMuchSmallerThan(1.0, tol));
}

void test_Jintegrate_against_numdiff(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random values for the initial state and its rate of change
  const Eigen::VectorXd& x = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jint_1(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_2(state->get_ndx(), state->get_ndx());
  state->Jintegrate(x, dx, Jint_1, Jint_2);

  // Computing the partial derivatives of the difference function numerically
  crocoddyl::StateNumDiff state_num_diff(state);
  Eigen::MatrixXd Jint_num_1(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jint_num_2(state->get_ndx(), state->get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_1, Jint_num_2);

  // Checking the partial derivatives against NumDiff
  // The previous tolerance was 10*disturbance
  double tol = NUMDIFF_MODIFIER * state_num_diff.get_disturbance();
  BOOST_CHECK((Jint_1 - Jint_num_1).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((Jint_2 - Jint_num_2).isMuchSmallerThan(1.0, tol));
}

void test_Jdiff_and_Jintegrate_are_inverses(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random states
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& dx = Eigen::VectorXd::Random(state->get_ndx());
  Eigen::VectorXd x2(state->get_nx());
  state->integrate(x1, dx, x2);

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdx(state->get_ndx(), state->get_ndx());
  state->Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd J2(state->get_ndx(), state->get_ndx());
  state->Jdiff(x1, x2, J1, J2);

  // Checking that Jdiff and Jintegrate are inverses
  Eigen::MatrixXd dX_dDX = Jdx;
  Eigen::MatrixXd dDX_dX = J2;
  BOOST_CHECK((dX_dDX - dDX_dX.inverse()).isMuchSmallerThan(1.0, 1e-9));
}

void test_velocity_from_Jintegrate_Jdiff(StateTypes::Type state_type, PinocchioModelTypes::Type model_type) {
  StateFactory factory(state_type, model_type);
  const boost::shared_ptr<crocoddyl::StateAbstract>& state = factory.create();
  // Generating random states
  const Eigen::VectorXd& x1 = state->rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state->get_ndx());
  Eigen::VectorXd x2(state->get_nx());
  state->integrate(x1, dx, x2);
  Eigen::VectorXd eps = Eigen::VectorXd::Random(state->get_ndx());
  double h = 1e-8;

  // Computing the partial derivatives of the integrate and difference function
  Eigen::MatrixXd Jx(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Jdx(state->get_ndx(), state->get_ndx());
  state->Jintegrate(x1, dx, Jx, Jdx);
  Eigen::MatrixXd J1(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd J2(state->get_ndx(), state->get_ndx());
  state->Jdiff(x1, x2, J1, J2);

  // Checking that computed velocity from Jintegrate
  const Eigen::MatrixXd& dX_dDX = Jdx;
  Eigen::VectorXd x2eps(state->get_nx());
  state->integrate(x1, dx + eps * h, x2eps);
  Eigen::VectorXd x2_eps(state->get_ndx());
  state->diff(x2, x2eps, x2_eps);
  BOOST_CHECK((dX_dDX * eps - x2_eps / h).isMuchSmallerThan(1.0, 1e-3));

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
  BOOST_CHECK((J2 * eps - (-dx + dxi) / h).isMuchSmallerThan(1.0, 1e-3));
}

//----------------------------------------------------------------------------//

void register_state_unit_tests(StateTypes::Type state_type, PinocchioModelTypes::Type model_type, test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_state_dimension, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_integrate_against_difference, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_difference_against_integrate, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_firstsecond, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jint_firstsecond, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_num_diff_firstsecond, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jint_num_diff_firstsecond, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_against_numdiff, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jintegrate_against_numdiff, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_and_Jintegrate_are_inverses, state_type, model_type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_velocity_from_Jintegrate_Jdiff, state_type, model_type)));
}

bool init_function() {
  for (size_t i = 0; i < StateTypes::all.size(); ++i) {
    if (StateTypes::all[i] == StateTypes::StateMultibody) {
      for (size_t j = 0; j < PinocchioModelTypes::all.size(); ++j) {
        std::ostringstream test_name;
        test_name << "test_" << StateTypes::all[i] << PinocchioModelTypes::all[j];
        test_suite* ts = BOOST_TEST_SUITE(test_name.str());
        std::cout << "Running " << test_name.str() << std::endl;
        register_state_unit_tests(StateTypes::all[i], PinocchioModelTypes::all[j], *ts);
        framework::master_test_suite().add(ts);
      }
    } else {
      std::ostringstream test_name;
      test_name << "test_" << StateTypes::all[i];
      test_suite* ts = BOOST_TEST_SUITE(test_name.str());
      std::cout << "Running " << test_name.str() << std::endl;
      register_state_unit_tests(StateTypes::all[i], PinocchioModelTypes::NbPinocchioModelTypes, *ts);
      framework::master_test_suite().add(ts);
    }
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
