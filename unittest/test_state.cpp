///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <Eigen/Dense>

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>

#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/states/unicycle.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

using namespace boost::unit_test;

class StateAbstractFactory {
 public:
  crocoddyl::StateAbstract& get_state() { return *state_; }
  boost::shared_ptr<crocoddyl::StateAbstract> state_;
};

class StateVectorFactory : public StateAbstractFactory {
 public:
  StateVectorFactory(int nx) : StateAbstractFactory() {
    state_vector_ = boost::make_shared<crocoddyl::StateVector>(nx);
    state_ = state_vector_;
  }
  boost::shared_ptr<crocoddyl::StateVector> state_vector_;
};

class StateMultibodyFactory : public StateAbstractFactory {
 public:
  StateMultibodyFactory(const std::string& urdf_file = "", bool free_flyer = true) : StateAbstractFactory() {
    pinocchio_model_.reset(new pinocchio::Model());
    if (urdf_file.size() != 0) {
      if (free_flyer) {
        free_flyer_.reset(new pinocchio::JointModelFreeFlyer());
        pinocchio::urdf::buildModel(urdf_file, *free_flyer_, *pinocchio_model_, true);
        pinocchio_model_->lowerPositionLimit.head<3>().fill(-1.0);
        pinocchio_model_->upperPositionLimit.head<3>().fill(1.0);
      } else {
        pinocchio::urdf::buildModel(urdf_file, *pinocchio_model_);
      }
    } else {
      pinocchio::buildModels::humanoidRandom(*pinocchio_model_, free_flyer);
    }

    state_multibody_.reset(new crocoddyl::StateMultibody(*pinocchio_model_));
    state_ = state_multibody_;
    std::cout << "created the state" << std::endl;
  }
  ~StateMultibodyFactory()
  {
    std::cout << "deleting the StateMultibodyFactory ";
    if (free_flyer_)
    {
      std::cout << "with free flyer.";
    }
    else
    {
      std::cout << "without free flyer.";
    }
    std::cout << std::endl;
  }
  boost::shared_ptr<pinocchio::JointModelFreeFlyer> free_flyer_;
  boost::shared_ptr<crocoddyl::StateMultibody> state_multibody_;
  boost::shared_ptr<pinocchio::Model> pinocchio_model_;
};

void test_state_dimension(StateAbstractFactory& factory, int nx) {
  std::cout << "test_state_dimension" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Checking the dimension of zero and random states
  BOOST_CHECK(state.zero().size() == nx);
  BOOST_CHECK(state.rand().size() == nx);
  std::cout << "test_state_dimension end" << std::endl;
}

void test_integrate_against_difference(StateAbstractFactory& factory) {
  std::cout << "test_integrate_against_difference" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
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
  std::cout << "test_integrate_against_difference end" << std::endl;
}

void test_difference_against_integrate(StateAbstractFactory& factory) {
  std::cout << "test_difference_against_integrate" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Generating random states
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Computing dx by differentiation of its integrate
  Eigen::VectorXd xidx(state.get_nx());
  state.integrate(x, dx, xidx);
  Eigen::VectorXd dxd(state.get_ndx());
  state.diff(x, xidx, dxd);

  // Checking that both states agree
  BOOST_CHECK((dxd - dx).isMuchSmallerThan(1.0, 1e-9));
  std::cout << "test_difference_against_integrate end" << std::endl;
}

void test_Jdiff_firstsecond(StateAbstractFactory& factory) {
  std::cout << "test_Jdiff_firstsecond" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_second(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, Jdiff_first, Jdiff_tmp, crocoddyl::first);
  state.Jdiff(x1, x2, Jdiff_tmp, Jdiff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_both_second(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, Jdiff_both_first, Jdiff_both_second);

  BOOST_CHECK((Jdiff_first - Jdiff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jdiff_second - Jdiff_both_second).isMuchSmallerThan(1.0, 1e-9));
  std::cout << "test_Jdiff_firstsecond end" << std::endl;
}

void test_Jint_firstsecond(StateAbstractFactory& factory) {
  std::cout << "test_Jint_firstsecond" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_second(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x, dx, Jint_first, Jint_tmp, crocoddyl::first);
  state.Jintegrate(x, dx, Jint_tmp, Jint_second, crocoddyl::second);

  // Computing the partial derivatives of the interence function separately
  Eigen::MatrixXd Jint_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_both_second(state.get_ndx(), state.get_ndx());
  state.Jintegrate(x, dx, Jint_both_first, Jint_both_second);

  BOOST_CHECK((Jint_first - Jint_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jint_second - Jint_both_second).isMuchSmallerThan(1.0, 1e-9));
  std::cout << "test_Jint_firstsecond end" << std::endl;
}

void test_Jdiff_num_diff_firstsecond(StateAbstractFactory& factory) {
  std::cout << "test_Jdiff_num_diff_firstsecond" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_first, Jdiff_num_diff_tmp, crocoddyl::first);
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_tmp, Jdiff_num_diff_second, crocoddyl::second);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jdiff_num_diff_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_num_diff_both_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jdiff(x1, x2, Jdiff_num_diff_both_first, Jdiff_num_diff_both_second);

  BOOST_CHECK((Jdiff_num_diff_first - Jdiff_num_diff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jdiff_num_diff_second - Jdiff_num_diff_both_second).isMuchSmallerThan(1.0, 1e-9));
  std::cout << "test_Jdiff_num_diff_firstsecond end" << std::endl;
}

void test_Jint_num_diff_firstsecond(StateAbstractFactory& factory) {
  std::cout << "test_Jint_num_diff_firstsecond" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x = state.rand();
  Eigen::VectorXd dx = Eigen::VectorXd::Random(state.get_ndx());

  // Get the num diff state
  crocoddyl::StateNumDiff state_num_diff(state);

  // Computing the partial derivatives of the difference function separately
  Eigen::MatrixXd Jint_num_diff_tmp(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_diff_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_diff_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_first, Jint_num_diff_tmp, crocoddyl::first);
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_tmp, Jint_num_diff_second, crocoddyl::second);

  // Computing the partial derivatives of the interence function separately
  Eigen::MatrixXd Jint_num_diff_both_first(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jint_num_diff_both_second(state.get_ndx(), state.get_ndx());
  state_num_diff.Jintegrate(x, dx, Jint_num_diff_both_first, Jint_num_diff_both_second);

  BOOST_CHECK((Jint_num_diff_first - Jint_num_diff_both_first).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((Jint_num_diff_second - Jint_num_diff_both_second).isMuchSmallerThan(1.0, 1e-9));
  std::cout << "test_Jint_num_diff_firstsecond end" << std::endl;
}

void test_Jdiff_against_numdiff(StateAbstractFactory& factory, double num_diff_modifier) {
  std::cout << "test_Jdiff_against_numdiff" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
  // Generating random values for the initial and terminal states
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd x2 = state.rand();

  // Computing the partial derivatives of the difference function analytically
  Eigen::MatrixXd Jdiff_1(state.get_ndx(), state.get_ndx());
  Eigen::MatrixXd Jdiff_2(state.get_ndx(), state.get_ndx());
  state.Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::first);
  state.Jdiff(x1, x2, Jdiff_1, Jdiff_2, crocoddyl::second);

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
  std::cout << "test_Jdiff_against_numdiff end" << std::endl;
}

void test_Jintegrate_against_numdiff(StateAbstractFactory& factory, double num_diff_modifier) {
  std::cout << "test_Jintegrate_against_numdiff" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
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
  std::cout << "test_Jintegrate_against_numdiff end" << std::endl;
}

void test_Jdiff_and_Jintegrate_are_inverses(StateAbstractFactory& factory) {
  std::cout << "test_Jdiff_and_Jintegrate_are_inverses" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
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
  std::cout << "test_Jdiff_and_Jintegrate_are_inverses end" << std::endl;
}

void test_velocity_from_Jintegrate_Jdiff(StateAbstractFactory& factory) {
  std::cout << "test_velocity_from_Jintegrate_Jdiff" << std::endl;
  crocoddyl::StateAbstract& state = factory.get_state();
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
  Eigen::VectorXd x2_eps(state.get_ndx());
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
  std::cout << "test_velocity_from_Jintegrate_Jdiff end" << std::endl;
}

void register_state_vector_unit_tests() {
  int nx = 10;
  double num_diff_modifier = 1e4;

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_state_dimension, StateVectorFactory(nx), nx)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_integrate_against_difference, StateVectorFactory(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_difference_against_integrate, StateVectorFactory(nx))));

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_Jdiff_firstsecond, StateVectorFactory(nx))));

  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_Jint_firstsecond, StateVectorFactory(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_num_diff_firstsecond, StateVectorFactory(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jint_num_diff_firstsecond, StateVectorFactory(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_against_numdiff, StateVectorFactory(nx), num_diff_modifier)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jintegrate_against_numdiff, StateVectorFactory(nx), num_diff_modifier)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_and_Jintegrate_are_inverses, StateVectorFactory(nx))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_velocity_from_Jintegrate_Jdiff, StateVectorFactory(nx))));
}

void register_state_multibody_unit_tests(const std::string& urdf_file = "", bool free_flyer = true) {
  double num_diff_modifier = 1e4;
  StateMultibodyFactory factory = StateMultibodyFactory(urdf_file, free_flyer);

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_state_dimension, StateMultibodyFactory(urdf_file, free_flyer),
                                  factory.pinocchio_model_->nq + factory.pinocchio_model_->nv)));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_integrate_against_difference, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_difference_against_integrate, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_firstsecond, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jint_firstsecond, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jdiff_num_diff_firstsecond, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_Jint_num_diff_firstsecond, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_Jdiff_against_numdiff, StateMultibodyFactory(urdf_file, free_flyer), num_diff_modifier)));

  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_Jintegrate_against_numdiff, StateMultibodyFactory(urdf_file, free_flyer), num_diff_modifier)));

  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_Jdiff_and_Jintegrate_are_inverses, StateMultibodyFactory(urdf_file, free_flyer))));

  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_velocity_from_Jintegrate_Jdiff, StateMultibodyFactory(urdf_file, free_flyer))));
}

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_state_vector_unit_tests();
  register_state_multibody_unit_tests(TALOS_ARM_URDF, false);
  // register_state_multibody_unit_tests(HYQ_URDF, true);
  // register_state_multibody_unit_tests(TALOS_URDF);
  register_state_multibody_unit_tests();  // random humanoid
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
