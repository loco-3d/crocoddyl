///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/math/quaternion.hpp>
#include "crocoddyl/multibody/cop-support.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

void test_constructor() {
  // Common parameters
  Eigen::Vector2d box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));

  // No rotation
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

  // Create the CoP support support
  crocoddyl::CoPSupport support(R, box);

  BOOST_CHECK(support.get_R().isApprox(R));
  BOOST_CHECK(support.get_box().isApprox(box));
  BOOST_CHECK(static_cast<std::size_t>(support.get_A().rows()) == 4);
  BOOST_CHECK(static_cast<std::size_t>(support.get_lb().size()) == 4);
  BOOST_CHECK(static_cast<std::size_t>(support.get_ub().size()) == 4);

  // With rotation
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  R = q.toRotationMatrix();

  // Create the wrench support
  support = crocoddyl::CoPSupport(R, box);

  BOOST_CHECK(support.get_R().isApprox(R));
  BOOST_CHECK(support.get_box().isApprox(box));
  BOOST_CHECK(static_cast<std::size_t>(support.get_A().rows()) == 4);
  BOOST_CHECK(static_cast<std::size_t>(support.get_lb().size()) == 4);
  BOOST_CHECK(static_cast<std::size_t>(support.get_ub().size()) == 4);

  // Create the wrench support from a reference
  {
    crocoddyl::CoPSupport support_from_reference(support);

    BOOST_CHECK(support.get_R().isApprox(support_from_reference.get_R()));
    BOOST_CHECK(support.get_box().isApprox(support_from_reference.get_box()));
    BOOST_CHECK(support.get_A().isApprox(support_from_reference.get_A()));
    BOOST_CHECK(support.get_ub().isApprox(support_from_reference.get_ub()));
    BOOST_CHECK(support.get_lb().isApprox(support_from_reference.get_lb()));
  }

  // Create the wrench support through the copy operator
  {
    crocoddyl::CoPSupport support_from_copy;
    support_from_copy = support;

    BOOST_CHECK(support.get_R().isApprox(support_from_copy.get_R()));
    BOOST_CHECK(support.get_box().isApprox(support_from_copy.get_box()));
    BOOST_CHECK(support.get_A().isApprox(support_from_copy.get_A()));
    BOOST_CHECK(support.get_ub().isApprox(support_from_copy.get_ub()));
    BOOST_CHECK(support.get_lb().isApprox(support_from_copy.get_lb()));
  }
}

void test_A_matrix_with_rotation_change() {
  // Common parameters
  Eigen::Vector2d box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));

  // No rotation
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  crocoddyl::CoPSupport support_1(R, box);

  // With rotation
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  R = q.toRotationMatrix();
  crocoddyl::CoPSupport support_2(R, box);

  for (std::size_t i = 0; i < 4; ++i) {
    BOOST_CHECK((support_1.get_A().row(i).head(3) - support_2.get_A().row(i).head(3) * R).isZero(1e-9));
  }
}

void test_cop_within_the_support_region() {
  // Create the CoP support with a random orientation
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Vector2d box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::CoPSupport support(R, box);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(support.get_lb(), support.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value with a force along the contact normal
  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.head(3) = random_real_in_range(0., 100.) * R.col(2);
  Eigen::VectorXd r = support.get_A() * wrench;
  activation.calc(data, r);
  BOOST_CHECK(data->a_value == 0.);

  // Create the CoP support with no rotation
  R = Eigen::Matrix3d::Identity();
  support = crocoddyl::CoPSupport(R, box);

  // Compute the activation value with a small enough torque in X
  wrench.setZero();
  wrench(5) = random_real_in_range(0., 100.);
  wrench(0) = (box(0) - 1e-9) * wrench(5) / 2.;
  r = support.get_A() * wrench;
  activation.calc(data, r);
  BOOST_CHECK(data->a_value == 0.);

  // Compute the activation value with a small enough torque in Y
  wrench.setZero();
  wrench(5) = random_real_in_range(0., 100.);
  wrench(1) = (box(1) - 1e-9) * wrench(5) / 2.;
  r = support.get_A() * wrench;
  activation.calc(data, r);
  BOOST_CHECK(data->a_value == 0.);
}

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_A_matrix_with_rotation_change)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_cop_within_the_support_region)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
