///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/math/quaternion.hpp>
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

void test_constructor() {
  // Common parameters
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;

  // No rotation
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

  // Create the friction cone with rotation and surface normal
  crocoddyl::FrictionCone cone(R, mu, nf, inner_appr);

  BOOST_CHECK(cone.get_R().isApprox(R));
  BOOST_CHECK(cone.get_mu() == mu);
  BOOST_CHECK(cone.get_nf() == nf);
  BOOST_CHECK(cone.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_A().rows()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_lb().size()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_ub().size()) == nf + 1);

  // With rotation
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  R = q.toRotationMatrix();

  // Create the friction cone
  cone = crocoddyl::FrictionCone(R, mu, nf, inner_appr);

  BOOST_CHECK(cone.get_R().isApprox(R));
  BOOST_CHECK(cone.get_mu() == mu);
  BOOST_CHECK(cone.get_nf() == nf);
  BOOST_CHECK(cone.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_A().rows()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_lb().size()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_ub().size()) == nf + 1);

  // Create the friction cone from a reference
  {
    crocoddyl::FrictionCone cone_reference(cone);

    BOOST_CHECK(cone.get_nf() == cone_reference.get_nf());
    BOOST_CHECK(cone.get_A().isApprox(cone_reference.get_A()));
    BOOST_CHECK(cone.get_ub().isApprox(cone_reference.get_ub()));
    BOOST_CHECK(cone.get_lb().isApprox(cone_reference.get_lb()));
    BOOST_CHECK(cone.get_R().isApprox(cone_reference.get_R()));
    BOOST_CHECK(std::abs(cone.get_mu() - cone_reference.get_mu()) < 1e-9);
    BOOST_CHECK(cone.get_inner_appr() == cone_reference.get_inner_appr());
    BOOST_CHECK(std::abs(cone.get_min_nforce() - cone_reference.get_min_nforce()) < 1e-9);
    BOOST_CHECK(std::abs(cone.get_max_nforce() - cone_reference.get_max_nforce()) < 1e-9);
  }

  // Create the friction cone through the copy operator
  {
    crocoddyl::FrictionCone cone_copy;
    cone_copy = cone;

    BOOST_CHECK(cone.get_nf() == cone_copy.get_nf());
    BOOST_CHECK(cone.get_A().isApprox(cone_copy.get_A()));
    BOOST_CHECK(cone.get_ub().isApprox(cone_copy.get_ub()));
    BOOST_CHECK(cone.get_lb().isApprox(cone_copy.get_lb()));
    BOOST_CHECK(cone.get_R().isApprox(cone_copy.get_R()));
    BOOST_CHECK(std::abs(cone.get_mu() - cone_copy.get_mu()) < 1e-9);
    BOOST_CHECK(cone.get_inner_appr() == cone_copy.get_inner_appr());
    BOOST_CHECK(std::abs(cone.get_min_nforce() - cone_copy.get_min_nforce()) < 1e-9);
    BOOST_CHECK(std::abs(cone.get_max_nforce() - cone_copy.get_max_nforce()) < 1e-9);
  }
}

void test_inner_approximation_of_friction_cone() {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = true;
  crocoddyl::FrictionCone cone(R, mu, nf, inner_appr);
  const Eigen::VectorXd A_mu = -cone.get_A().col(2);
  for (std::size_t i = 0; i < nf; ++i) {
    BOOST_CHECK_CLOSE(A_mu(i), mu * cos((2 * M_PI / static_cast<double>(nf)) / 2.), 1e-9);
  }
}

void test_A_matrix_with_rotation_change() {
  // Common parameters
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;

  // No rotation
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  crocoddyl::FrictionCone cone_1(R, mu, nf, inner_appr);

  // With rotation
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  R = q.toRotationMatrix();
  crocoddyl::FrictionCone cone_2(R, mu, nf, inner_appr);

  for (std::size_t i = 0; i < 5; ++i) {
    BOOST_CHECK((cone_1.get_A().row(i) - cone_2.get_A().row(i) * R).isMuchSmallerThan(1.0, 1e-9));
  }
}

void test_force_along_friction_cone_normal() {
  // Create the friction cone
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  Eigen::Matrix3d R = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;
  crocoddyl::FrictionCone cone(R, mu, nf, inner_appr);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::Vector3d force = random_real_in_range(0., 100.) * R.col(2);
  Eigen::VectorXd r = cone.get_A() * force;
  activation.calc(data, r);

  // The activation value has to be zero since the force is inside the friction cone
  BOOST_CHECK(data->a_value == 0.);
}

void test_negative_force_along_friction_cone_normal() {
  // Create the friction cone
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  Eigen::Matrix3d R = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;
  crocoddyl::FrictionCone cone(R, mu, nf, inner_appr);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::Vector3d force = -random_real_in_range(0., 100.) * R.col(2);
  Eigen::VectorXd r = cone.get_A() * force;

  // The first nf elements of the residual has to be positive since the force is outside the
  // friction cone. Additionally, the last value has to be equals to the force norm but with
  // negative value since the forces is aligned and in opposite direction to the friction
  // cone orientation
  for (std::size_t i = 0; i < nf; ++i) {
    BOOST_CHECK(r(i) > 0.);
  }
  BOOST_CHECK_CLOSE(r(nf), -force.norm(), 1e-9);

  // The activation value has to be positive since the force is outside the friction cone
  activation.calc(data, r);
  BOOST_CHECK(data->a_value > 0.);
}

void test_force_parallel_to_friction_cone_normal() {
  // Create the friction cone
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;
  crocoddyl::FrictionCone cone(R, mu, nf, inner_appr);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::Vector3d force = -random_real_in_range(0., 100.) * Eigen::Vector3d::UnitX();
  Eigen::VectorXd r = cone.get_A() * force;

  // The last value of the residual is equals to zero since the force is parallel to the
  // friction cone orientation
  BOOST_CHECK_CLOSE(r(nf), 0., 1e-9);

  // The activation value has to be positive since the force is outside the friction cone
  activation.calc(data, r);
  BOOST_CHECK(data->a_value > 0.);
}

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_inner_approximation_of_friction_cone)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_A_matrix_with_rotation_change)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_force_along_friction_cone_normal)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_negative_force_along_friction_cone_normal)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_force_parallel_to_friction_cone_normal)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
