///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

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
  Eigen::Vector3d normal = R * Eigen::Vector3d::UnitZ();

  // Create the friction cone with rotation and surface normal
  crocoddyl::FrictionCone cone_1(R, mu, nf, inner_appr);
  crocoddyl::FrictionCone cone_2(normal, mu, nf, inner_appr);

  BOOST_CHECK((cone_1.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_1.get_mu() == mu);
  BOOST_CHECK(cone_1.get_nf() == nf);
  BOOST_CHECK(cone_1.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_A().rows()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_lb().size()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_ub().size()) == nf + 1);
  BOOST_CHECK((cone_2.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_2.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_2.get_mu() == mu);
  BOOST_CHECK(cone_2.get_nf() == nf);
  BOOST_CHECK(cone_2.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_A().rows()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_lb().size()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_ub().size()) == nf + 1);
  BOOST_CHECK((cone_1.get_A() - cone_2.get_A()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_lb() - cone_2.get_lb()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_ub() - cone_2.get_ub()).isMuchSmallerThan(1.0, 1e-9));

  // With rotation
  normal = Eigen::Vector3d(0., sqrt(2) / 2, sqrt(2) / 2);
  R = Eigen::Quaternion<double>::FromTwoVectors(normal, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  // Create the friction cone with rotation and surface contact
  cone_1 = crocoddyl::FrictionCone(R, mu, nf, inner_appr);
  cone_2 = crocoddyl::FrictionCone(normal, mu, nf, inner_appr);

  BOOST_CHECK((cone_1.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_1.get_mu() == mu);
  BOOST_CHECK(cone_1.get_nf() == nf);
  BOOST_CHECK(cone_1.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_A().rows()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_lb().size()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_ub().size()) == nf + 1);
  BOOST_CHECK((cone_2.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_2.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_2.get_mu() == mu);
  BOOST_CHECK(cone_2.get_nf() == nf);
  BOOST_CHECK(cone_2.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_A().rows()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_lb().size()) == nf + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_ub().size()) == nf + 1);
  BOOST_CHECK((cone_1.get_A() - cone_2.get_A()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_lb() - cone_2.get_lb()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_ub() - cone_2.get_ub()).isMuchSmallerThan(1.0, 1e-9));
}

void test_inner_approximation_of_friction_cone() {
  Eigen::Vector3d cone_normal = Eigen::Vector3d::UnitZ();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = true;
  crocoddyl::FrictionCone cone(cone_normal, mu, nf, inner_appr);
  const Eigen::Vector3d A_mu = -cone.get_A().col(2);
  BOOST_CHECK_CLOSE(A_mu(0), mu * cos((2 * M_PI / static_cast<double>(nf)) / 2.), 1e-9);
  BOOST_CHECK_CLOSE(A_mu(1), mu * cos((2 * M_PI / static_cast<double>(nf)) / 2.), 1e-9);
  BOOST_CHECK_CLOSE(A_mu(2), mu * cos((2 * M_PI / static_cast<double>(nf)) / 2.), 1e-9);
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
  Eigen::Vector3d normal(0., sqrt(2) / 2, sqrt(2) / 2);
  R = Eigen::Quaternion<double>::FromTwoVectors(normal, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  crocoddyl::FrictionCone cone_2(R, mu, nf, inner_appr);

  for (std::size_t i = 0; i < 5; ++i) {
    BOOST_CHECK((cone_1.get_A().row(i) - cone_2.get_A().row(i) * R.transpose()).isMuchSmallerThan(1.0, 1e-9));
  }
}

void test_force_along_friction_cone_normal() {
  // Create the friction cone
  Eigen::Vector3d cone_normal = Eigen::Vector3d::Random();
  cone_normal /= cone_normal.norm();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;
  crocoddyl::FrictionCone cone(cone_normal, mu, nf, inner_appr);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::Vector3d force = random_real_in_range(0., 100.) * cone_normal;
  Eigen::VectorXd r = cone.get_A() * force;
  activation.calc(data, r);

  // The activation value has to be zero since the force is inside the friction cone
  BOOST_CHECK(data->a_value == 0.);
}

void test_negative_force_along_friction_cone_normal() {
  // Create the friction cone
  Eigen::Vector3d cone_normal = Eigen::Vector3d::Random();
  cone_normal /= cone_normal.norm();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;
  crocoddyl::FrictionCone cone(cone_normal, mu, nf, inner_appr);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::Vector3d force = -random_real_in_range(0., 100.) * cone_normal;
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
  Eigen::Vector3d cone_normal = Eigen::Vector3d::UnitZ();
  cone_normal /= cone_normal.norm();
  double mu = random_real_in_range(0.01, 1.);
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;
  crocoddyl::FrictionCone cone(cone_normal, mu, nf, inner_appr);

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
