///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

void test_constructor() {
  // Create the wrench cone
  Eigen::Matrix3d cone_rotation = Eigen::Matrix3d::Identity();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(cone_rotation, mu, cone_box);

  BOOST_CHECK((cone.get_rot() - cone_rotation).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone.get_mu() == mu / sqrt(2.));
  BOOST_CHECK(cone.get_nf() == 16);
  BOOST_CHECK((cone.get_box() - cone_box).isMuchSmallerThan(1.0, 1e-9));

  BOOST_CHECK(static_cast<std::size_t>(cone.get_A().rows()) == cone.get_nf() + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_lb().size()) == cone.get_nf() + 1);
  BOOST_CHECK(static_cast<std::size_t>(cone.get_ub().size()) == cone.get_nf() + 1);
}

void test_force_along_wrench_cone_normal() {
  // Create the wrench cone
  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Matrix3d cone_rotation = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(cone_rotation, mu, cone_box);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.head(3) = random_real_in_range(0., 100.) * cone_rotation.col(2);
  Eigen::VectorXd r = cone.get_A() * wrench;
  activation.calc(data, r);
  
  // The activation value has to be zero since the wrench is inside the wrench cone
  BOOST_CHECK(data->a_value == 0.);
}

void test_negative_force_along_wrench_cone_normal() {
  // Create the wrench cone
  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Matrix3d cone_rotation = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(cone_rotation, mu, cone_box);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.head(3) = -random_real_in_range(0., 100.) * cone_rotation.col(2);
  Eigen::VectorXd r = cone.get_A() * wrench;
  activation.calc(data, r);

  // The first nf elements of the residual has to be positive since the force is outside the
  // wrench cone. Additionally, the last value has to be equals to the force norm but with
  // negative value since the wrench is aligned and in opposite direction to the wrench
  // cone orientation
  for (std::size_t i = 0; i < cone.get_nf(); ++i) {
    BOOST_CHECK(r(i) > 0.);
  }
  
  // The activation value has to be positive since the wrench is outside the wrench cone
  activation.calc(data, r);
  BOOST_CHECK(data->a_value > 0.);
}

void test_force_parallel_to_wrench_cone_normal() {
  // Create the wrench cone
  Eigen::Matrix3d cone_rotation = Eigen::Matrix3d::Identity();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(cone_rotation, mu, cone_box);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.head(3) = -random_real_in_range(0., 100.) * Eigen::Vector3d::UnitX();
  Eigen::VectorXd r = cone.get_A() * wrench;

  // The activation value has to be positive since the force is outside the wrench cone
  activation.calc(data, r);
  BOOST_CHECK(data->a_value > 0.);
}

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_force_along_wrench_cone_normal)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_negative_force_along_wrench_cone_normal)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_force_parallel_to_wrench_cone_normal)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
