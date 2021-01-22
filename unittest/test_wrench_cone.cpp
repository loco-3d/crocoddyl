///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/multibody/cop-support.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "pinocchio/math/quaternion.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

void test_constructor() {
  // Common parameters
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = false;

  // No rotation
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d normal = R * Eigen::Vector3d::UnitZ();

  // Create the wrench cone with rotation and surface normal
  crocoddyl::WrenchCone cone_1(R, mu, box, nf, inner_appr);
  crocoddyl::WrenchCone cone_2(normal, mu, box, nf, inner_appr);

  BOOST_CHECK((cone_1.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_1.get_mu() == mu);
  BOOST_CHECK(cone_1.get_nf() == nf);
  BOOST_CHECK((cone_1.get_box() - cone_2.get_box()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_1.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_A().rows()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_lb().size()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_ub().size()) == nf + 13);
  BOOST_CHECK((cone_2.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_2.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_2.get_mu() == mu);
  BOOST_CHECK(cone_2.get_nf() == nf);
  BOOST_CHECK(cone_2.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_A().rows()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_lb().size()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_ub().size()) == nf + 13);
  BOOST_CHECK((cone_1.get_A() - cone_2.get_A()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_lb() - cone_2.get_lb()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_ub() - cone_2.get_ub()).isMuchSmallerThan(1.0, 1e-9));

  // With rotation
  normal = Eigen::Vector3d(0., sqrt(2) / 2, sqrt(2) / 2);
  R = Eigen::Quaternion<double>::FromTwoVectors(normal, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  // Create the wrench cone with rotation and surface normal
  cone_1 = crocoddyl::WrenchCone(R, mu, box, nf, inner_appr);
  cone_2 = crocoddyl::WrenchCone(normal, mu, box, nf, inner_appr);

  BOOST_CHECK((cone_1.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_1.get_mu() == mu);
  BOOST_CHECK(cone_1.get_nf() == nf);
  BOOST_CHECK((cone_1.get_box() - cone_2.get_box()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_1.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_A().rows()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_lb().size()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_1.get_ub().size()) == nf + 13);
  BOOST_CHECK((cone_2.get_R() - R).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_2.get_nsurf() - normal).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(cone_2.get_mu() == mu);
  BOOST_CHECK(cone_2.get_nf() == nf);
  BOOST_CHECK(cone_2.get_inner_appr() == inner_appr);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_A().rows()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_lb().size()) == nf + 13);
  BOOST_CHECK(static_cast<std::size_t>(cone_2.get_ub().size()) == nf + 13);
  BOOST_CHECK((cone_1.get_A() - cone_2.get_A()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_lb() - cone_2.get_lb()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((cone_1.get_ub() - cone_2.get_ub()).isMuchSmallerThan(1.0, 1e-9));
}

void test_against_friction_cone() {
  // Common parameters
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  Eigen::Matrix3d R = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  std::size_t nf = 2 * random_int_in_range(2, 16);
  bool inner_appr = true;

  // Create wrench and friction cone
  crocoddyl::WrenchCone wrench_cone(R, mu, box, nf, inner_appr, 0., 100.);
  crocoddyl::FrictionCone friction_cone(R, mu, nf, inner_appr, 0., 100.);

  BOOST_CHECK((wrench_cone.get_R() - friction_cone.get_R()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK(wrench_cone.get_mu() == friction_cone.get_mu());
  BOOST_CHECK(wrench_cone.get_nf() == friction_cone.get_nf());
  for (std::size_t i = 0; i < nf + 1; ++i) {
    BOOST_CHECK((wrench_cone.get_A().row(i).head(3) - friction_cone.get_A().row(i)).isMuchSmallerThan(1.0, 1e-9));
  }
  BOOST_CHECK((wrench_cone.get_lb().head(nf + 1) - friction_cone.get_lb()).isMuchSmallerThan(1.0, 1e-9));
  BOOST_CHECK((wrench_cone.get_ub().head(nf + 1) - friction_cone.get_ub()).isMuchSmallerThan(1.0, 1e-9));
}

void test_against_cop_support() {
  // // Common parameters
  // Eigen::Quaterniond q;
  // pinocchio::quaternion::uniformRandom(q);
  // Eigen::Matrix3d R = q.toRotationMatrix();
  // double mu = random_real_in_range(0.01, 1.);
  // Eigen::Vector2d box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  // std::size_t nf = 2 * random_int_in_range(2, 16);
  // bool inner_appr = true;

  // // Create wrench and friction cone
  // crocoddyl::WrenchCone wrench_cone(R, mu, box, nf, inner_appr, 0., 100.);
  // crocoddyl::CoPSupport cop_support(R, box);

  // BOOST_CHECK((wrench_cone.get_R() - cop_support.get_R()).isMuchSmallerThan(1.0, 1e-9));
  // for (std::size_t i = 0; i < 4; ++i) {
  //   BOOST_CHECK((wrench_cone.get_A().row(nf + i + 1).head(3) - cop_support.get_A().row(i)).isMuchSmallerThan(1.0, 1e-9));
  // }
  // BOOST_CHECK((wrench_cone.get_lb().segment(nf + 1, 4) - cop_support.get_lb()).isMuchSmallerThan(1.0, 1e-9));
  // BOOST_CHECK((wrench_cone.get_ub().segment(nf + 1, 4) - cop_support.get_ub()).isMuchSmallerThan(1.0, 1e-9));
}

void test_force_along_wrench_cone_normal() {
  // Create the wrench cone
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  Eigen::Matrix3d R = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(R, mu, cone_box);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.head(3) = random_real_in_range(0., 100.) * R.row(2).transpose();
  Eigen::VectorXd r = cone.get_A() * wrench;
  activation.calc(data, r);

  // The activation value has to be zero since the wrench is inside the wrench cone
  BOOST_CHECK(data->a_value == 0.);
}

void test_negative_force_along_wrench_cone_normal() {
  // Create the wrench cone
  Eigen::Quaterniond q;
  pinocchio::quaternion::uniformRandom(q);
  Eigen::Matrix3d R = q.toRotationMatrix();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(R, mu, cone_box);

  // Create the activation for quadratic barrier
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  crocoddyl::ActivationModelQuadraticBarrier activation(bounds);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = activation.createData();

  // Compute the activation value
  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.head(3) = -random_real_in_range(0., 100.) * R.row(2).transpose();
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
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  double mu = random_real_in_range(0.01, 1.);
  Eigen::Vector2d cone_box = Eigen::Vector2d(random_real_in_range(0.01, 0.1), random_real_in_range(0.01, 0.1));
  crocoddyl::WrenchCone cone(R, mu, cone_box);

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
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_against_friction_cone)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_against_cop_support)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_force_along_wrench_cone_normal)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_negative_force_along_wrench_cone_normal)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_force_parallel_to_wrench_cone_normal)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
