///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <sstream>

#include "crocoddyl/core/integrator/time.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename Scalar>
void test_construction_mutation_copy_and_print() {
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;

  IntegratorTime default_time;
  BOOST_CHECK_SMALL(default_time.get_time_step() - Scalar(1e-3),
                    Scalar(10) * Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(default_time.get_time_step2() - Scalar(1e-6),
                    Scalar(10) * Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK(!default_time.get_timeopt());

  IntegratorTime zero_time(Scalar(0), true);
  BOOST_CHECK_SMALL(zero_time.get_time_step(),
                    Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(zero_time.get_time_step2(),
                    Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK(zero_time.get_timeopt());

  zero_time.set_time_step(Scalar(0.2));
  zero_time.set_timeopt(false);
  BOOST_CHECK_SMALL(zero_time.get_time_step() - Scalar(0.2),
                    Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(zero_time.get_time_step2() - Scalar(0.04),
                    Scalar(10) * Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK(!zero_time.get_timeopt());

  IntegratorTime copied(zero_time);
  copied.set_time_step(Scalar(0.3));
  BOOST_CHECK_SMALL(zero_time.get_time_step() - Scalar(0.2),
                    Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(copied.get_time_step2() - Scalar(0.09),
                    Scalar(10) * Eigen::NumTraits<Scalar>::epsilon());

  BOOST_CHECK_THROW(IntegratorTime(Scalar(-1e-3)), crocoddyl::Exception);
  BOOST_CHECK_THROW(zero_time.set_time_step(Scalar(-1e-3)),
                    crocoddyl::Exception);

  std::ostringstream stream;
  stream << copied;
  BOOST_CHECK(stream.str().find("IntegratorTime") != std::string::npos);
}

void test_scalar_cast() {
  const crocoddyl::IntegratorTime source(0.3, true);
  const crocoddyl::IntegratorTimeTpl<float> target = source.cast<float>();
  BOOST_CHECK_SMALL(static_cast<double>(target.get_time_step()) - 0.3, 1e-7);
  BOOST_CHECK_SMALL(static_cast<double>(target.get_time_step2()) - 0.09, 1e-7);
  BOOST_CHECK(target.get_timeopt());
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_integrator_time");
  ts->add(BOOST_TEST_CASE(&test_construction_mutation_copy_and_print<double>));
  ts->add(BOOST_TEST_CASE(&test_construction_mutation_copy_and_print<float>));
  ts->add(BOOST_TEST_CASE(&test_scalar_cast));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
