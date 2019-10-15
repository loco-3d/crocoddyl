///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

/**
 * To be included last in the test_XXX.cpp,
 * otherwise it interferes with pinocchio boost::variant.
 */

#ifndef CROCODDYL_UNITTEST_COMMON_HPP_
#define CROCODDYL_UNITTEST_COMMON_HPP_

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <Eigen/Dense>
#include <boost/bind.hpp>
#include <boost/test/included/unit_test.hpp>

namespace crocoddyl_unit_test {


} // namespace crocoddyl_unit_test

#endif // CROCODDYL_UNITTEST_COMMON_HPP_
