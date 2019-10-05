///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <iterator>
#include <Eigen/Dense>
#include <pinocchio/fwd.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/smooth-abs.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/numdiff/activation.hpp"
#include "crocoddyl_unittest_common.hpp"

/**
 * c = sum( a(ri) )
 * c' = sum( [a(ri)]' ) = sum( ri' a'(ri) ) = R' [ a'(ri) ]_i
 * c'' = R' [a'(ri) ]_i' = R' [a''(ri) ] R
 *
 * ex
 * a(x) =  x**2/x
 * a'(x) = x
 * a''(x) = 1
 *
 * sum(a(ri)) = sum(ri**2/2) = .5*r'r
 * sum(ri' a'(ri)) = sum(ri' ri) = R' r
 * sum(ri' a''(ri) ri') = R' r
 * c'' = R'R
 */

using namespace boost::unit_test;

struct TestTypes {
  enum Type {
    ActivationModelQuadraticBarrier,
    ActivationModelQuad, /*ActivationModelSmoothAbs,*/
    ActivationModelWeightedQuad,
    NbTestTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbTestTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<TestTypes::Type> TestTypes::all(TestTypes::init_all());

class Factory {
 public:
  Factory(TestTypes::Type type) {
    test_type_ = type;

    nr_ = 5;
    num_diff_modifier_ = 1e4;
    bounds_ = NULL;
    Eigen::VectorXd lb = Eigen::VectorXd::Random(nr_);
    Eigen::VectorXd ub = lb + Eigen::VectorXd::Ones(nr_) + Eigen::VectorXd::Random(nr_);
    Eigen::VectorXd weights = Eigen::VectorXd::Random(nr_);

    switch (test_type_) {
      case TestTypes::ActivationModelQuadraticBarrier:
        bounds_ = new crocoddyl::ActivationBounds(lb, ub);
        model_ = new crocoddyl::ActivationModelQuadraticBarrier(*bounds_);
        break;
      case TestTypes::ActivationModelQuad:
        model_ = new crocoddyl::ActivationModelQuad(nr_);
        break;
      // case TestTypes::ActivationModelSmoothAbs:
      //   model_ = new crocoddyl::ActivationModelSmoothAbs(nr_);
      //   break;
      case TestTypes::ActivationModelWeightedQuad:
        model_ = new crocoddyl::ActivationModelWeightedQuad(weights);
        break;
      default:
        throw std::runtime_error(__FILE__ ": Wrong TestTypes::Type given");
        break;
    }
  }

  ~Factory() {
    switch (test_type_) {
      case TestTypes::ActivationModelQuadraticBarrier:
        crocoddyl_unit_test::delete_pointer((crocoddyl::ActivationModelQuadraticBarrier*)model_);
        break;
      case TestTypes::ActivationModelQuad:
        crocoddyl_unit_test::delete_pointer((crocoddyl::ActivationModelQuad*)model_);
        break;
      // case TestTypes::ActivationModelSmoothAbs:
      //   crocoddyl_unit_test::delete_pointer((crocoddyl::ActivationModelSmoothAbs)model_ );
      //   break;
      case TestTypes::ActivationModelWeightedQuad:
        crocoddyl_unit_test::delete_pointer((crocoddyl::ActivationModelWeightedQuad*)model_);
        break;
      default:
        throw std::runtime_error(__FILE__ ": Wrong TestTypes::Type given");
        break;
    }
    model_ = NULL;
    crocoddyl_unit_test::delete_pointer(bounds_);
    bounds_ = NULL;
  }

  crocoddyl::ActivationModelAbstract* get_model() { return model_; }
  unsigned int get_nr() { return nr_; }
  double get_num_diff_modifier() { return num_diff_modifier_; }

 private:
  double num_diff_modifier_;
  unsigned int nr_;
  crocoddyl::ActivationModelAbstract* model_;
  crocoddyl::ActivationBounds* bounds_;
  TestTypes::Type test_type_;
};

//----------------------------------------------------------------------------//

void test_construct_data(TestTypes::Type test_type) {
  // create the model
  Factory factory(test_type);
  crocoddyl::ActivationModelAbstract* model = factory.get_model();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();
}

void test_calc_returns_a_value(TestTypes::Type test_type) {
  // create the model
  Factory factory(test_type);
  crocoddyl::ActivationModelAbstract* model = factory.get_model();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();

  // Generating random input vector
  Eigen::VectorXd r = Eigen::VectorXd::Random(model->get_nr());
  data->a_value = nan("");

  // Getting the state dimension from calc() call
  model->calc(data, r);

  // Checking that calc returns a value
  BOOST_CHECK(!std::isnan(data->a_value));
}

void test_partial_derivatives_against_numdiff(TestTypes::Type test_type) {
  // create the model
  Factory factory(test_type);
  crocoddyl::ActivationModelAbstract* model = factory.get_model();

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();

  crocoddyl::ActivationModelNumDiff model_num_diff(*model);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd r = Eigen::VectorXd::Random(model->get_nr());

  // Computing the action derivatives
  model->calcDiff(data, r);
  model_num_diff.calcDiff(data_num_diff, r);

  // Checking the partial derivatives against NumDiff
  double tol = factory.get_num_diff_modifier() * model_num_diff.get_disturbance();
  BOOST_CHECK(std::abs(data->a_value - data_num_diff->a_value) < tol);
  BOOST_CHECK((data->Ar - data_num_diff->Ar).isMuchSmallerThan(1.0, tol));

  // numerical differentiation of the Hessian is not good enough to be tested.
  // BOOST_CHECK((data->Arr - data_num_diff->Arr).isMuchSmallerThan(1.0, tol));
}

//----------------------------------------------------------------------------//

void register_unit_tests(TestTypes::Type type) {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_construct_data, type)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_value, type)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, type)));
}

bool init_function() {
  for (size_t i = 0; i < TestTypes::all.size(); ++i) {
    register_unit_tests(TestTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
