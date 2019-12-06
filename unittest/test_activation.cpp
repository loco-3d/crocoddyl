///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft
//                          University of Edinburgh, INRIA
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

using namespace boost::unit_test;

struct TestTypes {
  enum Type {
    ActivationModelQuadraticBarrier,
    ActivationModelQuad,
    ActivationModelSmoothAbs,
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
    Eigen::VectorXd lb = Eigen::VectorXd::Random(nr_);
    Eigen::VectorXd ub = lb + Eigen::VectorXd::Ones(nr_) + Eigen::VectorXd::Random(nr_);
    Eigen::VectorXd weights = Eigen::VectorXd::Random(nr_);

    switch (test_type_) {
      case TestTypes::ActivationModelQuadraticBarrier:
        model_ = boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(lb, ub));
        break;
      case TestTypes::ActivationModelQuad:
        model_ = boost::make_shared<crocoddyl::ActivationModelQuad>(nr_);
        break;
      case TestTypes::ActivationModelSmoothAbs:
        model_ = boost::make_shared<crocoddyl::ActivationModelSmoothAbs>(nr_);
        break;
      case TestTypes::ActivationModelWeightedQuad:
        model_ = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);
        break;
      default:
        throw std::runtime_error(__FILE__ ":\n Construct wrong TestTypes::Type");
        break;
    }
  }

  ~Factory() {}

  boost::shared_ptr<crocoddyl::ActivationModelAbstract> get_model() { return model_; }
  const std::size_t& get_nr() { return nr_; }
  double get_num_diff_modifier() { return num_diff_modifier_; }

 private:
  double num_diff_modifier_;
  std::size_t nr_;
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> model_;
  TestTypes::Type test_type_;
};

//----------------------------------------------------------------------------//

void test_construct_data(TestTypes::Type test_type) {
  // create the model
  Factory factory(test_type);
  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model = factory.get_model();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();
}

void test_calc_returns_a_value(TestTypes::Type test_type) {
  // create the model
  Factory factory(test_type);
  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model = factory.get_model();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();

  // Generating random input vector
  const Eigen::VectorXd& r = Eigen::VectorXd::Random(model->get_nr());
  data->a_value = nan("");

  // Getting the state dimension from calc() call
  model->calc(data, r);

  // Checking that calc returns a value
  BOOST_CHECK(!std::isnan(data->a_value));
}

void test_partial_derivatives_against_numdiff(TestTypes::Type test_type) {
  // create the model
  Factory factory(test_type);
  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model = factory.get_model();

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data = model->createData();

  crocoddyl::ActivationModelNumDiff model_num_diff(model);
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd& r = Eigen::VectorXd::Random(model->get_nr());

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

void register_unit_tests(TestTypes::Type type, test_suite& ts) {
  ts.add(BOOST_TEST_CASE(boost::bind(&test_construct_data, type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_value, type)));
  ts.add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, type)));
}

bool init_function() {
  for (size_t i = 0; i < TestTypes::all.size(); ++i) {
    const std::string test_name = "test_" + std::to_string(i);
    test_suite* ts = BOOST_TEST_SUITE(test_name);
    register_unit_tests(TestTypes::all[i], *ts);
    framework::master_test_suite().add(ts);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
