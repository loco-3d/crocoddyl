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
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"

using namespace boost::unit_test;

struct TestTypes {
  enum Type { DifferentialActionModelLQR, NbTestTypes };
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

class DifferentialActionModelFactory {
 public:
  DifferentialActionModelFactory(TestTypes::Type type) {
    // build the DifferentialActionModelLQR
    nq_ = 40;
    nu_ = 40;
    driftfree_ = true;
    num_diff_modifier_ = 1e4;
    test_type_ = type;

    switch (test_type_) {
      case TestTypes::DifferentialActionModelLQR:
        diff_action_model_ = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(nq_, nu_, driftfree_);
        break;
      default:
        throw std::runtime_error(__FILE__ ": Wrong TestTypes::Type given");
        break;
    }
  }

  ~DifferentialActionModelFactory() {}

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> get_diff_action_model() { return diff_action_model_; }

  double get_num_diff_modifier() { return num_diff_modifier_; }

 private:
  std::size_t nq_;
  std::size_t nu_;
  double num_diff_modifier_;
  bool driftfree_;
  TestTypes::Type test_type_;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> diff_action_model_;
};

//----------------------------------------------------------------------------//

void test_construct_data(TestTypes::Type test_type) {
  // create the model
  DifferentialActionModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& model = factory.get_diff_action_model();

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data = model->createData();
}

void test_calc_returns_state(TestTypes::Type test_type) {
  // create the model
  DifferentialActionModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& model = factory.get_diff_action_model();

  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data = model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(static_cast<std::size_t>(data->xout.size()) == model->get_state()->get_nv());
}

void test_calc_returns_a_cost(TestTypes::Type test_type) {
  // create the model
  DifferentialActionModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& model = factory.get_diff_action_model();

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data = model->createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_partial_derivatives_against_numdiff(TestTypes::Type test_type) {
  // create the model
  DifferentialActionModelFactory factory(test_type);
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& model = factory.get_diff_action_model();

  // create the corresponding data object and set the cost to nan
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data = model->createData();

  crocoddyl::DifferentialActionModelNumDiff model_num_diff(*model.get());
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  const Eigen::VectorXd& x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calcDiff(data, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = factory.get_num_diff_modifier() * model_num_diff.get_disturbance();
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isMuchSmallerThan(1.0, tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isMuchSmallerThan(1.0, tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data_num_diff->Lxu).isMuchSmallerThan(1.0, tol));
    BOOST_CHECK((data_num_diff->Luu).isMuchSmallerThan(1.0, tol));
  }
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(TestTypes::Type test_type) {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_construct_data, test_type)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, test_type)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, test_type)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, test_type)));
}

bool init_function() {
  for (size_t i = 0; i < TestTypes::all.size(); ++i) {
    register_action_model_unit_tests(TestTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
