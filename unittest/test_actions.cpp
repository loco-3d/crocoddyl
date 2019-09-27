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
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl_test/test_common.hpp"

using namespace boost::unit_test;

struct ActionModelTypes
{
  enum Type{
    ActionModelUnicycle,
    ActionModelLQR
  };
  static std::vector<Type> init_all()
  {
    std::vector<Type> v;
    v.clear();
    v.push_back(ActionModelUnicycle);
    v.push_back(ActionModelLQR);
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<ActionModelTypes::Type> ActionModelTypes::all(ActionModelTypes::init_all());

class ActionModelFactory{
public:
  ActionModelFactory(ActionModelTypes::Type type)
  {
    nx_ = 80;
    nu_ = 40;
    driftfree_ = true;
    num_diff_modifier_ = 1e4;
    action_model_ = NULL;
    action_model_unicycle_ = NULL;
    diff_action_model_lqr_ = NULL;
    action_model_lqr_ = NULL;
    switch (type)
    {
    case ActionModelTypes::ActionModelUnicycle :
      std::cout << "created an ActionModelUnicycle" << std::endl;
      action_model_unicycle_ = new crocoddyl::ActionModelUnicycle();
      action_model_ = action_model_unicycle_;
      break;

    case ActionModelTypes::ActionModelLQR :
      std::cout << "created an ActionModelLQR" << std::endl;
      action_model_lqr_ = new crocoddyl::ActionModelLQR(nx_, nu_, driftfree_);
      action_model_ = action_model_lqr_;
      break;
    
    default:
      throw std::runtime_error("test_actions.cpp: This type of ActionModel requested has not been implemented yet.");
      break;
    }
  }

  ~ActionModelFactory(){
    std::cout << "delete factory" << std::endl;
    crocoddyl_unit_test::delete_pointer(action_model_unicycle_);
    crocoddyl_unit_test::delete_pointer(diff_action_model_lqr_);
    crocoddyl_unit_test::delete_pointer(action_model_lqr_);
    action_model_ = NULL;
    std::cout << "delete factory done" << std::endl;
  }

  crocoddyl::ActionModelAbstract* get_action_model(){return action_model_;}

  double num_diff_modifier_;

private:
  int nx_;
  int nu_;
  bool driftfree_;

  crocoddyl::ActionModelUnicycle* action_model_unicycle_;
  crocoddyl::DifferentialActionModelLQR* diff_action_model_lqr_;
  crocoddyl::ActionModelLQR* action_model_lqr_;
  crocoddyl::ActionModelAbstract* action_model_;
};

void test_construct_data(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory (action_model_type);
  crocoddyl::ActionModelAbstract* model = factory.get_action_model();
  std::cout << "model address: " << model << std::endl;

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model->createData();

  std::cout << "test_construct_data end" << std::endl;
}

void test_calc_returns_state(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory (action_model_type);
  crocoddyl::ActionModelAbstract* model = factory.get_action_model();

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model->createData();

  // Generating random state and control vectors
  Eigen::VectorXd x = model->get_state().rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(data->get_xnext().size() == model->get_state().get_nx());
}

void test_calc_returns_a_cost(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory (action_model_type);
  crocoddyl::ActionModelAbstract* model = factory.get_action_model();

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model->createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  Eigen::VectorXd x = model->get_state().rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_partial_derivatives_against_numdiff(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory (action_model_type);
  crocoddyl::ActionModelAbstract* model = factory.get_action_model();

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data = model->createData();

  crocoddyl::ActionModelNumDiff model_num_diff(*model);
  boost::shared_ptr<crocoddyl::ActionDataAbstract> data_num_diff = model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state().rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calcDiff(data, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = factory.num_diff_modifier_ * model_num_diff.get_disturbance();
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

void register_action_model_unit_tests(ActionModelTypes::Type action_model_type) {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_construct_data, action_model_type)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, action_model_type)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, action_model_type)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_partial_derivatives_against_numdiff, action_model_type)));
}

bool init_function() {
  for (size_t i = 0 ; i < ActionModelTypes::all.size() ; ++i){
    register_action_model_unit_tests(ActionModelTypes::all[i]);
  }  
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
