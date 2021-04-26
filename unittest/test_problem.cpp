///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "factory/action.hpp"
#include "factory/diff_action.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> > datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas, model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->rand();

  // check the state and cost in each node
  problem1.calc(xs, us);
  problem2.calc(xs, us);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK(problem1.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK(problem2.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK((problem1.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
  }
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
  model->calc(data, xs.back());
  BOOST_CHECK(problem1.get_terminalData()->cost == data->cost);
  BOOST_CHECK(problem2.get_terminalData()->cost == data->cost);
  BOOST_CHECK((problem1.get_terminalData()->xnext - data->xnext).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->xnext - data->xnext).isZero(1e-9));
}

void test_calc_diffAction(DifferentialActionModelTypes::Type action_model_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel = factory.create(action_model_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(diffModel);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> > datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas, model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->rand();

  // check the state and cost in each node
  problem1.calc(xs, us);
  problem2.calc(xs, us);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK(problem1.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK(problem2.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK((problem1.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
  }
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
  model->calc(data, xs.back());
  BOOST_CHECK(problem1.get_terminalData()->cost == data->cost);
  BOOST_CHECK(problem2.get_terminalData()->cost == data->cost);
  BOOST_CHECK((problem1.get_terminalData()->xnext - data->xnext).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->xnext - data->xnext).isZero(1e-9));
}

void test_calcDiff(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> > datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas, model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->rand();

  // check the state and cost in each node
  problem1.calc(xs, us);
  problem2.calc(xs, us);
  problem1.calcDiff(xs, us);
  problem2.calcDiff(xs, us);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    model->calc(data, xs[i], us[i]);
    model->calcDiff(data, xs[i], us[i]);
    BOOST_CHECK((problem1.get_runningDatas()[i]->Fx - data->Fx).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Fx - data->Fx).isZero(1e-9));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Fu - data->Fu).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Fu - data->Fu).isZero(1e-9));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lx - data->Lx).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lx - data->Lx).isZero(1e-9));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lu - data->Lu).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lu - data->Lu).isZero(1e-9));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lxx - data->Lxx).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lxx - data->Lxx).isZero(1e-9));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lxu - data->Lxu).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lxu - data->Lxu).isZero(1e-9));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Luu - data->Luu).isZero(1e-9));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Luu - data->Luu).isZero(1e-9));
  }
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
  model->calc(data, xs.back());
  model->calcDiff(data, xs.back());
  BOOST_CHECK((problem1.get_terminalData()->Fx - data->Fx).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->Fx - data->Fx).isZero(1e-9));
  BOOST_CHECK((problem1.get_terminalData()->Lx - data->Lx).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->Lx - data->Lx).isZero(1e-9));
  BOOST_CHECK((problem1.get_terminalData()->Lxx - data->Lxx).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->Lxx - data->Lxx).isZero(1e-9));
}

void test_calcDiff_diffAction(DifferentialActionModelTypes::Type action_model_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel = factory.create(action_model_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(diffModel);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> > datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas, model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->rand();

  // check the state and cost in each node
  problem1.calc(xs, us);
  problem2.calc(xs, us);
  problem1.calcDiff(xs, us);
  problem2.calcDiff(xs, us);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    model->calc(data, xs[i], us[i]);
    model->calcDiff(data, xs[i], us[i]);
    BOOST_CHECK((problem1.get_runningDatas()[i]->Fx - data->Fx).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Fx - data->Fx).isZero(1e-7));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Fu - data->Fu).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Fu - data->Fu).isZero(1e-7));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lx - data->Lx).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lx - data->Lx).isZero(1e-7));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lu - data->Lu).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lu - data->Lu).isZero(1e-7));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lxx - data->Lxx).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lxx - data->Lxx).isZero(1e-7));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Lxu - data->Lxu).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Lxu - data->Lxu).isZero(1e-7));
    BOOST_CHECK((problem1.get_runningDatas()[i]->Luu - data->Luu).isZero(1e-7));
    BOOST_CHECK((problem2.get_runningDatas()[i]->Luu - data->Luu).isZero(1e-7));
  }
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
  model->calc(data, xs.back());
  model->calcDiff(data, xs.back());
  BOOST_CHECK((problem1.get_terminalData()->Fx - data->Fx).isZero(1e-7));
  BOOST_CHECK((problem2.get_terminalData()->Fx - data->Fx).isZero(1e-7));
  BOOST_CHECK((problem1.get_terminalData()->Lx - data->Lx).isZero(1e-7));
  BOOST_CHECK((problem2.get_terminalData()->Lx - data->Lx).isZero(1e-7));
  BOOST_CHECK((problem1.get_terminalData()->Lxx - data->Lxx).isZero(1e-7));
  BOOST_CHECK((problem2.get_terminalData()->Lxx - data->Lxx).isZero(1e-7));
}

void test_rollout(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create the shooting problem
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  crocoddyl::ShootingProblem problem(x0, models, model);

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->zero();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->zero();

  // check the state and cost in each node
  problem.rollout(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK((xs[i + 1] - data->xnext).isZero(1e-7));
  }
}

void test_rollout_diffAction(DifferentialActionModelTypes::Type action_model_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel = factory.create(action_model_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(diffModel);

  // create the shooting problem
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  crocoddyl::ShootingProblem problem(x0, models, model);

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->zero();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->zero();

  // check the state and cost in each node
  problem.rollout(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK((xs[i + 1] - data->xnext).isZero(1e-7));
  }
}

void test_quasiStatic(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = factory.create(action_model_type);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> > datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas, model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    xs[i].tail(model->get_state()->get_nv()) *= 0;
    us[i] = Eigen::VectorXd::Zero(model->get_nu());
  }

  // check the state and cost in each node
  problem1.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
  problem2.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
}

void test_quasiStatic_diffAction(DifferentialActionModelTypes::Type action_model_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel = factory.create(action_model_type);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(diffModel);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models(T, model);
  std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> > datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas, model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    xs[i].tail(model->get_state()->get_nv()) *= 0;
    us[i] = Eigen::VectorXd::Zero(model->get_nu());
  }

  // check the state and cost in each node
  problem1.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
  problem2.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(ActionModelTypes::Type action_model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc, action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff, action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_quasiStatic, action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_rollout, action_model_type)));
  framework::master_test_suite().add(ts);
}

void register_diff_action_model_unit_tests(DifferentialActionModelTypes::Type action_model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_diffAction, action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff_diffAction, action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_quasiStatic_diffAction, action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_rollout_diffAction, action_model_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    register_action_model_unit_tests(ActionModelTypes::all[i]);
  }
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    register_diff_action_model_unit_tests(DifferentialActionModelTypes::all[i]);
  }
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
