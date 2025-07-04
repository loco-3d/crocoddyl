///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "factory/action.hpp"
#include "factory/diff_action.hpp"
#include "factory/integrator.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
  std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>> datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas,
                                      model->createData());

  // Run the print function
  std::ostringstream tmp;
  tmp << problem1;

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->rand();

  // check the state and cost in each node
  double cost = problem1.calc(xs, us);
  problem2.calc(xs, us);
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK(problem1.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK(problem2.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
    BOOST_CHECK(
        (problem2.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
  }
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  model->calc(data, xs.back());
  BOOST_CHECK(problem1.get_terminalData()->cost == data->cost);
  BOOST_CHECK(problem2.get_terminalData()->cost == data->cost);
  BOOST_CHECK((problem1.get_terminalData()->xnext - data->xnext).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->xnext - data->xnext).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ShootingProblemTpl<float> casted_problem1 = problem1.cast<float>();
  crocoddyl::ShootingProblemTpl<float> casted_problem2 = problem2.cast<float>();
  std::vector<Eigen::VectorXf> xs_f(T + 1);
  std::vector<Eigen::VectorXf> us_f(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs_f[i] = xs[i].cast<float>();
    us_f[i] = us[i].cast<float>();
  }
  xs_f.back() = xs.back().cast<float>();
  float cost_f = casted_problem1.calc(xs_f, us_f);
  casted_problem2.calc(xs_f, us_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>>&
        casted_model = model->cast<float>();
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>>&
        casted_data = casted_model->createData();
    casted_model->calc(casted_data, xs_f[i], us_f[i]);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->cost ==
                casted_data->cost);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->cost ==
                casted_data->cost);
    BOOST_CHECK(
        (casted_problem1.get_runningDatas()[i]->xnext - casted_data->xnext)
            .isZero(1e-9f));
    BOOST_CHECK(
        (casted_problem2.get_runningDatas()[i]->xnext - casted_data->xnext)
            .isZero(1e-9f));
    BOOST_CHECK(float(problem1.get_runningDatas()[i]->cost) -
                    casted_data->cost <=
                tol_f);
    BOOST_CHECK((problem1.get_runningDatas()[i]->xnext.cast<float>() -
                 casted_data->xnext)
                    .isZero(tol_f));
  }
  BOOST_CHECK(std::abs(float(cost) - cost_f) <= tol_f);
#endif
}

void test_calc_diffAction(DifferentialActionModelTypes::Type action_model_type,
                          IntegratorTypes::Type integrator_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel =
      factory.create(action_model_type);
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory_int.create(integrator_type, diffModel);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
  std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>> datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas,
                                      model->createData());

  // create random trajectory
  std::vector<Eigen::VectorXd> xs(T + 1);
  std::vector<Eigen::VectorXd> us(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs[i] = model->get_state()->rand();
    us[i] = Eigen::VectorXd::Random(model->get_nu());
  }
  xs.back() = model->get_state()->rand();

  // check the state and cost in each node
  double cost = problem1.calc(xs, us);
  problem2.calc(xs, us);
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK(problem1.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK(problem2.get_runningDatas()[i]->cost == data->cost);
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
    BOOST_CHECK(
        (problem2.get_runningDatas()[i]->xnext - data->xnext).isZero(1e-9));
  }
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  model->calc(data, xs.back());
  BOOST_CHECK(problem1.get_terminalData()->cost == data->cost);
  BOOST_CHECK(problem2.get_terminalData()->cost == data->cost);
  BOOST_CHECK((problem1.get_terminalData()->xnext - data->xnext).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->xnext - data->xnext).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ShootingProblemTpl<float> casted_problem1 = problem1.cast<float>();
  crocoddyl::ShootingProblemTpl<float> casted_problem2 = problem2.cast<float>();
  std::vector<Eigen::VectorXf> xs_f(T + 1);
  std::vector<Eigen::VectorXf> us_f(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs_f[i] = xs[i].cast<float>();
    us_f[i] = us[i].cast<float>();
  }
  xs_f.back() = xs.back().cast<float>();
  float cost_f = casted_problem1.calc(xs_f, us_f);
  casted_problem2.calc(xs_f, us_f);
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>>&
        casted_model = model->cast<float>();
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>>&
        casted_data = casted_model->createData();
    casted_model->calc(casted_data, xs_f[i], us_f[i]);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->cost ==
                casted_data->cost);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->cost ==
                casted_data->cost);
    BOOST_CHECK(
        (casted_problem1.get_runningDatas()[i]->xnext - casted_data->xnext)
            .isZero(1e-9f));
    BOOST_CHECK(
        (casted_problem2.get_runningDatas()[i]->xnext - casted_data->xnext)
            .isZero(1e-9f));
    BOOST_CHECK(std::abs(float(problem1.get_runningDatas()[i]->cost) -
                         casted_data->cost) <= tol_f);
    BOOST_CHECK((problem1.get_runningDatas()[i]->xnext.cast<float>() -
                 casted_data->xnext)
                    .isZero(tol_f));
  }
  BOOST_CHECK(std::abs(float(cost) - cost_f) <= tol_f);
#endif
}

void test_calcDiff(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
  std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>> datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas,
                                      model->createData());

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
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
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
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  model->calc(data, xs.back());
  model->calcDiff(data, xs.back());
  BOOST_CHECK((problem1.get_terminalData()->Fx - data->Fx).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->Fx - data->Fx).isZero(1e-9));
  BOOST_CHECK((problem1.get_terminalData()->Lx - data->Lx).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->Lx - data->Lx).isZero(1e-9));
  BOOST_CHECK((problem1.get_terminalData()->Lxx - data->Lxx).isZero(1e-9));
  BOOST_CHECK((problem2.get_terminalData()->Lxx - data->Lxx).isZero(1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ShootingProblemTpl<float> casted_problem1 = problem1.cast<float>();
  crocoddyl::ShootingProblemTpl<float> casted_problem2 = problem2.cast<float>();
  std::vector<Eigen::VectorXf> xs_f(T + 1);
  std::vector<Eigen::VectorXf> us_f(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs_f[i] = xs[i].cast<float>();
    us_f[i] = us[i].cast<float>();
  }
  xs_f.back() = xs.back().cast<float>();
  casted_problem1.calc(xs_f, us_f);
  casted_problem1.calcDiff(xs_f, us_f);
  casted_problem2.calc(xs_f, us_f);
  casted_problem2.calcDiff(xs_f, us_f);
  float tol_f = 10.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>>&
        casted_model = model->cast<float>();
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>>&
        casted_data = casted_model->createData();
    casted_model->calc(casted_data, xs_f[i], us_f[i]);
    casted_model->calcDiff(casted_data, xs_f[i], us_f[i]);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Fx == casted_data->Fx);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Fx == casted_data->Fx);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Fu == casted_data->Fu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Fu == casted_data->Fu);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lx == casted_data->Lx);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lx == casted_data->Lx);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lu == casted_data->Lu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lu == casted_data->Lu);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lxx == casted_data->Lxx);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lxx == casted_data->Lxx);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lxu == casted_data->Lxu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lxu == casted_data->Lxu);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Luu == casted_data->Luu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Luu == casted_data->Luu);
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Fx.cast<float>() - casted_data->Fx)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Fu.cast<float>() - casted_data->Fu)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lx.cast<float>() - casted_data->Lx)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lu.cast<float>() - casted_data->Lu)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lxx.cast<float>() - casted_data->Lxx)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lxu.cast<float>() - casted_data->Lxu)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Luu.cast<float>() - casted_data->Luu)
            .isZero(tol_f));
  }
#endif
}

void test_calcDiff_diffAction(
    DifferentialActionModelTypes::Type action_model_type,
    IntegratorTypes::Type integrator_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel =
      factory.create(action_model_type);
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory_int.create(integrator_type, diffModel);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
  std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>> datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas,
                                      model->createData());

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
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
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
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  model->calc(data, xs.back());
  model->calcDiff(data, xs.back());
  BOOST_CHECK((problem1.get_terminalData()->Fx - data->Fx).isZero(1e-7));
  BOOST_CHECK((problem2.get_terminalData()->Fx - data->Fx).isZero(1e-7));
  BOOST_CHECK((problem1.get_terminalData()->Lx - data->Lx).isZero(1e-7));
  BOOST_CHECK((problem2.get_terminalData()->Lx - data->Lx).isZero(1e-7));
  BOOST_CHECK((problem1.get_terminalData()->Lxx - data->Lxx).isZero(1e-7));
  BOOST_CHECK((problem2.get_terminalData()->Lxx - data->Lxx).isZero(1e-7));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ShootingProblemTpl<float> casted_problem1 = problem1.cast<float>();
  crocoddyl::ShootingProblemTpl<float> casted_problem2 = problem2.cast<float>();
  std::vector<Eigen::VectorXf> xs_f(T + 1);
  std::vector<Eigen::VectorXf> us_f(T);
  for (std::size_t i = 0; i < T; ++i) {
    xs_f[i] = xs[i].cast<float>();
    us_f[i] = us[i].cast<float>();
  }
  xs_f.back() = xs.back().cast<float>();
  casted_problem1.calc(xs_f, us_f);
  casted_problem1.calcDiff(xs_f, us_f);
  casted_problem2.calc(xs_f, us_f);
  casted_problem2.calcDiff(xs_f, us_f);
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>>&
        casted_model = model->cast<float>();
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>>&
        casted_data = casted_model->createData();
    casted_model->calc(casted_data, xs_f[i], us_f[i]);
    casted_model->calcDiff(casted_data, xs_f[i], us_f[i]);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Fx == casted_data->Fx);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Fx == casted_data->Fx);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Fu == casted_data->Fu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Fu == casted_data->Fu);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lx == casted_data->Lx);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lx == casted_data->Lx);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lu == casted_data->Lu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lu == casted_data->Lu);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lxx == casted_data->Lxx);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lxx == casted_data->Lxx);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lxu == casted_data->Lxu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lxu == casted_data->Lxu);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Luu == casted_data->Luu);
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Luu == casted_data->Luu);
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Fx.cast<float>() - casted_data->Fx)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Fu.cast<float>() - casted_data->Fu)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lx.cast<float>() - casted_data->Lx)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lu.cast<float>() - casted_data->Lu)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lxx.cast<float>() - casted_data->Lxx)
            .isZero(20.f * tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Lxu.cast<float>() - casted_data->Lxu)
            .isZero(tol_f));
    BOOST_CHECK(
        (problem1.get_runningDatas()[i]->Luu.cast<float>() - casted_data->Luu)
            .isZero(tol_f));
  }
#endif
}

void test_rollout(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);

  // create the shooting problem
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
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
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK((xs[i + 1] - data->xnext).isZero(1e-7));
  }
}

void test_rollout_diffAction(
    DifferentialActionModelTypes::Type action_model_type,
    IntegratorTypes::Type integrator_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel =
      factory.create(action_model_type);
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory_int.create(integrator_type, diffModel);

  // create the shooting problem
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
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
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    model->calc(data, xs[i], us[i]);
    BOOST_CHECK((xs[i + 1] - data->xnext).isZero(1e-7));
  }
}

void test_quasiStatic(ActionModelTypes::Type action_model_type) {
  // create the model
  ActionModelFactory factory;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory.create(action_model_type);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
  std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>> datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas,
                                      model->createData());

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
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
  problem2.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
}

void test_quasiStatic_diffAction(
    DifferentialActionModelTypes::Type action_model_type,
    IntegratorTypes::Type integrator_type) {
  // create the model
  DifferentialActionModelFactory factory;
  const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& diffModel =
      factory.create(action_model_type);
  IntegratorFactory factory_int;
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      factory_int.create(integrator_type, diffModel);

  // create two shooting problems (with and without data allocation)
  std::size_t T = 20;
  const Eigen::VectorXd& x0 = model->get_state()->rand();
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(T, model);
  std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>> datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  crocoddyl::ShootingProblem problem1(x0, models, model);
  crocoddyl::ShootingProblem problem2(x0, models, model, datas,
                                      model->createData());

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
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
  problem2.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK((u - us[i]).isZero(1e-7));
  }
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(
    ActionModelTypes::Type action_model_type) {
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

void register_diff_action_model_unit_tests(
    DifferentialActionModelTypes::Type action_model_type,
    IntegratorTypes::Type integrator_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_model_type << "_" << integrator_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_diffAction, action_model_type, integrator_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff_diffAction,
                                      action_model_type, integrator_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_quasiStatic_diffAction,
                                      action_model_type, integrator_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_rollout_diffAction,
                                      action_model_type, integrator_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    register_action_model_unit_tests(ActionModelTypes::all[i]);
  }
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    for (size_t j = 0; j < IntegratorTypes::all.size(); ++j) {
      register_diff_action_model_unit_tests(
          DifferentialActionModelTypes::all[i], IntegratorTypes::all[j]);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
