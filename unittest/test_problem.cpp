///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2026, University of Edinburgh Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "factory/action.hpp"
#include "factory/control.hpp"
#include "factory/dynamics.hpp"
#include "factory/integrator.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_calc_model(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
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
  BOOST_CHECK(isCloseAbsRel(problem1.get_terminalData()->xnext, data->xnext,
                            1e-9, 1e-9));
  BOOST_CHECK(isCloseAbsRel(problem2.get_terminalData()->xnext, data->xnext,
                            1e-9, 1e-9));

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
    BOOST_CHECK(
        isCloseAbsRel(problem1.get_runningDatas()[i]->xnext.cast<float>(),
                      casted_data->xnext, tol_f, tol_f));
  }
  BOOST_CHECK(std::abs(float(cost) - cost_f) <= tol_f);
#endif
}

void test_calcDiff_model(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
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
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Fx, data->Fx,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Fx, data->Fx,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Fu, data->Fu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Fu, data->Fu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lx, data->Lx,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Lx, data->Lx,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lu, data->Lu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Lu, data->Lu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lxx, data->Lxx,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Lxx, data->Lxx,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lxu, data->Lxu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Lxu, data->Lxu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Luu, data->Luu,
                              1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(problem2.get_runningDatas()[i]->Luu, data->Luu,
                              1e-9, 1e-9));
  }
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();
  model->calc(data, xs.back());
  model->calcDiff(data, xs.back());
  BOOST_CHECK(
      isCloseAbsRel(problem1.get_terminalData()->Fx, data->Fx, 1e-9, 1e-9));
  BOOST_CHECK(
      isCloseAbsRel(problem2.get_terminalData()->Fx, data->Fx, 1e-9, 1e-9));
  BOOST_CHECK(
      isCloseAbsRel(problem1.get_terminalData()->Lx, data->Lx, 1e-9, 1e-9));
  BOOST_CHECK(
      isCloseAbsRel(problem2.get_terminalData()->Lx, data->Lx, 1e-9, 1e-9));
  BOOST_CHECK(
      isCloseAbsRel(problem1.get_terminalData()->Lxx, data->Lxx, 1e-9, 1e-9));
  BOOST_CHECK(
      isCloseAbsRel(problem2.get_terminalData()->Lxx, data->Lxx, 1e-9, 1e-9));

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
  float tol_f_jac = 100.f * tol_f;
  float tol_f_hess = 100.f * tol_f;
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>>&
        casted_model = model->cast<float>();
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float>>&
        casted_data = casted_model->createData();
    casted_model->calc(casted_data, xs_f[i], us_f[i]);
    casted_model->calcDiff(casted_data, xs_f[i], us_f[i]);
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Fx.isApprox(
        casted_data->Fx, tol_f_jac));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Fx.isApprox(
        casted_data->Fx, tol_f_jac));
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Fu.isApprox(
        casted_data->Fu, tol_f_jac));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Fu.isApprox(
        casted_data->Fu, tol_f_jac));
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lx.isApprox(
        casted_data->Lx, tol_f));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lx.isApprox(
        casted_data->Lx, tol_f));
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lu.isApprox(
        casted_data->Lu, tol_f));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lu.isApprox(
        casted_data->Lu, tol_f));
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lxx.isApprox(
        casted_data->Lxx, tol_f_hess));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lxx.isApprox(
        casted_data->Lxx, tol_f_hess));
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Lxu.isApprox(
        casted_data->Lxu, tol_f_hess));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Lxu.isApprox(
        casted_data->Lxu, tol_f_hess));
    BOOST_CHECK(casted_problem1.get_runningDatas()[i]->Luu.isApprox(
        casted_data->Luu, tol_f_hess));
    BOOST_CHECK(casted_problem2.get_runningDatas()[i]->Luu.isApprox(
        casted_data->Luu, tol_f_hess));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Fx.cast<float>(),
                              casted_data->Fx, tol_f_jac, tol_f_jac));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Fu.cast<float>(),
                              casted_data->Fu, tol_f_jac, tol_f_jac));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lx.cast<float>(),
                              casted_data->Lx, tol_f, tol_f));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lu.cast<float>(),
                              casted_data->Lu, tol_f, tol_f));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lxx.cast<float>(),
                              casted_data->Lxx, tol_f_hess, tol_f_hess));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Lxu.cast<float>(),
                              casted_data->Lxu, tol_f_hess, tol_f_hess));
    BOOST_CHECK(isCloseAbsRel(problem1.get_runningDatas()[i]->Luu.cast<float>(),
                              casted_data->Luu, tol_f_hess, tol_f_hess));
  }
#endif
}

void test_rollout_model(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
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
    BOOST_CHECK(isCloseAbsRel(xs[i + 1], data->xnext, 1e-7, 1e-7));
  }
}

void test_quasiStatic_model(
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model) {
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
    BOOST_CHECK(isCloseAbsRel(u, us[i], 1e-7, 1e-7));
  }
  problem2.quasiStatic(us, xs);
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        model->createData();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
    model->quasiStatic(data, u, xs[i]);
    BOOST_CHECK(isCloseAbsRel(u, us[i], 1e-7, 1e-7));
  }
}

//----------------------------------------------------------------------------//

std::shared_ptr<crocoddyl::ActionModelAbstract> create_dynamics_action_model(
    const DynamicsModelTypes::Type dynamics_type,
    const IntegratorTypes::Type integrator_type) {
  const DynamicsModelFactoryResult components =
      DynamicsModelFactory().create(dynamics_type);
  const std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract>
      control = ControlFactory().create(ControlTypes::PolyZero,
                                        components.dynamics->get_nu());
  return IntegratorFactory().create(integrator_type, components.dynamics,
                                    components.costs, components.constraints,
                                    control);
}

void test_calc(ActionModelTypes::Type action_model_type) {
  test_calc_model(ActionModelFactory().create(action_model_type));
}

void test_calc(const DynamicsModelTypes::Type dynamics_type,
               const IntegratorTypes::Type integrator_type) {
  test_calc_model(create_dynamics_action_model(dynamics_type, integrator_type));
}

void test_calcDiff(ActionModelTypes::Type action_model_type) {
  test_calcDiff_model(ActionModelFactory().create(action_model_type));
}

void test_calcDiff(const DynamicsModelTypes::Type dynamics_type,
                   const IntegratorTypes::Type integrator_type) {
  test_calcDiff_model(
      create_dynamics_action_model(dynamics_type, integrator_type));
}

void test_rollout(ActionModelTypes::Type action_model_type) {
  test_rollout_model(ActionModelFactory().create(action_model_type));
}

void test_rollout(const DynamicsModelTypes::Type dynamics_type,
                  const IntegratorTypes::Type integrator_type) {
  test_rollout_model(
      create_dynamics_action_model(dynamics_type, integrator_type));
}

void test_quasiStatic(ActionModelTypes::Type action_model_type) {
  test_quasiStatic_model(ActionModelFactory().create(action_model_type));
}

void test_quasiStatic(const DynamicsModelTypes::Type dynamics_type,
                      const IntegratorTypes::Type integrator_type) {
  test_quasiStatic_model(
      create_dynamics_action_model(dynamics_type, integrator_type));
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(
    ActionModelTypes::Type action_model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_model_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(static_cast<void (*)(ActionModelTypes::Type)>(&test_calc),
                  action_model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(static_cast<void (*)(ActionModelTypes::Type)>(&test_calcDiff),
                  action_model_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(
      static_cast<void (*)(ActionModelTypes::Type)>(&test_quasiStatic),
      action_model_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(static_cast<void (*)(ActionModelTypes::Type)>(&test_rollout),
                  action_model_type)));
  framework::master_test_suite().add(ts);
}

void register_action_model_unit_tests(
    const DynamicsModelTypes::Type dynamics_type,
    const IntegratorTypes::Type integrator_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << dynamics_type << "_" << integrator_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(
      static_cast<void (*)(DynamicsModelTypes::Type, IntegratorTypes::Type)>(
          &test_calc),
      dynamics_type, integrator_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(
      static_cast<void (*)(DynamicsModelTypes::Type, IntegratorTypes::Type)>(
          &test_calcDiff),
      dynamics_type, integrator_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(
      static_cast<void (*)(DynamicsModelTypes::Type, IntegratorTypes::Type)>(
          &test_quasiStatic),
      dynamics_type, integrator_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(
      static_cast<void (*)(DynamicsModelTypes::Type, IntegratorTypes::Type)>(
          &test_rollout),
      dynamics_type, integrator_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    register_action_model_unit_tests(ActionModelTypes::all[i]);
  }
  for (std::vector<DynamicsModelTypes::Type>::const_iterator dynamics =
           DynamicsModelTypes::all.begin();
       dynamics != DynamicsModelTypes::all.end(); ++dynamics) {
    for (std::vector<IntegratorTypes::Type>::const_iterator integrator =
             IntegratorTypes::all.begin();
         integrator != IntegratorTypes::all.end(); ++integrator) {
      register_action_model_unit_tests(*dynamics, *integrator);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
