///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC
#endif

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>
#include <limits>
#include <pinocchio/multibody/sample-models.hpp>

#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/numdiff/observer.hpp"
#include "crocoddyl/core/observer/rk.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/parameters.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/params/log-cholesky.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

typedef crocoddyl::StateMultibody StateMultibody;
typedef crocoddyl::ActuationModelMultibody ActuationModelMultibody;
typedef crocoddyl::ActuationModelAbstract ActuationModelAbstract;
typedef crocoddyl::DynamicsModelConstrainedForward
    DynamicsModelConstrainedForward;
typedef crocoddyl::ImplicitConstraintModelMultiple
    ImplicitConstraintModelMultiple;
typedef crocoddyl::IntegratedObserverModelRK IntegratedObserverModelRK;
typedef crocoddyl::IntegratedObserverDataRK IntegratedObserverDataRK;
typedef crocoddyl::ObserverModelAbstract ObserverModelAbstract;
typedef crocoddyl::ObserverModelNumDiff ObserverModelNumDiff;
typedef crocoddyl::CostModelSum CostModelSum;
typedef crocoddyl::ParameterManager ParameterManager;
typedef crocoddyl::MultibodyInertialParams MultibodyInertialParams;
typedef crocoddyl::LogCholeskyParametrization LogCholeskyParametrization;

std::shared_ptr<StateMultibody> create_state() {
  std::shared_ptr<pinocchio::Model> model =
      std::make_shared<pinocchio::Model>();
  pinocchio::buildModels::humanoidRandom(*model, true);
  model->lowerPositionLimit.template segment<7>(0).fill(-1.);
  model->upperPositionLimit.template segment<7>(0).fill(1.);
  return std::make_shared<StateMultibody>(model);
}

std::shared_ptr<ParameterManager> create_inertial_params(
    const std::shared_ptr<StateMultibody>& state) {
  const std::shared_ptr<LogCholeskyParametrization> parametrization =
      std::make_shared<LogCholeskyParametrization>();
  const std::vector<std::string> body_names(
      state->get_pinocchio()->names.begin() + 1,
      state->get_pinocchio()->names.begin() + 2);
  Eigen::VectorXd p_seed(10);
  p_seed << 0.2, -0.1, 0.15, -0.2, 0.1, -0.25, 0.3, 0.05, -0.08, 0.12;
  const std::shared_ptr<
      LogCholeskyParametrization::InertialParametrizationDataAbstract>
      data = parametrization->createData();
  Eigen::VectorXd psi(10);
  parametrization->fromParametrization(data, psi, p_seed);
  state->get_pinocchio()
      ->inertias[state->get_pinocchio()->getJointId(body_names[0])] =
      pinocchio::Inertia::FromDynamicParameters(psi);
  const std::shared_ptr<MultibodyInertialParams> inertia =
      std::make_shared<MultibodyInertialParams>(state, parametrization,
                                                body_names);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("inertia", inertia);
  return manager;
}

std::shared_ptr<IntegratedObserverModelRK> create_model(
    const crocoddyl::RKType rktype) {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics->get_nu());
  return std::make_shared<IntegratedObserverModelRK>(dynamics, costs, nullptr,
                                                     1e-3, rktype);
}

std::shared_ptr<IntegratedObserverModelRK> create_dissipative_model(
    const crocoddyl::RKType rktype,
    std::shared_ptr<ParameterManager>& params_out) {
  typedef crocoddyl::JointDynamicsModelAbstract JointModel;
  typedef crocoddyl::JointDynamicsModelFriction Friction;
  const std::shared_ptr<StateMultibody> state = create_state();

  const pinocchio::JointIndex njoints =
      static_cast<pinocchio::JointIndex>(state->get_pinocchio()->njoints);
  pinocchio::JointIndex joint_id = 1;
  for (; joint_id < njoints; ++joint_id) {
    if (state->get_pinocchio()->joints[joint_id].nv() == 1) {
      break;
    }
  }
  if (joint_id >= njoints) {
    throw_pretty("Invalid test model: no single-DoF joint was found");
  }
  Eigen::Vector2d friction_p;
  friction_p << std::log(0.3), std::log(4.0);
  const std::shared_ptr<Friction> friction = std::make_shared<Friction>(
      joint_id,
      static_cast<std::size_t>(state->get_pinocchio()->joints[joint_id].nq()),
      friction_p, crocoddyl::JointFrictionType::Coulomb);
  const std::vector<std::shared_ptr<JointModel> > joints(1, friction);
  const std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state, joints);
  const std::shared_ptr<crocoddyl::ActuationMultibodyParams> actuation_params =
      std::make_shared<crocoddyl::ActuationMultibodyParams>(actuation);
  params_out = std::make_shared<ParameterManager>(state);
  params_out->addParam("actuation", actuation_params);

  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::size_t observer_nu = state->get_ndx() + dynamics->get_nu();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu, params_out->get_np());
  return std::make_shared<IntegratedObserverModelRK>(dynamics, costs, nullptr,
                                                     2e-2, rktype);
}

std::shared_ptr<IntegratedObserverModelRK> create_numdiff_model(
    const crocoddyl::RKType rktype,
    std::shared_ptr<DynamicsModelConstrainedForward>& dynamics_out,
    std::shared_ptr<ParameterManager>& params_out) {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  dynamics_out = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, constraints);
  params_out = create_inertial_params(state);

  const std::size_t observer_nu = state->get_ndx() + dynamics_out->get_nu();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu, params_out->get_np());
  const std::shared_ptr<crocoddyl::ResidualModelState> x_residual =
      std::make_shared<crocoddyl::ResidualModelState>(state, state->zero(),
                                                      observer_nu);
  const std::shared_ptr<crocoddyl::ResidualModelControl> u_residual =
      std::make_shared<crocoddyl::ResidualModelControl>(state, observer_nu);
  const std::shared_ptr<crocoddyl::ResidualModelParameters> p_residual =
      std::make_shared<crocoddyl::ResidualModelParameters>(
          state, params_out->zero(), observer_nu);
  costs->addCost(
      "xReg", std::make_shared<crocoddyl::CostModelResidual>(state, x_residual),
      1.);
  costs->addCost(
      "uReg", std::make_shared<crocoddyl::CostModelResidual>(state, u_residual),
      1.);
  costs->addCost(
      "pReg", std::make_shared<crocoddyl::CostModelResidual>(state, p_residual),
      1.);
  return std::make_shared<IntegratedObserverModelRK>(dynamics_out, costs,
                                                     nullptr, 1e-3, rktype);
}

std::shared_ptr<IntegratedObserverModelRK>
create_numdiff_model_with_parameter_constraints(
    const crocoddyl::RKType rktype,
    std::shared_ptr<DynamicsModelConstrainedForward>& dynamics_out,
    std::shared_ptr<ParameterManager>& params_out) {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> dynamics_constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  dynamics_out = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, dynamics_constraints);
  params_out = create_inertial_params(state);

  const std::size_t observer_nu = state->get_ndx() + dynamics_out->get_nu();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu, params_out->get_np());
  const std::shared_ptr<crocoddyl::ConstraintModelManager> constraints =
      std::make_shared<crocoddyl::ConstraintModelManager>(state, observer_nu,
                                                          params_out->get_np());
  const std::shared_ptr<crocoddyl::ResidualModelParameters> p_residual =
      std::make_shared<crocoddyl::ResidualModelParameters>(
          state, params_out->zero(), observer_nu);
  constraints->addConstraint(
      "parameter_equality",
      std::make_shared<crocoddyl::ConstraintModelResidual>(state, p_residual));
  constraints->addConstraint(
      "parameter_inequality",
      std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, p_residual, -Eigen::VectorXd::Ones(params_out->get_np()),
          Eigen::VectorXd::Ones(params_out->get_np())));
  return std::make_shared<IntegratedObserverModelRK>(dynamics_out, costs,
                                                     constraints, 1e-3, rktype);
}

std::shared_ptr<IntegratedObserverModelRK>
create_continuous_estimation_state_tracking_model(
    const crocoddyl::RKType rktype,
    std::shared_ptr<DynamicsModelConstrainedForward>& dynamics_out,
    std::shared_ptr<ParameterManager>& params_out) {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  dynamics_out = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, constraints, 0,
      crocoddyl::DynamicsType::ContinuousEstimation);
  params_out = create_inertial_params(state);

  const std::size_t observer_nu = state->get_ndx();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu, params_out->get_np());
  const std::shared_ptr<crocoddyl::ResidualModelState> x_residual =
      std::make_shared<crocoddyl::ResidualModelState>(state, state->rand(),
                                                      observer_nu);
  costs->addCost(
      "xObs", std::make_shared<crocoddyl::CostModelResidual>(state, x_residual),
      10.);
  return std::make_shared<IntegratedObserverModelRK>(dynamics_out, costs,
                                                     nullptr, 1e-2, rktype);
}

void test_observer_rk_dissipative_energy(const crocoddyl::RKType rktype) {
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_dissipative_model(rktype, params);
  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model), params);
  const std::shared_ptr<IntegratedObserverDataRK> data =
      std::dynamic_pointer_cast<IntegratedObserverDataRK>(
          model->createData(params->createData()));
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data_nd =
      model_nd.createData(params->createData());
  BOOST_REQUIRE(data != nullptr);
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = 5e-2 * Eigen::VectorXd::Random(model->get_nu());
  Eigen::VectorXd p = params->zero();
  p.array() += 0.05;
  model->update_p(data, p);
  model_nd.update_p(data_nd, p);

  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  model_nd.calc(data_nd, x, w);
  model_nd.calcDiff(data_nd, x, w);

  double expected_energy = 0.;
  if (rktype == crocoddyl::two) {
    expected_energy = data->dynamics_stage[1]->dissipative_P[0];
  } else if (rktype == crocoddyl::three) {
    expected_energy = (data->dynamics_stage[0]->dissipative_P[0] +
                       3. * data->dynamics_stage[2]->dissipative_P[0]) /
                      4.;
  } else {
    expected_energy = (data->dynamics_stage[0]->dissipative_P[0] +
                       2. * data->dynamics_stage[1]->dissipative_P[0] +
                       2. * data->dynamics_stage[2]->dissipative_P[0] +
                       data->dynamics_stage[3]->dissipative_P[0]) /
                      6.;
  }
  expected_energy *= model->get_dt();

  const double tol = std::pow(model_nd.get_disturbance(), 1. / 3.);
  const double nonzero_tol = 10. * std::numeric_limits<double>::epsilon();
  BOOST_CHECK_SMALL(data->dissipative_E[0] - expected_energy, 1e-12);
  BOOST_CHECK_GT(data->Ex.norm(), nonzero_tol);
  BOOST_CHECK_GT(data->Eu.norm(), nonzero_tol);
  BOOST_CHECK_GT(data->Ep.norm(), nonzero_tol);
  const std::shared_ptr<crocoddyl::ObserverDataAbstract> obs_data_nd =
      std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstract>(data_nd);
  BOOST_REQUIRE(obs_data_nd != nullptr);
  BOOST_CHECK((data->Ex - obs_data_nd->Ex).isZero(tol));
  BOOST_CHECK((data->Eu - obs_data_nd->Eu).isZero(tol));
  BOOST_CHECK((data->Ep - obs_data_nd->Ep).isZero(tol));
}

void test_observer_rk_numdiff(const crocoddyl::RKType rktype) {
  const std::shared_ptr<IntegratedObserverModelRK> model = create_model(rktype);
  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model));
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data_nd =
      model_nd.createData();

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = 1e-2 * Eigen::VectorXd::Random(model->get_nu());

  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  model_nd.calc(data_nd, x, w);
  model_nd.calcDiff(data_nd, x, w);

  const double tol = std::pow(model_nd.get_disturbance(), 1. / 3.);
  BOOST_CHECK(
      (data->xnext - data_nd->xnext)
          .isZero(std::sqrt(2.0 * std::numeric_limits<double>::epsilon())));
  BOOST_CHECK((data->Fx - data_nd->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_nd->Fu).isZero(tol));
  BOOST_CHECK(data->Fp.isZero(1e-12));
  BOOST_CHECK((data->Lx - data_nd->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_nd->Lu).isZero(tol));
  BOOST_CHECK(data->Lp.isZero(1e-12));

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model->calc(data, x, w);
      model->calcDiff(data, x, w);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

void test_observer_rk_terminal_path() {
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_model(crocoddyl::four);
  const std::shared_ptr<IntegratedObserverDataRK> data =
      std::dynamic_pointer_cast<IntegratedObserverDataRK>(model->createData());
  BOOST_REQUIRE(data != nullptr);
  const crocoddyl::DataCollectorObserver* shared =
      dynamic_cast<const crocoddyl::DataCollectorObserver*>(
          data->dynamics->shared);
  BOOST_REQUIRE(shared != nullptr);
  BOOST_CHECK(!shared->hasObserverData());
  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  model->calc(data, x);
  model->calcDiff(data, x);

  BOOST_CHECK(data->xnext.isApprox(x, 1e-12));
  BOOST_CHECK(data->dx.isZero(1e-12));
  BOOST_CHECK(data->Fx.isApprox(
      Eigen::MatrixXd::Identity(state->get_ndx(), state->get_ndx()), 1e-12));
  BOOST_CHECK(data->Fu.isZero(1e-12));
  BOOST_CHECK(data->Fp.isZero(1e-12));
  BOOST_CHECK(data->dissipative_E.isZero(1e-12));
  BOOST_CHECK(data->Ex.isZero(1e-12));
  BOOST_CHECK(data->Eu.isZero(1e-12));
  BOOST_CHECK(data->Ep.isZero(1e-12));
}

void test_observer_rk_parameter_derivatives_running(
    const crocoddyl::RKType rktype) {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_numdiff_model(rktype, dynamics, params);

  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model), params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data_nd =
      model_nd.createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = 1e-2 * Eigen::VectorXd::Random(model->get_nu());
  const Eigen::VectorXd p = params->rand();
  const Eigen::VectorXd tau =
      Eigen::VectorXd::Random(dynamics->get_actuation()->get_nu());

  model->update_tau(tau);
  model_nd.update_tau(tau);
  model->update_p(data, p);
  model_nd.update_p(data_nd, p);

  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  model_nd.calc(data_nd, x, w);
  model_nd.calcDiff(data_nd, x, w);

  const double tol = std::pow(model_nd.get_disturbance(), 1. / 3.);
  BOOST_CHECK(
      (data->xnext - data_nd->xnext)
          .isZero(std::sqrt(2.0 * std::numeric_limits<double>::epsilon())));
  BOOST_CHECK((data->Fx - data_nd->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_nd->Fu).isZero(tol));
  BOOST_CHECK((data->Fp - data_nd->Fp).isZero(tol));
  BOOST_CHECK((data->Lx - data_nd->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_nd->Lu).isZero(tol));
  BOOST_CHECK((data->Lp - data_nd->Lp).isZero(tol));
  if (model_nd.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_nd->Lxx).isZero(tol));
    BOOST_CHECK((data->Luu - data_nd->Luu).isZero(tol));
    BOOST_CHECK((data->Lpp - data_nd->Lpp).isZero(tol));
    BOOST_CHECK((data->Lxu - data_nd->Lxu).isZero(tol));
    BOOST_CHECK((data->Lpx - data_nd->Lpx).isZero(tol));
    BOOST_CHECK((data->Lpu - data_nd->Lpu).isZero(tol));
  }
  BOOST_CHECK((data->Gx - data_nd->Gx).isZero(tol));
  BOOST_CHECK((data->Gu - data_nd->Gu).isZero(tol));
  BOOST_CHECK((data->Gp - data_nd->Gp).isZero(tol));
  BOOST_CHECK((data->Hx - data_nd->Hx).isZero(tol));
  BOOST_CHECK((data->Hu - data_nd->Hu).isZero(tol));
  BOOST_CHECK((data->Hp - data_nd->Hp).isZero(tol));
}

void test_observer_rk_parameter_derivatives_terminal(
    const crocoddyl::RKType rktype) {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_numdiff_model(rktype, dynamics, params);

  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model), params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data_nd =
      model_nd.createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd p = params->rand();

  model->update_p(data, p);
  model_nd.update_p(data_nd, p);

  model->calc(data, x);
  model->calcDiff(data, x);
  model_nd.calc(data_nd, x);
  model_nd.calcDiff(data_nd, x);

  const double tol = std::pow(model_nd.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->xnext - x).isZero(tol));
  BOOST_CHECK((data->xnext - data_nd->xnext).isZero(tol));
  BOOST_CHECK((data->Lx - data_nd->Lx).isZero(tol));
  BOOST_CHECK((data->Lp - data_nd->Lp).isZero(tol));
  if (model_nd.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_nd->Lxx).isZero(tol));
    BOOST_CHECK((data->Lpp - data_nd->Lpp).isZero(tol));
    BOOST_CHECK((data->Lpx - data_nd->Lpx).isZero(tol));
  }
}

void test_observer_rk_constraint_parameter_derivatives_propagate(
    const crocoddyl::RKType rktype) {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_numdiff_model_with_parameter_constraints(rktype, dynamics, params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = 1e-2 * Eigen::VectorXd::Random(model->get_nu());
  const Eigen::VectorXd p = params->rand();
  const Eigen::VectorXd tau =
      Eigen::VectorXd::Random(dynamics->get_actuation()->get_nu());

  model->update_tau(tau);
  model->update_p(data, p);
  model->calc(data, x, w);
  data->Gp.setConstant(42.);
  data->Hp.setConstant(42.);
  model->calcDiff(data, x, w);

  const std::size_t np = params->get_np();
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.cols()), np);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.cols()), np);
  BOOST_CHECK(data->Gp.bottomRows(np).isApprox(
      Eigen::MatrixXd::Identity(np, np), 1e-12));
  BOOST_CHECK(data->Hp.bottomRows(np).isApprox(
      Eigen::MatrixXd::Identity(np, np), 1e-12));
}

void test_observer_rk_continuous_estimation_cost_derivatives(
    const crocoddyl::RKType rktype) {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_continuous_estimation_state_tracking_model(rktype, dynamics,
                                                        params);

  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model), params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data_nd =
      model_nd.createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = 1e-2 * Eigen::VectorXd::Random(model->get_nu());
  const Eigen::VectorXd p = params->rand();
  const Eigen::VectorXd tau =
      Eigen::VectorXd::Random(dynamics->get_actuation()->get_nu());

  model->update_tau(tau);
  model_nd.update_tau(tau);
  model->update_p(data, p);
  model_nd.update_p(data_nd, p);

  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  model_nd.calc(data_nd, x, w);
  model_nd.calcDiff(data_nd, x, w);

  const double tol = std::pow(model_nd.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->Fx - data_nd->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_nd->Fu).isZero(tol));
  BOOST_CHECK((data->Fp - data_nd->Fp).isZero(tol));
  BOOST_CHECK((data->Lx - data_nd->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_nd->Lu).isZero(tol));
  BOOST_CHECK((data->Lp - data_nd->Lp).isZero(tol));
}

void test_observer_rk_calc_diff_no_malloc_with_parameters() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelRK> model =
      create_numdiff_model_with_parameter_constraints(crocoddyl::four, dynamics,
                                                      params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = 1e-2 * Eigen::VectorXd::Random(model->get_nu());
  const Eigen::VectorXd p = params->rand();
  const Eigen::VectorXd tau =
      Eigen::VectorXd::Random(dynamics->get_actuation()->get_nu());

  model->update_tau(tau);
  model->update_p(data, p);
  model->calc(data, x, w);
  model->calcDiff(data, x, w);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    model->calcDiff(data, x, w);
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  crocoddyl::IntegratedObserverModelRKTpl<float> model_float =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::ParameterManagerTpl<float> > params_float =
      model_float.get_params();
  BOOST_REQUIRE(params_float != nullptr);
  const std::shared_ptr<crocoddyl::DynamicsModelConstrainedForwardTpl<float> >
      dynamics_float = std::dynamic_pointer_cast<
          crocoddyl::DynamicsModelConstrainedForwardTpl<float> >(
          model_float.get_dynamics());
  const std::shared_ptr<crocoddyl::MultibodyInertialParamsTpl<float> >
      inertia_float = std::dynamic_pointer_cast<
          crocoddyl::MultibodyInertialParamsTpl<float> >(
          params_float->get_dynamics_params().at("inertia")->get_param());
  BOOST_REQUIRE(dynamics_float != nullptr);
  BOOST_REQUIRE(inertia_float != nullptr);
  BOOST_CHECK(inertia_float->get_state() == dynamics_float->get_state());
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float> > data_float =
      model_float.createData(params_float->createData());
  model_float.set_params(data_float, params_float);
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float> > state_float =
      std::dynamic_pointer_cast<crocoddyl::StateMultibodyTpl<float> >(
          model_float.get_state());
  BOOST_REQUIRE(state_float != nullptr);
  const Eigen::VectorXf x_float = state_float->rand();
  const Eigen::VectorXf w_float = Eigen::VectorXf::Random(model_float.get_nu());
  model_float.update_tau(Eigen::VectorXf::Random(model_float.get_ntau()));
  model_float.update_p(data_float, params_float->rand());
  model_float.calc(data_float, x_float, w_float);
  model_float.calcDiff(data_float, x_float, w_float);

  const bool float_malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model_float.calc(data_float, x_float, w_float);
      model_float.calcDiff(data_float, x_float, w_float);
    }
    Eigen::internal::set_is_malloc_allowed(float_malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(float_malloc_was_allowed);
    throw;
  }
}

}  // namespace

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_dissipative_energy, crocoddyl::two)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_dissipative_energy, crocoddyl::three)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_dissipative_energy, crocoddyl::four)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_observer_rk_numdiff, crocoddyl::two)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_numdiff, crocoddyl::three)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(boost::bind(&test_observer_rk_numdiff, crocoddyl::four)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_rk_terminal_path));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_observer_rk_parameter_derivatives_running, crocoddyl::two)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_observer_rk_parameter_derivatives_running, crocoddyl::three)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_observer_rk_parameter_derivatives_running, crocoddyl::four)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_observer_rk_parameter_derivatives_terminal, crocoddyl::two)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_observer_rk_parameter_derivatives_terminal, crocoddyl::three)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(
      &test_observer_rk_parameter_derivatives_terminal, crocoddyl::four)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_constraint_parameter_derivatives_propagate,
                  crocoddyl::two)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_constraint_parameter_derivatives_propagate,
                  crocoddyl::three)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_constraint_parameter_derivatives_propagate,
                  crocoddyl::four)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_continuous_estimation_cost_derivatives,
                  crocoddyl::two)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_continuous_estimation_cost_derivatives,
                  crocoddyl::three)));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      boost::bind(&test_observer_rk_continuous_estimation_cost_derivatives,
                  crocoddyl::four)));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_rk_calc_diff_no_malloc_with_parameters));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
