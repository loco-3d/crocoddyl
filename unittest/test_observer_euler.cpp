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
#include "crocoddyl/core/observer/euler.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/parameters.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/params/log-cholesky.hpp"
#include "crocoddyl/multibody/residuals/power.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

typedef crocoddyl::StateMultibody StateMultibody;
typedef crocoddyl::ActuationModelMultibody ActuationModelMultibody;
typedef crocoddyl::ActuationModelAbstract ActuationModelAbstract;
typedef crocoddyl::DynamicsModelConstrainedForward
    DynamicsModelConstrainedForward;
typedef crocoddyl::DynamicsDataConstrainedForward
    DynamicsDataConstrainedForward;
typedef crocoddyl::ImplicitConstraintModelMultiple
    ImplicitConstraintModelMultiple;
typedef crocoddyl::IntegratedObserverModelEuler IntegratedObserverModelEuler;
typedef crocoddyl::IntegratedObserverDataEuler IntegratedObserverDataEuler;
typedef crocoddyl::ObserverModelAbstract ObserverModelAbstract;
typedef crocoddyl::ObserverModelNumDiff ObserverModelNumDiff;
typedef crocoddyl::ObserverDataNumDiff ObserverDataNumDiff;
typedef crocoddyl::CostModelSum CostModelSum;
typedef crocoddyl::ParameterDataManager ParameterDataManager;
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

std::shared_ptr<IntegratedObserverModelEuler> create_model(
    std::shared_ptr<DynamicsModelConstrainedForward>& dynamics_out) {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  dynamics_out = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, constraints);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics_out->get_nu());
  return std::make_shared<IntegratedObserverModelEuler>(dynamics_out, costs);
}

std::shared_ptr<IntegratedObserverModelEuler> create_power_model(
    std::shared_ptr<DynamicsModelConstrainedForward>& dynamics_out) {
  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(
          crocoddyl::unittest::StateModelFactory().create(
              crocoddyl::unittest::StateModelTypes::StateMultibody_TalosArm));
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  dynamics_out = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, constraints);
  const std::size_t observer_nu = state->get_ndx() + dynamics_out->get_nu();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu);
  const std::shared_ptr<crocoddyl::ResidualModelPower> power_residual =
      std::make_shared<crocoddyl::ResidualModelPower>(state, observer_nu, 0);
  costs->addCost(
      "power",
      std::make_shared<crocoddyl::CostModelResidual>(state, power_residual),
      1.);
  return std::make_shared<IntegratedObserverModelEuler>(dynamics_out, costs);
}

std::shared_ptr<IntegratedObserverModelEuler> create_numdiff_model(
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
  return std::make_shared<IntegratedObserverModelEuler>(dynamics_out, costs);
}

std::shared_ptr<IntegratedObserverModelEuler>
create_numdiff_model_with_parameter_constraints(
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
  return std::make_shared<IntegratedObserverModelEuler>(dynamics_out, costs,
                                                        constraints);
}

void test_observer_euler_calc_matches_manual_discretization() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_model(dynamics);
  const std::shared_ptr<IntegratedObserverDataEuler> data =
      std::dynamic_pointer_cast<IntegratedObserverDataEuler>(
          model->createData());
  BOOST_REQUIRE(data != nullptr);
  const crocoddyl::DataCollectorObserver* shared =
      dynamic_cast<const crocoddyl::DataCollectorObserver*>(
          data->dynamics->shared);
  BOOST_REQUIRE(shared != nullptr);
  BOOST_CHECK(shared->hasObserverData());

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, w);

  const std::shared_ptr<DynamicsDataConstrainedForward> dyn_data =
      std::dynamic_pointer_cast<DynamicsDataConstrainedForward>(
          dynamics->createData());
  BOOST_REQUIRE(dyn_data != nullptr);
  dynamics->calc(dyn_data, x, w.tail(dynamics->get_nu()));

  const std::size_t nv = state->get_nv();
  const double dt = model->get_dt();
  Eigen::VectorXd expected_dx(state->get_ndx());
  expected_dx.head(nv).noalias() = x.tail(nv) * dt + dyn_data->vdot * dt * dt;
  expected_dx.tail(nv).noalias() = dyn_data->vdot * dt;
  expected_dx += w.head(state->get_ndx());
  Eigen::VectorXd expected_xnext(state->get_nx());
  state->integrate(x, expected_dx, expected_xnext);

  BOOST_CHECK(data->dx.isApprox(expected_dx, 1e-12));
  BOOST_CHECK(data->xnext.isApprox(expected_xnext, 1e-12));
  BOOST_CHECK(
      data->dissipative_E.isApprox(dyn_data->dissipative_P * dt, 1e-12));
}

void test_observer_euler_calc_diff_matches_manual_jacobians() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_model(dynamics);
  const std::shared_ptr<IntegratedObserverDataEuler> data =
      std::dynamic_pointer_cast<IntegratedObserverDataEuler>(
          model->createData());
  BOOST_REQUIRE(data != nullptr);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, w);
  model->calcDiff(data, x, w);

  const std::shared_ptr<DynamicsDataConstrainedForward> dyn_data =
      std::dynamic_pointer_cast<DynamicsDataConstrainedForward>(
          dynamics->createData());
  BOOST_REQUIRE(dyn_data != nullptr);
  dynamics->calc(dyn_data, x, w.tail(dynamics->get_nu()));
  dynamics->calcDiff(dyn_data, x, w.tail(dynamics->get_nu()));

  const std::size_t nv = state->get_nv();
  const std::size_t ndx = state->get_ndx();
  const std::size_t dynamics_nu = dynamics->get_nu();
  const double dt = model->get_dt();

  Eigen::MatrixXd expected_Fx = Eigen::MatrixXd::Zero(ndx, ndx);
  expected_Fx.topRows(nv).noalias() = dyn_data->Fx * dt * dt;
  expected_Fx.bottomRows(nv).noalias() = dyn_data->Fx * dt;
  expected_Fx.topRightCorner(nv, nv).diagonal().array() += dt;
  state->JintegrateTransport(x, data->dx, expected_Fx, crocoddyl::second);
  state->Jintegrate(x, data->dx, expected_Fx, expected_Fx, crocoddyl::first,
                    crocoddyl::addto);

  Eigen::MatrixXd expected_Fu = Eigen::MatrixXd::Zero(ndx, model->get_nu());
  const Eigen::MatrixXd Jdx =
      state->Jintegrate_Js(x, data->dx, crocoddyl::second)[0];
  expected_Fu.leftCols(ndx) = Jdx;
  Eigen::MatrixXd ddx_du(ndx, dynamics_nu);
  ddx_du.topRows(nv).noalias() = dyn_data->Fu * dt * dt;
  ddx_du.bottomRows(nv).noalias() = dyn_data->Fu * dt;
  expected_Fu.rightCols(dynamics_nu).noalias() = Jdx * ddx_du;

  BOOST_CHECK(data->Fx.isApprox(expected_Fx, 1e-10));
  BOOST_CHECK(data->Fu.isApprox(expected_Fu, 1e-10));
  BOOST_CHECK(data->Fp.isZero(1e-12));
  BOOST_CHECK(data->dE_dv.isApprox(dyn_data->dP_dv * dt, 1e-12));
  BOOST_CHECK(data->dE_dp.isApprox(dyn_data->dP_dp * dt, 1e-12));
}

void test_observer_euler_terminal_path() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_model(dynamics);
  const std::shared_ptr<IntegratedObserverDataEuler> data =
      std::dynamic_pointer_cast<IntegratedObserverDataEuler>(
          model->createData());
  BOOST_REQUIRE(data != nullptr);

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
  BOOST_CHECK(data->dE_dv.isZero(1e-12));
  BOOST_CHECK(data->dE_dp.isZero(1e-12));
}

void test_observer_euler_power_cost_uses_integrated_energy() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_power_model(dynamics);
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
  const Eigen::VectorXd w = 5e-2 * Eigen::VectorXd::Random(model->get_nu());

  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  model_nd.calc(data_nd, x, w);
  model_nd.calcDiff(data_nd, x, w);

  const std::shared_ptr<IntegratedObserverDataEuler> observer_data =
      std::dynamic_pointer_cast<IntegratedObserverDataEuler>(data);
  BOOST_REQUIRE(observer_data != nullptr);
  const std::shared_ptr<crocoddyl::CostDataAbstract> power_cost =
      observer_data->costs->costs.at("power");
  BOOST_REQUIRE(power_cost != nullptr);
  BOOST_CHECK_GT(std::abs(power_cost->residual->r[0]), 1e-12);
  BOOST_CHECK_SMALL(data->cost - model->get_dt() * power_cost->cost, 1e-12);

  const double tol = std::pow(model_nd.get_disturbance(), 1. / 3.);
  BOOST_CHECK_SMALL(data->cost - data_nd->cost, tol);
  BOOST_CHECK_MESSAGE(
      (data->Lx - data_nd->Lx).isZero(tol),
      "max|Lx-Lx_nd|=" << (data->Lx - data_nd->Lx).cwiseAbs().maxCoeff()
                       << " tol=" << tol);
  BOOST_CHECK_MESSAGE(
      (data->Lu - data_nd->Lu).isZero(tol),
      "max|Lu-Lu_nd|=" << (data->Lu - data_nd->Lu).cwiseAbs().maxCoeff()
                       << " noise_cols="
                       << (data->Lu.head(state->get_ndx()) -
                           data_nd->Lu.head(state->get_ndx()))
                              .cwiseAbs()
                              .maxCoeff()
                       << " control_cols="
                       << (data->Lu.tail(dynamics->get_nu()) -
                           data_nd->Lu.tail(dynamics->get_nu()))
                              .cwiseAbs()
                              .maxCoeff()
                       << " tol=" << tol);
}

void test_observer_numdiff_partial_derivatives_running() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model(dynamics, params);

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
  const Eigen::VectorXd w = Eigen::VectorXd::Random(model->get_nu());
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
    BOOST_CHECK_MESSAGE(
        (data->Lpx - data_nd->Lpx).isZero(tol),
        "max|Lpx-Lpx_nd|=" << (data->Lpx - data_nd->Lpx).cwiseAbs().maxCoeff()
                           << " tol=" << tol
                           << " analytic_finite=" << data->Lpx.allFinite()
                           << " numdiff_finite=" << data_nd->Lpx.allFinite());
    BOOST_CHECK_MESSAGE(
        (data->Lpu - data_nd->Lpu).isZero(tol),
        "max|Lpu-Lpu_nd|=" << (data->Lpu - data_nd->Lpu).cwiseAbs().maxCoeff()
                           << " tol=" << tol
                           << " analytic_finite=" << data->Lpu.allFinite()
                           << " numdiff_finite=" << data_nd->Lpu.allFinite());
  }
  BOOST_CHECK((data->Gx - data_nd->Gx).isZero(tol));
  BOOST_CHECK((data->Gu - data_nd->Gu).isZero(tol));
  BOOST_CHECK((data->Gp - data_nd->Gp).isZero(tol));
  BOOST_CHECK((data->Hx - data_nd->Hx).isZero(tol));
  BOOST_CHECK((data->Hu - data_nd->Hu).isZero(tol));
  BOOST_CHECK((data->Hp - data_nd->Hp).isZero(tol));
  const std::shared_ptr<crocoddyl::ObserverDataAbstract> obs_data =
      std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstract>(data);
  const std::shared_ptr<crocoddyl::ObserverDataAbstract> obs_data_nd =
      std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstract>(data_nd);
  BOOST_REQUIRE(obs_data != nullptr);
  BOOST_REQUIRE(obs_data_nd != nullptr);
  BOOST_CHECK((obs_data->dE_dv - obs_data_nd->dE_dv).isZero(tol));
  BOOST_CHECK((obs_data->dE_dp - obs_data_nd->dE_dp).isZero(tol));
}

void test_observer_numdiff_partial_derivatives_terminal() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model(dynamics, params);

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
  BOOST_CHECK((data->Gx - data_nd->Gx).isZero(tol));
  BOOST_CHECK((data->Gp - data_nd->Gp).isZero(tol));
  BOOST_CHECK((data->Hx - data_nd->Hx).isZero(tol));
  BOOST_CHECK((data->Hp - data_nd->Hp).isZero(tol));
}

void test_observer_constraint_parameter_derivatives_propagate() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model_with_parameter_constraints(dynamics, params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = Eigen::VectorXd::Random(model->get_nu());
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

void test_observer_euler_calc_diff_no_malloc_with_parameter_constraints() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model_with_parameter_constraints(dynamics, params);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model->createData(params->createData());
  model->set_params(data, params);

  const std::shared_ptr<StateMultibody> state =
      std::dynamic_pointer_cast<StateMultibody>(model->get_state());
  BOOST_REQUIRE(state != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = Eigen::VectorXd::Random(model->get_nu());
  const Eigen::VectorXd p = params->rand();
  const Eigen::VectorXd tau =
      Eigen::VectorXd::Random(dynamics->get_actuation()->get_nu());

  model->update_tau(tau);
  model->update_p(data, p);
  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  model->calc(data, x);
  model->calcDiff(data, x);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model->calc(data, x, w);
      model->calcDiff(data, x, w);
      model->calc(data, x);
      model->calcDiff(data, x);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  crocoddyl::IntegratedObserverModelEulerTpl<float> model_float =
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
  model_float.calc(data_float, x_float);
  model_float.calcDiff(data_float, x_float);

  const bool float_malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model_float.calc(data_float, x_float, w_float);
      model_float.calcDiff(data_float, x_float, w_float);
      model_float.calc(data_float, x_float);
      model_float.calcDiff(data_float, x_float);
    }
    Eigen::internal::set_is_malloc_allowed(float_malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(float_malloc_was_allowed);
    throw;
  }
}

void test_integrated_observer_create_data_reuses_external_params_manager() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model(dynamics, params);
  const std::shared_ptr<ParameterDataManager> params_data =
      params->createData();
  const std::shared_ptr<IntegratedObserverDataEuler> data =
      std::dynamic_pointer_cast<IntegratedObserverDataEuler>(
          model->createData(params_data));
  BOOST_REQUIRE(data != nullptr);

  const std::shared_ptr<DynamicsDataConstrainedForward> dyn =
      std::dynamic_pointer_cast<DynamicsDataConstrainedForward>(data->dynamics);
  BOOST_REQUIRE(dyn != nullptr);
  BOOST_CHECK(dyn->params == params_data);
  BOOST_CHECK(dyn->shared_params == params_data->params);

  model->set_params(data, params);
  BOOST_CHECK(dyn->params == params_data);
  BOOST_CHECK(dyn->shared_params == params_data->params);

  const Eigen::VectorXd p = params->rand();
  model->update_p(data, p);
  BOOST_CHECK(params_data->params->p.isApprox(p, 1e-12));
}

void test_observer_numdiff_create_data_propagates_external_params_manager() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model(dynamics, params);
  const std::shared_ptr<ParameterDataManager> params_data =
      params->createData();

  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model), params);
  const std::shared_ptr<ObserverDataNumDiff> data =
      std::dynamic_pointer_cast<ObserverDataNumDiff>(
          model_nd.createData(params_data));
  BOOST_REQUIRE(data != nullptr);

  auto check_shared =
      [&](const std::shared_ptr<crocoddyl::ActionDataAbstract>& inner) {
        const std::shared_ptr<IntegratedObserverDataEuler> inner_obs =
            std::dynamic_pointer_cast<IntegratedObserverDataEuler>(inner);
        BOOST_REQUIRE(inner_obs != nullptr);
        const std::shared_ptr<DynamicsDataConstrainedForward> inner_dyn =
            std::dynamic_pointer_cast<DynamicsDataConstrainedForward>(
                inner_obs->dynamics);
        BOOST_REQUIRE(inner_dyn != nullptr);
        BOOST_CHECK(inner_dyn->params == params_data);
        BOOST_CHECK(inner_dyn->shared_params == params_data->params);
      };

  check_shared(data->data_0);
  BOOST_REQUIRE(!data->data_x.empty());
  BOOST_REQUIRE(!data->data_w.empty());
  BOOST_REQUIRE(!data->data_p.empty());
  check_shared(data->data_x.front());
  check_shared(data->data_w.front());
  check_shared(data->data_p.front());

  const Eigen::VectorXd p = params->rand();
  model_nd.update_p(data, p);
  BOOST_CHECK(params_data->params->p.isApprox(p, 1e-12));
}

void test_observer_numdiff_cast() {
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics;
  std::shared_ptr<ParameterManager> params;
  const std::shared_ptr<IntegratedObserverModelEuler> model =
      create_numdiff_model(dynamics, params);
  ObserverModelNumDiff model_nd(
      std::static_pointer_cast<ObserverModelAbstract>(model), params);

  crocoddyl::ObserverModelNumDiffTpl<float> model_float =
      model_nd.cast<float>();
  const std::shared_ptr<crocoddyl::ParameterManagerTpl<float> > params_float =
      model_float.get_params();
  BOOST_REQUIRE(params_float != nullptr);
  BOOST_CHECK_EQUAL(model_float.get_np(), params->get_np());
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<float> > data_float =
      model_float.createData(params_float->createData());
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float> > state_float =
      std::dynamic_pointer_cast<crocoddyl::StateMultibodyTpl<float> >(
          model_float.get_state());
  BOOST_REQUIRE(state_float != nullptr);
  const Eigen::VectorXf x = state_float->rand();
  const Eigen::VectorXf w = Eigen::VectorXf::Random(model_float.get_nu());
  model_float.update_tau(Eigen::VectorXf::Random(model_float.get_ntau()));
  model_float.update_p(data_float, params_float->rand());
  model_float.calc(data_float, x, w);
  model_float.calcDiff(data_float, x, w);
  BOOST_CHECK(data_float->Fx.allFinite());
  BOOST_CHECK(data_float->Fu.allFinite());
  BOOST_CHECK(data_float->Fp.allFinite());
}

}  // namespace

void register_unit_tests() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_euler_calc_matches_manual_discretization));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_euler_calc_diff_matches_manual_jacobians));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_euler_terminal_path));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_euler_power_cost_uses_integrated_energy));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_numdiff_partial_derivatives_running));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_numdiff_partial_derivatives_terminal));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_observer_constraint_parameter_derivatives_propagate));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_observer_euler_calc_diff_no_malloc_with_parameter_constraints));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_integrated_observer_create_data_reuses_external_params_manager));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_observer_numdiff_create_data_propagates_external_params_manager));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_numdiff_cast));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
