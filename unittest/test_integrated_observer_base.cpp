///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/multibody/sample-models.hpp>

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/integ-observer-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/constrained-inverse.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/params/log-cholesky.hpp"
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
typedef crocoddyl::DynamicsModelConstrainedInverse
    DynamicsModelConstrainedInverse;
typedef crocoddyl::ImplicitConstraintModelMultiple
    ImplicitConstraintModelMultiple;
typedef crocoddyl::ContactModel ContactModel;
typedef crocoddyl::IntegratedObserverModelAbstract
    IntegratedObserverModelAbstract;
typedef crocoddyl::IntegratedObserverDataAbstract
    IntegratedObserverDataAbstract;
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

ContactModel::MaskArray translation_mask() {
  return ContactModel::MaskArray{{true, true, true, false, false, false}};
}

pinocchio::FrameIndex frame_id(const std::shared_ptr<StateMultibody>& state,
                               const std::string& name) {
  return state->get_pinocchio()->getFrameId(name);
}

std::shared_ptr<ContactModel> create_contact(
    const std::shared_ptr<StateMultibody>& state,
    const pinocchio::FrameIndex id, const std::size_t nu) {
  return std::make_shared<ContactModel>(
      state, id, pinocchio::SE3::Identity(), pinocchio::LOCAL_WORLD_ALIGNED, nu,
      Eigen::Vector2d::Zero(), translation_mask());
}

Eigen::MatrixXd expected_contact_noise_projector(
    const std::shared_ptr<StateMultibody>& state,
    const pinocchio::Data& pinocchio_data, const Eigen::MatrixXd& Jc,
    const std::size_t nc) {
  const std::size_t nv = state->get_nv();
  Eigen::MatrixXd Kinv = Eigen::MatrixXd::Zero(nv + nc, nv + nc);
  pinocchio::getKKTContactDynamicMatrixInverse(
      *state->get_pinocchio(), pinocchio_data, Jc.topRows(nc), Kinv);
  Kinv.block(nv, 0, nc, nv + nc) *= -1.;

  const Eigen::MatrixXd Nc = Kinv.topLeftCorner(nv, nv) * pinocchio_data.M;
  Eigen::MatrixXd projector =
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx());
  projector.topLeftCorner(nv, nv) = Nc;
  projector.bottomRightCorner(nv, nv) = Nc;
  return projector;
}

Eigen::MatrixXd expected_contact_noise_projector_recomputed(
    const std::shared_ptr<StateMultibody>& state,
    pinocchio::Data& pinocchio_data, const Eigen::MatrixXd& Jc,
    const std::size_t nc, const Eigen::Ref<const Eigen::VectorXd>& q) {
  const std::size_t nv = state->get_nv();
  Eigen::MatrixXd Kinv = Eigen::MatrixXd::Zero(nv + nc, nv + nc);
  pinocchio::computeKKTContactDynamicMatrixInverse(
      *state->get_pinocchio(), pinocchio_data, q, Jc.topRows(nc), Kinv, 0.);
  Kinv.block(nv, 0, nc, nv + nc) *= -1.;

  const Eigen::MatrixXd Nc = Kinv.topLeftCorner(nv, nv) * pinocchio_data.M;
  Eigen::MatrixXd projector =
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx());
  projector.topLeftCorner(nv, nv) = Nc;
  projector.bottomRightCorner(nv, nv) = Nc;
  return projector;
}

class DummyIntegratedObserverModel : public IntegratedObserverModelAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_CAST(crocoddyl::ActionModelBase,
                              DummyIntegratedObserverModel)

  typedef Eigen::VectorXd VectorXs;

  DummyIntegratedObserverModel(
      const std::shared_ptr<crocoddyl::DynamicsModelAbstract>& dynamics,
      const std::shared_ptr<CostModelSum>& costs)
      : IntegratedObserverModelAbstract(dynamics, costs, nullptr, 1e-3) {}

  virtual void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>&) override {
    data->xnext = x;
    data->cost = 0.;
  }

  virtual void calcDiff(
      const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>&,
      const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setZero();
    data->Fu.setZero();
    data->Fp.setZero();
    data->Lx.setZero();
    data->Lu.setZero();
    data->Lp.setZero();
    data->Lxx.setZero();
    data->Lxu.setZero();
    data->Luu.setZero();
    data->Lpp.setZero();
    data->Lpx.setZero();
    data->Lpu.setZero();
  }

  const Eigen::MatrixXd& call_compute_noise_projector(
      const std::shared_ptr<IntegratedObserverDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) const {
    return compute_noise_projector(data, x);
  }

  const Eigen::VectorXd& call_compute_projected_noise(
      const std::shared_ptr<IntegratedObserverDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& w) const {
    return compute_projected_noise(data, x, w);
  }

  const Eigen::MatrixXd& call_compute_projected_noise_jacobian(
      const std::shared_ptr<IntegratedObserverDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& w) const {
    return compute_projected_noise_jacobian(data, x, w);
  }
};

void test_integrated_observer_base_create_data_and_shared_memory() {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::size_t observer_nu = state->get_ndx() + dynamics->get_nu();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu);
  DummyIntegratedObserverModel model(dynamics, costs);

  BOOST_CHECK_EQUAL(model.get_ntau(), actuation->get_nu());
  BOOST_CHECK_EQUAL(model.get_nu(), observer_nu);
  BOOST_CHECK_EQUAL(model.get_np(), dynamics->get_np());

  const std::shared_ptr<crocoddyl::ActionDataAbstract> data_base =
      model.createData();
  const std::shared_ptr<IntegratedObserverDataAbstract> data =
      std::dynamic_pointer_cast<IntegratedObserverDataAbstract>(data_base);
  BOOST_REQUIRE(data != nullptr);
  BOOST_REQUIRE(data->dynamics != nullptr);
  BOOST_REQUIRE(data->costs != nullptr);
  BOOST_REQUIRE(data->constraintsW != nullptr);

  crocoddyl::DataCollectorObserver* const shared =
      dynamic_cast<crocoddyl::DataCollectorObserver*>(data->dynamics->shared);
  BOOST_REQUIRE(shared != nullptr);
  BOOST_CHECK(!shared->hasObserverData());
}

void test_integrated_observer_base_projector_identity_without_constraints() {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics->get_nu());
  DummyIntegratedObserverModel model(dynamics, costs);
  const std::shared_ptr<IntegratedObserverDataAbstract> data =
      std::dynamic_pointer_cast<IntegratedObserverDataAbstract>(
          model.createData());
  BOOST_REQUIRE(data != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd w = Eigen::VectorXd::Random(state->get_ndx());

  BOOST_CHECK(model.call_compute_noise_projector(data, x).isApprox(
      Eigen::MatrixXd::Identity(state->get_ndx(), state->get_ndx()), 1e-12));
  BOOST_CHECK(
      model.call_compute_projected_noise(data, x, w).isApprox(w, 1e-12));
  BOOST_CHECK(
      model.call_compute_projected_noise_jacobian(data, x, w).isZero(1e-12));
}

void test_integrated_observer_base_projector_uses_active_contacts() {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  constraints->addConstraint(
      "rf", create_contact(state, frame_id(state, "rleg6_joint"),
                           actuation->get_nu()));
  constraints->addConstraint(
      "lf",
      create_contact(state, frame_id(state, "lleg6_joint"),
                     actuation->get_nu()),
      false);
  BOOST_REQUIRE_EQUAL(constraints->get_nc(), 3);
  BOOST_REQUIRE_EQUAL(constraints->get_nc_total(), 6);

  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics->get_nu());
  DummyIntegratedObserverModel model(dynamics, costs);
  const std::shared_ptr<IntegratedObserverDataAbstract> data =
      std::dynamic_pointer_cast<IntegratedObserverDataAbstract>(
          model.createData());
  BOOST_REQUIRE(data != nullptr);
  const std::shared_ptr<DynamicsDataConstrainedForward> dyn_data =
      std::dynamic_pointer_cast<DynamicsDataConstrainedForward>(data->dynamics);
  BOOST_REQUIRE(dyn_data != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(dynamics->get_nu());
  const Eigen::VectorXd w = Eigen::VectorXd::Random(state->get_ndx());
  dynamics->calc(data->dynamics, x, u);

  const Eigen::MatrixXd expected_projector = expected_contact_noise_projector(
      state, dyn_data->pinocchio, dyn_data->multibody.constraints->Jc,
      constraints->get_nc());
  const Eigen::VectorXd expected_projected_noise = expected_projector * w;

  BOOST_CHECK(model.call_compute_noise_projector(data, x).isApprox(
      expected_projector, 1e-10));
  BOOST_CHECK(!data->noise_projector.isApprox(
      Eigen::MatrixXd::Identity(state->get_ndx(), state->get_ndx()), 1e-10));
  BOOST_CHECK(model.call_compute_projected_noise(data, x, w)
                  .isApprox(expected_projected_noise, 1e-10));
  BOOST_CHECK(
      model.call_compute_projected_noise_jacobian(data, x, w).allFinite());
  BOOST_CHECK(!data->dprojected_noise_dx.isZero(1e-12));
}

void test_integrated_observer_base_inverse_projector_recomputes_kkt() {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::size_t active_nc = 3;
  const std::size_t dynamics_nu = state->get_nv() + active_nc;
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state, dynamics_nu);
  constraints->addConstraint(
      "rf", create_contact(state, frame_id(state, "rleg6_joint"), dynamics_nu));
  constraints->addConstraint(
      "lf", create_contact(state, frame_id(state, "lleg6_joint"), dynamics_nu),
      false);
  BOOST_REQUIRE_EQUAL(constraints->get_nc(), active_nc);
  BOOST_REQUIRE_EQUAL(constraints->get_nc_total(), 2 * active_nc);

  const std::shared_ptr<DynamicsModelConstrainedInverse> dynamics =
      std::make_shared<DynamicsModelConstrainedInverse>(
          state, actuation, constraints, 0,
          crocoddyl::DynamicsType::ContinuousEstimation);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics->get_nu());
  DummyIntegratedObserverModel model(dynamics, costs);
  const std::shared_ptr<IntegratedObserverDataAbstract> data =
      std::dynamic_pointer_cast<IntegratedObserverDataAbstract>(
          model.createData());
  BOOST_REQUIRE(data != nullptr);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(dynamics->get_nu());
  const Eigen::VectorXd tau_meas = Eigen::VectorXd::Random(actuation->get_nu());
  const Eigen::VectorXd w = Eigen::VectorXd::Random(state->get_ndx());
  model.update_tau(tau_meas);
  dynamics->calc(data->dynamics, x, u);

  const std::shared_ptr<crocoddyl::DynamicsDataConstrainedInverse> dyn_data =
      std::dynamic_pointer_cast<crocoddyl::DynamicsDataConstrainedInverse>(
          data->dynamics);
  BOOST_REQUIRE(dyn_data != nullptr);

  pinocchio::Data pin_data(*state->get_pinocchio());
  pinocchio::computeAllTerms(*state->get_pinocchio(), pin_data,
                             x.head(state->get_nq()), x.tail(state->get_nv()));
  const Eigen::MatrixXd expected_projector =
      expected_contact_noise_projector_recomputed(
          state, pin_data, dyn_data->multibody.constraints->Jc,
          constraints->get_nc(), x.head(state->get_nq()));
  const Eigen::VectorXd expected_projected_noise = expected_projector * w;

  BOOST_CHECK(model.call_compute_noise_projector(data, x).isApprox(
      expected_projector, 1e-10));
  BOOST_CHECK(model.call_compute_projected_noise(data, x, w)
                  .isApprox(expected_projected_noise, 1e-10));
  BOOST_CHECK(
      model.call_compute_projected_noise_jacobian(data, x, w).allFinite());
}

void test_integrated_observer_base_set_params_resizes_data() {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics->get_nu());
  DummyIntegratedObserverModel model(dynamics, costs);
  const std::shared_ptr<IntegratedObserverDataAbstract> data =
      std::dynamic_pointer_cast<IntegratedObserverDataAbstract>(
          model.createData());
  BOOST_REQUIRE(data != nullptr);

  const std::shared_ptr<ParameterManager> params =
      create_inertial_params(state);
  model.set_params(data, params);

  BOOST_CHECK_EQUAL(model.get_np(), params->get_np());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fp.cols()),
                    params->get_np());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Ep.cols()),
                    params->get_np());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->dynamics->Fp.cols()),
                    params->get_np());
}

void test_integrated_observer_base_update_tau_propagates_to_dynamics() {
  const std::shared_ptr<StateMultibody> state = create_state();
  const std::shared_ptr<ActuationModelAbstract> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints);
  const std::shared_ptr<CostModelSum> costs = std::make_shared<CostModelSum>(
      state, state->get_ndx() + dynamics->get_nu());
  DummyIntegratedObserverModel model(dynamics, costs);

  Eigen::VectorXd tau = Eigen::VectorXd::Random(actuation->get_nu());
  model.update_tau(tau);
  BOOST_CHECK(model.get_tau_meas().isApprox(tau));
  BOOST_CHECK(dynamics->get_tau_meas().isApprox(tau));
}

}  // namespace

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_integrated_observer_base_create_data_and_shared_memory));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_integrated_observer_base_projector_identity_without_constraints));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_integrated_observer_base_projector_uses_active_contacts));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_integrated_observer_base_inverse_projector_recomputes_kkt));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_integrated_observer_base_set_params_resizes_data));
  framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_integrated_observer_base_update_tau_propagates_to_dynamics));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
