///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/data/actuation.hpp"
#include "crocoddyl/core/data/joint.hpp"
#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

template <typename Scalar>
std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > create_state() {
  StateModelFactory factory;
  const std::shared_ptr<crocoddyl::StateAbstract> state =
      factory.create(StateModelTypes::StateMultibody_RandomHumanoid);
  return std::static_pointer_cast<crocoddyl::StateMultibodyTpl<Scalar> >(
      state->template cast<Scalar>());
}

template <typename Scalar>
void test_params_data_layout_resize_and_active() {
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      create_state<Scalar>();
  ParamsData params(state, 2, 3);

  BOOST_CHECK_EQUAL(params.np, 5);
  BOOST_CHECK_EQUAL(params.np_action, 2);
  BOOST_CHECK_EQUAL(params.np_dynamics, 3);
  BOOST_CHECK(params.active);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.p.size()), 5);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.dx_dp.rows()),
                    state->get_ndx());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.dx_dp.cols()), 2);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.dtau_dp.rows()),
                    state->get_nv());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.dtau_dp.cols()), 3);
  BOOST_CHECK(params.p.isZero());
  BOOST_CHECK(params.dx_dp.isZero());
  BOOST_CHECK(params.dtau_dp.isZero());

  params.p.head(params.np_action).setOnes();
  params.p.tail(params.np_dynamics).setConstant(Scalar(2));
  BOOST_CHECK(params.p.head(params.np_action).isOnes());
  BOOST_CHECK(params.p.tail(params.np_dynamics).isConstant(Scalar(2)));

  params.p.setOnes();
  params.dx_dp.setOnes();
  params.dtau_dp.setOnes();
  params.active = false;
  params.setZero();
  BOOST_CHECK(params.p.isZero());
  BOOST_CHECK(params.dx_dp.isZero());
  BOOST_CHECK(params.dtau_dp.isZero());
  BOOST_CHECK(!params.active);

  params.resize(4, 2);
  BOOST_CHECK_EQUAL(params.np, 6);
  BOOST_CHECK_EQUAL(params.np_action, 4);
  BOOST_CHECK_EQUAL(params.np_dynamics, 2);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.p.size()), 6);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.dx_dp.cols()), 4);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(params.dtau_dp.cols()), 2);
  BOOST_CHECK(params.p.isZero());
  BOOST_CHECK(params.dx_dp.isZero());
  BOOST_CHECK(params.dtau_dp.isZero());
  BOOST_CHECK(!params.active);

  crocoddyl::ActionModelParamsDataAbstractTpl<Scalar> action_params(state, 4);
  BOOST_CHECK_EQUAL(action_params.np, 4);
  BOOST_CHECK_EQUAL(action_params.np_action, 4);
  BOOST_CHECK_EQUAL(action_params.np_dynamics, 0);
  BOOST_CHECK_EQUAL(action_params.dx_dp.cols(), 4);
  BOOST_CHECK_EQUAL(action_params.dtau_dp.cols(), 0);

  crocoddyl::DynamicsParamsDataAbstractTpl<Scalar> dynamics_params(state, 3);
  BOOST_CHECK_EQUAL(dynamics_params.np, 3);
  BOOST_CHECK_EQUAL(dynamics_params.np_action, 0);
  BOOST_CHECK_EQUAL(dynamics_params.np_dynamics, 3);
  BOOST_CHECK_EQUAL(dynamics_params.dx_dp.cols(), 0);
  BOOST_CHECK_EQUAL(dynamics_params.dtau_dp.rows(), state->get_nv());
  BOOST_CHECK_EQUAL(dynamics_params.dtau_dp.cols(), 3);
  BOOST_CHECK(dynamics_params.p.isZero());
  BOOST_CHECK(dynamics_params.dtau_dp.isZero());
}

void test_params_data_copy_and_scalar_values() {
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      create_state<double>();
  crocoddyl::ParamsDataAbstract params(state, 2, 3);
  params.p.setRandom();
  params.dx_dp.setRandom();
  params.dtau_dp.setRandom();
  params.active = false;

  crocoddyl::ParamsDataAbstract copied(params);
  BOOST_CHECK(copied.p.isApprox(params.p));
  BOOST_CHECK(copied.dx_dp.isApprox(params.dx_dp));
  BOOST_CHECK(copied.dtau_dp.isApprox(params.dtau_dp));
  BOOST_CHECK_EQUAL(copied.np, params.np);
  BOOST_CHECK_EQUAL(copied.np_action, params.np_action);
  BOOST_CHECK_EQUAL(copied.np_dynamics, params.np_dynamics);
  BOOST_CHECK_EQUAL(copied.active, params.active);

  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float> > state_float =
      create_state<float>();
  crocoddyl::ParamsDataAbstractTpl<float> casted(state_float, 2, 3);
  casted.p = params.p.cast<float>();
  casted.dx_dp = params.dx_dp.cast<float>();
  casted.dtau_dp = params.dtau_dp.cast<float>();
  casted.active = params.active;
  BOOST_CHECK(casted.p.isApprox(params.p.cast<float>()));
  BOOST_CHECK(casted.dx_dp.isApprox(params.dx_dp.cast<float>()));
  BOOST_CHECK(casted.dtau_dp.isApprox(params.dtau_dp.cast<float>()));
  BOOST_CHECK_EQUAL(casted.active, params.active);
}

template <typename Scalar>
void test_core_parameter_collectors() {
  typedef crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModel;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> MultibodyActuation;
  typedef crocoddyl::JointDataAbstractTpl<Scalar> JointData;
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      create_state<Scalar>();
  const std::shared_ptr<ActuationModel> actuation =
      std::make_shared<MultibodyActuation>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<Scalar> >
      actuation_data = actuation->createData();
  const std::shared_ptr<JointData> joint_data =
      std::make_shared<JointData>(state, actuation, actuation->get_nu());
  const std::shared_ptr<ParamsData> params =
      std::make_shared<ParamsData>(state, 2, 3);

  crocoddyl::DataCollectorParamsTpl<Scalar> params_collector(params);
  BOOST_CHECK(params_collector.params == params);
  BOOST_CHECK(params_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorParamsTpl<Scalar> params_copy(params_collector);
  BOOST_CHECK(params_copy.params == params);
  BOOST_CHECK(params_copy.parameter_data == nullptr);

  crocoddyl::DataCollectorActuationParamsTpl<Scalar> actuation_collector(
      actuation_data, params);
  BOOST_CHECK(actuation_collector.actuation == actuation_data);
  BOOST_CHECK(actuation_collector.params == params);
  BOOST_CHECK(actuation_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorActuationParamsTpl<Scalar> actuation_copy(
      actuation_collector);
  BOOST_CHECK(actuation_copy.actuation == actuation_data);
  BOOST_CHECK(actuation_copy.params == params);
  BOOST_CHECK(actuation_copy.parameter_data == nullptr);

  crocoddyl::DataCollectorJointParamsTpl<Scalar> joint_collector(joint_data,
                                                                 params);
  BOOST_CHECK(joint_collector.joint == joint_data);
  BOOST_CHECK(joint_collector.params == params);
  BOOST_CHECK(joint_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorJointParamsTpl<Scalar> joint_copy(
      joint_collector);
  BOOST_CHECK(joint_copy.joint == joint_data);
  BOOST_CHECK(joint_copy.params == params);
  BOOST_CHECK(joint_copy.parameter_data == nullptr);

  crocoddyl::DataCollectorJointActuationParamsTpl<Scalar>
      joint_actuation_collector(actuation_data, joint_data, params);
  BOOST_CHECK(joint_actuation_collector.actuation == actuation_data);
  BOOST_CHECK(joint_actuation_collector.joint == joint_data);
  BOOST_CHECK(joint_actuation_collector.params == params);
  BOOST_CHECK(joint_actuation_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorJointActuationParamsTpl<Scalar>
      joint_actuation_copy(joint_actuation_collector);
  BOOST_CHECK(joint_actuation_copy.actuation == actuation_data);
  BOOST_CHECK(joint_actuation_copy.joint == joint_data);
  BOOST_CHECK(joint_actuation_copy.params == params);
  BOOST_CHECK(joint_actuation_copy.parameter_data == nullptr);
}

template <typename Scalar>
void test_multibody_parameter_collectors() {
  typedef crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModel;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> MultibodyActuation;
  typedef crocoddyl::JointDataAbstractTpl<Scalar> JointData;
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      create_state<Scalar>();
  const std::shared_ptr<ActuationModel> actuation =
      std::make_shared<MultibodyActuation>(state);
  const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<Scalar> >
      actuation_data = actuation->createData();
  const std::shared_ptr<JointData> joint_data =
      std::make_shared<JointData>(state, actuation, actuation->get_nu());
  const std::shared_ptr<ParamsData> params =
      std::make_shared<ParamsData>(state, 2, 3);

  pinocchio::DataTpl<Scalar> pinocchio_data(*state->get_pinocchio());

  crocoddyl::DataCollectorMultibodyParamsTpl<Scalar> multibody_collector(
      &pinocchio_data, params);
  BOOST_CHECK(multibody_collector.pinocchio == &pinocchio_data);
  BOOST_CHECK(multibody_collector.params == params);
  BOOST_CHECK(multibody_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorMultibodyParamsTpl<Scalar> multibody_copy(
      multibody_collector);
  BOOST_CHECK(multibody_copy.pinocchio == &pinocchio_data);
  BOOST_CHECK(multibody_copy.params == params);
  BOOST_CHECK(multibody_copy.parameter_data == nullptr);

  crocoddyl::DataCollectorActMultibodyParamsTpl<Scalar> act_multibody_collector(
      &pinocchio_data, actuation_data, params);
  BOOST_CHECK(act_multibody_collector.pinocchio == &pinocchio_data);
  BOOST_CHECK(act_multibody_collector.actuation == actuation_data);
  BOOST_CHECK(act_multibody_collector.params == params);
  BOOST_CHECK(act_multibody_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorActMultibodyParamsTpl<Scalar>
      act_multibody_copy(act_multibody_collector);
  BOOST_CHECK(act_multibody_copy.pinocchio == &pinocchio_data);
  BOOST_CHECK(act_multibody_copy.actuation == actuation_data);
  BOOST_CHECK(act_multibody_copy.params == params);
  BOOST_CHECK(act_multibody_copy.parameter_data == nullptr);

  crocoddyl::DataCollectorJointActMultibodyParamsTpl<Scalar>
      joint_act_multibody_collector(&pinocchio_data, actuation_data, joint_data,
                                    params);
  BOOST_CHECK(joint_act_multibody_collector.pinocchio == &pinocchio_data);
  BOOST_CHECK(joint_act_multibody_collector.actuation == actuation_data);
  BOOST_CHECK(joint_act_multibody_collector.joint == joint_data);
  BOOST_CHECK(joint_act_multibody_collector.params == params);
  BOOST_CHECK(joint_act_multibody_collector.parameter_data == nullptr);
  const crocoddyl::DataCollectorJointActMultibodyParamsTpl<Scalar>
      joint_act_multibody_copy(joint_act_multibody_collector);
  BOOST_CHECK(joint_act_multibody_copy.pinocchio == &pinocchio_data);
  BOOST_CHECK(joint_act_multibody_copy.actuation == actuation_data);
  BOOST_CHECK(joint_act_multibody_copy.joint == joint_data);
  BOOST_CHECK(joint_act_multibody_copy.params == params);
  BOOST_CHECK(joint_act_multibody_copy.parameter_data == nullptr);
}

template <typename Scalar>
void test_params_data_same_size_no_allocation() {
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  typedef crocoddyl::DynamicsParamsDataAbstractTpl<Scalar> DynamicsParamsData;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      create_state<Scalar>();
  ParamsData params(state, 2, 3);
  DynamicsParamsData dynamics_params(state, 3);
  const Scalar* const p_ptr = params.p.data();
  const Scalar* const dx_dp_ptr = params.dx_dp.data();
  const Scalar* const dtau_dp_ptr = params.dtau_dp.data();
  const Scalar* const dynamics_p_ptr = dynamics_params.p.data();
  const Scalar* const dynamics_dtau_dp_ptr = dynamics_params.dtau_dp.data();

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      params.resize(2, 3);
      params.setZero();
      dynamics_params.resize(0, 3);
      dynamics_params.setZero();
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  BOOST_CHECK_EQUAL(params.p.data(), p_ptr);
  BOOST_CHECK_EQUAL(params.dx_dp.data(), dx_dp_ptr);
  BOOST_CHECK_EQUAL(params.dtau_dp.data(), dtau_dp_ptr);
  BOOST_CHECK_EQUAL(dynamics_params.p.data(), dynamics_p_ptr);
  BOOST_CHECK_EQUAL(dynamics_params.dtau_dp.data(), dynamics_dtau_dp_ptr);
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_params_data");
  ts->add(BOOST_TEST_CASE(&test_params_data_layout_resize_and_active<double>));
  ts->add(BOOST_TEST_CASE(&test_params_data_layout_resize_and_active<float>));
  ts->add(BOOST_TEST_CASE(&test_params_data_copy_and_scalar_values));
  ts->add(BOOST_TEST_CASE(&test_core_parameter_collectors<double>));
  ts->add(BOOST_TEST_CASE(&test_core_parameter_collectors<float>));
  ts->add(BOOST_TEST_CASE(&test_multibody_parameter_collectors<double>));
  ts->add(BOOST_TEST_CASE(&test_multibody_parameter_collectors<float>));
  ts->add(BOOST_TEST_CASE(&test_params_data_same_size_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_params_data_same_size_no_allocation<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
