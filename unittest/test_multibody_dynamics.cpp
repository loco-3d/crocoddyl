///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>
#include <type_traits>

#include "crocoddyl/core/integrator/time.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/actuations/joint-friction.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/constrained-inverse.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/kinematic-loop.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

template <typename Scalar>
struct DynamicsFixture {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::JointDynamicsModelAbstractTpl<Scalar> JointModel;
  typedef crocoddyl::JointDynamicsModelFrictionTpl<Scalar> Friction;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Constraints;
  typedef crocoddyl::ContactModelTpl<Scalar> Contact;
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Loop;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  static std::shared_ptr<State> state() {
    const std::shared_ptr<crocoddyl::StateMultibody> state64 =
        std::static_pointer_cast<crocoddyl::StateMultibody>(
            StateModelFactory().create(
                StateModelTypes::StateMultibody_TalosArm));
    return std::make_shared<State>(state64->template cast<Scalar>());
  }

  static typename Contact::MaskArray contact_mask(const std::size_t nc) {
    typename Contact::MaskArray mask = {
        {false, false, false, false, false, false}};
    for (std::size_t i = 0; i < nc; ++i) {
      mask[i] = true;
    }
    return mask;
  }

  static typename Loop::MaskArray loop_mask(const std::size_t nc) {
    typename Loop::MaskArray mask = {
        {false, false, false, false, false, false}};
    for (std::size_t i = 0; i < nc; ++i) {
      mask[i] = true;
    }
    return mask;
  }

  static std::shared_ptr<Contact> contact(const std::shared_ptr<State>& state,
                                          const std::size_t nu,
                                          const std::size_t nc,
                                          const Scalar gain = Scalar(0.1)) {
    typedef pinocchio::SE3Tpl<Scalar> SE3;
    typename Contact::Vector2s gains;
    gains << gain, Scalar(0.2);
    const pinocchio::FrameIndex frame_id = static_cast<pinocchio::FrameIndex>(
        state->get_pinocchio()->frames.size() - 1);
    return std::make_shared<Contact>(state, frame_id, SE3::Identity(),
                                     pinocchio::LOCAL_WORLD_ALIGNED, nu, gains,
                                     contact_mask(nc));
  }

  static std::shared_ptr<Loop> loop(const std::shared_ptr<State>& state,
                                    const std::size_t nu,
                                    const std::size_t nc) {
    typedef pinocchio::SE3Tpl<Scalar> SE3;
    typename Loop::Vector2s gains;
    gains << Scalar(0.15), Scalar(0.25);
    return std::make_shared<Loop>(state, 1, SE3::Identity(), 2, SE3::Identity(),
                                  pinocchio::LOCAL, nu, gains, loop_mask(nc));
  }

  static std::shared_ptr<Actuation> friction_actuation(
      const std::shared_ptr<State>& state) {
    const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
        state->get_pinocchio();
    pinocchio::JointIndex joint_id = 1;
    for (; joint_id < static_cast<pinocchio::JointIndex>(pin_model->njoints);
         ++joint_id) {
      if (pin_model->joints[joint_id].nv() == 1) {
        break;
      }
    }
    if (joint_id == static_cast<pinocchio::JointIndex>(pin_model->njoints)) {
      throw_pretty("Test model has no one-DoF joint");
    }
    VectorXs mu(3);
    mu << std::log(Scalar(0.15)), std::log(Scalar(3.)), std::log(Scalar(0.2));
    std::vector<std::shared_ptr<JointModel> > joints;
    joints.push_back(std::make_shared<Friction>(
        joint_id, static_cast<std::size_t>(pin_model->joints[joint_id].nq()),
        mu, crocoddyl::JointFrictionType::CoulombViscous));
    return std::make_shared<Actuation>(state, joints);
  }
};

template <typename Scalar>
Scalar derivative_tolerance() {
  return std::is_same<Scalar, float>::value ? Scalar(2e-2) : Scalar(8e-4);
}

template <typename Scalar>
Scalar finite_difference_step() {
  return std::is_same<Scalar, float>::value ? Scalar(5e-3) : Scalar(1e-6);
}

template <typename Scalar, typename Model, typename Data>
void check_forward_finite_differences(const std::shared_ptr<Model>& model,
                                      const std::shared_ptr<Data>& data,
                                      const typename Model::VectorXs& x,
                                      const typename Model::VectorXs& u) {
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;
  const Scalar eps = finite_difference_step<Scalar>();
  MatrixXs Fx(data->Fx.rows(), data->Fx.cols());
  MatrixXs Fu(data->Fu.rows(), data->Fu.cols());
  const std::shared_ptr<Data> plus =
      std::dynamic_pointer_cast<Data>(model->createData());
  const std::shared_ptr<Data> minus =
      std::dynamic_pointer_cast<Data>(model->createData());
  const Eigen::Index nc =
      static_cast<Eigen::Index>(model->get_constraints()->get_nc());
  MatrixXs df_dx(nc, Fx.cols());
  MatrixXs df_du(nc, Fu.cols());
  MatrixXs dP_dv(1, model->get_state()->get_nv());
  for (Eigen::Index i = 0; i < Fx.cols(); ++i) {
    VectorXs dx = VectorXs::Zero(model->get_state()->get_ndx());
    dx[i] = eps;
    VectorXs xp(model->get_state()->get_nx());
    VectorXs xm(model->get_state()->get_nx());
    model->get_state()->integrate(x, dx, xp);
    model->get_state()->integrate(x, -dx, xm);
    model->calc(plus, xp, u);
    model->calc(minus, xm, u);
    Fx.col(i).noalias() = (plus->vdot - minus->vdot) / (Scalar(2) * eps);
    df_dx.col(i).noalias() = (plus->pinocchio.lambda_c.head(nc) -
                              minus->pinocchio.lambda_c.head(nc)) /
                             (Scalar(2) * eps);
    if (i >= static_cast<Eigen::Index>(model->get_state()->get_nv())) {
      dP_dv.col(i - static_cast<Eigen::Index>(model->get_state()->get_nv())) =
          (plus->dissipative_P - minus->dissipative_P) / (Scalar(2) * eps);
    }
  }
  for (Eigen::Index i = 0; i < Fu.cols(); ++i) {
    VectorXs up = u;
    VectorXs um = u;
    up[i] += eps;
    um[i] -= eps;
    model->calc(plus, x, up);
    model->calc(minus, x, um);
    Fu.col(i).noalias() = (plus->vdot - minus->vdot) / (Scalar(2) * eps);
    df_du.col(i).noalias() = (plus->pinocchio.lambda_c.head(nc) -
                              minus->pinocchio.lambda_c.head(nc)) /
                             (Scalar(2) * eps);
  }
  BOOST_CHECK(data->Fx.isApprox(Fx, derivative_tolerance<Scalar>()));
  BOOST_CHECK(data->Fu.isApprox(Fu, derivative_tolerance<Scalar>()));
  BOOST_CHECK(data->df_dx.topRows(nc).isApprox(
      df_dx, Scalar(2) * derivative_tolerance<Scalar>()));
  BOOST_CHECK(data->df_du.topRows(nc).isApprox(
      df_du, Scalar(2) * derivative_tolerance<Scalar>()));
  BOOST_CHECK(data->dP_dv.isApprox(dP_dv, derivative_tolerance<Scalar>()));
}

template <typename Scalar>
void test_constrained_forward() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataConstrainedForwardTpl<Scalar> Data;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, actuation->get_nu());
  constraints->addConstraint("contact",
                             Fixture::contact(state, actuation->get_nu(), 3));
  constraints->addConstraint(
      "loop", Fixture::loop(state, actuation->get_nu(), 2), false);
  const std::shared_ptr<Model> model =
      std::make_shared<Model>(state, actuation, constraints, 0,
                              crocoddyl::DynamicsType::ContinuousControl);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model->createData());
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(model->checkData(data));
  BOOST_CHECK(data->shared == &data->multibody);
  BOOST_CHECK(data->multibody.pinocchio == &data->pinocchio);
  BOOST_CHECK_EQUAL(constraints->get_nc(), 3u);
  BOOST_CHECK_EQUAL(constraints->get_nc_total(), 5u);

  const VectorXs x0 = state->zero();
  const VectorXs dx =
      VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.2), Scalar(0.3));
  VectorXs x(state->get_nx());
  state->integrate(x0, dx, x);
  const VectorXs u =
      VectorXs::LinSpaced(model->get_nu(), Scalar(-0.2), Scalar(0.3));
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  BOOST_CHECK(data->vdot.allFinite());
  BOOST_CHECK(data->Fx.allFinite());
  BOOST_CHECK(data->Fu.allFinite());
  BOOST_CHECK_EQUAL(data->multibody.constraints->Jc.rows(), 5);
  check_forward_finite_differences<Scalar>(model, data, x, u);

  Data copied(*data);
  BOOST_CHECK(copied.shared == &copied.multibody);
  BOOST_CHECK(copied.multibody.pinocchio == &copied.pinocchio);
  BOOST_CHECK(copied.multibody.constraints == data->multibody.constraints);
  BOOST_CHECK(copied.Fx.isApprox(data->Fx));
  data->Fx.setZero();
  BOOST_CHECK(!copied.Fx.isZero());

  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  const bool was_malloc_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      model->calc(data, x, u);
      model->calcDiff(data, x, u);
    }
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
    throw;
  }
}

template <typename Scalar>
void test_forward_terminal_estimation_loop_and_quasistatic() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataConstrainedForwardTpl<Scalar> Data;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, actuation->get_nu());
  constraints->addConstraint("loop",
                             Fixture::loop(state, actuation->get_nu(), 2));
  Model control(state, actuation, constraints);
  const std::shared_ptr<Data> control_data =
      std::dynamic_pointer_cast<Data>(control.createData());
  const VectorXs x0 = state->zero();
  const VectorXs dx =
      VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.02), Scalar(0.03));
  VectorXs x(state->get_nx());
  state->integrate(x0, dx, x);
  const VectorXs u = VectorXs::Zero(control.get_nu());
  control.calc(control_data, x, u);
  control.calcDiff(control_data, x, u);
  const bool was_malloc_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      control.calc(control_data, x, u);
      control.calcDiff(control_data, x, u);
    }
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
    throw;
  }
  control_data->Fu.setConstant(Scalar(4));
  control.calc(control_data, x);
  control.calcDiff(control_data, x);
  BOOST_CHECK(control_data->Fu.isConstant(Scalar(4)));

  VectorXs ustatic = VectorXs::Zero(control.get_nu());
  control.quasiStatic(control_data, ustatic, x);
  BOOST_CHECK(ustatic.allFinite());

  Model estimation(state, actuation, constraints, 0,
                   crocoddyl::DynamicsType::ContinuousEstimation);
  const std::shared_ptr<Data> estimation_data =
      std::dynamic_pointer_cast<Data>(estimation.createData());
  BOOST_CHECK_EQUAL(estimation.get_nu(), 0u);
  const VectorXs tau =
      VectorXs::LinSpaced(actuation->get_nu(), Scalar(-0.1), Scalar(0.2));
  estimation.update_tau(tau);
  estimation.calc(estimation_data, x, VectorXs());
  estimation.calcDiff(estimation_data, x, VectorXs());
  BOOST_CHECK(estimation_data->vdot.allFinite());
  BOOST_CHECK(estimation_data->Fu.cols() == 0);
  BOOST_CHECK_THROW(estimation.update_tau(VectorXs::Zero(tau.size() + 1)),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_constrained_inverse() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedInverseTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataConstrainedInverseTpl<Scalar> Data;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename Fixture::VectorXs VectorXs;
  typedef typename Fixture::MatrixXs MatrixXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::size_t nc = 3;
  const std::size_t nu = state->get_nv() + nc;
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, nu);
  constraints->addConstraint("contact", Fixture::contact(state, nu, nc));
  constraints->addConstraint("loop", Fixture::loop(state, nu, 2), false);
  Model model(state, actuation, constraints);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model.createData());
  BOOST_REQUIRE(data != nullptr);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  const std::shared_ptr<Data> forwarded =
      std::dynamic_pointer_cast<Data>(model.createData(params->createData()));
  BOOST_REQUIRE(forwarded != nullptr);
  BOOST_CHECK(forwarded->params != nullptr);
  BOOST_CHECK_EQUAL(model.get_nu(), nu);
  BOOST_CHECK_EQUAL(model.get_nh(), nc);

  const VectorXs x = state->rand();
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(-0.2), Scalar(0.3));
  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  BOOST_CHECK(data->h.allFinite());
  BOOST_CHECK(data->Hx.allFinite());
  BOOST_CHECK(data->Hu.allFinite());
  BOOST_CHECK(data->Fu.isIdentity());

  const Scalar eps = finite_difference_step<Scalar>();
  MatrixXs Hx(data->Hx.rows(), data->Hx.cols());
  MatrixXs Hu(data->Hu.rows(), data->Hu.cols());
  const std::shared_ptr<Data> plus =
      std::dynamic_pointer_cast<Data>(model.createData());
  const std::shared_ptr<Data> minus =
      std::dynamic_pointer_cast<Data>(model.createData());
  for (Eigen::Index i = 0; i < Hx.cols(); ++i) {
    VectorXs dx = VectorXs::Zero(state->get_ndx());
    dx[i] = eps;
    VectorXs xp(state->get_nx());
    VectorXs xm(state->get_nx());
    state->integrate(x, dx, xp);
    state->integrate(x, -dx, xm);
    model.calc(plus, xp, u);
    model.calc(minus, xm, u);
    Hx.col(i).noalias() = (plus->h - minus->h) / (Scalar(2) * eps);
  }
  for (Eigen::Index i = 0; i < Hu.cols(); ++i) {
    VectorXs up = u;
    VectorXs um = u;
    up[i] += eps;
    um[i] -= eps;
    model.calc(plus, x, up);
    model.calc(minus, x, um);
    Hu.col(i).noalias() = (plus->h - minus->h) / (Scalar(2) * eps);
  }
  BOOST_CHECK((data->Hx - Hx).isZero(derivative_tolerance<Scalar>()));
  BOOST_CHECK((data->Hu - Hu).isZero(derivative_tolerance<Scalar>()));

  Data copied(*data);
  BOOST_CHECK(copied.shared == &copied.multibody);
  BOOST_CHECK(copied.multibody.pinocchio == &copied.pinocchio);
  BOOST_CHECK(copied.Hx.isApprox(data->Hx));

  data->Hu.setConstant(Scalar(5));
  model.calc(data, x);
  model.calcDiff(data, x);
  BOOST_CHECK(data->Hu.isConstant(Scalar(5)));

  Model estimation(state, actuation, constraints, 0,
                   crocoddyl::DynamicsType::ContinuousEstimation);
  const std::shared_ptr<Data> estimation_data =
      std::dynamic_pointer_cast<Data>(estimation.createData());
  estimation.update_tau(VectorXs::Constant(actuation->get_nu(), Scalar(0.1)));
  estimation.calc(estimation_data, x, u);
  estimation.calcDiff(estimation_data, x, u);
  BOOST_CHECK_EQUAL(estimation.get_nh(), state->get_nv() + nc);
  BOOST_CHECK(estimation_data->h.allFinite());

  VectorXs ustatic = VectorXs::Zero(model.get_nu());
  model.quasiStatic(data, ustatic, x);
  BOOST_CHECK(ustatic.allFinite());
}

template <typename Scalar>
void test_parameters_and_no_allocation() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataConstrainedForwardTpl<Scalar> Data;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> ActuationParams;
  typedef typename Fixture::VectorXs VectorXs;
  typedef typename Fixture::MatrixXs MatrixXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      Fixture::friction_actuation(state);
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, actuation->get_nu());
  Model model(state, actuation, constraints);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model.createData());
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  params->addParam("actuation", std::make_shared<ActuationParams>(actuation));
  const std::shared_ptr<Data> forwarded =
      std::dynamic_pointer_cast<Data>(model.createData(params->createData()));
  BOOST_REQUIRE(forwarded != nullptr);
  BOOST_CHECK(forwarded->params != nullptr);
  model.set_params(data, params);
  BOOST_CHECK_EQUAL(model.get_np(), params->get_np());
  BOOST_CHECK(data->params != nullptr);
  BOOST_CHECK(data->shared_params == data->params->params);
  VectorXs p = params->zero();
  p.array() += Scalar(0.05);
  model.update_p(data, p);
  BOOST_CHECK(data->shared_params->p.isApprox(p));

  const VectorXs x = state->rand();
  const VectorXs u = VectorXs::Constant(model.get_nu(), Scalar(0.3));
  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  BOOST_CHECK(data->Fp.allFinite());
  BOOST_CHECK(data->dP_dp.allFinite());

  const Scalar eps = finite_difference_step<Scalar>();
  MatrixXs Fp(data->Fp.rows(), data->Fp.cols());
  const std::shared_ptr<Data> plus =
      std::dynamic_pointer_cast<Data>(model.createData());
  const std::shared_ptr<Data> minus =
      std::dynamic_pointer_cast<Data>(model.createData());
  model.set_params(plus, params);
  model.set_params(minus, params);
  for (Eigen::Index i = 0; i < p.size(); ++i) {
    VectorXs pp = p;
    VectorXs pm = p;
    pp[i] += eps;
    pm[i] -= eps;
    model.update_p(plus, pp);
    model.calc(plus, x, u);
    model.update_p(minus, pm);
    model.calc(minus, x, u);
    Fp.col(i).noalias() = (plus->vdot - minus->vdot) / (Scalar(2) * eps);
  }
  model.update_p(data, p);
  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  BOOST_CHECK((data->Fp - Fp).isZero(derivative_tolerance<Scalar>()));

  const bool was_malloc_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      model.update_p(data, p);
      model.calc(data, x, u);
      model.calcDiff(data, x, u);
    }
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
    throw;
  }
}

template <typename Scalar>
void test_parameterized_control_and_estimation_modes() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Forward;
  typedef crocoddyl::DynamicsDataConstrainedForwardTpl<Scalar> ForwardData;
  typedef crocoddyl::DynamicsModelConstrainedInverseTpl<Scalar> Inverse;
  typedef crocoddyl::DynamicsDataConstrainedInverseTpl<Scalar> InverseData;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> ActuationParams;
  typedef typename Fixture::VectorXs VectorXs;
  typedef typename Fixture::MatrixXs MatrixXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      Fixture::friction_actuation(state);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  params->addParam("actuation", std::make_shared<ActuationParams>(actuation));
  VectorXs p = params->zero();
  p.array() += Scalar(0.07);
  VectorXs x = state->zero();
  VectorXs dx =
      VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.03), Scalar(0.04));
  state->integrate(x, dx, x);
  const Scalar eps = finite_difference_step<Scalar>();

  const crocoddyl::DynamicsType modes[] = {
      crocoddyl::DynamicsType::ContinuousControl,
      crocoddyl::DynamicsType::ContinuousEstimation};
  for (const crocoddyl::DynamicsType mode : modes) {
    const std::shared_ptr<Constraints> forward_constraints =
        std::make_shared<Constraints>(state, actuation->get_nu());
    const std::shared_ptr<Forward> forward = std::make_shared<Forward>(
        state, actuation, forward_constraints, 0, mode);
    const std::shared_ptr<ParameterDataManager> external = params->createData();
    const std::shared_ptr<ForwardData> data =
        std::dynamic_pointer_cast<ForwardData>(forward->createData(external));
    forward->set_params(data, params);
    BOOST_CHECK(data->params == external);
    forward->update_p(data, p);
    const VectorXs u =
        mode == crocoddyl::DynamicsType::ContinuousControl
            ? VectorXs::LinSpaced(forward->get_nu(), Scalar(-0.2), Scalar(0.3))
            : VectorXs();
    if (mode == crocoddyl::DynamicsType::ContinuousEstimation) {
      forward->update_tau(VectorXs::LinSpaced(actuation->get_nu(),
                                              Scalar(-0.15), Scalar(0.25)));
    }
    forward->calc(data, x, u);
    forward->calcDiff(data, x, u);
    BOOST_CHECK(!data->Fp.isZero());
    MatrixXs Fp(data->Fp.rows(), data->Fp.cols());
    MatrixXs dP_dp(data->dP_dp.rows(), data->dP_dp.cols());
    const std::shared_ptr<ForwardData> plus =
        std::dynamic_pointer_cast<ForwardData>(forward->createData());
    const std::shared_ptr<ForwardData> minus =
        std::dynamic_pointer_cast<ForwardData>(forward->createData());
    for (Eigen::Index i = 0; i < p.size(); ++i) {
      VectorXs pp = p;
      VectorXs pm = p;
      pp[i] += eps;
      pm[i] -= eps;
      forward->update_p(plus, pp);
      forward->calc(plus, x, u);
      forward->update_p(minus, pm);
      forward->calc(minus, x, u);
      Fp.col(i).noalias() = (plus->vdot - minus->vdot) / (Scalar(2) * eps);
      dP_dp.col(i).noalias() =
          (plus->dissipative_P - minus->dissipative_P) / (Scalar(2) * eps);
    }
    forward->update_p(data, p);
    forward->calc(data, x, u);
    forward->calcDiff(data, x, u);
    BOOST_CHECK_SMALL((data->Fp - Fp).cwiseAbs().maxCoeff(),
                      derivative_tolerance<Scalar>());
    BOOST_CHECK_SMALL((data->dP_dp - dP_dp).cwiseAbs().maxCoeff(),
                      derivative_tolerance<Scalar>());

#ifdef NDEBUG
    typedef typename std::conditional<std::is_same<Scalar, double>::value,
                                      float, double>::type NewScalar;
    typedef crocoddyl::ActuationModelMultibodyTpl<NewScalar> ActuationNew;
    typedef crocoddyl::ActuationMultibodyParamsTpl<NewScalar>
        ActuationParamsNew;
    const crocoddyl::DynamicsModelConstrainedForwardTpl<NewScalar> casted =
        forward->template cast<NewScalar>();
    BOOST_CHECK_EQUAL(casted.get_np(), forward->get_np());
    BOOST_CHECK(casted.get_params() != nullptr);
    const std::shared_ptr<ActuationNew> casted_actuation =
        std::dynamic_pointer_cast<ActuationNew>(casted.get_actuation());
    const std::shared_ptr<ActuationParamsNew> casted_actuation_params =
        std::dynamic_pointer_cast<ActuationParamsNew>(
            casted.get_params()
                ->get_dynamics_params()
                .at("actuation")
                ->get_param());
    BOOST_REQUIRE(casted_actuation != nullptr);
    BOOST_REQUIRE(casted_actuation_params != nullptr);
    BOOST_CHECK(casted_actuation_params->get_actuation() == casted_actuation);
#endif

    const std::shared_ptr<Constraints> inverse_constraints =
        std::make_shared<Constraints>(state, state->get_nv());
    const std::shared_ptr<Inverse> inverse = std::make_shared<Inverse>(
        state, actuation, inverse_constraints, 0, mode);
    const std::shared_ptr<InverseData> inverse_data =
        std::dynamic_pointer_cast<InverseData>(inverse->createData());
    inverse->set_params(inverse_data, params);
    inverse->update_p(inverse_data, p);
    if (mode == crocoddyl::DynamicsType::ContinuousEstimation) {
      inverse->update_tau(
          VectorXs::LinSpaced(actuation->get_nu(), Scalar(-0.1), Scalar(0.2)));
    }
    const VectorXs inverse_u =
        VectorXs::LinSpaced(inverse->get_nu(), Scalar(-0.05), Scalar(0.08));
    inverse->calc(inverse_data, x, inverse_u);
    inverse->calcDiff(inverse_data, x, inverse_u);
    if (mode == crocoddyl::DynamicsType::ContinuousEstimation) {
      BOOST_CHECK(!inverse_data->Hp.isZero());
    }
    MatrixXs Hp(inverse_data->Hp.rows(), inverse_data->Hp.cols());
    const std::shared_ptr<InverseData> inverse_plus =
        std::dynamic_pointer_cast<InverseData>(inverse->createData());
    const std::shared_ptr<InverseData> inverse_minus =
        std::dynamic_pointer_cast<InverseData>(inverse->createData());
    for (Eigen::Index i = 0; i < p.size(); ++i) {
      VectorXs pp = p;
      VectorXs pm = p;
      pp[i] += eps;
      pm[i] -= eps;
      inverse->update_p(inverse_plus, pp);
      inverse->calc(inverse_plus, x, inverse_u);
      inverse->update_p(inverse_minus, pm);
      inverse->calc(inverse_minus, x, inverse_u);
      Hp.col(i).noalias() =
          (inverse_plus->h - inverse_minus->h) / (Scalar(2) * eps);
    }
    inverse->update_p(inverse_data, p);
    inverse->calc(inverse_data, x, inverse_u);
    inverse->calcDiff(inverse_data, x, inverse_u);
    if (Hp.size() != 0) {
      BOOST_CHECK_SMALL((inverse_data->Hp - Hp).cwiseAbs().maxCoeff(),
                        derivative_tolerance<Scalar>());
    }

    const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
    try {
      Eigen::internal::set_is_malloc_allowed(false);
      for (std::size_t i = 0; i < 100; ++i) {
        inverse->update_p(inverse_data, p);
        inverse->calc(inverse_data, x, inverse_u);
        inverse->calcDiff(inverse_data, x, inverse_u);
      }
      Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    } catch (...) {
      Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
      throw;
    }

#ifdef NDEBUG
    const crocoddyl::DynamicsModelConstrainedInverseTpl<NewScalar>
        casted_inverse = inverse->template cast<NewScalar>();
    BOOST_CHECK_EQUAL(casted_inverse.get_np(), inverse->get_np());
    BOOST_CHECK(casted_inverse.get_params() != nullptr);
    const std::shared_ptr<ActuationNew> casted_inverse_actuation =
        std::dynamic_pointer_cast<ActuationNew>(casted_inverse.get_actuation());
    const std::shared_ptr<ActuationParamsNew> casted_inverse_params =
        std::dynamic_pointer_cast<ActuationParamsNew>(
            casted_inverse.get_params()
                ->get_dynamics_params()
                .at("actuation")
                ->get_param());
    BOOST_REQUIRE(casted_inverse_actuation != nullptr);
    BOOST_REQUIRE(casted_inverse_params != nullptr);
    BOOST_CHECK(casted_inverse_params->get_actuation() ==
                casted_inverse_actuation);
#endif
  }
}

template <typename Scalar>
void test_action_only_parameters_clear_forward_fp() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataConstrainedForwardTpl<Scalar> Data;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, actuation->get_nu());
  Model model(state, actuation, constraints);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model.createData());
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  params->addParam(
      "time", std::make_shared<TimeParams>(
                  state, std::make_shared<IntegratorTime>(Scalar(1e-3), true)));
  model.set_params(data, params);
  const VectorXs p = params->zero();
  model.update_p(data, p);
  const VectorXs x = state->zero();
  const VectorXs u = VectorXs::Zero(model.get_nu());
  model.calc(data, x, u);
  data->Fp.setConstant(Scalar(7));
  model.calcDiff(data, x, u);
  BOOST_CHECK(data->Fp.isZero());
}

template <typename Scalar>
void test_constraint_layout_modes_and_exception_safety() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Forward;
  typedef crocoddyl::DynamicsDataConstrainedForwardTpl<Scalar> ForwardData;
  typedef crocoddyl::DynamicsModelConstrainedInverseTpl<Scalar> Inverse;
  typedef crocoddyl::DynamicsDataConstrainedInverseTpl<Scalar> InverseData;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const VectorXs x = state->rand();
  const bool modes[] = {false, true};
  for (const bool compute_all : modes) {
    const std::shared_ptr<Constraints> forward_constraints =
        std::make_shared<Constraints>(state, actuation->get_nu());
    forward_constraints->addConstraint(
        "a_inactive", Fixture::contact(state, actuation->get_nu(), 2), false);
    forward_constraints->addConstraint(
        "z_active", Fixture::contact(state, actuation->get_nu(), 3));
    forward_constraints->setComputeAllConstraints(compute_all);
    Forward forward(state, actuation, forward_constraints);
    const std::shared_ptr<ForwardData> forward_data =
        std::dynamic_pointer_cast<ForwardData>(forward.createData());
    const VectorXs forward_u = VectorXs::Zero(forward.get_nu());
    forward.calc(forward_data, x, forward_u);
    forward.calcDiff(forward_data, x, forward_u);
    BOOST_CHECK_EQUAL(forward_data->multibody.constraints->Jc.rows(), 5);
    if (compute_all) {
      BOOST_CHECK(forward_data->multibody.constraints->Jc.topRows(2).isZero());
      BOOST_CHECK(
          !forward_data->multibody.constraints->Jc.bottomRows(3).isZero());
    } else {
      BOOST_CHECK(!forward_data->multibody.constraints->Jc.topRows(3).isZero());
      BOOST_CHECK(
          forward_data->multibody.constraints->Jc.bottomRows(2).isZero());
    }
    BOOST_CHECK_EQUAL(forward_constraints->getComputeAllConstraints(),
                      compute_all);

    const std::shared_ptr<ForwardData> invalid_forward =
        std::dynamic_pointer_cast<ForwardData>(forward.createData());
    invalid_forward->multibody.constraints->constraints.erase("a_inactive");
    BOOST_CHECK_THROW(forward.calc(invalid_forward, x, forward_u),
                      crocoddyl::Exception);
    BOOST_CHECK_EQUAL(forward_constraints->getComputeAllConstraints(),
                      compute_all);

    const std::size_t inverse_nu = state->get_nv() + 3;
    const std::shared_ptr<Constraints> inverse_constraints =
        std::make_shared<Constraints>(state, inverse_nu);
    inverse_constraints->addConstraint(
        "a_inactive", Fixture::contact(state, inverse_nu, 2), false);
    inverse_constraints->addConstraint("z_active",
                                       Fixture::contact(state, inverse_nu, 3));
    inverse_constraints->setComputeAllConstraints(compute_all);
    Inverse inverse(state, actuation, inverse_constraints);
    const std::shared_ptr<InverseData> inverse_data =
        std::dynamic_pointer_cast<InverseData>(inverse.createData());
    const VectorXs inverse_u = VectorXs::Zero(inverse.get_nu());
    inverse.calc(inverse_data, x, inverse_u);
    inverse.calcDiff(inverse_data, x, inverse_u);
    BOOST_CHECK_EQUAL(inverse_data->multibody.constraints->Jc.rows(), 5);
    if (compute_all) {
      BOOST_CHECK(inverse_data->multibody.constraints->Jc.topRows(2).isZero());
      BOOST_CHECK(
          !inverse_data->multibody.constraints->Jc.bottomRows(3).isZero());
    } else {
      BOOST_CHECK(!inverse_data->multibody.constraints->Jc.topRows(3).isZero());
      BOOST_CHECK(
          inverse_data->multibody.constraints->Jc.bottomRows(2).isZero());
    }
    const std::shared_ptr<InverseData> invalid_inverse =
        std::dynamic_pointer_cast<InverseData>(inverse.createData());
    invalid_inverse->multibody.constraints->constraints.erase("a_inactive");
    BOOST_CHECK_THROW(inverse.calc(invalid_inverse, x, inverse_u),
                      crocoddyl::Exception);
    BOOST_CHECK_EQUAL(inverse_constraints->getComputeAllConstraints(),
                      compute_all);
  }
}

template <typename Scalar>
void test_errors_and_casts() {
  typedef DynamicsFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Actuation Actuation;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Forward;
  typedef crocoddyl::DynamicsModelConstrainedInverseTpl<Scalar> Inverse;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::shared_ptr<Constraints> forward_constraints =
      std::make_shared<Constraints>(state, actuation->get_nu());
  BOOST_CHECK_THROW(Forward(nullptr, actuation, forward_constraints),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Forward(state, nullptr, forward_constraints),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Forward(state, actuation, nullptr), crocoddyl::Exception);
  const std::shared_ptr<Constraints> wrong_forward =
      std::make_shared<Constraints>(state, actuation->get_nu() + 1);
  BOOST_CHECK_THROW(Forward(state, actuation, wrong_forward),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Forward(state, actuation, forward_constraints, 0,
                            crocoddyl::DynamicsType::DiscreteTime),
                    crocoddyl::Exception);

  const std::shared_ptr<Constraints> inverse_constraints =
      std::make_shared<Constraints>(state, state->get_nv());
  Inverse inverse(state, actuation, inverse_constraints);
  const std::shared_ptr<typename Inverse::DynamicsDataAbstract> data =
      inverse.createData();
  const VectorXs x = state->rand();
  BOOST_CHECK_THROW(inverse.calc(data, VectorXs::Zero(x.size() + 1),
                                 VectorXs::Zero(inverse.get_nu())),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(inverse.calc(data, x, VectorXs::Zero(inverse.get_nu() + 1)),
                    crocoddyl::Exception);

#ifdef NDEBUG
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type NewScalar;
  const Forward forward(state, actuation, forward_constraints);
  const crocoddyl::DynamicsModelConstrainedForwardTpl<NewScalar> casted =
      forward.template cast<NewScalar>();
  BOOST_CHECK_EQUAL(casted.get_nu(), forward.get_nu());
  BOOST_CHECK_EQUAL(casted.get_constraints()->get_nc_total(),
                    forward.get_constraints()->get_nc_total());
  const crocoddyl::DynamicsModelConstrainedInverseTpl<NewScalar>
      casted_inverse = inverse.template cast<NewScalar>();
  BOOST_CHECK_EQUAL(casted_inverse.get_nu(), inverse.get_nu());
  BOOST_CHECK_EQUAL(casted_inverse.get_nh(), inverse.get_nh());
#endif
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_multibody_dynamics");
  ts->add(BOOST_TEST_CASE(&test_constrained_forward<double>));
  ts->add(BOOST_TEST_CASE(&test_constrained_forward<float>));
  ts->add(BOOST_TEST_CASE(
      &test_forward_terminal_estimation_loop_and_quasistatic<double>));
  ts->add(BOOST_TEST_CASE(
      &test_forward_terminal_estimation_loop_and_quasistatic<float>));
  ts->add(BOOST_TEST_CASE(&test_constrained_inverse<double>));
  ts->add(BOOST_TEST_CASE(&test_constrained_inverse<float>));
  ts->add(BOOST_TEST_CASE(&test_parameters_and_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_parameters_and_no_allocation<float>));
  ts->add(BOOST_TEST_CASE(
      &test_parameterized_control_and_estimation_modes<double>));
  ts->add(
      BOOST_TEST_CASE(&test_parameterized_control_and_estimation_modes<float>));
  ts->add(
      BOOST_TEST_CASE(&test_action_only_parameters_clear_forward_fp<double>));
  ts->add(
      BOOST_TEST_CASE(&test_action_only_parameters_clear_forward_fp<float>));
  ts->add(BOOST_TEST_CASE(
      &test_constraint_layout_modes_and_exception_safety<double>));
  ts->add(BOOST_TEST_CASE(
      &test_constraint_layout_modes_and_exception_safety<float>));
  ts->add(BOOST_TEST_CASE(&test_errors_and_casts<double>));
  ts->add(BOOST_TEST_CASE(&test_errors_and_casts<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
