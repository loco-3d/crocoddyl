///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <type_traits>

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/actuations/joint-friction.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
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
struct ImpulseFixture {
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

  static typename Contact::MaskArray mask(const std::size_t nc) {
    typename Contact::MaskArray result = {
        {false, false, false, false, false, false}};
    for (std::size_t i = 0; i < nc; ++i) {
      result[i] = true;
    }
    return result;
  }

  static std::shared_ptr<Contact> contact(const std::shared_ptr<State>& state,
                                          const std::size_t nc) {
    typedef pinocchio::SE3Tpl<Scalar> SE3;
    const typename Contact::Vector2s gains = Contact::Vector2s::Zero();
    const pinocchio::FrameIndex frame_id = static_cast<pinocchio::FrameIndex>(
        state->get_pinocchio()->frames.size() - 1);
    return std::make_shared<Contact>(state, frame_id, SE3::Identity(),
                                     pinocchio::LOCAL_WORLD_ALIGNED, 0, gains,
                                     mask(nc));
  }

  static std::shared_ptr<Loop> loop(const std::shared_ptr<State>& state,
                                    const std::size_t nc) {
    typedef pinocchio::SE3Tpl<Scalar> SE3;
    typename Loop::MaskArray loop_mask = {
        {false, false, false, false, false, false}};
    for (std::size_t i = 0; i < nc; ++i) {
      loop_mask[i] = true;
    }
    const typename Loop::Vector2s gains = Loop::Vector2s::Zero();
    return std::make_shared<Loop>(state, 1, SE3::Identity(), 2, SE3::Identity(),
                                  pinocchio::LOCAL, 0, gains, loop_mask);
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
Scalar fd_step() {
  return std::is_same<Scalar, float>::value ? Scalar(5e-3) : Scalar(1e-6);
}

template <typename Scalar>
Scalar fd_tolerance() {
  return std::is_same<Scalar, float>::value ? Scalar(2e-2) : Scalar(1e-3);
}

template <typename Scalar>
void test_running_terminal_and_derivatives() {
  typedef ImpulseFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataImpulseForwardTpl<Scalar> Data;
  typedef typename Fixture::VectorXs VectorXs;
  typedef typename Fixture::MatrixXs MatrixXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, 0);
  constraints->addConstraint("contact", Fixture::contact(state, 3));
  constraints->addConstraint("loop", Fixture::loop(state, 2), false);
  Model model(state, constraints);
  Model restitution_model(state, constraints, 0, Scalar(0.15));
  Model damped_model(state, constraints, 0, Scalar(0), Scalar(0.2));
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model.createData());
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(model.checkData(data));
  BOOST_CHECK(model.get_dyn_type() == crocoddyl::DynamicsType::DiscreteTime);
  BOOST_CHECK_EQUAL(model.get_nu(), 0u);
  BOOST_CHECK_EQUAL(model.get_r_coeff(), Scalar(0));
  BOOST_CHECK_EQUAL(restitution_model.get_r_coeff(), Scalar(0.15));
  BOOST_CHECK_EQUAL(damped_model.get_damping_factor(), Scalar(0.2));
  BOOST_CHECK(data->shared == &data->multibody);
  BOOST_CHECK(data->multibody.pinocchio == &data->pinocchio);
  BOOST_REQUIRE(data->joint != nullptr);
  BOOST_CHECK(data->multibody.joint == data->joint);
  BOOST_CHECK_EQUAL(data->joint->tau.size(), 0);
  BOOST_CHECK_EQUAL(data->joint->dtau_dx.rows(), 0);
  BOOST_CHECK_EQUAL(data->joint->dtau_dx.cols(), state->get_ndx());
  BOOST_CHECK_EQUAL(data->joint->dtau_du.rows(), 0);
  BOOST_CHECK_EQUAL(data->joint->dtau_du.cols(), 0);
  BOOST_CHECK_EQUAL(data->joint->a.size(), state->get_nv());
  BOOST_CHECK_EQUAL(data->joint->da_dx.rows(), state->get_nv());
  BOOST_CHECK_EQUAL(data->joint->da_dx.cols(), state->get_ndx());
  BOOST_CHECK_EQUAL(data->joint->da_du.rows(), state->get_nv());
  BOOST_CHECK_EQUAL(data->joint->da_du.cols(), 0);

  const VectorXs x0 = state->zero();
  const VectorXs dx =
      VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.2), Scalar(0.3));
  VectorXs x(state->get_nx());
  state->integrate(x0, dx, x);
  const VectorXs u;
  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  BOOST_CHECK(data->vdot.allFinite());
  BOOST_CHECK(data->Fx.allFinite());
  BOOST_CHECK(data->Fu.cols() == 0);

  const Scalar eps = fd_step<Scalar>();
  MatrixXs Fx(data->Fx.rows(), data->Fx.cols());
  const Eigen::Index nc =
      static_cast<Eigen::Index>(model.get_constraints()->get_nc());
  MatrixXs df_dx(nc, data->Fx.cols());
  MatrixXs dP_dv(1, state->get_nv());
  const std::shared_ptr<Data> plus =
      std::dynamic_pointer_cast<Data>(model.createData());
  const std::shared_ptr<Data> minus =
      std::dynamic_pointer_cast<Data>(model.createData());
  VectorXs dy(state->get_ndx());
  for (Eigen::Index i = 0; i < Fx.cols(); ++i) {
    VectorXs dx = VectorXs::Zero(state->get_ndx());
    dx[i] = eps;
    VectorXs xp(state->get_nx());
    VectorXs xm(state->get_nx());
    state->integrate(x, dx, xp);
    state->integrate(x, -dx, xm);
    model.calc(plus, xp, u);
    model.calc(minus, xm, u);
    state->diff(minus->vdot, plus->vdot, dy);
    Fx.col(i) = dy / (Scalar(2) * eps);
    df_dx.col(i).noalias() = (plus->pinocchio.impulse_c.head(nc) -
                              minus->pinocchio.impulse_c.head(nc)) /
                             (Scalar(2) * eps);
    if (i >= static_cast<Eigen::Index>(state->get_nv())) {
      dP_dv.col(i - static_cast<Eigen::Index>(state->get_nv())) =
          (plus->dissipative_P - minus->dissipative_P) / (Scalar(2) * eps);
    }
  }
  BOOST_CHECK_SMALL((data->Fx - Fx).cwiseAbs().maxCoeff(),
                    fd_tolerance<Scalar>());
  BOOST_CHECK_SMALL((data->df_dx.topRows(nc) - df_dx).cwiseAbs().maxCoeff(),
                    Scalar(2) * fd_tolerance<Scalar>());
  BOOST_CHECK_SMALL((data->dP_dv - dP_dv).cwiseAbs().maxCoeff(),
                    fd_tolerance<Scalar>());

  Data copied(*data);
  BOOST_CHECK(copied.shared == &copied.multibody);
  BOOST_CHECK(copied.multibody.pinocchio == &copied.pinocchio);
  BOOST_CHECK(copied.Fx.isApprox(data->Fx));
  data->Fx.setZero();
  BOOST_CHECK(!copied.Fx.isZero());

  model.calc(data, x);
  model.calcDiff(data, x);
  BOOST_CHECK(data->vdot.isZero());
  BOOST_CHECK(data->Fx.isZero());
  BOOST_CHECK(data->Fp.isZero());
}

template <typename Scalar>
void test_zero_full_reduced_and_parameters() {
  typedef ImpulseFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataImpulseForwardTpl<Scalar> Data;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> ActuationParams;
  typedef typename Fixture::VectorXs VectorXs;
  typedef typename Fixture::MatrixXs MatrixXs;

  const std::shared_ptr<State> state = Fixture::state();
  const VectorXs x = state->rand();
  const VectorXs u;

  const std::shared_ptr<Constraints> empty =
      std::make_shared<Constraints>(state, 0);
  Model identity(state, empty);
  const std::shared_ptr<Data> identity_data =
      std::dynamic_pointer_cast<Data>(identity.createData());
  identity.calc(identity_data, x, u);
  BOOST_CHECK(identity_data->vdot.isApprox(x, fd_tolerance<Scalar>()));
  VectorXs ustatic;
  identity.quasiStatic(identity_data, ustatic, x);
  BOOST_CHECK_EQUAL(ustatic.size(), 0);

  const std::shared_ptr<Constraints> full =
      std::make_shared<Constraints>(state, 0);
  full->addConstraint("contact", Fixture::contact(state, 6));
  full->addConstraint("loop", Fixture::loop(state, 2), false);
  Model full_model(state, full);
  const std::shared_ptr<Data> full_data =
      std::dynamic_pointer_cast<Data>(full_model.createData());
  full_model.calc(full_data, x, u);
  full_model.calcDiff(full_data, x, u);
  BOOST_CHECK_EQUAL(full->get_nc(), 6u);
  BOOST_CHECK_EQUAL(full->get_nc_total(), 8u);
  BOOST_CHECK(full_data->vdot.allFinite());

  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  const std::shared_ptr<typename Fixture::Actuation> actuation =
      Fixture::friction_actuation(state);
  params->addParam("actuation", std::make_shared<ActuationParams>(actuation));
  const std::shared_ptr<Data> forwarded = std::dynamic_pointer_cast<Data>(
      full_model.createData(params->createData()));
  BOOST_REQUIRE(forwarded != nullptr);
  BOOST_CHECK(forwarded->params != nullptr);
  full_model.set_params(full_data, params);
  VectorXs p = params->zero();
  p.array() += Scalar(0.06);
  full_model.update_p(full_data, p);
  full_model.calc(full_data, x, u);
  full_model.calcDiff(full_data, x, u);
  BOOST_CHECK_EQUAL(full_model.get_np(), params->get_np());
  BOOST_CHECK(full_data->shared_params->p.isApprox(p));
  BOOST_CHECK(full_data->Fp.isZero(fd_tolerance<Scalar>()));
  BOOST_CHECK(full_data->dP_dp.isZero(fd_tolerance<Scalar>()));

  const Scalar eps = fd_step<Scalar>();
  MatrixXs Fp(full_data->Fp.rows(), full_data->Fp.cols());
  const std::shared_ptr<Data> plus =
      std::dynamic_pointer_cast<Data>(full_model.createData());
  const std::shared_ptr<Data> minus =
      std::dynamic_pointer_cast<Data>(full_model.createData());
  for (Eigen::Index i = 0; i < p.size(); ++i) {
    VectorXs pp = p;
    VectorXs pm = p;
    pp[i] += eps;
    pm[i] -= eps;
    full_model.update_p(plus, pp);
    full_model.calc(plus, x, u);
    full_model.update_p(minus, pm);
    full_model.calc(minus, x, u);
    Fp.col(i).noalias() = (plus->vdot - minus->vdot) / (Scalar(2) * eps);
  }
  full_model.update_p(full_data, p);
  full_model.calc(full_data, x, u);
  full_model.calcDiff(full_data, x, u);
  BOOST_CHECK_SMALL((full_data->Fp - Fp).cwiseAbs().maxCoeff(),
                    fd_tolerance<Scalar>());

#ifdef NDEBUG
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type NewScalar;
  const crocoddyl::DynamicsModelImpulseForwardTpl<NewScalar> casted =
      full_model.template cast<NewScalar>();
  BOOST_CHECK_EQUAL(casted.get_np(), full_model.get_np());
  BOOST_CHECK(casted.get_params() != nullptr);
  BOOST_CHECK_EQUAL(casted.get_damping_factor(),
                    static_cast<NewScalar>(full_model.get_damping_factor()));
#endif

  const bool was_malloc_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      full_model.update_p(full_data, p);
      full_model.calc(full_data, x, u);
      full_model.calcDiff(full_data, x, u);
    }
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
    throw;
  }
}

template <typename Scalar>
void test_constraint_layout_modes_and_exception_safety() {
  typedef ImpulseFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataImpulseForwardTpl<Scalar> Data;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const VectorXs x = state->rand();
  const VectorXs u;
  const bool modes[] = {false, true};
  for (const bool compute_all : modes) {
    const std::shared_ptr<Constraints> constraints =
        std::make_shared<Constraints>(state, 0);
    constraints->addConstraint("a_inactive", Fixture::contact(state, 2), false);
    constraints->addConstraint("z_active", Fixture::contact(state, 3));
    constraints->setComputeAllConstraints(compute_all);
    Model model(state, constraints);
    const std::shared_ptr<Data> data =
        std::dynamic_pointer_cast<Data>(model.createData());
    model.calc(data, x, u);
    model.calcDiff(data, x, u);
    BOOST_CHECK_EQUAL(data->multibody.constraints->Jc.rows(), 5);
    if (compute_all) {
      BOOST_CHECK(data->multibody.constraints->Jc.topRows(2).isZero());
      BOOST_CHECK(!data->multibody.constraints->Jc.bottomRows(3).isZero());
    } else {
      BOOST_CHECK(!data->multibody.constraints->Jc.topRows(3).isZero());
      BOOST_CHECK(data->multibody.constraints->Jc.bottomRows(2).isZero());
    }
    const std::shared_ptr<Data> invalid =
        std::dynamic_pointer_cast<Data>(model.createData());
    invalid->multibody.constraints->constraints.erase("a_inactive");
    BOOST_CHECK_THROW(model.calc(invalid, x, u), crocoddyl::Exception);
    BOOST_CHECK_EQUAL(constraints->getComputeAllConstraints(), compute_all);
  }
}

template <typename Scalar>
void test_errors_and_casts() {
  typedef ImpulseFixture<Scalar> Fixture;
  typedef typename Fixture::State State;
  typedef typename Fixture::Constraints Constraints;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> Model;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<State> state = Fixture::state();
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, 0);
  BOOST_CHECK_THROW(Model(nullptr, constraints), crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(state, nullptr), crocoddyl::Exception);
  const std::shared_ptr<Constraints> wrong =
      std::make_shared<Constraints>(state, 1);
  BOOST_CHECK_THROW(Model(state, wrong), crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(state, constraints, 0, Scalar(0), Scalar(-1)),
                    crocoddyl::Exception);
  Model model(state, constraints);
  const std::shared_ptr<typename Model::DynamicsDataAbstract> data =
      model.createData();
  const VectorXs x = state->rand();
  BOOST_CHECK_THROW(model.calc(data, VectorXs::Zero(x.size() + 1), VectorXs()),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model.calc(data, x, VectorXs::Zero(1)),
                    crocoddyl::Exception);

#ifdef NDEBUG
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type NewScalar;
  const crocoddyl::DynamicsModelImpulseForwardTpl<NewScalar> casted =
      model.template cast<NewScalar>();
  BOOST_CHECK_EQUAL(casted.get_nu(), 0u);
  BOOST_CHECK_EQUAL(casted.get_constraints()->get_nc_total(), 0u);
#endif
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_impulse_dynamics");
  ts->add(BOOST_TEST_CASE(&test_running_terminal_and_derivatives<double>));
  ts->add(BOOST_TEST_CASE(&test_running_terminal_and_derivatives<float>));
  ts->add(BOOST_TEST_CASE(&test_zero_full_reduced_and_parameters<double>));
  ts->add(BOOST_TEST_CASE(&test_zero_full_reduced_and_parameters<float>));
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
