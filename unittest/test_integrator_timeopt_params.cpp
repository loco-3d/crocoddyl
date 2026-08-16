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
#include <pinocchio/multibody/sample-models.hpp>
#include <type_traits>

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename Scalar>
std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > create_state() {
  typedef pinocchio::Model PinocchioModel;
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  const std::shared_ptr<PinocchioModel> model =
      std::make_shared<PinocchioModel>();
  pinocchio::buildModels::humanoidRandom(*model, false);
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::make_shared<crocoddyl::StateMultibody>(model);
  return std::make_shared<State>(state->template cast<Scalar>());
}

template <typename _Scalar>
class TimeDynamicsProbeTpl
    : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase, TimeDynamicsProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  explicit TimeDynamicsProbeTpl(std::shared_ptr<StateAbstract> state)
      : Base(state, crocoddyl::DynamicsType::ContinuousControl, 0,
             state->get_nv()) {}

  using Base::calc;
  using Base::calcDiff;

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    data->vdot.noalias() = x.tail(this->state_->get_nv()) + u;
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setZero();
    data->Fx.rightCols(this->state_->get_nv()).setIdentity();
    data->Fu.setIdentity();
  }

  template <typename NewScalar>
  TimeDynamicsProbeTpl<NewScalar> cast() const {
    return TimeDynamicsProbeTpl<NewScalar>(
        this->state_->template cast<NewScalar>());
  }
};

template <typename Scalar>
struct TimeFixtureTpl {
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>
      DifferentialAction;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Action;
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef TimeDynamicsProbeTpl<Scalar> Dynamics;

  TimeFixtureTpl()
      : state(create_state<Scalar>()),
        dynamics(std::make_shared<Dynamics>(state)),
        actuation(std::make_shared<Actuation>(state)),
        costs(std::make_shared<CostModelSum>(state, actuation->get_nu())),
        differential(
            std::make_shared<DifferentialAction>(state, actuation, costs)),
        action(std::make_shared<Action>(differential, Scalar(0.02))),
        time(action->get_integrator_time()) {
    time->set_timeopt(true);
  }

  std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state;
  std::shared_ptr<Dynamics> dynamics;
  std::shared_ptr<Actuation> actuation;
  std::shared_ptr<CostModelSum> costs;
  std::shared_ptr<DifferentialAction> differential;
  std::shared_ptr<Action> action;
  std::shared_ptr<IntegratorTime> time;
};

template <typename Scalar>
void test_construction_data_update_bounds_copy_and_failures() {
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> Model;
  typedef crocoddyl::IntegratorTimeoptParamsDataTpl<Scalar> Data;
  typedef crocoddyl::ActionModelParamsDataAbstractTpl<Scalar> ActionParamsData;
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  typedef typename Model::VectorXs VectorXs;

  TimeFixtureTpl<Scalar> fixture;
  Model model(fixture.state, fixture.time);
  BOOST_CHECK_EQUAL(model.get_np(), 1u);
  BOOST_CHECK(model.get_integrator_time() == fixture.time);
  BOOST_CHECK((std::is_base_of<ActionParamsData, Data>::value));

  const std::shared_ptr<ParamsData> data_base = model.createData();
  const std::shared_ptr<Data> data = std::dynamic_pointer_cast<Data>(data_base);
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(std::dynamic_pointer_cast<ActionParamsData>(data_base) !=
              nullptr);
  BOOST_CHECK(model.checkData(data));
  BOOST_CHECK_EQUAL(data->np, 1u);
  BOOST_CHECK_EQUAL(data->np_action, 1u);
  BOOST_CHECK_EQUAL(data->np_dynamics, 0u);
  BOOST_CHECK(data->active);
  const std::shared_ptr<Data> wrong_dimensions =
      std::dynamic_pointer_cast<Data>(model.createData());
  wrong_dimensions->resize(2, 0);
  BOOST_CHECK(!model.checkData(wrong_dimensions));
  Data copied_data(*data);
  copied_data.p.setOnes();
  BOOST_CHECK(data->p.isZero());

  const Scalar dt = Scalar(0.03);
  VectorXs p(1);
  using std::log;
  p[0] = log(dt);
  model.update(data, p);
  BOOST_CHECK(data->p.isApprox(p));
  BOOST_CHECK_SMALL(data->dt - dt,
                    Scalar(20) * Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(data->dt_dp - dt,
                    Scalar(20) * Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(fixture.time->get_time_step() - dt,
                    Scalar(20) * Eigen::NumTraits<Scalar>::epsilon());

  const VectorXs lb = VectorXs::Constant(1, Scalar(-8));
  const VectorXs ub = VectorXs::Constant(1, Scalar(-2));
  model.set_lb(lb);
  model.set_ub(ub);
  BOOST_CHECK(model.get_lb().isApprox(lb));
  BOOST_CHECK(model.get_ub().isApprox(ub));
  BOOST_CHECK_THROW(model.set_lb(VectorXs::Zero(2)), crocoddyl::Exception);
  BOOST_CHECK_THROW(model.set_ub(VectorXs::Zero(2)), crocoddyl::Exception);

  const VectorXs random = model.rand();
  using std::exp;
  BOOST_CHECK_GE(exp(random[0]), Scalar(1e-4));
  BOOST_CHECK_LE(exp(random[0]), Scalar(1e-2));

  data->active = false;
  data->dt = Scalar(1);
  data->dt_dp = Scalar(2);
  data->resize(1, 0);
  BOOST_CHECK(!data->active);
  BOOST_CHECK_SMALL(data->dt, Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(data->dt_dp, Eigen::NumTraits<Scalar>::epsilon());
  data->dt = Scalar(1);
  data->dt_dp = Scalar(2);
  data->setZero();
  BOOST_CHECK_SMALL(data->dt, Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK_SMALL(data->dt_dp, Eigen::NumTraits<Scalar>::epsilon());

  Model copied(model);
  const VectorXs copied_p = VectorXs::Constant(1, log(Scalar(0.04)));
  copied.update(copied.createData(), copied_p);
  BOOST_CHECK_SMALL(fixture.time->get_time_step() - Scalar(0.04),
                    Scalar(30) * Eigen::NumTraits<Scalar>::epsilon());

  BOOST_CHECK_THROW(model.update(data, VectorXs::Zero(2)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model.update(std::make_shared<ParamsData>(1, 0), p),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      Model(std::shared_ptr<typename Model::StateAbstract>(), fixture.time),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(fixture.state, std::shared_ptr<IntegratorTime>()),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(fixture.state, std::make_shared<IntegratorTime>(
                                             Scalar(0.01), false)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      Model(std::make_shared<crocoddyl::StateVectorTpl<Scalar> >(3),
            fixture.time),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(std::make_shared<Data>(static_cast<Model*>(nullptr)),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_shared_and_copied_time_running_terminal_and_sensitivity() {
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> Model;
  typedef crocoddyl::IntegratorTimeoptParamsDataTpl<Scalar> Data;
  typedef typename TimeFixtureTpl<Scalar>::Action Action;
  typedef crocoddyl::IntegratedActionDataEulerTpl<Scalar> ActionData;
  typedef typename Model::VectorXs VectorXs;

  TimeFixtureTpl<Scalar> fixture;
  const std::shared_ptr<Action> action2 =
      std::make_shared<Action>(*fixture.action);
  const std::shared_ptr<Action> copied_action =
      std::make_shared<Action>(fixture.action->template cast<Scalar>());
  Model model(fixture.state, fixture.time);
  BOOST_CHECK(model.get_integrator_time() == fixture.time);
  BOOST_CHECK(action2->get_integrator_time() == fixture.time);
  BOOST_CHECK(copied_action->get_integrator_time() != fixture.time);

  const std::shared_ptr<Data> params =
      std::dynamic_pointer_cast<Data>(model.createData());
  BOOST_REQUIRE(params != nullptr);
  const std::shared_ptr<ActionData> data1 =
      std::dynamic_pointer_cast<ActionData>(fixture.action->createData());
  const std::shared_ptr<ActionData> data2 =
      std::dynamic_pointer_cast<ActionData>(action2->createData());
  const std::shared_ptr<ActionData> copied_data =
      std::dynamic_pointer_cast<ActionData>(copied_action->createData());
  BOOST_REQUIRE(data1 != nullptr);
  BOOST_REQUIRE(data2 != nullptr);
  BOOST_REQUIRE(copied_data != nullptr);
  data1->dynamics = fixture.dynamics->createData();
  data2->dynamics = fixture.dynamics->createData();
  BOOST_CHECK(fixture.action->checkData(data1));

  const VectorXs x = fixture.state->rand();
  const VectorXs u =
      VectorXs::LinSpaced(fixture.action->get_nu(), Scalar(0.1), Scalar(0.5));
  VectorXs p(1);
  using std::log;
  p[0] = log(Scalar(0.025));
  model.update(params, p);
  fixture.action->calc(data1, x, u);
  action2->calc(data2, x, u);
  copied_action->calc(copied_data, x, u);
  BOOST_CHECK(data1->xnext.isApprox(data2->xnext));
  BOOST_CHECK_SMALL(copied_action->get_dt() - Scalar(0.02),
                    Scalar(20) * Eigen::NumTraits<Scalar>::epsilon());

  const VectorXs first_xnext = data1->xnext;
  fixture.time->set_time_step(Scalar(0.04));
  fixture.action->calc(data1, x, u);
  action2->calc(data2, x, u);
  BOOST_CHECK(data1->xnext.isApprox(data2->xnext));
  BOOST_CHECK(!data1->xnext.isApprox(first_xnext));
  BOOST_CHECK_SMALL(fixture.action->get_dt() - Scalar(0.04),
                    Scalar(20) * Eigen::NumTraits<Scalar>::epsilon());

  fixture.action->calc(data1, x);
  BOOST_CHECK(data1->xnext.isApprox(x));
  action2->calc(data2, x);
  BOOST_CHECK(data2->xnext.isApprox(x));

  p[0] = log(Scalar(0.03));
  model.update(params, p);
  fixture.action->calc(data1, x, u);
  data1->dynamics->vdot = data1->differential->xout;
  typename Model::MatrixXs dx_dp(fixture.state->get_ndx(), 1);
  model.computeParamSensitivity(data1, params, dx_dp, x, u);
  const VectorXs analytical = dx_dp.col(0);

  const Scalar disturbance =
      std::is_same<Scalar, float>::value ? Scalar(2e-3) : Scalar(1e-4);
  VectorXs p_minus = p;
  VectorXs p_plus = p;
  p_minus[0] -= disturbance;
  p_plus[0] += disturbance;
  model.update(params, p_minus);
  fixture.action->calc(data1, x, u);
  const VectorXs xnext_minus = data1->xnext;
  model.update(params, p_plus);
  fixture.action->calc(data2, x, u);
  const VectorXs xnext_plus = data2->xnext;
  VectorXs numerical(fixture.state->get_ndx());
  fixture.state->diff(xnext_minus, xnext_plus, numerical);
  numerical /= Scalar(2) * disturbance;
  const Scalar tolerance =
      std::is_same<Scalar, float>::value ? Scalar(2e-3) : Scalar(5e-6);
  BOOST_CHECK_MESSAGE(
      (analytical - numerical).cwiseAbs().maxCoeff() < tolerance,
      "Time sensitivity error: "
          << (analytical - numerical).cwiseAbs().maxCoeff());

  BOOST_CHECK_THROW(model.computeParamSensitivity(
                        std::shared_ptr<typename Action::ActionDataAbstract>(),
                        params, dx_dp, x, u),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model.computeParamSensitivity(
                        data1, params, dx_dp, VectorXs::Zero(x.size() + 1), u),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_live_constraint_forwarding() {
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> ConstraintManager;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> ConstraintResidual;
  typedef crocoddyl::ResidualModelControlTpl<Scalar> ResidualControl;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> ResidualState;
  typedef
      typename TimeFixtureTpl<Scalar>::DifferentialAction DifferentialAction;
  typedef typename TimeFixtureTpl<Scalar>::Action Action;
  typedef typename Action::VectorXs VectorXs;

  TimeFixtureTpl<Scalar> fixture;
  const std::shared_ptr<ConstraintManager> constraints =
      std::make_shared<ConstraintManager>(fixture.state,
                                          fixture.actuation->get_nu());
  const std::shared_ptr<DifferentialAction> differential =
      std::make_shared<DifferentialAction>(fixture.state, fixture.actuation,
                                           fixture.costs, constraints);
  const std::shared_ptr<Action> action =
      std::make_shared<Action>(differential, Scalar(0.02));
  BOOST_CHECK_EQUAL(action->get_ng(), 0u);
  BOOST_CHECK_EQUAL(action->get_nh(), 0u);

  const std::shared_ptr<ResidualControl> residual =
      std::make_shared<ResidualControl>(fixture.state,
                                        fixture.actuation->get_nu());
  const std::shared_ptr<ResidualState> state_residual =
      std::make_shared<ResidualState>(fixture.state,
                                      fixture.actuation->get_nu());
  const VectorXs lower = VectorXs::Constant(residual->get_nr(), Scalar(-0.5));
  const VectorXs upper = VectorXs::Constant(residual->get_nr(), Scalar(0.5));
  constraints->addConstraint("inequality",
                             std::make_shared<ConstraintResidual>(
                                 fixture.state, residual, lower, upper, false));
  constraints->addConstraint(
      "equality", std::make_shared<ConstraintResidual>(fixture.state,
                                                       state_residual, true));

  BOOST_CHECK_EQUAL(action->get_ng(), differential->get_ng());
  BOOST_CHECK_EQUAL(action->get_nh(), differential->get_nh());
  BOOST_CHECK_EQUAL(action->get_ng_T(), differential->get_ng_T());
  BOOST_CHECK_EQUAL(action->get_nh_T(), differential->get_nh_T());
  BOOST_CHECK(action->get_g_lb().isApprox(differential->get_g_lb()));
  BOOST_CHECK(action->get_g_ub().isApprox(differential->get_g_ub()));
  BOOST_CHECK_EQUAL(action->get_ng(), residual->get_nr());
  BOOST_CHECK_EQUAL(action->get_nh(), state_residual->get_nr());
  BOOST_CHECK_EQUAL(action->get_ng_T(), 0u);
  BOOST_CHECK_EQUAL(action->get_nh_T(), state_residual->get_nr());

  constraints->removeConstraint("inequality");
  constraints->removeConstraint("equality");
  BOOST_CHECK_EQUAL(action->get_ng(), 0u);
  BOOST_CHECK_EQUAL(action->get_nh(), 0u);
}

template <typename Scalar>
void test_manager_activation_and_no_allocation() {
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> Model;
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef typename Model::VectorXs VectorXs;

  TimeFixtureTpl<Scalar> fixture;
  const std::shared_ptr<Model> model = std::make_shared<Model>(
      fixture.state, fixture.action->get_integrator_time());
  Manager manager(fixture.state);
  manager.addParam("time", model);
  BOOST_CHECK_EQUAL(manager.get_np(), 1u);
  BOOST_CHECK_EQUAL(manager.get_np_action(), 1u);
  BOOST_CHECK_EQUAL(manager.get_np_dynamics(), 0u);

  const std::shared_ptr<typename Manager::ParameterDataManager> manager_data =
      manager.createData();
  typedef crocoddyl::IntegratedActionDataEulerTpl<Scalar> ActionData;
  const std::shared_ptr<ActionData> action_data =
      std::dynamic_pointer_cast<ActionData>(fixture.action->createData());
  BOOST_REQUIRE(action_data != nullptr);
  action_data->dynamics = fixture.dynamics->createData();
  const VectorXs x = fixture.state->rand();
  const VectorXs u =
      VectorXs::LinSpaced(fixture.action->get_nu(), Scalar(0.1), Scalar(0.5));
  VectorXs p(1);
  using std::log;
  p[0] = log(Scalar(0.035));
  manager.update(manager_data, p);
  fixture.action->calc(action_data, x, u);
  action_data->dynamics->vdot = action_data->differential->xout;
  typename Manager::MatrixXs dx_dp(fixture.state->get_ndx(), 1);
  manager.calcDiff_action(manager_data, action_data, dx_dp, x, u);
  BOOST_CHECK_SMALL(fixture.time->get_time_step() - Scalar(0.035),
                    Scalar(30) * Eigen::NumTraits<Scalar>::epsilon());
  BOOST_CHECK(!dx_dp.isZero());

  manager.changeParamStatus("time", false);
  BOOST_CHECK_EQUAL(manager.get_np(), 0u);
  BOOST_CHECK_EQUAL(manager.get_np_action(), 0u);
  manager_data->resize(&manager);
  manager.update(manager_data, VectorXs::Zero(0));
  BOOST_CHECK_SMALL(fixture.time->get_time_step() - Scalar(0.035),
                    Scalar(30) * Eigen::NumTraits<Scalar>::epsilon());
  manager.changeParamStatus("time", true);
  manager_data->resize(&manager);
  manager.update(manager_data, p);

  const std::shared_ptr<typename Model::ParamsDataAbstract> params =
      manager_data->action_params.at("time");
  const Scalar* const p_ptr = params->p.data();
  const Scalar* const dx_dp_ptr = dx_dp.data();
  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      manager.update(manager_data, p);
      manager.calcDiff_action(manager_data, action_data, dx_dp, x, u);
      params->setZero();
      params->resize(1, 0);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_CHECK_EQUAL(params->p.data(), p_ptr);
  BOOST_CHECK_EQUAL(dx_dp.data(), dx_dp_ptr);
}

void test_scalar_cast() {
  TimeFixtureTpl<double> fixture;
  crocoddyl::IntegratorTimeoptParams model(fixture.state, fixture.time);
  crocoddyl::IntegratorTimeoptParamsTpl<float> casted = model.cast<float>();
  BOOST_CHECK_EQUAL(casted.get_np(), 1u);
  BOOST_CHECK_EQUAL(casted.get_state()->get_nx(), model.get_state()->get_nx());
  BOOST_CHECK_SMALL(
      static_cast<double>(casted.get_integrator_time()->get_time_step()) -
          fixture.time->get_time_step(),
      1e-7);
  BOOST_CHECK(static_cast<const void*>(casted.get_integrator_time().get()) !=
              static_cast<const void*>(fixture.time.get()));
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_integrator_timeopt_params");
  ts->add(BOOST_TEST_CASE(
      &test_construction_data_update_bounds_copy_and_failures<double>));
  ts->add(BOOST_TEST_CASE(
      &test_construction_data_update_bounds_copy_and_failures<float>));
  ts->add(BOOST_TEST_CASE(
      &test_shared_and_copied_time_running_terminal_and_sensitivity<double>));
  ts->add(BOOST_TEST_CASE(
      &test_shared_and_copied_time_running_terminal_and_sensitivity<float>));
  ts->add(BOOST_TEST_CASE(&test_live_constraint_forwarding<double>));
  ts->add(BOOST_TEST_CASE(&test_live_constraint_forwarding<float>));
  ts->add(BOOST_TEST_CASE(&test_manager_activation_and_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_manager_activation_and_no_allocation<float>));
  ts->add(BOOST_TEST_CASE(&test_scalar_cast));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
