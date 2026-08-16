///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <string>
#include <type_traits>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/observer/discretized.hpp"
#include "crocoddyl/core/optctrl/observation.hpp"
#include "crocoddyl/core/optctrl/parametrized-shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/parameters.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename _Scalar>
class CountingActionParamsTpl
    : public crocoddyl::ActionModelParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ParamsModelBase, CountingActionParamsTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionModelParamsAbstractTpl<Scalar> Base;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  CountingActionParamsTpl(std::shared_ptr<StateAbstract> state,
                          const std::size_t np)
      : Base(state, np), update_calls(0) {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (data == nullptr || static_cast<std::size_t>(p.size()) != this->np_) {
      throw_pretty("Invalid argument: inconsistent action parameter update");
    }
    data->p = p;
    ++update_calls;
  }

  void computeParamSensitivity(const std::shared_ptr<ActionDataAbstract>&,
                               const std::shared_ptr<ParamsDataAbstract>&,
                               Eigen::Ref<MatrixXs> dx_dp,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) override {
    dx_dp.setZero();
  }

  template <typename NewScalar>
  CountingActionParamsTpl<NewScalar> cast() const {
    return CountingActionParamsTpl<NewScalar>(
        this->state_->template cast<NewScalar>(), this->np_);
  }

  std::size_t update_calls;
};

template <typename _Scalar>
class CountingDynamicsParamsTpl
    : public crocoddyl::DynamicsParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ParamsModelBase, CountingDynamicsParamsTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsParamsAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  CountingDynamicsParamsTpl(std::shared_ptr<StateAbstract> state,
                            const std::size_t np)
      : Base(state, np), update_calls(0) {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (data == nullptr || static_cast<std::size_t>(p.size()) != this->np_) {
      throw_pretty("Invalid argument: inconsistent dynamics parameter update");
    }
    data->p = p;
    ++update_calls;
  }

  void computeJointTorqueRegressor(const std::shared_ptr<DynamicsDataAbstract>&,
                                   const std::shared_ptr<ParamsDataAbstract>&,
                                   Eigen::Ref<MatrixXs> dtau_dp,
                                   const Eigen::Ref<const VectorXs>&,
                                   const Eigen::Ref<const VectorXs>&) override {
    dtau_dp.setZero();
  }

  template <typename NewScalar>
  CountingDynamicsParamsTpl<NewScalar> cast() const {
    return CountingDynamicsParamsTpl<NewScalar>(
        this->state_->template cast<NewScalar>(), this->np_);
  }

  std::size_t update_calls;
};

template <typename _Scalar>
class DiscreteDynamicsProblemProbeTpl
    : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase,
                         DiscreteDynamicsProblemProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::ParameterDataManager ParameterDataManager;
  typedef typename Base::ParameterManager ParameterManager;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  using Base::calc;
  using Base::calcDiff;

  DiscreteDynamicsProblemProbeTpl(std::shared_ptr<StateAbstract> state,
                                  const std::size_t np)
      : Base(state, crocoddyl::DynamicsType::DiscreteTime, np, 1) {}

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>&) override {
    data->vdot = x;
    data->vdot[0] += this->tau_meas_[0];
    const crocoddyl::DataCollectorParamsTpl<Scalar>* params =
        dynamic_cast<const crocoddyl::DataCollectorParamsTpl<Scalar>*>(
            data->shared);
    if (params == nullptr || params->params == nullptr ||
        static_cast<std::size_t>(params->params->p.size()) != this->np_) {
      throw_pretty("Invalid argument: parameter collector is inconsistent");
    }
    for (std::size_t j = 0; j < this->np_; ++j) {
      data->vdot.array() +=
          Scalar(j + 1) * params->params->p[static_cast<Eigen::Index>(j)];
    }
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setIdentity();
    data->Fu.setZero();
  }

  void calcDiff_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                  const Eigen::Ref<const VectorXs>&,
                  const Eigen::Ref<const VectorXs>&) override {
    for (std::size_t j = 0; j < this->np_; ++j) {
      data->Fp.col(static_cast<Eigen::Index>(j)).setConstant(Scalar(j + 1));
    }
  }

  using Base::createData;
  std::shared_ptr<DynamicsDataAbstract> createData() override {
    return std::make_shared<DynamicsDataAbstract>(this);
  }

  std::shared_ptr<DynamicsDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override {
    if (params_data == nullptr || params_data->params == nullptr ||
        params_data->params->np != this->np_) {
      throw_pretty("Invalid argument: parameter data is inconsistent");
    }
    const std::shared_ptr<DynamicsDataAbstract> data = createData();
    data->shared = params_data.get();
    return data;
  }

  void set_params(const std::shared_ptr<DynamicsDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override {
    const crocoddyl::DataCollectorParamsTpl<Scalar>* params_data =
        data != nullptr
            ? dynamic_cast<const crocoddyl::DataCollectorParamsTpl<Scalar>*>(
                  data->shared)
            : nullptr;
    if (params_data == nullptr || params_data->params == nullptr ||
        params == nullptr || params->get_np() != this->np_ ||
        params_data->params->np != this->np_) {
      throw_pretty("Invalid argument: dynamics parameters are inconsistent");
    }
  }

  void update_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override {
    if (data == nullptr || static_cast<std::size_t>(p.size()) != this->np_) {
      throw_pretty("Invalid argument: dynamics parameter vector is invalid");
    }
  }

  template <typename NewScalar>
  DiscreteDynamicsProblemProbeTpl<NewScalar> cast() const {
    DiscreteDynamicsProblemProbeTpl<NewScalar> model(
        this->state_->template cast<NewScalar>(), this->np_);
    model.update_tau(this->tau_meas_.template cast<NewScalar>());
    return model;
  }
};

template <typename Scalar>
std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar> > create_action_params(
    const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar> >& state,
    const std::size_t np,
    std::shared_ptr<CountingActionParamsTpl<Scalar> >& item) {
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  item = std::make_shared<CountingActionParamsTpl<Scalar> >(state, np);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  params->addParam("action", item);
  return params;
}

template <typename Scalar>
std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar> > create_dynamics_params(
    const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar> >& state,
    const std::size_t np,
    std::shared_ptr<CountingDynamicsParamsTpl<Scalar> >& item) {
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  item = std::make_shared<CountingDynamicsParamsTpl<Scalar> >(state, np);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  params->addParam("dynamics", item);
  return params;
}

template <typename Scalar>
std::shared_ptr<crocoddyl::ConstraintModelManagerTpl<Scalar> >
create_parameter_constraints(
    const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar> >& state,
    const std::size_t nu, const std::size_t np) {
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> Constraints;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> Constraint;
  typedef crocoddyl::ResidualModelControlTpl<Scalar> ControlResidual;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ParameterResidual;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, nu, np);
  constraints->addConstraint(
      "a_control", std::make_shared<Constraint>(
                       state, std::make_shared<ControlResidual>(state, nu)));
  constraints->addConstraint(
      "b_parameter",
      std::make_shared<Constraint>(state, std::make_shared<ParameterResidual>(
                                              state, VectorXs::Zero(np), nu)));
  return constraints;
}

template <typename Scalar>
void test_parametrized_shooting_problem() {
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModel;
  typedef crocoddyl::ParametrizedShootingProblemTpl<Scalar> Problem;
  typedef crocoddyl::ProblemAbstractTpl<Scalar> ProblemAbstract;
  typedef crocoddyl::ShootingProblemTpl<Scalar> ShootingProblem;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<ActionModel> phase0_a =
      std::make_shared<ActionModel>(4, 2, 1, 1, 1);
  const std::shared_ptr<ActionModel> phase0_b =
      std::make_shared<ActionModel>(4, 2, 1, 1, 1);
  const std::shared_ptr<ActionModel> phase1 =
      std::make_shared<ActionModel>(4, 1, 2, 1, 1);
  const std::shared_ptr<ActionModel> terminal =
      std::make_shared<ActionModel>(4, 1, 2, 1, 1);
  std::shared_ptr<CountingActionParamsTpl<Scalar> > item0;
  std::shared_ptr<CountingActionParamsTpl<Scalar> > item1;
  const std::shared_ptr<ParameterManager> params0 =
      create_action_params<Scalar>(phase0_a->get_state(), 1, item0);
  const std::shared_ptr<ParameterManager> params1 =
      create_action_params<Scalar>(phase1->get_state(), 2, item1);
  const std::shared_ptr<typename Problem::ConstraintModelManager> constraints0 =
      create_parameter_constraints<Scalar>(phase0_a->get_state(), 2, 1);
  const std::shared_ptr<typename Problem::ConstraintModelManager> constraints1 =
      create_parameter_constraints<Scalar>(phase1->get_state(), 1, 2);
  const std::vector<
      std::vector<std::shared_ptr<typename Problem::ActionModelAbstract> > >
      phases{{phase0_a, phase0_b}, {phase1}};
  const VectorXs x0 = VectorXs::LinSpaced(4, Scalar(-0.3), Scalar(0.3));
  Problem problem(
      x0, phases, terminal, {params0, params1},
      std::vector<std::shared_ptr<typename Problem::ConstraintModelManager> >{
          constraints0, constraints1});
  ProblemAbstract& base = problem;
  ShootingProblem& shooting_base = problem;

  BOOST_CHECK_EQUAL(base.get_T(), 3);
  BOOST_CHECK_EQUAL(base.get_n_phases(), 2);
  BOOST_CHECK(problem.get_phase_idxs() == std::vector<std::size_t>({0, 2}));
  BOOST_CHECK(problem.get_phase_edxs() == std::vector<std::size_t>({2, 3}));
  BOOST_CHECK_EQUAL(problem.get_running_phase_models(0).size(), 2);
  BOOST_CHECK_EQUAL(problem.get_running_phase_datas(1).size(), 1);
  BOOST_CHECK(problem.has_parameter_constraints());
  BOOST_CHECK(problem.get_params()[0] == params0);
  BOOST_CHECK(problem.get_params()[1] == params1);

  const VectorXs p0 = VectorXs::Constant(1, Scalar(0.25));
  VectorXs p1(2);
  p1 << Scalar(-0.2), Scalar(0.4);
  const std::size_t item0_updates = item0->update_calls;
  const std::size_t item1_updates = item1->update_calls;
  problem.update_p(p0, 0);
  problem.update_p(p1, 1);
  BOOST_CHECK_EQUAL(item0->update_calls, item0_updates + 1);
  BOOST_CHECK_EQUAL(item1->update_calls, item1_updates + 1);
  BOOST_CHECK(problem.get_params_data()[0]->params->p.isApprox(p0));
  BOOST_CHECK(problem.get_params_data()[1]->params->p.isApprox(p1));
  BOOST_CHECK(problem.get_params_data()[0]->parameter_data ==
              problem.get_params_data()[0].get());
  BOOST_CHECK(problem.get_params_data()[1]->parameter_data ==
              problem.get_params_data()[1].get());
  for (std::size_t t = 0; t < 2; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataLQRTpl<Scalar> > data =
        std::dynamic_pointer_cast<crocoddyl::ActionDataLQRTpl<Scalar> >(
            problem.get_runningDatas()[t]);
    BOOST_REQUIRE(data != nullptr);
    BOOST_CHECK(data->params == problem.get_params_data()[0]);
  }
  const std::shared_ptr<crocoddyl::ActionDataLQRTpl<Scalar> > phase1_data =
      std::dynamic_pointer_cast<crocoddyl::ActionDataLQRTpl<Scalar> >(
          problem.get_runningDatas()[2]);
  const std::shared_ptr<crocoddyl::ActionDataLQRTpl<Scalar> > terminal_data =
      std::dynamic_pointer_cast<crocoddyl::ActionDataLQRTpl<Scalar> >(
          problem.get_terminalData());
  BOOST_REQUIRE(phase1_data != nullptr);
  BOOST_REQUIRE(terminal_data != nullptr);
  BOOST_CHECK(phase1_data->params == problem.get_params_data()[1]);
  BOOST_CHECK(terminal_data->params == problem.get_params_data()[1]);
  BOOST_CHECK(problem.get_parameter_constraints_datas()[0]->shared ==
              problem.get_params_data()[0].get());

  std::vector<VectorXs> us(3);
  us[0] = VectorXs::Constant(2, Scalar(0.1));
  us[1] = VectorXs::Constant(2, Scalar(-0.2));
  us[2] = VectorXs::Constant(1, Scalar(0.3));
  std::vector<VectorXs> xs = problem.rollout_us(us);
  const Scalar cost = problem.calc(xs, us);
  BOOST_CHECK(std::isfinite(static_cast<double>(cost)));
  BOOST_CHECK_EQUAL(problem.calcDiff(xs, us), cost);
  BOOST_CHECK(problem.get_runningDatas()[0]->Fp.cols() == 1);
  BOOST_CHECK(problem.get_runningDatas()[2]->Fp.cols() == 2);
  BOOST_CHECK(problem.get_terminalData()->Fp.cols() == 2);

  constraints1->calc(problem.get_parameter_constraints_datas()[1], xs[2],
                     us[2]);
  constraints1->calcDiff(problem.get_parameter_constraints_datas()[1], xs[2],
                         us[2]);
  BOOST_CHECK(problem.get_parameter_constraints_datas()[1]
                  ->h.head(us[2].size())
                  .isApprox(us[2]));
  BOOST_CHECK(
      problem.get_parameter_constraints_datas()[1]->h.tail(p1.size()).isApprox(
          p1));
  BOOST_CHECK(problem.get_parameter_constraints_datas()[1]
                  ->Hp.bottomRows(p1.size())
                  .isIdentity());

  const std::vector<std::shared_ptr<typename Problem::ActionModelAbstract> >
      original_models = problem.get_runningModels();
  const std::vector<std::shared_ptr<typename Problem::ActionDataAbstract> >
      original_datas = problem.get_runningDatas();
  const std::shared_ptr<typename Problem::ActionModelAbstract>
      original_terminal_model = problem.get_terminalModel();
  const std::shared_ptr<typename Problem::ActionDataAbstract>
      original_terminal_data = problem.get_terminalData();
  const std::vector<std::size_t> original_phase_idxs = problem.get_phase_idxs();
  const std::vector<std::size_t> original_phase_edxs = problem.get_phase_edxs();
  const std::vector<std::shared_ptr<ParameterManager> > original_params =
      problem.get_params();
  const std::vector<std::shared_ptr<typename Problem::ParameterDataManager> >
      original_params_data = problem.get_params_data();
  const std::vector<std::shared_ptr<typename Problem::ConstraintModelManager> >
      original_constraints = problem.get_parameter_constraints_models();
  const std::vector<std::shared_ptr<typename Problem::ConstraintDataManager> >
      original_constraints_data = problem.get_parameter_constraints_datas();
  const auto requires_reconstruction = [](const crocoddyl::Exception& e) {
    return std::string(e.what()).find("must be reconstructed") !=
           std::string::npos;
  };
  const auto check_structural_identity = [&]() {
    BOOST_CHECK(problem.get_runningModels() == original_models);
    BOOST_CHECK(problem.get_runningDatas() == original_datas);
    BOOST_CHECK(problem.get_terminalModel() == original_terminal_model);
    BOOST_CHECK(problem.get_terminalData() == original_terminal_data);
    BOOST_CHECK(problem.get_phase_idxs() == original_phase_idxs);
    BOOST_CHECK(problem.get_phase_edxs() == original_phase_edxs);
    BOOST_CHECK(problem.get_params() == original_params);
    BOOST_CHECK(problem.get_params_data() == original_params_data);
    BOOST_CHECK(problem.get_parameter_constraints_models() ==
                original_constraints);
    BOOST_CHECK(problem.get_parameter_constraints_datas() ==
                original_constraints_data);
    for (std::size_t t = 0; t < 2; ++t) {
      const std::shared_ptr<crocoddyl::ActionDataLQRTpl<Scalar> > data =
          std::dynamic_pointer_cast<crocoddyl::ActionDataLQRTpl<Scalar> >(
              problem.get_runningDatas()[t]);
      BOOST_REQUIRE(data != nullptr);
      BOOST_CHECK(data->params == original_params_data[0]);
    }
    BOOST_CHECK(phase1_data->params == original_params_data[1]);
    BOOST_CHECK(terminal_data->params == original_params_data[1]);
    problem.update_p(p0, 0);
    problem.update_p(p1, 1);
    BOOST_CHECK(original_params_data[0]->params->p.isApprox(p0));
    BOOST_CHECK(original_params_data[1]->params->p.isApprox(p1));
    xs = problem.rollout_us(us);
    const Scalar current_cost = problem.calc(xs, us);
    BOOST_CHECK(std::isfinite(static_cast<double>(current_cost)));
    BOOST_CHECK_EQUAL(problem.calcDiff(xs, us), current_cost);
  };

  BOOST_CHECK_EXCEPTION(problem.circularAppend(phase0_a, original_datas[0]),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(problem.circularAppend(phase0_a), crocoddyl::Exception,
                        requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(problem.updateNode(0, phase0_a, original_datas[0]),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(problem.updateModel(0, phase0_a), crocoddyl::Exception,
                        requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(problem.set_runningModels(original_models),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(problem.set_terminalModel(terminal),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(
      shooting_base.circularAppend(phase0_a, original_datas[0]),
      crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(shooting_base.circularAppend(phase0_a),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(
      shooting_base.updateNode(0, phase0_a, original_datas[0]),
      crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(shooting_base.updateModel(0, phase0_a),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(shooting_base.set_runningModels(original_models),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();
  BOOST_CHECK_EXCEPTION(shooting_base.set_terminalModel(terminal),
                        crocoddyl::Exception, requires_reconstruction);
  check_structural_identity();

  VectorXs modified_x0 = x0;
  modified_x0.array() += Scalar(0.1);
  problem.set_x0(modified_x0);
  BOOST_CHECK(problem.get_x0().isApprox(modified_x0));
  problem.set_x0(x0);
  problem.set_nthreads(1);
  BOOST_CHECK_EQUAL(problem.get_nthreads(), 1);
  problem.set_is_updated(true);
  BOOST_CHECK(problem.is_updated());
  BOOST_CHECK(!problem.is_updated());
  check_structural_identity();

  Problem copied(problem);
  BOOST_CHECK(copied.get_runningDatas()[0] == problem.get_runningDatas()[0]);
  BOOST_CHECK(copied.get_params_data()[1] == problem.get_params_data()[1]);
  BOOST_CHECK_EQUAL(copied.calc(xs, us), cost);

  BOOST_CHECK_THROW(problem.update_p(p0, 2), crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.get_running_phase_models(2), crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.calc(std::vector<VectorXs>(3), us),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.rollout_us(std::vector<VectorXs>(2)),
                    crocoddyl::Exception);

  problem.calc(xs, us);
  problem.calcDiff(xs, us);
  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      problem.update_p(p0, 0);
      problem.update_p(p1, 1);
      problem.calc(xs, us);
      problem.calcDiff(xs, us);
      problem.rollout(us, xs);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

template <typename Scalar>
void test_observation_problem() {
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef DiscreteDynamicsProblemProbeTpl<Scalar> Dynamics;
  typedef crocoddyl::CostModelSumTpl<Scalar> Costs;
  typedef crocoddyl::CostModelResidualTpl<Scalar> Cost;
  typedef crocoddyl::ResidualModelControlTpl<Scalar> ControlResidual;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ParameterResidual;
  typedef crocoddyl::DiscretizedObserverModelTpl<Scalar> Observer;
  typedef crocoddyl::ObservationProblemTpl<Scalar> Problem;
  typedef crocoddyl::ProblemAbstractTpl<Scalar> ProblemAbstract;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  std::shared_ptr<CountingDynamicsParamsTpl<Scalar> > item0;
  std::shared_ptr<CountingDynamicsParamsTpl<Scalar> > item1;
  const std::shared_ptr<ParameterManager> params0 =
      create_dynamics_params<Scalar>(state, 1, item0);
  const std::shared_ptr<ParameterManager> params1 =
      create_dynamics_params<Scalar>(state, 2, item1);

  const auto make_observer =
      [&state](const std::size_t np) -> std::shared_ptr<Observer> {
    const std::shared_ptr<Dynamics> dynamics =
        std::make_shared<Dynamics>(state, np);
    const std::shared_ptr<Costs> costs =
        std::make_shared<Costs>(state, state->get_ndx(), np);
    costs->addCost(
        "control",
        std::make_shared<Cost>(
            state, std::make_shared<ControlResidual>(state, state->get_ndx())),
        Scalar(1));
    costs->addCost("parameter",
                   std::make_shared<Cost>(
                       state, std::make_shared<ParameterResidual>(
                                  state, VectorXs::Zero(np), state->get_ndx())),
                   Scalar(1));
    return std::make_shared<Observer>(dynamics, costs, 1);
  };

  const std::shared_ptr<Observer> observer0a = make_observer(1);
  const std::shared_ptr<Observer> observer0b = make_observer(1);
  const std::shared_ptr<Observer> observer1 = make_observer(2);
  const std::shared_ptr<Observer> terminal = make_observer(2);
  const std::shared_ptr<typename Problem::ConstraintModelManager> constraints0 =
      create_parameter_constraints<Scalar>(state, state->get_ndx(), 1);
  const std::shared_ptr<typename Problem::ConstraintModelManager> constraints1 =
      create_parameter_constraints<Scalar>(state, state->get_ndx(), 2);
  const std::vector<
      std::vector<std::shared_ptr<typename Problem::ObserverModelAbstract> > >
      phases{{observer0a, observer0b}, {observer1}};
  std::vector<VectorXs> tau_meas(3);
  tau_meas[0] = VectorXs::Constant(1, Scalar(0.1));
  tau_meas[1] = VectorXs::Constant(1, Scalar(0.2));
  tau_meas[2] = VectorXs::Constant(1, Scalar(0.3));
  const VectorXs x0 = VectorXs::LinSpaced(4, Scalar(-0.2), Scalar(0.2));
  Problem problem(
      x0, tau_meas, phases, terminal, {params0, params1},
      std::vector<std::shared_ptr<typename Problem::ConstraintModelManager> >{
          constraints0, constraints1});
  ProblemAbstract& base = problem;

  BOOST_CHECK_EQUAL(base.get_T(), 3);
  BOOST_CHECK_EQUAL(base.get_n_phases(), 2);
  BOOST_CHECK_EQUAL(base.get_nthreads(), 1);
  BOOST_CHECK(problem.get_phase_idxs() == std::vector<std::size_t>({0, 2}));
  BOOST_CHECK(problem.get_phase_edxs() == std::vector<std::size_t>({2, 3}));
  BOOST_CHECK_EQUAL(problem.get_running_phase_models(0).size(), 2);
  BOOST_CHECK(problem.has_parameter_constraints());
  BOOST_CHECK(observer0a->get_tau_meas().isApprox(tau_meas[0]));
  BOOST_CHECK(observer0b->get_tau_meas().isApprox(tau_meas[1]));
  BOOST_CHECK(observer1->get_tau_meas().isApprox(tau_meas[2]));

  const VectorXs p0 = VectorXs::Constant(1, Scalar(0.25));
  VectorXs p1(2);
  p1 << Scalar(-0.2), Scalar(0.4);
  const std::size_t item0_updates = item0->update_calls;
  const std::size_t item1_updates = item1->update_calls;
  problem.update_p(p0, 0);
  problem.update_p(p1, 1);
  BOOST_CHECK_EQUAL(item0->update_calls, item0_updates + 1);
  BOOST_CHECK_EQUAL(item1->update_calls, item1_updates + 1);
  BOOST_CHECK(problem.get_params_data()[0]->params->p.isApprox(p0));
  BOOST_CHECK(problem.get_params_data()[1]->params->p.isApprox(p1));
  const std::shared_ptr<crocoddyl::DiscretizedObserverDataTpl<Scalar> >
      running0_data = std::dynamic_pointer_cast<
          crocoddyl::DiscretizedObserverDataTpl<Scalar> >(
          problem.get_runningDatas()[0]);
  const std::shared_ptr<crocoddyl::DiscretizedObserverDataTpl<Scalar> >
      running2_data = std::dynamic_pointer_cast<
          crocoddyl::DiscretizedObserverDataTpl<Scalar> >(
          problem.get_runningDatas()[2]);
  const std::shared_ptr<crocoddyl::DiscretizedObserverDataTpl<Scalar> >
      terminal_data = std::dynamic_pointer_cast<
          crocoddyl::DiscretizedObserverDataTpl<Scalar> >(
          problem.get_terminalData());
  BOOST_REQUIRE(running0_data != nullptr);
  BOOST_REQUIRE(running2_data != nullptr);
  BOOST_REQUIRE(terminal_data != nullptr);
  BOOST_CHECK(running0_data->dynamics->shared ==
              problem.get_params_data()[0].get());
  BOOST_CHECK(running2_data->dynamics->shared ==
              problem.get_params_data()[1].get());
  BOOST_CHECK(terminal_data->dynamics->shared ==
              problem.get_params_data()[1].get());

  std::vector<VectorXs> ws(3);
  ws[0] = VectorXs::LinSpaced(4, Scalar(0.1), Scalar(0.4));
  ws[1] = VectorXs::LinSpaced(4, Scalar(-0.4), Scalar(-0.1));
  ws[2] = VectorXs::LinSpaced(4, Scalar(0.5), Scalar(0.8));
  std::vector<VectorXs> xs = problem.rollout_us(ws);
  BOOST_CHECK(xs[1][0] > x0[0]);
  BOOST_CHECK(!xs[1].isApprox(x0));
  const Scalar cost = problem.calc(xs, ws);
  BOOST_CHECK(cost > Scalar(0));
  BOOST_CHECK_EQUAL(problem.calcDiff(xs, ws), cost);
  BOOST_CHECK(problem.get_runningDatas()[0]->Fu.isZero());
  BOOST_CHECK(problem.get_runningDatas()[0]->Fp.col(0).isOnes());
  BOOST_CHECK(problem.get_runningDatas()[2]->Fp.col(0).isOnes());
  BOOST_CHECK(problem.get_runningDatas()[2]->Fp.col(1).isConstant(Scalar(2)));

  const std::shared_ptr<typename Problem::ConstraintDataManager>& phase1_data =
      problem.get_parameter_constraints_datas()[1];
  BOOST_CHECK(phase1_data->h.head(ws[2].size()).isApprox(ws[2]));
  BOOST_CHECK(phase1_data->h.tail(p1.size()).isApprox(p1));
  BOOST_CHECK(phase1_data->Hu.topRows(ws[2].size()).isIdentity());
  BOOST_CHECK(phase1_data->Hp.bottomRows(p1.size()).isIdentity());

  const VectorXs new_tau = VectorXs::Constant(1, Scalar(0.7));
  problem.update_tau(1, new_tau);
  BOOST_CHECK(observer0b->get_tau_meas().isApprox(new_tau));
  std::vector<VectorXs> all_tau(3, VectorXs::Constant(1, Scalar(-0.1)));
  problem.update_us(all_tau);
  BOOST_CHECK(observer0a->get_tau_meas().isApprox(all_tau[0]));
  BOOST_CHECK(observer1->get_tau_meas().isApprox(all_tau[2]));

  Problem copied(problem);
  BOOST_CHECK(copied.get_runningDatas()[0] == problem.get_runningDatas()[0]);
  BOOST_CHECK(copied.get_params_data()[1] == problem.get_params_data()[1]);
  BOOST_CHECK_EQUAL(copied.calc(xs, ws), problem.calc(xs, ws));
  copied.set_is_updated(true);
  BOOST_CHECK(copied.is_updated());
  BOOST_CHECK(!copied.is_updated());

  BOOST_CHECK_THROW(problem.update_p(p0, 2), crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.update_tau(3, new_tau), crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.update_tau(0, VectorXs::Zero(2)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.update_us(std::vector<VectorXs>(2)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.calc(std::vector<VectorXs>(3), ws),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(problem.rollout_us(std::vector<VectorXs>(2)),
                    crocoddyl::Exception);

  problem.calc(xs, ws);
  problem.calcDiff(xs, ws);
  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      problem.update_p(p0, 0);
      problem.update_p(p1, 1);
      problem.calc(xs, ws);
      problem.calcDiff(xs, ws);
      problem.rollout(ws, xs);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

template <typename Scalar>
void test_problem_validation_and_layout_changes() {
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModel;
  typedef crocoddyl::ParametrizedShootingProblemTpl<Scalar> ShootingProblem;
  typedef crocoddyl::ObservationProblemTpl<Scalar> ObservationProblem;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef DiscreteDynamicsProblemProbeTpl<Scalar> Dynamics;
  typedef crocoddyl::CostModelSumTpl<Scalar> Costs;
  typedef crocoddyl::DiscretizedObserverModelTpl<Scalar> Observer;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<ActionModel> model =
      std::make_shared<ActionModel>(4, 2, 1, 0, 0);
  std::shared_ptr<CountingActionParamsTpl<Scalar> > item;
  const std::shared_ptr<ParameterManager> params =
      create_action_params<Scalar>(model->get_state(), 1, item);
  const VectorXs x0 = VectorXs::Zero(4);
  BOOST_CHECK_THROW(
      ShootingProblem(x0,
                      std::vector<std::vector<std::shared_ptr<
                          typename ShootingProblem::ActionModelAbstract> > >(),
                      model, std::vector<std::shared_ptr<ParameterManager> >()),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ShootingProblem(
          x0,
          std::vector<std::vector<std::shared_ptr<
              typename ShootingProblem::ActionModelAbstract> > >{{}},
          model, {params}),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ShootingProblem(
          x0,
          std::vector<
              std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
              nullptr},
          model, params),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ShootingProblem(
          x0,
          std::vector<
              std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
              model},
          nullptr, params),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ShootingProblem(
          x0,
          std::vector<
              std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
              model},
          model, std::shared_ptr<ParameterManager>()),
      crocoddyl::Exception);
  const std::shared_ptr<ActionModel> wrong_np_model =
      std::make_shared<ActionModel>(4, 2, 2, 0, 0);
  BOOST_CHECK_THROW(
      ShootingProblem(
          x0,
          std::vector<
              std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
              wrong_np_model},
          wrong_np_model, params),
      crocoddyl::Exception);
  const std::shared_ptr<typename ShootingProblem::ConstraintModelManager>
      wrong_shooting_nu =
          create_parameter_constraints<Scalar>(model->get_state(), 1, 1);
  BOOST_CHECK_THROW(
      ShootingProblem(
          x0,
          std::vector<
              std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
              model},
          model, params, wrong_shooting_nu),
      crocoddyl::Exception);

  const std::shared_ptr<CountingActionParamsTpl<Scalar> > inactive =
      std::make_shared<CountingActionParamsTpl<Scalar> >(model->get_state(), 1);
  params->addParam("inactive", inactive, false);
  ShootingProblem stale(
      x0,
      std::vector<
          std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
          model},
      model, params);
  params->changeParamStatus("inactive", true);
  BOOST_CHECK_THROW(stale.update_p(VectorXs::Zero(2)), crocoddyl::Exception);
  const std::shared_ptr<ActionModel> resized =
      std::make_shared<ActionModel>(4, 2, 2, 0, 0);
  ShootingProblem rebuilt(
      x0,
      std::vector<
          std::shared_ptr<typename ShootingProblem::ActionModelAbstract> >{
          resized},
      resized, params);
  rebuilt.update_p(VectorXs::Zero(2));
  BOOST_CHECK_EQUAL(rebuilt.get_params_data()[0]->params->p.size(), 2);

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  std::shared_ptr<CountingDynamicsParamsTpl<Scalar> > dynamics_item;
  const std::shared_ptr<ParameterManager> dynamics_params =
      create_dynamics_params<Scalar>(state, 1, dynamics_item);
  const std::shared_ptr<Dynamics> dynamics =
      std::make_shared<Dynamics>(state, 1);
  const std::shared_ptr<Costs> costs =
      std::make_shared<Costs>(state, state->get_ndx(), 1);
  const std::shared_ptr<Observer> observer =
      std::make_shared<Observer>(dynamics, costs, 1);
  const std::shared_ptr<Dynamics> wrong_np_dynamics =
      std::make_shared<Dynamics>(state, 2);
  const std::shared_ptr<Costs> wrong_np_costs =
      std::make_shared<Costs>(state, state->get_ndx(), 2);
  const std::shared_ptr<Observer> wrong_np_observer =
      std::make_shared<Observer>(wrong_np_dynamics, wrong_np_costs, 1);
  const std::vector<VectorXs> tau(1, VectorXs::Zero(1));
  BOOST_CHECK_THROW(
      ObservationProblem(
          x0, tau,
          std::vector<std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> > >(),
          observer, std::vector<std::shared_ptr<ParameterManager> >()),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ObservationProblem(
          x0, std::vector<VectorXs>(),
          std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> >{observer},
          observer, dynamics_params),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ObservationProblem(
          VectorXs::Zero(3), tau,
          std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> >{observer},
          observer, dynamics_params),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ObservationProblem(
          x0, tau,
          std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> >{nullptr},
          observer, dynamics_params),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ObservationProblem(
          x0, tau,
          std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> >{observer},
          nullptr, dynamics_params),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      ObservationProblem(
          x0, tau,
          std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> >{
              wrong_np_observer},
          wrong_np_observer, dynamics_params),
      crocoddyl::Exception);
  const std::shared_ptr<typename ObservationProblem::ConstraintModelManager>
      wrong_nu = create_parameter_constraints<Scalar>(state, 1, 1);
  BOOST_CHECK_THROW(
      ObservationProblem(
          x0, tau,
          std::vector<std::shared_ptr<
              typename ObservationProblem::ObserverModelAbstract> >{observer},
          observer, dynamics_params, wrong_nu),
      crocoddyl::Exception);
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_parameterized_problems");
  ts->add(BOOST_TEST_CASE(&test_parametrized_shooting_problem<double>));
  ts->add(BOOST_TEST_CASE(&test_parametrized_shooting_problem<float>));
  ts->add(BOOST_TEST_CASE(&test_observation_problem<double>));
  ts->add(BOOST_TEST_CASE(&test_observation_problem<float>));
  ts->add(BOOST_TEST_CASE(&test_problem_validation_and_layout_changes<double>));
  ts->add(BOOST_TEST_CASE(&test_problem_validation_and_layout_changes<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
