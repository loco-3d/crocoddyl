///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <array>
#include <type_traits>
#include <utility>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

class CoutRedirect {
 public:
  CoutRedirect() : previous_(std::cout.rdbuf(stream_.rdbuf())) {}
  ~CoutRedirect() { std::cout.rdbuf(previous_); }

  std::string str() const { return stream_.str(); }

 private:
  std::ostringstream stream_;
  std::streambuf* previous_;
};

template <typename _Scalar>
class ActionParamsProbeTpl
    : public crocoddyl::ActionModelParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ParamsModelBase, ActionParamsProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionModelParamsAbstractTpl<Scalar> Base;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  ActionParamsProbeTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t np, const Scalar scale)
      : Base(state, np),
        scale(scale),
        check_calls(0),
        update_calls(0),
        sensitivity_calls(0) {}

  bool checkData(
      const std::shared_ptr<ParamsDataAbstract>& data) const override {
    ++check_calls;
    return Base::checkData(data);
  }

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (data == nullptr || static_cast<std::size_t>(p.size()) != this->np_) {
      throw_pretty("Invalid argument: action parameter update is inconsistent");
    }
    data->p = p;
    ++update_calls;
  }

  void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>&,
      const std::shared_ptr<ParamsDataAbstract>& params,
      Eigen::Ref<MatrixXs> dx_dp, const Eigen::Ref<const VectorXs>&,
      const Eigen::Ref<const VectorXs>&) override {
    if (params == nullptr) {
      throw_pretty("Invalid argument: action parameter data is null");
    }
    for (std::size_t j = 0; j < this->np_; ++j) {
      dx_dp.col(static_cast<Eigen::Index>(j)).setConstant(scale + Scalar(j));
    }
    ++sensitivity_calls;
  }

  template <typename NewScalar>
  ActionParamsProbeTpl<NewScalar> cast() const {
    ActionParamsProbeTpl<NewScalar> model(
        this->state_->template cast<NewScalar>(), this->np_,
        crocoddyl::scalar_cast<NewScalar>(scale));
    model.set_lb(this->lb_.template cast<NewScalar>());
    model.set_ub(this->ub_.template cast<NewScalar>());
    return model;
  }

  Scalar scale;
  mutable std::size_t check_calls;
  std::size_t update_calls;
  std::size_t sensitivity_calls;
};

template <typename _Scalar>
class DynamicsParamsProbeTpl
    : public crocoddyl::DynamicsParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ParamsModelBase, DynamicsParamsProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsParamsAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  DynamicsParamsProbeTpl(std::shared_ptr<StateAbstract> state,
                         const std::size_t np, const Scalar scale)
      : Base(state, np),
        scale(scale),
        check_calls(0),
        update_calls(0),
        regressor_calls(0) {}

  bool checkData(
      const std::shared_ptr<ParamsDataAbstract>& data) const override {
    ++check_calls;
    return Base::checkData(data);
  }

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (data == nullptr || static_cast<std::size_t>(p.size()) != this->np_) {
      throw_pretty(
          "Invalid argument: dynamics parameter update is inconsistent");
    }
    data->p = p;
    ++update_calls;
  }

  void computeJointTorqueRegressor(
      const std::shared_ptr<DynamicsDataAbstract>&,
      const std::shared_ptr<ParamsDataAbstract>& params,
      Eigen::Ref<MatrixXs> dtau_dp, const Eigen::Ref<const VectorXs>&,
      const Eigen::Ref<const VectorXs>&) override {
    if (params == nullptr) {
      throw_pretty("Invalid argument: dynamics parameter data is null");
    }
    for (std::size_t j = 0; j < this->np_; ++j) {
      dtau_dp.col(static_cast<Eigen::Index>(j)).setConstant(scale + Scalar(j));
    }
    ++regressor_calls;
  }

  template <typename NewScalar>
  DynamicsParamsProbeTpl<NewScalar> cast() const {
    DynamicsParamsProbeTpl<NewScalar> model(
        this->state_->template cast<NewScalar>(), this->np_,
        crocoddyl::scalar_cast<NewScalar>(scale));
    model.set_lb(this->lb_.template cast<NewScalar>());
    model.set_ub(this->ub_.template cast<NewScalar>());
    return model;
  }

  Scalar scale;
  mutable std::size_t check_calls;
  std::size_t update_calls;
  std::size_t regressor_calls;
};

template <typename _Scalar>
class NodeDependentActionParamsTpl
    : public crocoddyl::ActionModelParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionModelParamsAbstractTpl<Scalar> Base;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::MatrixXs MatrixXs;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  NodeDependentActionParamsTpl(std::shared_ptr<StateAbstract> state,
                               const std::size_t np)
      : Base(state, np) {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    data->p = p;
  }

  void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>&,
      const std::shared_ptr<ParamsDataAbstract>& params,
      Eigen::Ref<MatrixXs> dx_dp, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override {
    for (std::size_t j = 0; j < this->np_; ++j) {
      dx_dp.col(static_cast<Eigen::Index>(j))
          .setConstant(x[0] + u[0] + params->p[static_cast<Eigen::Index>(j)]);
    }
  }
};

template <typename _Scalar>
class NodeDependentDynamicsParamsTpl
    : public crocoddyl::DynamicsParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsParamsAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::MatrixXs MatrixXs;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  NodeDependentDynamicsParamsTpl(std::shared_ptr<StateAbstract> state,
                                 const std::size_t np)
      : Base(state, np) {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    data->p = p;
  }

  void computeJointTorqueRegressor(
      const std::shared_ptr<DynamicsDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      Eigen::Ref<MatrixXs> dtau_dp, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override {
    for (std::size_t j = 0; j < this->np_; ++j) {
      dtau_dp.col(static_cast<Eigen::Index>(j))
          .setConstant(x[0] + data->vdot[0] + u[0] +
                       params->p[static_cast<Eigen::Index>(j)]);
    }
  }
};

template <typename Scalar>
class DynamicsDataModelStubTpl {
 public:
  typedef crocoddyl::StateAbstractTpl<Scalar> StateAbstract;

  DynamicsDataModelStubTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t nu)
      : state_(state), nu_(nu) {}

  const std::shared_ptr<StateAbstract>& get_state() const { return state_; }
  std::size_t get_np() const { return 0; }
  std::size_t get_nu() const { return nu_; }
  std::size_t get_ng() const { return 0; }
  std::size_t get_nh() const { return 0; }
  crocoddyl::DynamicsType get_dyn_type() const {
    return crocoddyl::DynamicsType::ContinuousControl;
  }

 private:
  std::shared_ptr<StateAbstract> state_;
  std::size_t nu_;
};

template <typename Scalar>
struct ParameterManagerFixtureTpl {
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef ActionParamsProbeTpl<Scalar> ActionParams;
  typedef DynamicsParamsProbeTpl<Scalar> DynamicsParams;
  typedef typename Manager::VectorXs VectorXs;

  ParameterManagerFixtureTpl()
      : state(std::make_shared<State>(4)), manager(state) {
    action_zeta = std::make_shared<ActionParams>(state, 1, Scalar(30));
    dynamics_zulu = std::make_shared<DynamicsParams>(state, 2, Scalar(50));
    action_alpha = std::make_shared<ActionParams>(state, 2, Scalar(10));
    dynamics_idle = std::make_shared<DynamicsParams>(state, 1, Scalar(70));
    dynamics_beta = std::make_shared<DynamicsParams>(state, 1, Scalar(40));
    action_middle = std::make_shared<ActionParams>(state, 1, Scalar(20));
    action_alpha->set_lb(VectorXs::Constant(2, Scalar(-2)));
    action_alpha->set_ub(VectorXs::Constant(2, Scalar(2)));
    dynamics_zulu->set_lb(VectorXs::Constant(2, Scalar(-5)));
    dynamics_zulu->set_ub(VectorXs::Constant(2, Scalar(5)));

    manager.addParam("zeta", action_zeta);
    manager.addParam("zulu", dynamics_zulu);
    manager.addParam("alpha", action_alpha);
    manager.addParam("idle", dynamics_idle, false);
    manager.addParam("beta", dynamics_beta);
    manager.addParam("middle", action_middle, false);
  }

  std::shared_ptr<State> state;
  Manager manager;
  std::shared_ptr<ActionParams> action_alpha;
  std::shared_ptr<ActionParams> action_middle;
  std::shared_ptr<ActionParams> action_zeta;
  std::shared_ptr<DynamicsParams> dynamics_beta;
  std::shared_ptr<DynamicsParams> dynamics_idle;
  std::shared_ptr<DynamicsParams> dynamics_zulu;
};

template <typename Scalar>
void test_order_update_derivatives_and_data() {
  typedef ParameterManagerFixtureTpl<Scalar> Fixture;
  typedef typename Fixture::Manager Manager;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModel;
  typedef crocoddyl::ActionModelParamsDataAbstractTpl<Scalar> ActionParamsData;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> DynamicsData;
  typedef crocoddyl::DynamicsParamsDataAbstractTpl<Scalar> DynamicsParamsData;
  typedef typename Manager::ParameterItem ParameterItem;
  typedef crocoddyl::ParamsAbstractTpl<Scalar> ParamsAbstract;
  typedef typename Manager::VectorXs VectorXs;

  static_assert(
      std::is_same<decltype(std::declval<const ParameterItem&>().get_name()),
                   const std::string&>::value,
      "ParameterItem name must be read-only");
  static_assert(
      std::is_same<decltype(std::declval<const ParameterItem&>().get_param()),
                   const std::shared_ptr<ParamsAbstract>&>::value,
      "ParameterItem pointer must be read-only");
  static_assert(
      std::is_same<decltype(std::declval<const ParameterItem&>().get_active()),
                   bool>::value,
      "ParameterItem activity must be returned by value");
  static_assert(!std::is_assignable<
                    decltype(std::declval<const ParameterItem&>().get_name()),
                    std::string>::value,
                "ParameterItem name must not be assignable");
  static_assert(!std::is_assignable<
                    decltype(std::declval<const ParameterItem&>().get_param()),
                    std::shared_ptr<ParamsAbstract> >::value,
                "ParameterItem pointer must not be assignable");
  static_assert(!std::is_assignable<
                    decltype(std::declval<const ParameterItem&>().get_active()),
                    bool>::value,
                "ParameterItem activity must not be assignable");
  static_assert(std::is_copy_constructible<ParameterItem>::value,
                "ParameterItem must remain copy constructible");
  static_assert(!std::is_copy_assignable<ParameterItem>::value,
                "ParameterItem metadata must not be replaced by assignment");
  static_assert(!std::is_move_assignable<ParameterItem>::value,
                "ParameterItem metadata must not be replaced by assignment");

  Fixture fixture;
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 6u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np_action(), 3u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np_dynamics(), 3u);
  BOOST_CHECK_EQUAL(fixture.manager.get_active_set().size(), 4u);
  BOOST_CHECK_EQUAL(fixture.manager.get_inactive_set().size(), 2u);
  BOOST_CHECK(fixture.manager.get_active_set().count("alpha") == 1u);
  BOOST_CHECK(fixture.manager.get_inactive_set().count("middle") == 1u);
  const std::shared_ptr<ParameterItem> alpha_item =
      fixture.manager.get_action_params().at("alpha");
  BOOST_CHECK_EQUAL(alpha_item->get_name(), "alpha");
  BOOST_CHECK(alpha_item->get_param() == fixture.action_alpha);
  BOOST_CHECK(alpha_item->get_active());
  const VectorXs replacement_lb = VectorXs::Constant(2, Scalar(-3));
  alpha_item->get_param()->set_lb(replacement_lb);
  BOOST_CHECK(fixture.action_alpha->get_lb().isApprox(replacement_lb));
  fixture.manager.changeParamStatus("alpha", false);
  BOOST_CHECK(!alpha_item->get_active());
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 4u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np_action(), 1u);
  BOOST_CHECK(fixture.manager.get_active_set().count("alpha") == 0u);
  BOOST_CHECK(fixture.manager.get_inactive_set().count("alpha") == 1u);
  fixture.manager.changeParamStatus("alpha", true);
  BOOST_CHECK(alpha_item->get_active());
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 6u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np_action(), 3u);
  BOOST_CHECK(fixture.manager.get_active_set().count("alpha") == 1u);
  BOOST_CHECK(fixture.manager.get_inactive_set().count("alpha") == 0u);

  const std::shared_ptr<typename Manager::ParameterDataManager> data =
      fixture.manager.createData();
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(data->parameter_data == data.get());
  BOOST_CHECK_EQUAL(data->action_params.size(), 3u);
  BOOST_CHECK_EQUAL(data->dynamics_params.size(), 3u);
  BOOST_CHECK(std::dynamic_pointer_cast<ActionParamsData>(
                  data->action_params.at("alpha")) != nullptr);
  BOOST_CHECK(std::dynamic_pointer_cast<ActionParamsData>(
                  data->action_params.at("middle")) != nullptr);
  BOOST_CHECK(std::dynamic_pointer_cast<DynamicsParamsData>(
                  data->dynamics_params.at("idle")) != nullptr);
  BOOST_CHECK(std::dynamic_pointer_cast<DynamicsParamsData>(
                  data->dynamics_params.at("zulu")) != nullptr);
  typename Manager::ParameterDataManager copied_data(*data);
  BOOST_CHECK(copied_data.parameter_data == &copied_data);
  BOOST_CHECK(copied_data.params == data->params);
  BOOST_CHECK(copied_data.action_params.at("alpha") ==
              data->action_params.at("alpha"));
  BOOST_CHECK(copied_data.dynamics_params.at("zulu") ==
              data->dynamics_params.at("zulu"));

  const std::size_t action_alpha_checks = fixture.action_alpha->check_calls;
  const std::size_t action_middle_checks = fixture.action_middle->check_calls;
  const std::size_t action_zeta_checks = fixture.action_zeta->check_calls;
  const std::size_t dynamics_beta_checks = fixture.dynamics_beta->check_calls;
  const std::size_t dynamics_idle_checks = fixture.dynamics_idle->check_calls;
  const std::size_t dynamics_zulu_checks = fixture.dynamics_zulu->check_calls;
  const VectorXs p = VectorXs::LinSpaced(6, Scalar(1), Scalar(6));
  fixture.manager.update(data, p);
  BOOST_CHECK_EQUAL(fixture.action_alpha->check_calls,
                    action_alpha_checks + 1u);
  BOOST_CHECK_EQUAL(fixture.action_middle->check_calls,
                    action_middle_checks + 1u);
  BOOST_CHECK_EQUAL(fixture.action_zeta->check_calls, action_zeta_checks + 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_beta->check_calls,
                    dynamics_beta_checks + 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_idle->check_calls,
                    dynamics_idle_checks + 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_zulu->check_calls,
                    dynamics_zulu_checks + 1u);
  BOOST_CHECK(data->params->p.isApprox(p));
  BOOST_CHECK(data->action_params.at("alpha")->p.isApprox(p.segment(0, 2)));
  BOOST_CHECK(data->action_params.at("zeta")->p.isApprox(p.segment(2, 1)));
  BOOST_CHECK(data->dynamics_params.at("beta")->p.isApprox(p.segment(3, 1)));
  BOOST_CHECK(data->dynamics_params.at("zulu")->p.isApprox(p.segment(4, 2)));
  BOOST_CHECK(data->action_params.at("middle")->p.isZero());
  BOOST_CHECK(data->dynamics_params.at("idle")->p.isZero());
  BOOST_CHECK_EQUAL(fixture.action_alpha->update_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.action_zeta->update_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.action_middle->update_calls, 0u);
  BOOST_CHECK_EQUAL(fixture.dynamics_beta->update_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_zulu->update_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_idle->update_calls, 0u);

  const std::size_t nu = 2;
  ActionModel action_model(4, nu);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > action_data =
      action_model.createData();
  DynamicsDataModelStubTpl<Scalar> dynamics_model(fixture.state, nu);
  const std::shared_ptr<DynamicsData> dynamics_data =
      std::make_shared<DynamicsData>(&dynamics_model);
  const VectorXs x = VectorXs::LinSpaced(4, Scalar(0.1), Scalar(0.4));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.5), Scalar(0.6));
  typename Manager::MatrixXs dx_dp(fixture.state->get_ndx(),
                                   fixture.manager.get_np_action());
  typename Manager::MatrixXs dtau_dp(fixture.state->get_nv(),
                                     fixture.manager.get_np_dynamics());
  dx_dp.setConstant(Scalar(999));
  dtau_dp.setConstant(Scalar(999));
  const std::size_t action_checks_before = fixture.action_alpha->check_calls;
  const std::size_t dynamics_checks_before = fixture.dynamics_beta->check_calls;
  fixture.manager.calcDiff_action(data, action_data, dx_dp, x, u);
  BOOST_CHECK_EQUAL(fixture.action_alpha->check_calls,
                    action_checks_before + 1u);
  BOOST_CHECK_EQUAL(fixture.action_middle->check_calls,
                    action_middle_checks + 2u);
  BOOST_CHECK_EQUAL(fixture.action_zeta->check_calls, action_zeta_checks + 2u);
  BOOST_CHECK_EQUAL(fixture.dynamics_beta->check_calls, dynamics_checks_before);
  BOOST_CHECK_EQUAL(fixture.dynamics_idle->check_calls,
                    dynamics_idle_checks + 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_zulu->check_calls,
                    dynamics_zulu_checks + 1u);
  fixture.manager.calcDiff_dynamics(data, dynamics_data, dtau_dp, x, u);
  BOOST_CHECK_EQUAL(fixture.action_alpha->check_calls,
                    action_checks_before + 1u);
  BOOST_CHECK_EQUAL(fixture.action_middle->check_calls,
                    action_middle_checks + 2u);
  BOOST_CHECK_EQUAL(fixture.action_zeta->check_calls, action_zeta_checks + 2u);
  BOOST_CHECK_EQUAL(fixture.dynamics_beta->check_calls,
                    dynamics_checks_before + 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_idle->check_calls,
                    dynamics_idle_checks + 2u);
  BOOST_CHECK_EQUAL(fixture.dynamics_zulu->check_calls,
                    dynamics_zulu_checks + 2u);
  BOOST_CHECK(dx_dp.col(0).isConstant(Scalar(10)));
  BOOST_CHECK(dx_dp.col(1).isConstant(Scalar(11)));
  BOOST_CHECK(dx_dp.col(2).isConstant(Scalar(30)));
  BOOST_CHECK(dtau_dp.col(0).isConstant(Scalar(40)));
  BOOST_CHECK(dtau_dp.col(1).isConstant(Scalar(50)));
  BOOST_CHECK(dtau_dp.col(2).isConstant(Scalar(51)));
  BOOST_CHECK_EQUAL(fixture.action_alpha->sensitivity_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.action_zeta->sensitivity_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.action_middle->sensitivity_calls, 0u);
  BOOST_CHECK_EQUAL(fixture.dynamics_beta->regressor_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_zulu->regressor_calls, 1u);
  BOOST_CHECK_EQUAL(fixture.dynamics_idle->regressor_calls, 0u);
  const typename Manager::MatrixXs returned_dx_dp =
      fixture.manager.calcDiff_action_x(data, action_data, x, u);
  const typename Manager::MatrixXs returned_dtau_dp =
      fixture.manager.calcDiff_dynamics_x(data, dynamics_data, x, u);
  BOOST_CHECK(returned_dx_dp.isApprox(dx_dp));
  BOOST_CHECK(returned_dtau_dp.isApprox(dtau_dp));
  BOOST_CHECK_EQUAL(fixture.action_alpha->sensitivity_calls, 2u);
  BOOST_CHECK_EQUAL(fixture.action_zeta->sensitivity_calls, 2u);
  BOOST_CHECK_EQUAL(fixture.action_middle->sensitivity_calls, 0u);
  BOOST_CHECK_EQUAL(fixture.dynamics_beta->regressor_calls, 2u);
  BOOST_CHECK_EQUAL(fixture.dynamics_zulu->regressor_calls, 2u);
  BOOST_CHECK_EQUAL(fixture.dynamics_idle->regressor_calls, 0u);

  BOOST_CHECK(fixture.manager.zero().isZero());
  const VectorXs random = fixture.manager.rand();
  BOOST_CHECK_EQUAL(random.size(), 6);
  BOOST_CHECK((random.array() >= Scalar(0)).all());
  BOOST_CHECK((random.array() <= Scalar(1)).all());
  std::ostringstream stream;
  stream << fixture.manager;
  BOOST_CHECK(!stream.str().empty());

  data->params->p.setOnes();
  data->params->active = false;
  for (typename Manager::ParameterDataManager::ParameterDataContainer::iterator
           it = data->action_params.begin();
       it != data->action_params.end(); ++it) {
    it->second->p.setOnes();
  }
  for (typename Manager::ParameterDataManager::ParameterDataContainer::iterator
           it = data->dynamics_params.begin();
       it != data->dynamics_params.end(); ++it) {
    it->second->p.setOnes();
  }
  data->action_params.at("middle")->active = false;
  data->setZero();
  BOOST_CHECK(data->params->p.isZero());
  BOOST_CHECK(!data->params->active);
  BOOST_CHECK(!data->action_params.at("middle")->active);
  for (typename Manager::ParameterDataManager::ParameterDataContainer::iterator
           it = data->action_params.begin();
       it != data->action_params.end(); ++it) {
    BOOST_CHECK(it->second->p.isZero());
  }
  for (typename Manager::ParameterDataManager::ParameterDataContainer::iterator
           it = data->dynamics_params.begin();
       it != data->dynamics_params.end(); ++it) {
    BOOST_CHECK(it->second->p.isZero());
  }

  BOOST_CHECK_THROW(fixture.manager.update(data, VectorXs::Zero(5)),
                    std::exception);
  BOOST_CHECK_THROW(
      fixture.manager.update(
          std::shared_ptr<typename Manager::ParameterDataManager>(), p),
      std::exception);
  BOOST_CHECK_THROW(
      fixture.manager.calcDiff_action(
          data, std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> >(),
          dx_dp, x, u),
      std::exception);
  BOOST_CHECK_THROW(fixture.manager.calcDiff_dynamics(
                        data, std::shared_ptr<DynamicsData>(), dtau_dp, x, u),
                    std::exception);
  BOOST_CHECK_THROW(fixture.manager.calcDiff_action(data, action_data, dx_dp,
                                                    VectorXs::Zero(3), u),
                    std::exception);
  BOOST_CHECK_THROW(fixture.manager.calcDiff_dynamics(
                        data, dynamics_data, dtau_dp, x, VectorXs::Zero(3)),
                    std::exception);
  typename Manager::MatrixXs wrong_dx_dp(fixture.state->get_ndx() + 1,
                                         fixture.manager.get_np_action());
  typename Manager::MatrixXs wrong_dtau_dp(
      fixture.state->get_nv(), fixture.manager.get_np_dynamics() + 1);
  BOOST_CHECK_THROW(
      fixture.manager.calcDiff_action(data, action_data, wrong_dx_dp, x, u),
      std::exception);
  BOOST_CHECK_THROW(fixture.manager.calcDiff_dynamics(data, dynamics_data,
                                                      wrong_dtau_dp, x, u),
                    std::exception);
}

template <typename Scalar>
void test_duplicate_missing_status_remove_and_resize() {
  typedef ParameterManagerFixtureTpl<Scalar> Fixture;
  typedef typename Fixture::Manager Manager;
  typedef typename Manager::VectorXs VectorXs;
  typedef crocoddyl::internal::ParameterDataManagerAccessTpl<Scalar>
      ParameterDataAccess;

  Fixture fixture;
  const std::shared_ptr<typename Manager::ParameterItem> original =
      fixture.manager.get_action_params().at("alpha");
  const std::size_t np = fixture.manager.get_np();
  const std::size_t np_action = fixture.manager.get_np_action();
  const std::size_t active_size = fixture.manager.get_active_set().size();
  std::string warnings;
  {
    CoutRedirect output;
    std::shared_ptr<typename Fixture::ActionParams> null_param;
    fixture.manager.addParam("alpha", null_param);
    fixture.manager.removeParam("missing");
    fixture.manager.changeParamStatus("missing", true);
    BOOST_CHECK(!fixture.manager.getParamStatus("missing"));
    warnings = output.str();
  }
  BOOST_CHECK(warnings.find("already existed") != std::string::npos);
  BOOST_CHECK(warnings.find("couldn't remove") != std::string::npos);
  BOOST_CHECK(warnings.find("couldn't change") != std::string::npos);
  BOOST_CHECK(warnings.find("couldn't get") != std::string::npos);
  BOOST_CHECK(fixture.manager.get_action_params().at("alpha") == original);
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), np);
  BOOST_CHECK_EQUAL(fixture.manager.get_np_action(), np_action);
  BOOST_CHECK_EQUAL(fixture.manager.get_active_set().size(), active_size);

  const std::shared_ptr<typename Manager::ParameterDataManager> data =
      fixture.manager.createData();
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "alpha"), 0u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "zeta"), 2u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "beta"), 3u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "zulu"), 4u);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(*data, "middle"),
                    std::exception);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(*data, "idle"),
                    std::exception);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(*data, "missing"),
                    std::exception);
  typename Manager::ParameterDataManager copied_data(*data);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(copied_data, "zulu"),
                    4u);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(copied_data, "middle"),
                    std::exception);
  data->action_params.at("alpha")->active = false;
  data->action_params.at("middle")->active = true;
  fixture.manager.changeParamStatus("alpha", false);
  BOOST_CHECK_EQUAL(fixture.manager.get_np_action(), 1u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 4u);
  BOOST_CHECK_THROW(fixture.manager.update(data, VectorXs::Zero(4)),
                    std::exception);
  data->resize(&fixture.manager);
  BOOST_CHECK_EQUAL(data->params->np_action, 1u);
  BOOST_CHECK_EQUAL(data->params->np_dynamics, 3u);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(*data, "alpha"),
                    std::exception);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "zeta"), 0u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "beta"), 1u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "zulu"), 2u);
  BOOST_CHECK(!data->action_params.at("alpha")->active);
  BOOST_CHECK(data->action_params.at("middle")->active);
  fixture.manager.update(data, VectorXs::LinSpaced(4, Scalar(1), Scalar(4)));
  fixture.manager.changeParamStatus("alpha", false);
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 4u);
  fixture.manager.changeParamStatus("alpha", true);
  data->resize(&fixture.manager);
  BOOST_CHECK_EQUAL(data->params->np, 6u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "alpha"), 0u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "zeta"), 2u);

  fixture.manager.changeParamStatus("middle", true);
  fixture.manager.changeParamStatus("zeta", false);
  data->resize(&fixture.manager);
  copied_data.resize(&fixture.manager);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "alpha"), 0u);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(*data, "middle"), 2u);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(*data, "zeta"),
                    std::exception);
  BOOST_CHECK_EQUAL(ParameterDataAccess::getActiveOffset(copied_data, "middle"),
                    2u);
  BOOST_CHECK_THROW(ParameterDataAccess::getActiveOffset(copied_data, "zeta"),
                    std::exception);
  BOOST_CHECK(!data->action_params.at("alpha")->active);
  BOOST_CHECK(data->action_params.at("middle")->active);
  fixture.manager.changeParamStatus("middle", false);
  fixture.manager.changeParamStatus("zeta", true);
  data->resize(&fixture.manager);

  fixture.manager.removeParam("middle");
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 6u);
  BOOST_CHECK_EQUAL(fixture.manager.get_inactive_set().size(), 1u);
  BOOST_CHECK_THROW(data->resize(&fixture.manager), std::exception);
  BOOST_CHECK_THROW(fixture.manager.update(data, VectorXs::Zero(6)),
                    std::exception);

  const std::shared_ptr<typename Manager::ParameterDataManager> fresh =
      fixture.manager.createData();
  fixture.manager.removeParam("beta");
  BOOST_CHECK_EQUAL(fixture.manager.get_np_dynamics(), 2u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 5u);
  BOOST_CHECK_THROW(fresh->resize(&fixture.manager), std::exception);

  Fixture added_fixture;
  const std::shared_ptr<typename Manager::ParameterDataManager> before_add =
      added_fixture.manager.createData();
  added_fixture.manager.addParam(
      "aardvark", std::make_shared<typename Fixture::ActionParams>(
                      added_fixture.state, 1, Scalar(5)));
  BOOST_CHECK_THROW(before_add->resize(&added_fixture.manager), std::exception);
  BOOST_CHECK_THROW(
      added_fixture.manager.update(
          before_add, VectorXs::Zero(added_fixture.manager.get_np())),
      std::exception);
}

template <typename Scalar>
void test_null_state_model_and_stale_data_rejection() {
  typedef ParameterManagerFixtureTpl<Scalar> Fixture;
  typedef typename Fixture::Manager Manager;
  typedef typename Manager::VectorXs VectorXs;
  typedef crocoddyl::StateVectorTpl<Scalar> State;

  BOOST_CHECK_THROW(Manager(std::shared_ptr<State>()), std::exception);
  BOOST_CHECK_THROW(
      typename Manager::ParameterItem(
          "null", std::shared_ptr<crocoddyl::ParamsAbstractTpl<Scalar> >()),
      std::exception);
  BOOST_CHECK_THROW(typename Manager::ParameterDataManager(
                        static_cast<const Manager*>(nullptr)),
                    std::exception);

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Manager empty(state);
  BOOST_CHECK_EQUAL(empty.get_np(), 0u);
  BOOST_CHECK(empty.get_action_params().empty());
  BOOST_CHECK(empty.get_dynamics_params().empty());
  BOOST_CHECK(empty.get_active_set().empty());
  BOOST_CHECK(empty.get_inactive_set().empty());
  BOOST_CHECK(empty.createData()->parameter_data != nullptr);

  Manager action_only(state);
  action_only.addParam(
      "action",
      std::make_shared<typename Fixture::ActionParams>(state, 2, Scalar(1)));
  BOOST_CHECK_EQUAL(action_only.get_np_action(), 2u);
  BOOST_CHECK_EQUAL(action_only.get_np_dynamics(), 0u);
  Manager dynamics_only(state);
  dynamics_only.addParam(
      "dynamics",
      std::make_shared<typename Fixture::DynamicsParams>(state, 3, Scalar(1)));
  BOOST_CHECK_EQUAL(dynamics_only.get_np_action(), 0u);
  BOOST_CHECK_EQUAL(dynamics_only.get_np_dynamics(), 3u);

  Fixture fixture;
  std::shared_ptr<typename Fixture::ActionParams> null_action;
  std::shared_ptr<typename Fixture::DynamicsParams> null_dynamics;
  BOOST_CHECK_THROW(fixture.manager.addParam("null_action", null_action),
                    std::exception);
  BOOST_CHECK_THROW(fixture.manager.addParam("null_dynamics", null_dynamics),
                    std::exception);
  const std::shared_ptr<State> other_state = std::make_shared<State>(5);
  BOOST_CHECK_THROW(
      fixture.manager.addParam("wrong_state",
                               std::make_shared<typename Fixture::ActionParams>(
                                   other_state, 1, Scalar(1))),
      std::exception);

  const VectorXs p = VectorXs::Zero(fixture.manager.get_np());
  std::shared_ptr<typename Manager::ParameterDataManager> data =
      fixture.manager.createData();
  data->parameter_data = nullptr;
  BOOST_CHECK_THROW(fixture.manager.update(data, p), std::exception);

  data = fixture.manager.createData();
  data->action_params.erase("alpha");
  BOOST_CHECK_THROW(fixture.manager.update(data, p), std::exception);
  BOOST_CHECK_THROW(data->resize(static_cast<const Manager*>(nullptr)),
                    std::exception);

  data = fixture.manager.createData();
  data->action_params.at("alpha").reset();
  BOOST_CHECK_THROW(fixture.manager.update(data, p), std::exception);
  BOOST_CHECK_THROW(data->setZero(), std::exception);

  data = fixture.manager.createData();
  data->action_params.at("alpha")->resize(3, 0);
  BOOST_CHECK_THROW(fixture.manager.update(data, p), std::exception);

  data = fixture.manager.createData();
  data->params->p.resize(fixture.manager.get_np() + 1);
  BOOST_CHECK_THROW(fixture.manager.update(data, p), std::exception);

  data = fixture.manager.createData();
  data->params.reset();
  BOOST_CHECK_THROW(fixture.manager.update(data, p), std::exception);
  BOOST_CHECK_THROW(data->resize(&fixture.manager), std::exception);
  BOOST_CHECK_THROW(data->setZero(), std::exception);
}

template <typename Scalar>
void test_manager_copy_and_nonempty_scalar_cast() {
  typedef ParameterManagerFixtureTpl<Scalar> Fixture;
  typedef typename Fixture::Manager Manager;
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type OtherScalar;
  typedef ActionParamsProbeTpl<OtherScalar> OtherActionParams;
  typedef DynamicsParamsProbeTpl<OtherScalar> OtherDynamicsParams;

  Fixture fixture;
  Manager copied(fixture.manager);
  BOOST_CHECK_EQUAL(copied.get_np(), fixture.manager.get_np());
  BOOST_CHECK_EQUAL(copied.get_np_action(), fixture.manager.get_np_action());
  BOOST_CHECK_EQUAL(copied.get_np_dynamics(),
                    fixture.manager.get_np_dynamics());
  BOOST_CHECK(copied.get_action_params().at("alpha") !=
              fixture.manager.get_action_params().at("alpha"));
  BOOST_CHECK(copied.get_action_params().at("alpha")->get_param() ==
              fixture.manager.get_action_params().at("alpha")->get_param());
  copied.changeParamStatus("alpha", false);
  BOOST_CHECK(!copied.getParamStatus("alpha"));
  BOOST_CHECK(fixture.manager.getParamStatus("alpha"));
  BOOST_CHECK_EQUAL(copied.get_np(), 4u);
  BOOST_CHECK_EQUAL(fixture.manager.get_np(), 6u);

  const crocoddyl::ParameterManagerTpl<OtherScalar> casted =
      fixture.manager.template cast<OtherScalar>();
  BOOST_CHECK_EQUAL(casted.get_np(), fixture.manager.get_np());
  BOOST_CHECK_EQUAL(casted.get_np_action(), fixture.manager.get_np_action());
  BOOST_CHECK_EQUAL(casted.get_np_dynamics(),
                    fixture.manager.get_np_dynamics());
  BOOST_CHECK(casted.get_active_set() == fixture.manager.get_active_set());
  BOOST_CHECK(casted.get_inactive_set() == fixture.manager.get_inactive_set());
  BOOST_CHECK_EQUAL(casted.get_action_params().begin()->first, "alpha");
  BOOST_CHECK_EQUAL(casted.get_dynamics_params().begin()->first, "beta");
  const std::shared_ptr<OtherActionParams> action =
      std::dynamic_pointer_cast<OtherActionParams>(
          casted.get_action_params().at("alpha")->get_param());
  const std::shared_ptr<OtherDynamicsParams> dynamics =
      std::dynamic_pointer_cast<OtherDynamicsParams>(
          casted.get_dynamics_params().at("zulu")->get_param());
  BOOST_REQUIRE(action != nullptr);
  BOOST_REQUIRE(dynamics != nullptr);
  BOOST_CHECK_EQUAL(action->scale, OtherScalar(10));
  BOOST_CHECK_EQUAL(dynamics->scale, OtherScalar(50));
  BOOST_CHECK(action->get_lb().isApprox(
      fixture.action_alpha->get_lb().template cast<OtherScalar>()));
  BOOST_CHECK(dynamics->get_ub().isApprox(
      fixture.dynamics_zulu->get_ub().template cast<OtherScalar>()));
}

template <typename Scalar>
void test_parameter_manager_no_allocation() {
  typedef ParameterManagerFixtureTpl<Scalar> Fixture;
  typedef typename Fixture::Manager Manager;
  typedef typename Manager::VectorXs VectorXs;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModel;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> DynamicsData;

  Fixture fixture;
  const std::shared_ptr<typename Manager::ParameterDataManager> data =
      fixture.manager.createData();
  const std::size_t nu = 2;
  ActionModel action_model(4, nu);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > action_data =
      action_model.createData();
  DynamicsDataModelStubTpl<Scalar> dynamics_model(fixture.state, nu);
  const std::shared_ptr<DynamicsData> dynamics_data =
      std::make_shared<DynamicsData>(&dynamics_model);
  const VectorXs p = VectorXs::LinSpaced(6, Scalar(1), Scalar(6));
  const VectorXs x = VectorXs::LinSpaced(4, Scalar(0.1), Scalar(0.4));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.5), Scalar(0.6));
  typename Manager::MatrixXs dx_dp(fixture.state->get_ndx(),
                                   fixture.manager.get_np_action());
  typename Manager::MatrixXs dtau_dp(fixture.state->get_nv(),
                                     fixture.manager.get_np_dynamics());
  fixture.manager.update(data, p);
  fixture.manager.calcDiff_action(data, action_data, dx_dp, x, u);
  fixture.manager.calcDiff_dynamics(data, dynamics_data, dtau_dp, x, u);
  data->setZero();
  data->resize(&fixture.manager);
  const Scalar* const p_ptr = data->params->p.data();
  const Scalar* const dx_dp_ptr = dx_dp.data();
  const Scalar* const dtau_dp_ptr = dtau_dp.data();

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      fixture.manager.update(data, p);
      fixture.manager.calcDiff_action(data, action_data, dx_dp, x, u);
      fixture.manager.calcDiff_dynamics(data, dynamics_data, dtau_dp, x, u);
      data->setZero();
      data->resize(&fixture.manager);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  BOOST_CHECK_EQUAL(data->params->p.data(), p_ptr);
  BOOST_CHECK_EQUAL(dx_dp.data(), dx_dp_ptr);
  BOOST_CHECK_EQUAL(dtau_dp.data(), dtau_dp_ptr);
}

template <typename Scalar>
void test_shared_parameter_context_with_parallel_node_workspaces() {
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModel;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> DynamicsData;
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef typename Manager::MatrixXs MatrixXs;
  typedef typename Manager::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Manager manager(state);
  manager.addParam(
      "node",
      std::make_shared<NodeDependentActionParamsTpl<Scalar> >(state, 2));
  manager.addParam(
      "node_dynamics",
      std::make_shared<NodeDependentDynamicsParamsTpl<Scalar> >(state, 2));
  const std::shared_ptr<typename Manager::ParameterDataManager> params_data =
      manager.createData();
  VectorXs p(4);
  p << Scalar(0.2), Scalar(0.6), Scalar(1.2), Scalar(1.6);
  manager.update(params_data, p);

  ActionModel action(4, 1);
  std::array<std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> >, 2>
      action_data = {{action.createData(), action.createData()}};
  std::array<VectorXs, 2> x = {
      {VectorXs::Constant(4, Scalar(1.)), VectorXs::Constant(4, Scalar(3.))}};
  const VectorXs u = VectorXs::Constant(1, Scalar(0.5));
  DynamicsDataModelStubTpl<Scalar> dynamics_model(state, 1);
  std::array<std::shared_ptr<DynamicsData>, 2> dynamics_data = {
      {std::make_shared<DynamicsData>(&dynamics_model),
       std::make_shared<DynamicsData>(&dynamics_model)}};
  dynamics_data[0]->vdot.setConstant(Scalar(2.));
  dynamics_data[1]->vdot.setConstant(Scalar(4.));
  std::array<MatrixXs, 2> dx_dp = {
      {MatrixXs(state->get_ndx(), 2), MatrixXs(state->get_ndx(), 2)}};
  std::array<MatrixXs, 2> dtau_dp = {
      {MatrixXs(state->get_nv(), 2), MatrixXs(state->get_nv(), 2)}};

  for (std::size_t repeat = 0; repeat < 100; ++repeat) {
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(2)
#endif
    for (std::size_t i = 0; i < 2; ++i) {
      manager.calcDiff_action(params_data, action_data[i], dx_dp[i], x[i], u);
      manager.calcDiff_dynamics(params_data, dynamics_data[i], dtau_dp[i], x[i],
                                u);
    }
  }

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      BOOST_CHECK(
          dx_dp[i]
              .col(static_cast<Eigen::Index>(j))
              .isConstant(x[i][0] + u[0] + p[static_cast<Eigen::Index>(j)]));
      BOOST_CHECK(dtau_dp[i]
                      .col(static_cast<Eigen::Index>(j))
                      .isConstant(x[i][0] + dynamics_data[i]->vdot[0] + u[0] +
                                  p[static_cast<Eigen::Index>(2 + j)]));
    }
  }
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_parameter_manager");
  ts->add(BOOST_TEST_CASE(&test_order_update_derivatives_and_data<double>));
  ts->add(BOOST_TEST_CASE(&test_order_update_derivatives_and_data<float>));
  ts->add(BOOST_TEST_CASE(
      &test_duplicate_missing_status_remove_and_resize<double>));
  ts->add(
      BOOST_TEST_CASE(&test_duplicate_missing_status_remove_and_resize<float>));
  ts->add(
      BOOST_TEST_CASE(&test_null_state_model_and_stale_data_rejection<double>));
  ts->add(
      BOOST_TEST_CASE(&test_null_state_model_and_stale_data_rejection<float>));
  ts->add(BOOST_TEST_CASE(&test_manager_copy_and_nonempty_scalar_cast<double>));
  ts->add(BOOST_TEST_CASE(&test_manager_copy_and_nonempty_scalar_cast<float>));
  ts->add(BOOST_TEST_CASE(&test_parameter_manager_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_parameter_manager_no_allocation<float>));
  ts->add(BOOST_TEST_CASE(
      &test_shared_parameter_context_with_parallel_node_workspaces<double>));
  ts->add(BOOST_TEST_CASE(
      &test_shared_parameter_context_with_parallel_node_workspaces<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
