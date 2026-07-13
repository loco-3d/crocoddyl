///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/params-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

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

  ActionParamsProbeTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t np)
      : Base(state, np), sensitivity_calls(0) {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    data->p = p;
  }

  void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>&,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override {
    params->dx_dp.setConstant(x.sum() + u.sum());
    ++sensitivity_calls;
  }

  template <typename NewScalar>
  ActionParamsProbeTpl<NewScalar> cast() const {
    ActionParamsProbeTpl<NewScalar> model(
        this->get_state()->template cast<NewScalar>(), this->get_np());
    model.set_lb(this->get_lb().template cast<NewScalar>());
    model.set_ub(this->get_ub().template cast<NewScalar>());
    return model;
  }

  std::size_t sensitivity_calls;
};

template <typename Scalar>
void test_params_model_defaults_bounds_data_and_copy() {
  typedef crocoddyl::ParamsAbstractTpl<Scalar> ParamsModel;
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef typename ParamsModel::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  ParamsModel empty(state);
  BOOST_CHECK_EQUAL(empty.get_np(), 0u);
  BOOST_CHECK_EQUAL(empty.zero().size(), 0);

  const std::size_t np = 3;
  ParamsModel model(state, np);
  BOOST_CHECK(model.get_state() == state);
  BOOST_CHECK_EQUAL(model.get_np(), np);
  BOOST_CHECK_EQUAL(model.get_lb().size(), static_cast<Eigen::Index>(np));
  BOOST_CHECK_EQUAL(model.get_ub().size(), static_cast<Eigen::Index>(np));
  BOOST_CHECK(model.get_lb().isConstant(-std::numeric_limits<Scalar>::max()));
  BOOST_CHECK(model.get_ub().isConstant(std::numeric_limits<Scalar>::max()));
  BOOST_CHECK(model.zero().isZero());
  const VectorXs random = model.rand();
  BOOST_CHECK_EQUAL(random.size(), static_cast<Eigen::Index>(np));
  BOOST_CHECK((random.array() >= Scalar(0.)).all());
  BOOST_CHECK((random.array() <= Scalar(1.)).all());

  const VectorXs lb = VectorXs::LinSpaced(np, Scalar(-3.), Scalar(-1.));
  const VectorXs ub = VectorXs::LinSpaced(np, Scalar(1.), Scalar(3.));
  model.set_lb(lb);
  model.set_ub(ub);
  BOOST_CHECK(model.get_lb().isApprox(lb));
  BOOST_CHECK(model.get_ub().isApprox(ub));
  BOOST_CHECK_THROW(model.set_lb(VectorXs::Zero(np + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_ub(VectorXs::Zero(np + 1)), std::exception);

  const std::shared_ptr<ParamsData> data = model.createData();
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK_EQUAL(data->np, np);
  BOOST_CHECK_EQUAL(data->np_action, np);
  BOOST_CHECK_EQUAL(data->np_dynamics, 0u);
  BOOST_CHECK(model.checkData(data));
  BOOST_CHECK(!model.checkData(std::make_shared<ParamsData>(state, np + 1, 0)));
  BOOST_CHECK(!model.checkData(std::shared_ptr<ParamsData>()));

  data->p.setOnes();
  model.update(data, VectorXs::Zero(np));
  BOOST_CHECK(data->p.isOnes());

  ParamsModel copied(model);
  BOOST_CHECK(copied.get_state() == state);
  BOOST_CHECK_EQUAL(copied.get_np(), np);
  BOOST_CHECK(copied.get_lb().isApprox(lb));
  BOOST_CHECK(copied.get_ub().isApprox(ub));
  std::ostringstream stream;
  stream << copied;
  BOOST_CHECK(!stream.str().empty());
  BOOST_CHECK_THROW(model.template cast<float>(), std::exception);
}

template <typename Scalar>
void test_action_params_model_update_sensitivity_data_and_copy() {
  typedef ActionParamsProbeTpl<Scalar> ParamsModel;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModel;
  typedef crocoddyl::ActionModelParamsDataAbstractTpl<Scalar> ActionParamsData;
  typedef crocoddyl::ParamsDataAbstractTpl<Scalar> ParamsData;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef typename ParamsModel::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 2;
  const std::size_t np = 3;
  const std::shared_ptr<State> state = std::make_shared<State>(nx);
  ParamsModel model(state, np);
  const VectorXs lb = VectorXs::Constant(np, Scalar(-2.));
  const VectorXs ub = VectorXs::Constant(np, Scalar(2.));
  model.set_lb(lb);
  model.set_ub(ub);

  const std::shared_ptr<ParamsData> params = model.createData();
  BOOST_REQUIRE(params != nullptr);
  BOOST_CHECK(std::dynamic_pointer_cast<ActionParamsData>(params) != nullptr);
  BOOST_CHECK_EQUAL(params->np, np);
  BOOST_CHECK_EQUAL(params->np_action, np);
  BOOST_CHECK_EQUAL(params->np_dynamics, 0u);
  const VectorXs p = VectorXs::LinSpaced(np, Scalar(0.2), Scalar(0.6));
  model.update(params, p);
  BOOST_CHECK(params->p.isApprox(p));

  ActionModel action(nx, nu);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > action_data =
      action.createData();
  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(0.1), Scalar(0.4));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.5), Scalar(0.6));
  model.computeParamSensitivity(action_data, params, x, u);
  BOOST_CHECK_EQUAL(model.sensitivity_calls, 1u);
  BOOST_CHECK(params->dx_dp.isConstant(x.sum() + u.sum()));

  ParamsModel copied(model);
  BOOST_CHECK_EQUAL(copied.get_np(), np);
  BOOST_CHECK(copied.get_lb().isApprox(lb));
  BOOST_CHECK(copied.get_ub().isApprox(ub));

  const crocoddyl::ParamsModelBase& base_model = model;
  const std::shared_ptr<crocoddyl::ParamsAbstractTpl<float> > casted =
      base_model.template cast<float>();
  BOOST_REQUIRE(casted != nullptr);
  BOOST_CHECK_EQUAL(casted->get_np(), np);
  BOOST_CHECK(casted->get_lb().isApprox(lb.template cast<float>()));
  BOOST_CHECK(casted->get_ub().isApprox(ub.template cast<float>()));
  const std::shared_ptr<crocoddyl::ParamsAbstractTpl<double> > roundtrip =
      casted->template cast<double>();
  BOOST_REQUIRE(roundtrip != nullptr);
  BOOST_CHECK_EQUAL(roundtrip->get_np(), np);
  BOOST_CHECK(roundtrip->get_lb().isApprox(lb.template cast<double>()));
  BOOST_CHECK(roundtrip->get_ub().isApprox(ub.template cast<double>()));
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_params");
  ts->add(BOOST_TEST_CASE(
      &test_params_model_defaults_bounds_data_and_copy<double>));
  ts->add(
      BOOST_TEST_CASE(&test_params_model_defaults_bounds_data_and_copy<float>));
  ts->add(BOOST_TEST_CASE(
      &test_action_params_model_update_sensitivity_data_and_copy<double>));
  ts->add(BOOST_TEST_CASE(
      &test_action_params_model_update_sensitivity_data_and_copy<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
