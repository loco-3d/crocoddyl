///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename _Scalar>
class DynamicsProbeTpl : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase, DynamicsProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  using Base::calc;
  using Base::calcDiff;
  using Base::calcDiff_xu;
  using Base::createData;

  DynamicsProbeTpl(std::shared_ptr<StateAbstract> state,
                   const crocoddyl::DynamicsType type =
                       crocoddyl::DynamicsType::ContinuousControl,
                   const std::size_t np = 0, const std::size_t nu = 0,
                   const std::size_t ng = 0, const std::size_t nh = 0)
      : Base(state, type, np, nu, ng, nh),
        calc_calls(0),
        xu_calls(0),
        p_calls(0) {}

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    const Scalar value = x.sum() + u.sum();
    data->vdot.setConstant(value);
    data->dissipative_P.setConstant(u.squaredNorm());
    data->h.setConstant(value);
    data->g.setConstant(-value);
    ++calc_calls;
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setConstant(Scalar(1));
    data->Fu.setIdentity();
    data->dP_dv.setConstant(Scalar(2));
    data->Hx.setConstant(Scalar(3));
    data->Hu.setConstant(Scalar(4));
    data->Gx.setConstant(Scalar(5));
    data->Gu.setConstant(Scalar(6));
    ++xu_calls;
  }

  void calcDiff_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                  const Eigen::Ref<const VectorXs>&,
                  const Eigen::Ref<const VectorXs>&) override {
    data->Fp.setConstant(Scalar(7));
    data->dP_dp.setConstant(Scalar(8));
    data->Hp.setConstant(Scalar(9));
    data->Gp.setConstant(Scalar(10));
    ++p_calls;
  }

  template <typename NewScalar>
  DynamicsProbeTpl<NewScalar> cast() const {
    DynamicsProbeTpl<NewScalar> model(
        this->get_state()->template cast<NewScalar>(), this->get_dyn_type(),
        this->get_np(), this->get_nu(), this->get_ng(), this->get_nh());
    model.set_p_lb(this->get_p_lb().template cast<NewScalar>());
    model.set_p_ub(this->get_p_ub().template cast<NewScalar>());
    model.update_tau(this->get_tau_meas().template cast<NewScalar>());
    return model;
  }

  std::size_t calc_calls;
  std::size_t xu_calls;
  std::size_t p_calls;
};

template <typename _Scalar>
class DynamicsQuasiStaticProbeTpl : public DynamicsProbeTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DynamicsProbeTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  using Base::calc;
  using Base::calcDiff_xu;

  explicit DynamicsQuasiStaticProbeTpl(std::shared_ptr<StateAbstract> state)
      : Base(state, crocoddyl::DynamicsType::ContinuousControl, 0, 2) {
    this->calc_calls = 0;
    this->xu_calls = 0;
  }

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    data->vdot = u.array().cube().matrix() + u + x.head(2);
    ++this->calc_calls;
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>& u) override {
    data->Fu.setZero();
    data->Fu.diagonal().array() = Scalar(1) + Scalar(3) * u.array().square();
    ++this->xu_calls;
  }
};

template <typename Scalar>
void test_model_contracts_running_terminal_and_bounds() {
  typedef DynamicsProbeTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> Data;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef typename Model::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Model model(state, crocoddyl::DynamicsType::ContinuousControl, 3, 2, 2, 1);
  BOOST_CHECK(model.get_state() == state);
  BOOST_CHECK_EQUAL(model.get_np(), 3u);
  BOOST_CHECK_EQUAL(model.get_nu(), 2u);
  BOOST_CHECK_EQUAL(model.get_ng(), 2u);
  BOOST_CHECK_EQUAL(model.get_nh(), 1u);
  BOOST_CHECK(model.get_dyn_type() ==
              crocoddyl::DynamicsType::ContinuousControl);
  BOOST_CHECK(model.get_tau_meas().isZero());
  BOOST_CHECK(model.get_p_lb().array().isInf().all());
  BOOST_CHECK((model.get_p_lb().array() < Scalar(0)).all());
  BOOST_CHECK(model.get_p_ub().array().isInf().all());
  BOOST_CHECK((model.get_p_ub().array() > Scalar(0)).all());

  const VectorXs lb = VectorXs::LinSpaced(3, Scalar(-3), Scalar(-1));
  const VectorXs ub = VectorXs::LinSpaced(3, Scalar(1), Scalar(3));
  const VectorXs tau = VectorXs::LinSpaced(2, Scalar(0.2), Scalar(0.4));
  model.set_p_lb(lb);
  model.set_p_ub(ub);
  model.update_tau(tau);
  BOOST_CHECK(model.get_p_lb().isApprox(lb));
  BOOST_CHECK(model.get_p_ub().isApprox(ub));
  BOOST_CHECK(model.get_tau_meas().isApprox(tau));
  BOOST_CHECK_THROW(model.set_p_lb(VectorXs::Zero(4)), std::exception);
  BOOST_CHECK_THROW(model.set_p_ub(VectorXs::Zero(4)), std::exception);
  BOOST_CHECK_THROW(model.update_tau(VectorXs::Zero(3)), std::exception);

  const std::shared_ptr<Data> data = model.createData();
  BOOST_REQUIRE(data != nullptr);
  std::shared_ptr<typename Model::ParameterDataManager> params_data;
  const std::shared_ptr<Data> forwarded_data = model.createData(params_data);
  BOOST_REQUIRE(forwarded_data != nullptr);
  BOOST_CHECK_EQUAL(forwarded_data->Fp.cols(), model.get_np());
  std::shared_ptr<typename Model::ParameterManager> params;
  BOOST_CHECK_THROW(model.set_params(forwarded_data, params),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model.update_p(forwarded_data, VectorXs::Zero(3)),
                    crocoddyl::Exception);
  BOOST_CHECK(!model.checkData(data));
  const VectorXs x = VectorXs::LinSpaced(4, Scalar(0.1), Scalar(0.4));
  const VectorXs u = VectorXs::LinSpaced(2, Scalar(0.5), Scalar(0.6));
  model.calc(data, x, u);
  BOOST_CHECK_EQUAL(model.calc_calls, 1u);
  BOOST_CHECK(data->vdot.isConstant(x.sum() + u.sum()));
  BOOST_CHECK(data->h.isConstant(x.sum() + u.sum()));
  BOOST_CHECK(data->g.isConstant(-x.sum() - u.sum()));

  model.calc(data, x);
  BOOST_CHECK_EQUAL(model.calc_calls, 2u);
  BOOST_CHECK(data->vdot.isConstant(x.sum()));

  model.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(model.xu_calls, 1u);
  BOOST_CHECK_EQUAL(model.p_calls, 1u);
  BOOST_CHECK(data->Fx.isOnes());
  BOOST_CHECK(data->Fu.isIdentity());
  BOOST_CHECK(data->Fp.isConstant(Scalar(7)));
  BOOST_CHECK(data->dP_dv.isConstant(Scalar(2)));
  BOOST_CHECK(data->dP_dp.isConstant(Scalar(8)));
  BOOST_CHECK(data->Hx.isConstant(Scalar(3)));
  BOOST_CHECK(data->Hu.isConstant(Scalar(4)));
  BOOST_CHECK(data->Hp.isConstant(Scalar(9)));
  BOOST_CHECK(data->Gx.isConstant(Scalar(5)));
  BOOST_CHECK(data->Gu.isConstant(Scalar(6)));
  BOOST_CHECK(data->Gp.isConstant(Scalar(10)));

  model.calcDiff(data, x);
  BOOST_CHECK_EQUAL(model.xu_calls, 2u);
  BOOST_CHECK_EQUAL(model.p_calls, 1u);
  BOOST_CHECK_THROW(model.calcDiff(data, VectorXs::Zero(5), u), std::exception);
  BOOST_CHECK_THROW(model.calcDiff(data, x, VectorXs::Zero(3)), std::exception);
  BOOST_CHECK_THROW(model.calcDiff(data, VectorXs::Zero(5)), std::exception);

  Model parameter_free(state, crocoddyl::DynamicsType::ContinuousEstimation, 0,
                       2, 0, 0);
  const std::shared_ptr<Data> parameter_free_data = parameter_free.createData();
  parameter_free.calcDiff(parameter_free_data, x, u);
  BOOST_CHECK_EQUAL(parameter_free.xu_calls, 1u);
  BOOST_CHECK_EQUAL(parameter_free.p_calls, 0u);
  parameter_free.set_dyn_type(crocoddyl::DynamicsType::ContinuousControl);
  BOOST_CHECK(parameter_free.get_dyn_type() ==
              crocoddyl::DynamicsType::ContinuousControl);

  VectorXs ustatic = VectorXs::Zero(2);
  model.quasiStatic(data, ustatic, VectorXs::Zero(4), 10, Scalar(1e-9));
  BOOST_CHECK(ustatic.isZero());
  VectorXs invalid_u = VectorXs::Zero(3);
  BOOST_CHECK_THROW(model.quasiStatic(data, invalid_u, VectorXs::Zero(4)),
                    std::exception);

  std::ostringstream stream;
  stream << model;
  BOOST_CHECK(!stream.str().empty());
}

template <typename Scalar>
void test_nontrivial_quasi_static_newton_iteration() {
  typedef DynamicsQuasiStaticProbeTpl<Scalar> Model;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef typename Model::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Model model(state);
  const std::shared_ptr<typename Model::DynamicsDataAbstract> data =
      model.createData();
  VectorXs x = VectorXs::Zero(4);
  x.head(2) << Scalar(0.5), Scalar(-0.25);
  VectorXs u = VectorXs::Zero(2);
  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(1e-6) : Scalar(1e-12);

  model.quasiStatic(data, u, x, 100, tol);
  BOOST_CHECK(!u.isZero());
  BOOST_CHECK_GT(model.calc_calls, 1u);
  BOOST_CHECK_GT(model.xu_calls, 1u);
  model.calc(data, x, u);
  BOOST_CHECK_SMALL(data->vdot.norm(), Scalar(10) * tol);
}

template <typename Scalar>
void check_data_layout(const crocoddyl::DynamicsType type,
                       const std::size_t expected_value_rows,
                       const std::size_t expected_jacobian_rows) {
  typedef DynamicsProbeTpl<Scalar> Model;
  typedef crocoddyl::DataCollectorAbstractTpl<Scalar> Collector;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> Data;
  typedef crocoddyl::StateVectorTpl<Scalar> State;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Model model(state, type, 3, 2, 2, 1);
  const std::shared_ptr<Data> data = model.createData();
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(data->shared == nullptr);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->vdot.size()),
                    expected_value_rows);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fx.rows()),
                    expected_jacobian_rows);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fx.cols()), 4u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fu.rows()),
                    expected_jacobian_rows);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fu.cols()), 2u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fp.rows()),
                    expected_jacobian_rows);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Fp.cols()), 3u);
  BOOST_CHECK_EQUAL(data->dissipative_P.rows(), 1);
  BOOST_CHECK_EQUAL(data->dP_dv.rows(), 1);
  BOOST_CHECK_EQUAL(data->dP_dv.cols(), 2);
  BOOST_CHECK_EQUAL(data->dP_dp.rows(), 1);
  BOOST_CHECK_EQUAL(data->dP_dp.cols(), 3);
  BOOST_CHECK_EQUAL(data->h.size(), 1);
  BOOST_CHECK_EQUAL(data->Hx.rows(), 1);
  BOOST_CHECK_EQUAL(data->Hx.cols(), 4);
  BOOST_CHECK_EQUAL(data->Hu.rows(), 1);
  BOOST_CHECK_EQUAL(data->Hu.cols(), 2);
  BOOST_CHECK_EQUAL(data->Hp.rows(), 1);
  BOOST_CHECK_EQUAL(data->Hp.cols(), 3);
  BOOST_CHECK_EQUAL(data->g.size(), 2);
  BOOST_CHECK_EQUAL(data->Gx.rows(), 2);
  BOOST_CHECK_EQUAL(data->Gx.cols(), 4);
  BOOST_CHECK_EQUAL(data->Gu.rows(), 2);
  BOOST_CHECK_EQUAL(data->Gu.cols(), 2);
  BOOST_CHECK_EQUAL(data->Gp.rows(), 2);
  BOOST_CHECK_EQUAL(data->Gp.cols(), 3);
  BOOST_CHECK_EQUAL(data->tmp_ustatic.size(), 2);
  BOOST_CHECK(data->vdot.isZero());
  BOOST_CHECK(data->Fx.isZero());
  BOOST_CHECK(data->Fu.isZero());
  BOOST_CHECK(data->Fp.isZero());
  BOOST_CHECK(data->dissipative_P.isZero());
  BOOST_CHECK(data->dP_dv.isZero());
  BOOST_CHECK(data->dP_dp.isZero());
  BOOST_CHECK(data->h.isZero());
  BOOST_CHECK(data->Hx.isZero());
  BOOST_CHECK(data->Hu.isZero());
  BOOST_CHECK(data->Hp.isZero());
  BOOST_CHECK(data->g.isZero());
  BOOST_CHECK(data->Gx.isZero());
  BOOST_CHECK(data->Gu.isZero());
  BOOST_CHECK(data->Gp.isZero());
  BOOST_CHECK(data->tmp_ustatic.isZero());

  data->vdot.setOnes();
  data->Fx.setOnes();
  data->Fu.setOnes();
  data->Fp.setOnes();
  data->dissipative_P.setOnes();
  data->dP_dv.setOnes();
  data->dP_dp.setOnes();
  data->h.setOnes();
  data->Hx.setOnes();
  data->Hu.setOnes();
  data->Hp.setOnes();
  data->g.setOnes();
  data->Gx.setOnes();
  data->Gu.setOnes();
  data->Gp.setOnes();
  data->tmp_ustatic.setOnes();
  Collector collector;
  data->shared = &collector;
  const Data copied(*data);
  BOOST_CHECK(copied.shared == &collector);
  BOOST_CHECK(copied.vdot.isApprox(data->vdot));
  BOOST_CHECK(copied.Fx.isApprox(data->Fx));
  BOOST_CHECK(copied.Fu.isApprox(data->Fu));
  BOOST_CHECK(copied.Fp.isApprox(data->Fp));
  BOOST_CHECK(copied.dissipative_P.isApprox(data->dissipative_P));
  BOOST_CHECK(copied.dP_dv.isApprox(data->dP_dv));
  BOOST_CHECK(copied.dP_dp.isApprox(data->dP_dp));
  BOOST_CHECK(copied.h.isApprox(data->h));
  BOOST_CHECK(copied.Hx.isApprox(data->Hx));
  BOOST_CHECK(copied.Hu.isApprox(data->Hu));
  BOOST_CHECK(copied.Hp.isApprox(data->Hp));
  BOOST_CHECK(copied.g.isApprox(data->g));
  BOOST_CHECK(copied.Gx.isApprox(data->Gx));
  BOOST_CHECK(copied.Gu.isApprox(data->Gu));
  BOOST_CHECK(copied.Gp.isApprox(data->Gp));
  BOOST_CHECK(copied.tmp_ustatic.isApprox(data->tmp_ustatic));

  data->setZero();
  BOOST_CHECK(data->shared == &collector);
  BOOST_CHECK(data->vdot.isZero());
  BOOST_CHECK(data->Fx.isZero());
  BOOST_CHECK(data->Fu.isZero());
  BOOST_CHECK(data->Fp.isZero());
  BOOST_CHECK(data->dissipative_P.isZero());
  BOOST_CHECK(data->dP_dv.isZero());
  BOOST_CHECK(data->dP_dp.isZero());
  BOOST_CHECK(data->h.isZero());
  BOOST_CHECK(data->Hx.isZero());
  BOOST_CHECK(data->Hu.isZero());
  BOOST_CHECK(data->Hp.isZero());
  BOOST_CHECK(data->g.isZero());
  BOOST_CHECK(data->Gx.isZero());
  BOOST_CHECK(data->Gu.isZero());
  BOOST_CHECK(data->Gp.isZero());
  BOOST_CHECK(data->tmp_ustatic.isZero());
}

template <typename Scalar>
void test_data_layout_initialization_copy_and_zero() {
  check_data_layout<Scalar>(crocoddyl::DynamicsType::ContinuousControl, 2, 2);
  check_data_layout<Scalar>(crocoddyl::DynamicsType::ContinuousEstimation, 2,
                            2);
  check_data_layout<Scalar>(crocoddyl::DynamicsType::DiscreteTime, 4, 4);
}

void test_model_scalar_casts() {
  typedef DynamicsProbeTpl<double> Model;
  typedef crocoddyl::StateVector State;
  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Model model(state, crocoddyl::DynamicsType::ContinuousEstimation, 3, 2, 2, 1);
  model.set_p_lb(Model::VectorXs::Constant(3, -2.));
  model.set_p_ub(Model::VectorXs::Constant(3, 2.));
  model.update_tau(Model::VectorXs::Constant(2, 0.5));

  const crocoddyl::DynamicsModelBase& base = model;
  const std::shared_ptr<crocoddyl::DynamicsModelAbstractTpl<float> > casted =
      base.cast<float>();
  BOOST_REQUIRE(casted != nullptr);
  BOOST_CHECK_EQUAL(casted->get_np(), 3u);
  BOOST_CHECK_EQUAL(casted->get_nu(), 2u);
  BOOST_CHECK_EQUAL(casted->get_ng(), 2u);
  BOOST_CHECK_EQUAL(casted->get_nh(), 1u);
  BOOST_CHECK(casted->get_dyn_type() ==
              crocoddyl::DynamicsType::ContinuousEstimation);
  BOOST_CHECK(casted->get_p_lb().isConstant(-2.f));
  BOOST_CHECK(casted->get_p_ub().isConstant(2.f));
  BOOST_CHECK(casted->get_tau_meas().isConstant(0.5f));

  const std::shared_ptr<crocoddyl::DynamicsModelAbstract> roundtrip =
      casted->cast<double>();
  BOOST_REQUIRE(roundtrip != nullptr);
  BOOST_CHECK_EQUAL(roundtrip->get_np(), 3u);
  BOOST_CHECK(roundtrip->get_p_lb().isConstant(-2.));
  BOOST_CHECK(roundtrip->get_tau_meas().isConstant(0.5));
}

template <typename Scalar>
void test_data_set_zero_no_allocation() {
  typedef DynamicsProbeTpl<Scalar> Model;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> Data;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Model model(state, crocoddyl::DynamicsType::ContinuousControl, 3, 2, 2, 1);
  const std::shared_ptr<Data> data = model.createData();
  data->vdot.setOnes();
  data->Fx.setOnes();
  data->Fp.setOnes();
  data->dP_dp.setOnes();

  const bool was_malloc_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      data->setZero();
    }
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(was_malloc_allowed);
    throw;
  }
  BOOST_CHECK(data->vdot.isZero());
  BOOST_CHECK(data->Fx.isZero());
  BOOST_CHECK(data->Fp.isZero());
  BOOST_CHECK(data->dP_dp.isZero());
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_dynamics");
  ts->add(BOOST_TEST_CASE(
      &test_model_contracts_running_terminal_and_bounds<double>));
  ts->add(BOOST_TEST_CASE(
      &test_model_contracts_running_terminal_and_bounds<float>));
  ts->add(
      BOOST_TEST_CASE(&test_nontrivial_quasi_static_newton_iteration<double>));
  ts->add(
      BOOST_TEST_CASE(&test_nontrivial_quasi_static_newton_iteration<float>));
  ts->add(
      BOOST_TEST_CASE(&test_data_layout_initialization_copy_and_zero<double>));
  ts->add(
      BOOST_TEST_CASE(&test_data_layout_initialization_copy_and_zero<float>));
  ts->add(BOOST_TEST_CASE(&test_model_scalar_casts));
  ts->add(BOOST_TEST_CASE(&test_data_set_zero_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_data_set_zero_no_allocation<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
