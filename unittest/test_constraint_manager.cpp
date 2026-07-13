///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/constraint.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

template <typename _Scalar>
class ConstraintManagerParameterResidualTpl
    : public crocoddyl::ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ResidualModelBase,
                         ConstraintManagerParameterResidualTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ResidualModelAbstractTpl<Scalar> Base;
  typedef typename Base::ResidualDataAbstract ResidualDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  ConstraintManagerParameterResidualTpl(std::shared_ptr<StateAbstract> state,
                                        const std::size_t nr,
                                        const std::size_t nu,
                                        const std::size_t np)
      : Base(state, nr, nu, true, true, nu != 0, np) {}

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>&,
            const Eigen::Ref<const VectorXs>&) override {
    for (Eigen::Index i = 0; i < data->r.size(); ++i) {
      data->r[i] = Scalar(i + 1);
    }
  }

  void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    for (Eigen::Index i = 0; i < data->Rx.rows(); ++i) {
      for (Eigen::Index j = 0; j < data->Rx.cols(); ++j) {
        data->Rx(i, j) = Scalar(10 * (i + 1) + j + 1);
      }
    }
    for (Eigen::Index i = 0; i < data->Ru.rows(); ++i) {
      for (Eigen::Index j = 0; j < data->Ru.cols(); ++j) {
        data->Ru(i, j) = Scalar(20 * (i + 1) + j + 1);
      }
    }
    for (Eigen::Index i = 0; i < data->Rp.rows(); ++i) {
      for (Eigen::Index j = 0; j < data->Rp.cols(); ++j) {
        data->Rp(i, j) = Scalar((i + 1) * (j + 1));
      }
    }
  }

  template <typename NewScalar>
  ConstraintManagerParameterResidualTpl<NewScalar> cast() const {
    return ConstraintManagerParameterResidualTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>(), this->get_nr(),
        this->get_nu(), this->get_np());
  }
};

typedef ConstraintManagerParameterResidualTpl<double>
    ConstraintManagerParameterResidual;

template <typename _Scalar>
class ConstraintManagerActionProbeTpl
    : public crocoddyl::ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ActionModelBase,
                         ConstraintManagerActionProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionModelAbstractTpl<Scalar> Base;
  typedef crocoddyl::ActionDataAbstractTpl<Scalar> Data;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  ConstraintManagerActionProbeTpl(std::shared_ptr<StateAbstract> state,
                                  const std::size_t nu, const std::size_t ng,
                                  const std::size_t nh, const std::size_t ng_T,
                                  const std::size_t nh_T, const std::size_t np)
      : Base(state, nu, 0, ng, nh, ng_T, nh_T, np) {}

  void calc(const std::shared_ptr<Data>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>&) override {
    data->xnext = x;
  }

  void calcDiff(const std::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {}

  template <typename NewScalar>
  ConstraintManagerActionProbeTpl<NewScalar> cast() const {
    return ConstraintManagerActionProbeTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>(), this->get_nu(),
        this->get_ng(), this->get_nh(), this->get_ng_T(), this->get_nh_T(),
        this->get_np());
  }
};

typedef ConstraintManagerActionProbeTpl<double> ConstraintManagerActionProbe;

void test_constructor(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  const std::shared_ptr<crocoddyl::StateAbstract> state =
      state_factory.create(state_type);
  crocoddyl::ConstraintModelManager model(state);
  crocoddyl::ConstraintModelManager legacy_model(state, state->get_nv());
  crocoddyl::ConstraintModelManager parameter_model(state, state->get_nv(), 3);

  // Run the print function
  std::ostringstream tmp;
  tmp << model;

  // Test the initial size of the map
  BOOST_CHECK(model.get_constraints().size() == 0);
  BOOST_CHECK_EQUAL(model.get_np(), 0u);
  BOOST_CHECK_EQUAL(legacy_model.get_np(), 0u);
  BOOST_CHECK_EQUAL(parameter_model.get_np(), 3u);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ConstraintModelManagerTpl<float> casted_model =
      model.cast<float>();
  BOOST_CHECK(casted_model.get_constraints().size() == 0);
  const std::shared_ptr<crocoddyl::StateAbstractTpl<float>>& casted_state =
      state->cast<float>();
  crocoddyl::ConstraintModelManagerTpl<float> casted_legacy(
      casted_state, casted_state->get_nv());
  crocoddyl::ConstraintModelManagerTpl<float> casted_parameter(
      casted_state, casted_state->get_nv(), 3);
  BOOST_CHECK_EQUAL(casted_legacy.get_np(), 0u);
  BOOST_CHECK_EQUAL(casted_parameter.get_np(), 3u);
#endif
}

void test_parameter_derivative_aggregation() {
  const std::size_t nu = 2;
  const std::size_t np = 3;
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  crocoddyl::ConstraintModelManager model(state, nu);

  const std::shared_ptr<ConstraintManagerParameterResidual> parameter_residual =
      std::make_shared<ConstraintManagerParameterResidual>(state, 2, nu, np);
  model.addConstraint("parameter_equality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, parameter_residual));
  BOOST_CHECK_EQUAL(model.get_np(), np);
  model.addConstraint("parameter_inequality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, parameter_residual, -Eigen::Vector2d::Ones(),
                          Eigen::Vector2d::Ones()));

  const std::shared_ptr<crocoddyl::ResidualModelControl> control_residual =
      std::make_shared<crocoddyl::ResidualModelControl>(state, nu);
  model.addConstraint("control_equality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, control_residual));
  model.addConstraint("control_inequality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, control_residual, -Eigen::VectorXd::Ones(nu),
                          Eigen::VectorXd::Ones(nu)));
  model.addConstraint("inactive_parameter_equality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, parameter_residual),
                      false);
  BOOST_CHECK_EQUAL(model.get_np(), np);

  crocoddyl::ConstraintModelManager mismatch_model(state, nu, np);
  const std::shared_ptr<ConstraintManagerParameterResidual> mismatch_residual =
      std::make_shared<ConstraintManagerParameterResidual>(state, 1, nu,
                                                           np + 1);
  BOOST_CHECK_THROW(mismatch_model.addConstraint(
                        "bad_parameter_size",
                        std::make_shared<crocoddyl::ConstraintModelResidual>(
                            state, mismatch_residual)),
                    crocoddyl::Exception);
  mismatch_model.addConstraint(
      "parameter_free", std::make_shared<crocoddyl::ConstraintModelResidual>(
                            state, control_residual));
  BOOST_CHECK_EQUAL(mismatch_model.get_np(), np);

  crocoddyl::DataCollectorAbstract shared_data;
  const std::shared_ptr<crocoddyl::ConstraintDataManager> data =
      model.createData(&shared_data);

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(nu);
  model.calc(data, x, u);
  data->Gp.setConstant(42.);
  data->Hp.setConstant(42.);
  model.calcDiff(data, x, u);

  Eigen::MatrixXd expected_Gp =
      Eigen::MatrixXd::Zero(model.get_ng(), model.get_np());
  Eigen::MatrixXd expected_Hp =
      Eigen::MatrixXd::Zero(model.get_nh(), model.get_np());
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;
  for (crocoddyl::ConstraintModelManager::ConstraintModelContainer::
           const_iterator it = model.get_constraints().begin();
       it != model.get_constraints().end(); ++it) {
    const std::shared_ptr<crocoddyl::ConstraintItem>& item = it->second;
    if (!item->active) {
      continue;
    }
    const std::shared_ptr<crocoddyl::ConstraintDataAbstract>& item_data =
        data->constraints.at(it->first);
    const std::size_t ng = item->constraint->get_ng();
    const std::size_t nh = item->constraint->get_nh();
    if (item->constraint->get_np() != 0) {
      expected_Gp.block(ng_i, 0, ng, np) = item_data->Gp;
      expected_Hp.block(nh_i, 0, nh, np) = item_data->Hp;
    }
    ng_i += ng;
    nh_i += nh;
  }

  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.rows()), model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.cols()), np);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.rows()), model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.cols()), np);
  BOOST_CHECK(data->Gp.isApprox(expected_Gp, 1e-12));
  BOOST_CHECK(data->Hp.isApprox(expected_Hp, 1e-12));

  crocoddyl::ConstraintDataManager data_copy(*data);
  BOOST_CHECK(data_copy.Gp.isApprox(data->Gp));
  BOOST_CHECK(data_copy.Hp.isApprox(data->Hp));
  data->set_Gp(Eigen::MatrixXd::Ones(model.get_ng(), np));
  data->set_Hp(Eigen::MatrixXd::Ones(model.get_nh(), np));
  BOOST_CHECK_THROW(data->set_Gp(Eigen::MatrixXd::Zero(model.get_ng(), np + 1)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(data->set_Hp(Eigen::MatrixXd::Zero(model.get_nh(), np + 1)),
                    crocoddyl::Exception);

#ifdef NDEBUG
  crocoddyl::ConstraintModelManagerTpl<float> casted_model =
      model.cast<float>();
  crocoddyl::DataCollectorAbstractTpl<float> casted_shared;
  const std::shared_ptr<crocoddyl::ConstraintDataManagerTpl<float>>&
      casted_data = casted_model.createData(&casted_shared);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model.calc(casted_data, x_f, u_f);
  casted_model.calcDiff(casted_data, x_f, u_f);
  BOOST_CHECK_EQUAL(casted_model.get_np(), np);
  BOOST_CHECK_EQUAL(casted_data->Gp.cols(), np);
  BOOST_CHECK_EQUAL(casted_data->Hp.cols(), np);
  BOOST_CHECK(casted_data->Gp.isApprox(expected_Gp.cast<float>(), 1e-6f));
  BOOST_CHECK(casted_data->Hp.isApprox(expected_Hp.cast<float>(), 1e-6f));
#endif
}

void test_parameter_derivative_calcDiff_no_malloc() {
  const std::size_t nu = 2;
  const std::size_t np = 3;
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  crocoddyl::ConstraintModelManager model(state, nu, np);
  const std::shared_ptr<ConstraintManagerParameterResidual> residual =
      std::make_shared<ConstraintManagerParameterResidual>(state, 2, nu, np);
  model.addConstraint(
      "parameter_equality",
      std::make_shared<crocoddyl::ConstraintModelResidual>(state, residual));
  model.addConstraint(
      "parameter_inequality",
      std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, residual, -Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones()));

  crocoddyl::DataCollectorAbstract shared_data;
  const std::shared_ptr<crocoddyl::ConstraintDataManager> data =
      model.createData(&shared_data);
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(nu);
  model.calc(data, x, u);
  model.calcDiff(data, x, u);

  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model.calcDiff(data, x, u);
    }
    data->resize(&model, false);
    for (std::size_t i = 0; i < 100; ++i) {
      model.calcDiff(data, x);
    }
    data->resize(&model);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
}

void test_parameter_running_terminal_resize() {
  const std::size_t nu = 2;
  const std::size_t np = 3;
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);

  crocoddyl::ConstraintModelManager model(state, nu, np);
  const std::shared_ptr<ConstraintManagerParameterResidual> parameter_only =
      std::make_shared<ConstraintManagerParameterResidual>(state, 2, nu, np);
  const std::shared_ptr<crocoddyl::ResidualModelAbstract> state_parameter =
      std::make_shared<crocoddyl::ResidualModelAbstract>(state, 2, nu, true,
                                                         true, false, np);
  model.addConstraint("terminal_equality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, state_parameter));
  model.addConstraint("terminal_inequality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, state_parameter, -Eigen::Vector2d::Ones(),
                          Eigen::Vector2d::Ones()));
  model.addConstraint("running_equality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, parameter_only, false));
  model.addConstraint("running_inequality",
                      std::make_shared<crocoddyl::ConstraintModelResidual>(
                          state, parameter_only, -Eigen::Vector2d::Ones(),
                          Eigen::Vector2d::Ones(), false));

  crocoddyl::DataCollectorAbstract shared_data;
  const std::shared_ptr<crocoddyl::ConstraintDataManager> data =
      model.createData(&shared_data);

  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->g.size()), model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gx.rows()), model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gu.rows()), model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.rows()), model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.cols()), model.get_np());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->h.size()), model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hx.rows()), model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hu.rows()), model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.rows()), model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.cols()), model.get_np());
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(nu);
  model.calc(data, x, u);
  data->Gp.setConstant(42.);
  data->Hp.setConstant(42.);
  model.calcDiff(data, x, u);
  BOOST_CHECK(!data->Gp.isZero());
  BOOST_CHECK(!data->Hp.isZero());

  data->resize(&model, false);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->g.size()), model.get_ng_T());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gx.rows()),
                    model.get_ng_T());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.rows()),
                    model.get_ng_T());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->h.size()), model.get_nh_T());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hx.rows()),
                    model.get_nh_T());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.rows()),
                    model.get_nh_T());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->g_internal.size()),
                    model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gx_internal.rows()),
                    model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp_internal.rows()),
                    model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->h_internal.size()),
                    model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hx_internal.rows()),
                    model.get_nh());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp_internal.rows()),
                    model.get_nh());
  model.calc(data, x);
  data->Gp.setConstant(42.);
  data->Hp.setConstant(42.);
  for (crocoddyl::ConstraintModelManager::ConstraintDataContainer::iterator it =
           data->constraints.begin();
       it != data->constraints.end(); ++it) {
    it->second->residual->Rp.setOnes();
  }
  model.calcDiff(data, x);
  BOOST_CHECK(!data->Gp.isZero());
  BOOST_CHECK(!data->Hp.isZero());

  data->resize(&model);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Gp.rows()), model.get_ng());
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Hp.rows()), model.get_nh());
}

template <typename Scalar>
void test_live_activation_resize() {
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> Manager;
  typedef crocoddyl::ConstraintDataManagerTpl<Scalar> Data;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> Constraint;
  typedef ConstraintManagerParameterResidualTpl<Scalar> Residual;
  typedef crocoddyl::DataCollectorAbstractTpl<Scalar> Collector;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::size_t nu = 2;
  const std::size_t np = 3;
  const std::shared_ptr<State> state = std::make_shared<State>(4);
  Manager model(state, nu, np);
  const std::shared_ptr<Residual> active =
      std::make_shared<Residual>(state, 1, nu, np);
  const std::shared_ptr<Residual> inactive =
      std::make_shared<Residual>(state, 2, nu, np);
  model.addConstraint("a_active_inequality",
                      std::make_shared<Constraint>(
                          state, active, VectorXs::Constant(1, Scalar(-1)),
                          VectorXs::Constant(1, Scalar(2))));
  model.addConstraint("b_active_equality",
                      std::make_shared<Constraint>(state, active));
  model.addConstraint("c_inactive_inequality",
                      std::make_shared<Constraint>(
                          state, inactive, VectorXs::Constant(2, Scalar(-3)),
                          VectorXs::Constant(2, Scalar(4))),
                      false);
  model.addConstraint("d_inactive_equality",
                      std::make_shared<Constraint>(state, inactive), false);

  Collector shared;
  const std::shared_ptr<Data> data = model.createData(&shared);
  BOOST_REQUIRE_EQUAL(data->g_internal.size(), 1);
  BOOST_REQUIRE_EQUAL(data->h_internal.size(), 1);
  model.changeConstraintStatus("c_inactive_inequality", true);
  model.changeConstraintStatus("d_inactive_equality", true);

  const VectorXs x = state->rand();
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(-0.2), Scalar(0.3));
  data->resize(&model);
  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(data->g.size(), 3);
  BOOST_CHECK_EQUAL(data->Gx.rows(), 3);
  BOOST_CHECK_EQUAL(data->Gu.rows(), 3);
  BOOST_CHECK_EQUAL(data->Gp.rows(), 3);
  BOOST_CHECK_EQUAL(data->h.size(), 3);
  BOOST_CHECK_EQUAL(data->Hx.rows(), 3);
  BOOST_CHECK_EQUAL(data->Hu.rows(), 3);
  BOOST_CHECK_EQUAL(data->Hp.rows(), 3);
  BOOST_CHECK(
      data->g.head(1).isApprox(data->constraints.at("a_active_inequality")->g));
  BOOST_CHECK(data->g.tail(2).isApprox(
      data->constraints.at("c_inactive_inequality")->g));
  BOOST_CHECK(
      data->h.head(1).isApprox(data->constraints.at("b_active_equality")->h));
  BOOST_CHECK(
      data->h.tail(2).isApprox(data->constraints.at("d_inactive_equality")->h));
  BOOST_CHECK(data->Gx.topRows(1).isApprox(
      data->constraints.at("a_active_inequality")->Gx));
  BOOST_CHECK(data->Gx.bottomRows(2).isApprox(
      data->constraints.at("c_inactive_inequality")->Gx));
  BOOST_CHECK(data->Gu.bottomRows(2).isApprox(
      data->constraints.at("c_inactive_inequality")->Gu));
  BOOST_CHECK(data->Gp.bottomRows(2).isApprox(
      data->constraints.at("c_inactive_inequality")->Gp));
  BOOST_CHECK(data->Hx.bottomRows(2).isApprox(
      data->constraints.at("d_inactive_equality")->Hx));
  BOOST_CHECK(data->Hu.bottomRows(2).isApprox(
      data->constraints.at("d_inactive_equality")->Hu));
  BOOST_CHECK(data->Hp.bottomRows(2).isApprox(
      data->constraints.at("d_inactive_equality")->Hp));
  BOOST_CHECK(model.get_lb().isApprox(
      (VectorXs(3) << Scalar(-1), Scalar(-3), Scalar(-3)).finished()));
  BOOST_CHECK(model.get_ub().isApprox(
      (VectorXs(3) << Scalar(2), Scalar(4), Scalar(4)).finished()));

  data->resize(&model, false);
  model.calc(data, x);
  model.calcDiff(data, x);
  BOOST_CHECK_EQUAL(data->g.size(), model.get_ng_T());
  BOOST_CHECK_EQUAL(data->h.size(), model.get_nh_T());
  BOOST_CHECK(data->g.tail(2).isApprox(
      data->constraints.at("c_inactive_inequality")->g));
  BOOST_CHECK(
      data->h.tail(2).isApprox(data->constraints.at("d_inactive_equality")->h));
  BOOST_CHECK(data->Gp.bottomRows(2).isApprox(
      data->constraints.at("c_inactive_inequality")->Gp));
  BOOST_CHECK(data->Hp.bottomRows(2).isApprox(
      data->constraints.at("d_inactive_equality")->Hp));

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      data->resize(&model);
      model.calc(data, x, u);
      model.calcDiff(data, x, u);
      data->resize(&model, false);
      model.calc(data, x);
      model.calcDiff(data, x);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

void test_addConstraint(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  crocoddyl::ConstraintModelManagerTpl<float> casted_model =
      model.cast<float>();

  // add an active constraint
  std::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint_1 =
      create_random_constraint(state_type);
  model.addConstraint("random_constraint_1", rand_constraint_1);
  std::size_t ng = rand_constraint_1->get_ng();
  std::size_t nh = rand_constraint_1->get_nh();
  std::size_t ng_T = rand_constraint_1->get_T_constraint() ? ng : 0;
  std::size_t nh_T = rand_constraint_1->get_T_constraint() ? nh : 0;
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);
  BOOST_CHECK(model.get_ng_T() == ng_T);
  BOOST_CHECK(model.get_nh_T() == nh_T);

  // add an inactive constraint
  std::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint_2 =
      create_random_constraint(state_type);
  model.addConstraint("random_constraint_2", rand_constraint_2, false);
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);
  BOOST_CHECK(model.get_ng_T() == ng_T);
  BOOST_CHECK(model.get_nh_T() == nh_T);

  // change the random constraint 2 status
  model.changeConstraintStatus("random_constraint_2", true);
  ng += rand_constraint_2->get_ng();
  nh += rand_constraint_2->get_nh();
  if (rand_constraint_2->get_T_constraint()) {
    ng_T += rand_constraint_2->get_ng();
    nh_T += rand_constraint_2->get_nh();
  }
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);
  BOOST_CHECK(model.get_ng_T() == ng_T);
  BOOST_CHECK(model.get_nh_T() == nh_T);

  // change the random constraint 1 status
  model.changeConstraintStatus("random_constraint_1", false);
  ng -= rand_constraint_1->get_ng();
  nh -= rand_constraint_1->get_nh();
  if (rand_constraint_1->get_T_constraint()) {
    ng_T -= rand_constraint_1->get_ng();
    nh_T -= rand_constraint_1->get_nh();
  }
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);
  BOOST_CHECK(model.get_ng_T() == ng_T);
  BOOST_CHECK(model.get_nh_T() == nh_T);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ConstraintModelAbstractTpl<float>>
      casted_rand_constraint_1 = rand_constraint_1->cast<float>();
  casted_model.addConstraint("random_constraint_1", casted_rand_constraint_1);
  ng = casted_rand_constraint_1->get_ng();
  nh = casted_rand_constraint_1->get_nh();
  ng_T = casted_rand_constraint_1->get_T_constraint() ? ng : 0;
  nh_T = casted_rand_constraint_1->get_T_constraint() ? nh : 0;
  BOOST_CHECK(casted_model.get_ng() == ng);
  BOOST_CHECK(casted_model.get_nh() == nh);
  BOOST_CHECK(casted_model.get_ng_T() == ng_T);
  BOOST_CHECK(casted_model.get_nh_T() == nh_T);
  std::shared_ptr<crocoddyl::ConstraintModelAbstractTpl<float>>
      casted_rand_constraint_2 = rand_constraint_2->cast<float>();
  casted_model.addConstraint("random_constraint_2", casted_rand_constraint_2,
                             false);
  BOOST_CHECK(casted_model.get_ng() == ng);
  BOOST_CHECK(casted_model.get_nh() == nh);
  BOOST_CHECK(casted_model.get_ng_T() == ng_T);
  BOOST_CHECK(casted_model.get_nh_T() == nh_T);
  casted_model.changeConstraintStatus("random_constraint_2", true);
  ng += casted_rand_constraint_2->get_ng();
  nh += casted_rand_constraint_2->get_nh();
  if (casted_rand_constraint_2->get_T_constraint()) {
    ng_T += casted_rand_constraint_2->get_ng();
    nh_T += casted_rand_constraint_2->get_nh();
  }
  BOOST_CHECK(casted_model.get_ng() == ng);
  BOOST_CHECK(casted_model.get_nh() == nh);
  BOOST_CHECK(casted_model.get_ng_T() == ng_T);
  BOOST_CHECK(casted_model.get_nh_T() == nh_T);
  casted_model.changeConstraintStatus("random_constraint_1", false);
  ng -= casted_rand_constraint_1->get_ng();
  nh -= casted_rand_constraint_1->get_nh();
  if (casted_rand_constraint_1->get_T_constraint()) {
    ng_T -= casted_rand_constraint_1->get_ng();
    nh_T -= casted_rand_constraint_1->get_nh();
  }
  BOOST_CHECK(casted_model.get_ng() == ng);
  BOOST_CHECK(casted_model.get_nh() == nh);
  BOOST_CHECK(casted_model.get_ng_T() == ng_T);
  BOOST_CHECK(casted_model.get_nh_T() == nh_T);
#endif
}

void test_addConstraint_error_message(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // create an constraint object
  std::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint =
      create_random_constraint(state_type);

  // add twice the same constraint object to the container
  model.addConstraint("random_constraint", rand_constraint);

  // test error message when we add a duplicate constraint
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addConstraint("random_constraint", rand_constraint);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't add the random_constraint "
                     "constraint item, it already existed."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the constraint status of an inexistent
  // constraint
  capture_ios.beginCapture();
  model.changeConstraintStatus("no_exist_constraint", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: we couldn't change the status of the "
                     "no_exist_constraint constraint item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeConstraint(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  crocoddyl::ConstraintModelManagerTpl<float> casted_model =
      model.cast<float>();

  // add an active constraint
  std::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint =
      create_random_constraint(state_type);
  model.addConstraint("random_constraint", rand_constraint);
  std::size_t ng = rand_constraint->get_ng();
  std::size_t nh = rand_constraint->get_nh();
  std::size_t ng_T = rand_constraint->get_T_constraint() ? ng : 0;
  std::size_t nh_T = rand_constraint->get_T_constraint() ? nh : 0;
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);
  BOOST_CHECK(model.get_ng_T() == ng_T);
  BOOST_CHECK(model.get_nh_T() == nh_T);

  // remove the constraint
  model.removeConstraint("random_constraint");
  BOOST_CHECK(model.get_ng() == 0);
  BOOST_CHECK(model.get_nh() == 0);
  BOOST_CHECK(model.get_ng_T() == 0);
  BOOST_CHECK(model.get_nh_T() == 0);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::ConstraintModelAbstractTpl<float>>
      casted_rand_constraint = rand_constraint->cast<float>();
  casted_model.addConstraint("random_constraint", casted_rand_constraint);
  ng = casted_rand_constraint->get_ng();
  nh = casted_rand_constraint->get_nh();
  ng_T = casted_rand_constraint->get_T_constraint() ? ng : 0;
  nh_T = casted_rand_constraint->get_T_constraint() ? nh : 0;
  BOOST_CHECK(casted_model.get_ng() == ng);
  BOOST_CHECK(casted_model.get_nh() == nh);
  BOOST_CHECK(casted_model.get_ng_T() == ng_T);
  BOOST_CHECK(casted_model.get_nh_T() == nh_T);
  casted_model.removeConstraint("random_constraint");
  BOOST_CHECK(casted_model.get_ng() == 0);
  BOOST_CHECK(casted_model.get_nh() == 0);
  BOOST_CHECK(casted_model.get_ng_T() == 0);
  BOOST_CHECK(casted_model.get_nh_T() == 0);
#endif
}

void test_removeConstraint_error_message(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // remove a none existing constraint form the container, we expect a cout
  // message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeConstraint("random_constraint");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't remove the random_constraint "
                     "constraint item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some constraint objects
  std::vector<std::shared_ptr<crocoddyl::ConstraintModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::ConstraintDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    const std::shared_ptr<crocoddyl::ConstraintModelAbstract>& m =
        create_random_constraint(state_type);
    model.addConstraint(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the constraint sum
  const std::shared_ptr<crocoddyl::ConstraintDataManager>& data =
      model.createData(&shared_data);

  // compute the constraint sum data for the case when all constraints are
  // defined as active
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1, u1);

  // check the constraint against single constraint computations
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(model.get_ng());
  Eigen::VectorXd h = Eigen::VectorXd::Zero(model.get_nh());
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    const std::size_t ng = models[i]->get_ng();
    const std::size_t nh = models[i]->get_nh();
    g.segment(ng_i, ng) = datas[i]->g;
    h.segment(nh_i, nh) = datas[i]->h;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(data->g.isApprox(g, 1e-9));
  BOOST_CHECK(data->h.isApprox(h, 1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ConstraintModelManagerTpl<float> casted_model =
      model.cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model.get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  std::vector<std::shared_ptr<crocoddyl::ConstraintModelAbstractTpl<float>>>
      casted_models;
  std::vector<std::shared_ptr<crocoddyl::ConstraintDataAbstractTpl<float>>>
      casted_datas;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models.push_back(models[i]->cast<float>());
    casted_datas.push_back(casted_models[i]->createData(&casted_shared_data));
  }
  const std::shared_ptr<crocoddyl::ConstraintDataManagerTpl<float>>&
      casted_data = casted_model.createData(&casted_shared_data);
  const Eigen::VectorXf& x1_f = x1.cast<float>();
  const Eigen::VectorXf& u1_f = u1.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x1_f);
  casted_model.calc(casted_data, x1_f, u1_f);
  ng_i = 0;
  nh_i = 0;
  Eigen::VectorXf g_f = Eigen::VectorXf::Zero(casted_model.get_ng());
  Eigen::VectorXf h_f = Eigen::VectorXf::Zero(casted_model.get_nh());
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models[i]->calc(casted_datas[i], x1_f, u1_f);
    const std::size_t ng = casted_models[i]->get_ng();
    const std::size_t nh = casted_models[i]->get_nh();
    g_f.segment(ng_i, ng) = casted_datas[i]->g;
    h_f.segment(nh_i, nh) = casted_datas[i]->h;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(casted_data->g.isApprox(g_f, 1e-9f));
  BOOST_CHECK(casted_data->h.isApprox(h_f, 1e-9f));
#endif
}

void test_calcDiff(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some constraint objects
  std::vector<std::shared_ptr<crocoddyl::ConstraintModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::ConstraintDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    const std::shared_ptr<crocoddyl::ConstraintModelAbstract>& m =
        create_random_constraint(state_type);
    model.addConstraint(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the constraint sum
  const std::shared_ptr<crocoddyl::ConstraintDataManager>& data =
      model.createData(&shared_data);

  // compute the constraint sum data for the case when all constraints are
  // defined as active
  Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1, u1);
  model.calcDiff(data, x1, u1);

  // check the constraint against single constraint computations
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;
  const std::size_t ndx = state->get_ndx();
  const std::size_t nu = model.get_nu();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(model.get_ng());
  Eigen::VectorXd h = Eigen::VectorXd::Zero(model.get_nh());
  Eigen::MatrixXd Gx = Eigen::MatrixXd::Zero(model.get_ng(), ndx);
  Eigen::MatrixXd Gu = Eigen::MatrixXd::Zero(model.get_ng(), nu);
  Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(model.get_nh(), ndx);
  Eigen::MatrixXd Hu = Eigen::MatrixXd::Zero(model.get_nh(), nu);
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    models[i]->calcDiff(datas[i], x1, u1);
    const std::size_t ng = models[i]->get_ng();
    const std::size_t nh = models[i]->get_nh();
    g.segment(ng_i, ng) = datas[i]->g;
    h.segment(nh_i, nh) = datas[i]->h;
    Gx.block(ng_i, 0, ng, ndx) = datas[i]->Gx;
    Gu.block(ng_i, 0, ng, nu) = datas[i]->Gu;
    Hx.block(nh_i, 0, nh, ndx) = datas[i]->Hx;
    Hu.block(nh_i, 0, nh, nu) = datas[i]->Hu;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(data->g.isApprox(g, 1e-9));
  BOOST_CHECK(data->h.isApprox(h, 1e-9));
  BOOST_CHECK(data->Gx.isApprox(Gx, 1e-9));
  BOOST_CHECK(data->Gu.isApprox(Gu, 1e-9));
  BOOST_CHECK(data->Hx.isApprox(Hx, 1e-9));
  BOOST_CHECK(data->Hu.isApprox(Hu, 1e-9));

  x1 = state->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  data->resize(&model, false);
  model.calc(data, x1);
  model.calcDiff(data, x1);

  const std::size_t ng_T = model.get_ng_T();
  const std::size_t nh_T = model.get_nh_T();
  ng_i = 0;
  nh_i = 0;
  g.conservativeResize(ng_T);
  h.conservativeResize(nh_T);
  Gx.conservativeResize(ng_T, ndx);
  Gu.conservativeResize(ng_T, nu);
  Hx.conservativeResize(nh_T, ndx);
  Hu.conservativeResize(nh_T, nu);
  for (std::size_t i = 0; i < 5; ++i) {
    if (models[i]->get_T_constraint()) {
      models[i]->calc(datas[i], x1);
      models[i]->calcDiff(datas[i], x1);
      const std::size_t ng = models[i]->get_ng();
      const std::size_t nh = models[i]->get_nh();
      g.segment(ng_i, ng) = datas[i]->g;
      h.segment(nh_i, nh) = datas[i]->h;
      Gx.block(ng_i, 0, ng, ndx) = datas[i]->Gx;
      Gu.block(ng_i, 0, ng, nu) = datas[i]->Gu;
      Hx.block(nh_i, 0, nh, ndx) = datas[i]->Hx;
      Hu.block(nh_i, 0, nh, nu) = datas[i]->Hu;
      ng_i += ng;
      nh_i += nh;
    }
  }
  BOOST_CHECK(data->g.isApprox(g, 1e-9));
  BOOST_CHECK(data->h.isApprox(h, 1e-9));
  BOOST_CHECK(data->Gx.isApprox(Gx, 1e-9));
  BOOST_CHECK(data->Hx.isApprox(Hx, 1e-9));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::ConstraintModelManagerTpl<float> casted_model =
      model.cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model.get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  std::vector<std::shared_ptr<crocoddyl::ConstraintModelAbstractTpl<float>>>
      casted_models;
  std::vector<std::shared_ptr<crocoddyl::ConstraintDataAbstractTpl<float>>>
      casted_datas;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models.push_back(models[i]->cast<float>());
    casted_datas.push_back(casted_models[i]->createData(&casted_shared_data));
  }
  const std::shared_ptr<crocoddyl::ConstraintDataManagerTpl<float>>&
      casted_data = casted_model.createData(&casted_shared_data);
  const Eigen::VectorXf& x1_f = x1.cast<float>();
  const Eigen::VectorXf& u1_f = u1.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x1_f);

  casted_model.calc(casted_data, x1_f, u1_f);
  casted_model.calcDiff(casted_data, x1_f, u1_f);

  ng_i = 0;
  nh_i = 0;
  Eigen::VectorXf g_f = Eigen::VectorXf::Zero(casted_model.get_ng());
  Eigen::VectorXf h_f = Eigen::VectorXf::Zero(casted_model.get_nh());
  Eigen::MatrixXf Gx_f = Eigen::MatrixXf::Zero(casted_model.get_ng(), ndx);
  Eigen::MatrixXf Gu_f = Eigen::MatrixXf::Zero(casted_model.get_ng(), nu);
  Eigen::MatrixXf Hx_f = Eigen::MatrixXf::Zero(casted_model.get_nh(), ndx);
  Eigen::MatrixXf Hu_f = Eigen::MatrixXf::Zero(casted_model.get_nh(), nu);
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models[i]->calc(casted_datas[i], x1_f, u1_f);
    casted_models[i]->calcDiff(casted_datas[i], x1_f, u1_f);
    const std::size_t ng = casted_models[i]->get_ng();
    const std::size_t nh = casted_models[i]->get_nh();
    g_f.segment(ng_i, ng) = casted_datas[i]->g;
    h_f.segment(nh_i, nh) = casted_datas[i]->h;
    Gx_f.block(ng_i, 0, ng, ndx) = casted_datas[i]->Gx;
    Gu_f.block(ng_i, 0, ng, nu) = casted_datas[i]->Gu;
    Hx_f.block(nh_i, 0, nh, ndx) = casted_datas[i]->Hx;
    Hu_f.block(nh_i, 0, nh, nu) = casted_datas[i]->Hu;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(casted_data->g.isApprox(g_f, 1e-9f));
  BOOST_CHECK(casted_data->h.isApprox(h_f, 1e-9f));
  BOOST_CHECK(casted_data->Gx.isApprox(Gx_f, 1e-9f));
  BOOST_CHECK(casted_data->Gu.isApprox(Gu_f, 1e-9f));
  BOOST_CHECK(casted_data->Hx.isApprox(Hx_f, 1e-9f));
  BOOST_CHECK(casted_data->Hu.isApprox(Hu_f, 1e-9f));
#endif
}

void test_get_constraints(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Data pinocchio_data(*state->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    model.addConstraint(os.str(), create_random_constraint(state_type), 1.);
  }

  // get the contacts
  const crocoddyl::ConstraintModelManager::ConstraintModelContainer&
      constraints = model.get_constraints();

  // test
  crocoddyl::ConstraintModelManager::ConstraintModelContainer::const_iterator
      it_m,
      end_m;
  unsigned i;
  for (i = 0, it_m = constraints.begin(), end_m = constraints.end();
       it_m != end_m; ++it_m, ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_shareMemory(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  const std::shared_ptr<crocoddyl::StateAbstract> state =
      state_factory.create(state_type);
  const std::size_t np = 3;
  const std::size_t nu = state->get_nv();
  crocoddyl::ConstraintModelManager constraint_model(state, nu, np);
  const std::shared_ptr<ConstraintManagerParameterResidual> residual =
      std::make_shared<ConstraintManagerParameterResidual>(state, 2, nu, np);
  constraint_model.addConstraint(
      "equality",
      std::make_shared<crocoddyl::ConstraintModelResidual>(state, residual));
  constraint_model.addConstraint(
      "inequality",
      std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, residual, -Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones()));
  crocoddyl::DataCollectorAbstract shared_data;
  const std::shared_ptr<crocoddyl::ConstraintDataManager>& constraint_data =
      constraint_model.createData(&shared_data);

  const std::size_t ng = constraint_model.get_ng();
  const std::size_t nh = constraint_model.get_nh();
  const std::size_t ndx = state->get_ndx();
  ConstraintManagerActionProbe action_model(state, nu, ng, nh,
                                            constraint_model.get_ng_T(),
                                            constraint_model.get_nh_T(), np);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& action_data =
      action_model.createData();

  ConstraintManagerActionProbe incompatible_action(
      state, nu, ng, nh, constraint_model.get_ng_T(),
      constraint_model.get_nh_T(), np + 1);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& incompatible_data =
      incompatible_action.createData();
  BOOST_CHECK_THROW(constraint_data->shareMemory(incompatible_data.get()),
                    crocoddyl::Exception);

  constraint_data->shareMemory(action_data.get());
  constraint_data->h = Eigen::VectorXd::Random(nh);
  constraint_data->g = Eigen::VectorXd::Random(ng);
  constraint_data->Gx = Eigen::MatrixXd::Random(ng, ndx);
  constraint_data->Gu = Eigen::MatrixXd::Random(ng, nu);
  constraint_data->Gp = Eigen::MatrixXd::Random(ng, np);
  constraint_data->Hx = Eigen::MatrixXd::Random(nh, ndx);
  constraint_data->Hu = Eigen::MatrixXd::Random(nh, nu);
  constraint_data->Hp = Eigen::MatrixXd::Random(nh, np);

  // check that the data has been shared
  BOOST_CHECK(action_data->g.isApprox(constraint_data->g, 1e-9));
  BOOST_CHECK(action_data->h.isApprox(constraint_data->h, 1e-9));
  BOOST_CHECK(action_data->Gx.isApprox(constraint_data->Gx, 1e-9));
  BOOST_CHECK(action_data->Gu.isApprox(constraint_data->Gu, 1e-9));
  BOOST_CHECK(action_data->Gp.isApprox(constraint_data->Gp, 1e-9));
  BOOST_CHECK(action_data->Hx.isApprox(constraint_data->Hx, 1e-9));
  BOOST_CHECK(action_data->Hu.isApprox(constraint_data->Hu, 1e-9));
  BOOST_CHECK(action_data->Hp.isApprox(constraint_data->Hp, 1e-9));

  // let's now resize the data
  constraint_data->resize(&action_model, action_data.get());

  // check that the shared data has been resized
  BOOST_CHECK(action_data->g.isApprox(constraint_data->g, 1e-9));
  BOOST_CHECK(action_data->h.isApprox(constraint_data->h, 1e-9));
  BOOST_CHECK(action_data->Gx.isApprox(constraint_data->Gx, 1e-9));
  BOOST_CHECK(action_data->Gu.isApprox(constraint_data->Gu, 1e-9));
  BOOST_CHECK(action_data->Gp.isApprox(constraint_data->Gp, 1e-9));
  BOOST_CHECK(action_data->Hx.isApprox(constraint_data->Hx, 1e-9));
  BOOST_CHECK(action_data->Hu.isApprox(constraint_data->Hu, 1e-9));
  BOOST_CHECK(action_data->Hp.isApprox(constraint_data->Hp, 1e-9));

  const std::shared_ptr<crocoddyl::ConstraintDataManager>& partial_data =
      constraint_model.createData(&shared_data);
  crocoddyl::DifferentialActionModelLQR differential_model(ndx, nu);
  const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&
      differential_data = differential_model.createData();
  differential_data->g.resize(ng);
  differential_data->Gx.resize(ng, ndx);
  differential_data->Gu.resize(ng, nu);
  differential_data->h.resize(nh);
  differential_data->Hx.resize(nh, ndx);
  differential_data->Hu.resize(nh, nu);
  partial_data->shareMemory(differential_data.get());
  partial_data->Gp.setOnes();
  partial_data->Hp.setOnes();
  BOOST_CHECK_EQUAL(partial_data->Gp.cols(), np);
  BOOST_CHECK_EQUAL(partial_data->Hp.cols(), np);
  BOOST_CHECK(partial_data->Gp_internal.isApprox(partial_data->Gp));
  BOOST_CHECK(partial_data->Hp_internal.isApprox(partial_data->Hp));
}

//----------------------------------------------------------------------------//

void register_unit_tests(StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_ConstraintModelManager"
            << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_constructor, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_addConstraint, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_addConstraint_error_message, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_removeConstraint, state_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_removeConstraint_error_message, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_get_constraints, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_shareMemory, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_derivative_aggregation));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_derivative_calcDiff_no_malloc));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_running_terminal_resize));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_live_activation_resize<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_live_activation_resize<float>));

  register_unit_tests(StateModelTypes::StateMultibody_TalosArm);
  register_unit_tests(StateModelTypes::StateMultibody_HyQ);
  register_unit_tests(StateModelTypes::StateMultibody_Talos);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
