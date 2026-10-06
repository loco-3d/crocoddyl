///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

template <typename _Scalar>
class ParameterOnlyResidualTpl
    : public crocoddyl::ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ResidualModelBase, ParameterOnlyResidualTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ResidualModelAbstractTpl<Scalar> Base;
  typedef typename Base::ResidualDataAbstract ResidualDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  ParameterOnlyResidualTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t nu, const std::size_t np)
      : Base(state, 2, nu, false, false, false, np),
        calc_calls(0),
        calc_diff_calls(0) {}

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>&,
            const Eigen::Ref<const VectorXs>&) override {
    ++calc_calls;
    data->r << Scalar(1), Scalar(-2);
  }

  void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    ++calc_diff_calls;
    data->Rx.setZero();
    data->Ru.setZero();
    if (this->get_np() != 0) {
      data->Rp << Scalar(1), Scalar(2), Scalar(-1), Scalar(3);
    }
  }

  template <typename NewScalar>
  ParameterOnlyResidualTpl<NewScalar> cast() const {
    typedef ParameterOnlyResidualTpl<NewScalar> ReturnType;
    return ReturnType(this->get_state()->template cast<NewScalar>(),
                      this->get_nu(), this->get_np());
  }

  std::size_t calc_calls;
  std::size_t calc_diff_calls;
};

typedef ParameterOnlyResidualTpl<double> ParameterOnlyResidual;

void test_parameter_only_residual_cost_running_and_terminal() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  const std::size_t nu = 2;
  const std::size_t np = 2;
  const std::shared_ptr<ParameterOnlyResidual> residual =
      std::make_shared<ParameterOnlyResidual>(state, nu, np);
  crocoddyl::CostModelResidual cost(state, residual);
  crocoddyl::DataCollectorAbstract shared;
  const std::shared_ptr<crocoddyl::CostDataAbstract> data =
      cost.createData(&shared);
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(nu);

  BOOST_CHECK_EQUAL(cost.get_np(), np);
  cost.calc(data, x, u);
  cost.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(residual->calc_calls, 1);
  BOOST_CHECK_EQUAL(residual->calc_diff_calls, 1);
  BOOST_CHECK_CLOSE(data->cost, 2.5, 1e-12);
  const Eigen::Vector2d expected_Lp =
      data->residual->Rp.transpose() * data->residual->r;
  const Eigen::Matrix2d expected_Lpp =
      data->residual->Rp.transpose() * data->residual->Rp;
  BOOST_CHECK(data->Lp.isApprox(expected_Lp, 1e-12));
  BOOST_CHECK(data->Lpp.isApprox(expected_Lpp, 1e-12));
  BOOST_CHECK(data->Lpx.isZero(0.));
  BOOST_CHECK(data->Lpu.isZero(0.));

  data->cost = 0.;
  data->Lp.setZero();
  data->Lpp.setZero();
  cost.calc(data, x);
  cost.calcDiff(data, x);
  BOOST_CHECK_EQUAL(residual->calc_calls, 2);
  BOOST_CHECK_EQUAL(residual->calc_diff_calls, 2);
  BOOST_CHECK_CLOSE(data->cost, 2.5, 1e-12);
  BOOST_CHECK(data->Lp.isApprox(expected_Lp, 1e-12));
  BOOST_CHECK(data->Lpp.isApprox(expected_Lpp, 1e-12));
  BOOST_CHECK(data->Lpx.isZero(0.));
  BOOST_CHECK(data->Lpu.isZero(0.));

#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::ResidualModelAbstract> residual_base =
      residual;
  const std::shared_ptr<crocoddyl::ResidualModelAbstractTpl<float>>&
      casted_residual = residual_base->cast<float>();
  BOOST_REQUIRE(casted_residual != nullptr);
  BOOST_CHECK_EQUAL(casted_residual->get_np(), np);
  crocoddyl::CostModelResidualTpl<float> casted_cost = cost.cast<float>();
  BOOST_CHECK_EQUAL(casted_cost.get_np(), np);
  crocoddyl::DataCollectorAbstractTpl<float> casted_shared;
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>> casted_data =
      casted_cost.createData(&casted_shared);
  BOOST_CHECK_EQUAL(casted_data->residual->Rp.cols(), np);
  BOOST_CHECK_EQUAL(casted_data->Lp.size(), np);
#endif
}

void test_dependency_free_terminal_cost_preserves_legacy_short_circuit() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  const std::shared_ptr<ParameterOnlyResidual> residual =
      std::make_shared<ParameterOnlyResidual>(state, 2, 0);
  crocoddyl::CostModelResidual cost(state, residual);
  crocoddyl::DataCollectorAbstract shared;
  const std::shared_ptr<crocoddyl::CostDataAbstract> data =
      cost.createData(&shared);
  const Eigen::VectorXd x = state->rand();
  data->cost = 42.;
  data->Lx.setConstant(42.);
  data->Lxx.setConstant(42.);

  cost.calc(data, x);
  cost.calcDiff(data, x);
  BOOST_CHECK_EQUAL(residual->calc_calls, 0);
  BOOST_CHECK_EQUAL(residual->calc_diff_calls, 0);
  BOOST_CHECK_EQUAL(data->cost, 0.);
  BOOST_CHECK(data->Lx.isZero(0.));
  BOOST_CHECK(data->Lxx.isZero(0.));
}

void test_calc_returns_a_cost(CostModelTypes::Type cost_type,
                              StateModelTypes::Type state_type,
                              ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);
  data->cost = nan("");

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Getting the cost value computed by calc()
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& pinocchio_model_f =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> pinocchio_data_f(pinocchio_model_f);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &pinocchio_data_f);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  casted_data->cost = float(nan(""));
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model_f, &pinocchio_data_f,
                                          x_f);
  casted_model->calc(casted_data, x_f, u_f);
  BOOST_CHECK(!std::isnan(casted_data->cost));
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
#endif
}

void test_calc_against_numdiff(CostModelTypes::Type cost_type,
                               StateModelTypes::Type state_type,
                               ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff =
      model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);

  // Checking the partial derivatives against NumDiff
  BOOST_CHECK(data->cost == data_num_diff->cost);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  casted_model->calc(casted_data, x_f, u_f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
#endif
}

void test_partial_derivatives_against_numdiff(
    CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
    ActivationModelTypes::Type activation_type) {
  using namespace boost::placeholders;

  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // Create the equivalent num diff model and data.
  crocoddyl::CostModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data_num_diff =
      model_num_diff.createData(&shared_data);

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // set the function that needs to be called at every step of the numdiff
  std::vector<crocoddyl::CostModelNumDiff::ReevaluationFunction> reevals;
  reevals.push_back(
      boost::bind(&crocoddyl::unittest::updateAllPinocchio<
                      double, 0, pinocchio::JointCollectionDefaultTpl>,
                  &pinocchio_model, &pinocchio_data, _1, _2));
  model_num_diff.set_reevals(reevals);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK(isCloseAbsRel(data->Lx, data_num_diff->Lx, tol, tol));
  BOOST_CHECK(isCloseAbsRel(data->Lu, data_num_diff->Lu, tol, tol));
  if (model_num_diff.get_with_gauss_approx()) {
    // The num diff is not precise enough to be tested here.
    BOOST_CHECK(isCloseAbsRel(data->Lxx, data_num_diff->Lxx, tol, tol));
    BOOST_CHECK(isCloseAbsRel(data->Lxu, data_num_diff->Lxu, tol, tol));
    BOOST_CHECK(isCloseAbsRel(data->Luu, data_num_diff->Luu, tol, tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data_num_diff->Luu).isZero(tol));
  }

  // Computing the cost derivatives
  x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);

  // Checking the partial derivatives against numdiff
  tol = std::max(std::pow(model_num_diff.get_disturbance(), 1. / 3.), 5e-2);
  BOOST_CHECK(isCloseAbsRel(data->Lx, data_num_diff->Lx, tol, tol));
  if (model_num_diff.get_with_gauss_approx()) {
    // The num diff is not precise enough to be tested here.
    BOOST_CHECK(isCloseAbsRel(data->Lxx, data_num_diff->Lxx, tol, tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isZero(tol));
  }

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
  BOOST_CHECK(
      isCloseAbsRel(data->Lx.cast<float>(), casted_data->Lx, tol_f, tol_f));
  BOOST_CHECK(
      isCloseAbsRel(data->Lu.cast<float>(), casted_data->Lu, tol_f, tol_f));
  BOOST_CHECK(
      isCloseAbsRel(data->Lxx.cast<float>(), casted_data->Lxx, tol_f, tol_f));
  BOOST_CHECK(
      isCloseAbsRel(data->Lxu.cast<float>(), casted_data->Lxu, tol_f, tol_f));
  BOOST_CHECK(
      isCloseAbsRel(data->Luu.cast<float>(), casted_data->Luu, tol_f, tol_f));
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x_f);
  model->calc(data, x);
  model->calcDiff(data, x);
  casted_model->calc(casted_data, x_f);
  casted_model->calcDiff(casted_data, x_f);
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
  BOOST_CHECK(
      isCloseAbsRel(data->Lx.cast<float>(), casted_data->Lx, tol_f, tol_f));
  BOOST_CHECK(
      isCloseAbsRel(data->Lxx.cast<float>(), casted_data->Lxx, tol_f, tol_f));
#endif
}

void test_dimensions_in_cost_sum(CostModelTypes::Type cost_type,
                                 StateModelTypes::Type state_type,
                                 ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);

  // Generating random values for the state and control
  const Eigen::VectorXd x = state->rand();

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  BOOST_CHECK(model->get_state()->get_nx() == cost_sum.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() == cost_sum.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == cost_sum.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() == cost_sum.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() == cost_sum.get_state()->get_nv());
  BOOST_CHECK(model->get_activation()->get_nr() == cost_sum.get_nr());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::CostModelSumTpl<float> casted_cost_sum = cost_sum.cast<float>();
  BOOST_CHECK(model->get_state()->get_nx() ==
              casted_cost_sum.get_state()->get_nx());
  BOOST_CHECK(model->get_state()->get_ndx() ==
              casted_cost_sum.get_state()->get_ndx());
  BOOST_CHECK(model->get_nu() == casted_cost_sum.get_nu());
  BOOST_CHECK(model->get_state()->get_nq() ==
              casted_cost_sum.get_state()->get_nq());
  BOOST_CHECK(model->get_state()->get_nv() ==
              casted_cost_sum.get_state()->get_nv());
  BOOST_CHECK(model->get_activation()->get_nr() == casted_cost_sum.get_nr());
#endif
}

void test_partial_derivatives_in_cost_sum(
    CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
    ActivationModelTypes::Type activation_type) {
  // create the model
  CostModelFactory factory;
  const std::shared_ptr<crocoddyl::CostModelAbstract>& model =
      factory.create(cost_type, state_type, activation_type);

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model->get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& data =
      model->createData(&shared_data);

  // create the cost sum model
  crocoddyl::CostModelSum cost_sum(state, model->get_nu());
  cost_sum.addCost("myCost", model, 1.);
  const std::shared_ptr<crocoddyl::CostDataSum>& data_sum =
      cost_sum.createData(&shared_data);

  // Generating random values for the state and control
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Compute all the pinocchio function needed for the models.
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x);

  // Computing the cost derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  cost_sum.calc(data_sum, x, u);
  cost_sum.calcDiff(data_sum, x, u);

  BOOST_CHECK((data->Lx - data_sum->Lx).isZero());
  BOOST_CHECK((data->Lu - data_sum->Lu).isZero());
  BOOST_CHECK((data->Lxx - data_sum->Lxx).isZero());
  BOOST_CHECK((data->Lxu - data_sum->Lxu).isZero());
  BOOST_CHECK((data->Luu - data_sum->Luu).isZero());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  const std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>& casted_model =
      model->cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model->get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  const std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>& casted_data =
      casted_model->createData(&casted_shared_data);
  crocoddyl::CostModelSumTpl<float> casted_cost_sum = cost_sum.cast<float>();
  const std::shared_ptr<crocoddyl::CostDataSumTpl<float>>& casted_data_sum =
      casted_cost_sum.createData(&casted_shared_data);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  casted_cost_sum.calc(casted_data_sum, x_f, u_f);
  casted_cost_sum.calcDiff(casted_data_sum, x_f, u_f);
  BOOST_CHECK((casted_data->Lx - casted_data_sum->Lx).isZero());
  BOOST_CHECK((casted_data->Lu - casted_data_sum->Lu).isZero());
  BOOST_CHECK((casted_data->Lxx - casted_data_sum->Lxx).isZero());
  BOOST_CHECK((casted_data->Lxu - casted_data_sum->Lxu).isZero());
  BOOST_CHECK((casted_data->Luu - casted_data_sum->Luu).isZero());
#endif
}

//----------------------------------------------------------------------------//

void register_cost_model_unit_tests(
    CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
    ActivationModelTypes::Type activation_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << cost_type << "_" << activation_type << "_"
            << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, cost_type,
                                      state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_against_numdiff, cost_type,
                                      state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                      cost_type, state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_dimensions_in_cost_sum, cost_type,
                                      state_type, activation_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_in_cost_sum,
                                      cost_type, state_type, activation_type)));
  framework::master_test_suite().add(ts);
}

void register_parameter_cost_base_unit_tests() {
  test_suite* ts = BOOST_TEST_SUITE("test_parameter_cost_base");
  ts->add(
      BOOST_TEST_CASE(&test_parameter_only_residual_cost_running_and_terminal));
  ts->add(BOOST_TEST_CASE(
      &test_dependency_free_terminal_cost_preserves_legacy_short_circuit));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // Test all costs available with all the activation types with all available
  // states types.
  for (size_t cost_type = 0; cost_type < CostModelTypes::all.size();
       ++cost_type) {
    for (size_t state_type =
             StateModelTypes::all[StateModelTypes::StateMultibody_TalosArm];
         state_type < StateModelTypes::all.size(); ++state_type) {
      for (size_t activation_type = 0;
           activation_type < ActivationModelTypes::all.size();
           ++activation_type) {
        register_cost_model_unit_tests(
            CostModelTypes::all[cost_type], StateModelTypes::all[state_type],
            ActivationModelTypes::all[activation_type]);
      }
    }
  }
  register_parameter_cost_base_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
