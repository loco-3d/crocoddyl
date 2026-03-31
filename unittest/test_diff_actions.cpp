///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, INRIA, University of
//                          Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "factory/diff_action.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

namespace {

template <typename DerivedA, typename DerivedB, typename Scalar>
bool isCloseAbsRel(const Eigen::MatrixBase<DerivedA>& value,
                   const Eigen::MatrixBase<DerivedB>& reference,
                   const Scalar abs_tol, const Scalar rel_tol) {
  if (value.size() == 0 || reference.size() == 0) {
    return value.size() == reference.size();
  }
  const auto value_s = value.template cast<Scalar>();
  const auto reference_s = reference.template cast<Scalar>();
  const Scalar scale = std::max(value_s.cwiseAbs().maxCoeff(),
                                reference_s.cwiseAbs().maxCoeff());
  const Scalar max_error = (value_s - reference_s).cwiseAbs().maxCoeff();
  return max_error <= abs_tol + rel_tol * std::max(Scalar(1), scale);
}

template <typename ValueScalar, typename ReferenceScalar, typename Scalar>
bool isCloseAbsRelScalar(const ValueScalar value,
                         const ReferenceScalar reference, const Scalar abs_tol,
                         const Scalar rel_tol) {
  const Scalar value_s = static_cast<Scalar>(value);
  const Scalar reference_s = static_cast<Scalar>(reference);
  const Scalar scale = std::max(std::abs(value_s), std::abs(reference_s));
  const Scalar abs_error = std::abs(value_s - reference_s);
  return abs_error <= abs_tol + rel_tol * std::max(Scalar(1), scale);
}

template <typename DerivedDouble, typename DerivedFloat>
bool isFloatCastClose(const Eigen::MatrixBase<DerivedDouble>& value,
                      const Eigen::MatrixBase<DerivedFloat>& casted_value,
                      const float abs_tol) {
  return isCloseAbsRel(value, casted_value, abs_tol,
                       200.f * std::numeric_limits<float>::epsilon());
}

template <typename DoubleScalar, typename FloatScalar>
bool isFloatCastCloseScalar(const DoubleScalar value,
                            const FloatScalar casted_value,
                            const float abs_tol) {
  return isCloseAbsRelScalar(value, casted_value, abs_tol,
                             200.f * std::numeric_limits<float>::epsilon());
}

void reseedDiffActionTestCase(
    const DifferentialActionModelTypes::Type action_type,
    const unsigned int salt) {
  const unsigned int seed =
      getUnitTestSeed() +
      1009u * (static_cast<unsigned int>(action_type) + 1u) + salt;
  seedUnitTestRandomGenerators(seed);
}

void sampleDiffActionOperatingPoint(
    const DifferentialActionModelTypes::Type,
    const std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>& model,
    const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
    Eigen::VectorXd* x, Eigen::VectorXd* u) {
  (void)data;
  *x = sampleUnitTestState(model->get_state(), 2.5e-1);
  *u = random_vector<double>(model->get_nu()) * 2.5e-1;
}

}  // namespace

void test_check_data(DifferentialActionModelTypes::Type action_type) {
  reseedDiffActionTestCase(action_type, 1u);
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  BOOST_CHECK(model->checkData(data));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstractTpl<float>>
      casted_model = model->cast<float>();
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>
      casted_data = casted_model->createData();
  BOOST_CHECK(casted_model->checkData(casted_data));
#endif
}

void test_calc_returns_state(DifferentialActionModelTypes::Type action_type) {
  reseedDiffActionTestCase(action_type, 2u);
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // create the corresponding data object
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd x = sampleUnitTestState(model->get_state(), 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(static_cast<std::size_t>(data->xout.size()) ==
              model->get_state()->get_nv());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstractTpl<float>>
      casted_model = model->cast<float>();
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>
      casted_data = casted_model->createData();
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  BOOST_CHECK(static_cast<std::size_t>(casted_data->xout.size()) ==
              casted_model->get_state()->get_nv());
  float tol_f = 10.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(isFloatCastClose(data->xout, casted_data->xout, tol_f));
#endif
}

void test_calc_returns_a_cost(DifferentialActionModelTypes::Type action_type) {
  reseedDiffActionTestCase(action_type, 3u);
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  const Eigen::VectorXd x = sampleUnitTestState(model->get_state(), 2.5e-1);
  const Eigen::VectorXd u = random_vector<double>(model->get_nu()) * 2.5e-1;
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstractTpl<float>>
      casted_model = model->cast<float>();
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>
      casted_data = casted_model->createData();
  casted_data->cost = float(nan(""));
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model->calc(casted_data, x_f, u_f);
  BOOST_CHECK(!std::isnan(casted_data->cost));
  float tol_f = 50.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(isFloatCastCloseScalar(data->cost, casted_data->cost, tol_f));
#endif
}

void test_quasi_static(DifferentialActionModelTypes::Type action_type) {
  if (action_type ==
      DifferentialActionModelTypes::
          DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed)
    return;
  reseedDiffActionTestCase(action_type, 4u);
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, false);

  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  // Getting the cost value computed by calc()
  Eigen::VectorXd x = sampleUnitTestState(model->get_state(), 2.5e-1);
  x.tail(model->get_state()->get_nv()).setZero();
  Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
  model->quasiStatic(data, u, x);
  model->calc(data, x, u);

  // Check for inactive contacts
  if (action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactFwdDynamics_HyQ ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactFwdDynamicsWithFriction_HyQ ||
      action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactFwdDynamics_Talos ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactFwdDynamicsWithFriction_Talos ||
      action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactInvDynamics_HyQ ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactInvDynamicsWithFriction_HyQ ||
      action_type == DifferentialActionModelTypes::
                         DifferentialActionModelContactInvDynamics_Talos ||
      action_type ==
          DifferentialActionModelTypes::
              DifferentialActionModelContactInvDynamicsWithFriction_Talos) {
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> m =
        std::static_pointer_cast<
            crocoddyl::DifferentialActionModelContactFwdDynamics>(model);
    m->get_contacts()->changeContactStatus("lf", false);

    model->quasiStatic(data, u, x);
    model->calc(data, x, u);

    // Checking that the acceleration is zero as supposed to be in a quasi
    // static condition
    BOOST_CHECK(data->xout.norm() <= 1e-8);

    // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
    std::shared_ptr<crocoddyl::DifferentialActionModelAbstractTpl<float>>
        casted_model = model->cast<float>();
    std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>
        casted_data = casted_model->createData();
    Eigen::VectorXf x_f = x.cast<float>();
    x_f.tail(casted_model->get_state()->get_nv()).setZero();
    Eigen::VectorXf u_f = Eigen::VectorXf::Zero(casted_model->get_nu());
    casted_model->quasiStatic(casted_data, u_f, x_f);
    casted_model->calc(casted_data, x_f, u_f);
    float tol_f =
        50.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
    BOOST_CHECK(casted_data->xout.norm() <= tol_f);
    BOOST_CHECK(isFloatCastClose(data->xout, casted_data->xout, tol_f));
#endif
  }
}

void test_partial_derivatives_against_numdiff(
    DifferentialActionModelTypes::Type action_type) {
  reseedDiffActionTestCase(action_type, 5u);
  // create the model
  DifferentialActionModelFactory factory;
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type);

  // create the corresponding data object and set the cost to nan
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  crocoddyl::DifferentialActionModelNumDiff model_num_diff(model);
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x;
  Eigen::VectorXd u;
  sampleDiffActionOperatingPoint(action_type, model, data, &x, &u);

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);
  // Tolerance defined as in
  // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
  double tol = 2. * std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  const double numdiff_rel_tol = 1e-6;
  BOOST_CHECK(isCloseAbsRel(data->h, data_num_diff->h, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->g, data_num_diff->g, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Fx, data_num_diff->Fx, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Fu, data_num_diff->Fu, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Lx, data_num_diff->Lx, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Lu, data_num_diff->Lu, tol, numdiff_rel_tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK(
        isCloseAbsRel(data->Lxx, data_num_diff->Lxx, tol, numdiff_rel_tol));
    BOOST_CHECK(
        isCloseAbsRel(data->Lxu, data_num_diff->Lxu, tol, numdiff_rel_tol));
    BOOST_CHECK(
        isCloseAbsRel(data->Luu, data_num_diff->Luu, tol, numdiff_rel_tol));
  }
  BOOST_CHECK(isCloseAbsRel(data->Hx, data_num_diff->Hx, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Hu, data_num_diff->Hu, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Gx, data_num_diff->Gx, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Gu, data_num_diff->Gu, tol, numdiff_rel_tol));

  // Computing the action derivatives
  x = sampleUnitTestState(model->get_state(), 2.5e-1);
  model->calc(data, x);
  model->calcDiff(data, x);
  model_num_diff.calc(data_num_diff, x);
  model_num_diff.calcDiff(data_num_diff, x);
  BOOST_CHECK(isCloseAbsRel(data->h, data_num_diff->h, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->g, data_num_diff->g, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Lx, data_num_diff->Lx, tol, numdiff_rel_tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK(
        isCloseAbsRel(data->Lxx, data_num_diff->Lxx, tol, numdiff_rel_tol));
  }
  BOOST_CHECK(isCloseAbsRel(data->Hx, data_num_diff->Hx, tol, numdiff_rel_tol));
  BOOST_CHECK(isCloseAbsRel(data->Gx, data_num_diff->Gx, tol, numdiff_rel_tol));

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstractTpl<float>>
      casted_model = model->cast<float>();
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>
      casted_data = casted_model->createData();
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  casted_model->calc(casted_data, x_f, u_f);
  casted_model->calcDiff(casted_data, x_f, u_f);
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(isFloatCastClose(data->h, casted_data->h, tol_f));
  BOOST_CHECK(isFloatCastClose(data->g, casted_data->g, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Fx, casted_data->Fx, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Fu, casted_data->Fu, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Lx, casted_data->Lx, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Lu, casted_data->Lu, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Gx, casted_data->Gx, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Gu, casted_data->Gu, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Hx, casted_data->Hx, tol_f));
  BOOST_CHECK(isFloatCastClose(data->Hu, casted_data->Hu, tol_f));
  crocoddyl::DifferentialActionModelNumDiffTpl<float> casted_model_num_diff =
      model_num_diff.cast<float>();
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<float>>
      casted_data_num_diff = casted_model_num_diff.createData();
  casted_model_num_diff.calc(casted_data_num_diff, x_f, u_f);
  casted_model_num_diff.calcDiff(casted_data_num_diff, x_f, u_f);
  tol_f = 80.0f * sqrt(casted_model_num_diff.get_disturbance());
  BOOST_CHECK(isFloatCastClose(casted_data->Gx.cast<double>(),
                               casted_data_num_diff->Gx, tol_f));
  BOOST_CHECK(isFloatCastClose(casted_data->Gu.cast<double>(),
                               casted_data_num_diff->Gu, tol_f));
  BOOST_CHECK(isFloatCastClose(casted_data->Hx.cast<double>(),
                               casted_data_num_diff->Hx, tol_f));
  BOOST_CHECK(isFloatCastClose(casted_data->Hu.cast<double>(),
                               casted_data_num_diff->Hu, tol_f));
#endif
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(
    DifferentialActionModelTypes::Type action_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_check_data, action_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, action_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, action_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_partial_derivatives_against_numdiff, action_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_quasi_static, action_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    register_action_model_unit_tests(DifferentialActionModelTypes::all[i]);
  }
  // register_action_model_unit_tests(DifferentialActionModelTypes::DifferentialActionModelContactInvDynamicsWithFriction_Talos);
  // register_action_model_unit_tests(DifferentialActionModelTypes::DifferentialActionModelContactInvDynamics_TalosArm);
  // register_action_model_unit_tests(DifferentialActionModelTypes::DifferentialActionModelContactInvDynamics_HyQ);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
