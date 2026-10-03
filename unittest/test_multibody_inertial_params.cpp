///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <Eigen/Eigenvalues>
#include <limits>
#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/multibody/sample-models.hpp>

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/constrained-inverse.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/exp-eigenvalue.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/params/log-cholesky.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename Scalar>
struct ScalarTraits {
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type OtherScalar;
  static Scalar derivativeDisturbance() {
    return std::is_same<Scalar, float>::value ? Scalar(2e-3) : Scalar(1e-7);
  }
  static double derivativeTolerance() {
    return std::is_same<Scalar, float>::value ? 2e-2 : 2e-5;
  }
  static double valueTolerance() {
    return std::is_same<Scalar, float>::value ? 3e-3 : 1e-10;
  }
};

template <typename Scalar>
std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > createState() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  std::shared_ptr<pinocchio::Model> model =
      std::make_shared<pinocchio::Model>();
  pinocchio::buildModels::humanoidRandom(*model, true);
  model->lowerPositionLimit.template segment<7>(0).fill(Scalar(-1));
  model->upperPositionLimit.template segment<7>(0).fill(Scalar(1));
  const crocoddyl::StateMultibody state(model);
  return std::make_shared<StateMultibody>(state.template cast<Scalar>());
}

template <typename Scalar>
std::vector<std::string> firstBodyNames(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> >& state,
    const std::size_t nbodies) {
  const std::vector<std::string>& names = state->get_pinocchio()->names;
  BOOST_REQUIRE_GT(names.size(), nbodies);
  return std::vector<std::string>(names.begin() + 1,
                                  names.begin() + 1 + nbodies);
}

template <typename Parametrization>
typename Parametrization::MatrixXs finiteDifferenceJacobian(
    Parametrization& parametrization,
    const typename Parametrization::VectorXs& p) {
  typedef typename Parametrization::Scalar Scalar;
  typedef typename Parametrization::VectorXs VectorXs;
  typedef typename Parametrization::MatrixXs MatrixXs;
  const Scalar disturbance = ScalarTraits<Scalar>::derivativeDisturbance();
  const std::shared_ptr<
      typename Parametrization::InertialParametrizationDataAbstract>
      data = parametrization.createData();
  MatrixXs jacobian = MatrixXs::Zero(10, 10);
  VectorXs plus = p;
  VectorXs minus = p;
  VectorXs psi_plus(10);
  VectorXs psi_minus(10);
  for (Eigen::Index i = 0; i < 10; ++i) {
    plus = p;
    minus = p;
    plus[i] += disturbance;
    minus[i] -= disturbance;
    parametrization.fromParametrization(data, psi_plus, plus);
    parametrization.fromParametrization(data, psi_minus, minus);
    jacobian.col(i) = (psi_plus - psi_minus) / (Scalar(2) * disturbance);
  }
  return jacobian;
}

template <typename Scalar>
void checkPhysicalVector(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& psi) {
  const pinocchio::InertiaTpl<Scalar> inertia =
      pinocchio::InertiaTpl<Scalar>::FromDynamicParameters(psi);
  BOOST_CHECK_GT(static_cast<double>(inertia.mass()), 0.);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3> > solver(
      inertia.inertia().matrix());
  BOOST_REQUIRE_EQUAL(solver.info(), Eigen::Success);
  BOOST_CHECK_GT(static_cast<double>(solver.eigenvalues().minCoeff()), 0.);
}

template <typename Scalar, typename Parametrization>
void checkParametrization(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& p) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
  typedef typename ScalarTraits<Scalar>::OtherScalar OtherScalar;
  Parametrization parametrization;
  const std::shared_ptr<
      typename Parametrization::InertialParametrizationDataAbstract>
      data = parametrization.createData();
  BOOST_REQUIRE(parametrization.checkData(data));
  BOOST_CHECK(!parametrization.checkData(
      std::shared_ptr<
          crocoddyl::InertialParametrizationDataAbstractTpl<Scalar> >()));

  VectorXs psi(10), recovered(10), psi_roundtrip(10);
  MatrixXs dpsi_dp(10, 10);
  parametrization.fromParametrization(data, psi, p);
  parametrization.toParametrization(recovered, psi);
  parametrization.fromParametrization(data, psi_roundtrip, recovered);
  parametrization.updateParametrizationDerivative(data, dpsi_dp, p, psi);
  BOOST_CHECK(psi_roundtrip.isApprox(
      psi, Scalar(ScalarTraits<Scalar>::valueTolerance())));
  BOOST_CHECK(
      dpsi_dp.isApprox(finiteDifferenceJacobian(parametrization, p),
                       Scalar(ScalarTraits<Scalar>::derivativeTolerance())));
  checkPhysicalVector(psi);

  Parametrization copied(parametrization);
  const auto casted = parametrization.template cast<OtherScalar>();
  BOOST_CHECK_EQUAL(copied.get_np(), 10);
  BOOST_CHECK_EQUAL(casted.get_np(), 10);

  VectorXs wrong(9);
  MatrixXs wrong_matrix(9, 10);
  BOOST_CHECK_THROW(parametrization.fromParametrization(data, wrong, p),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(parametrization.toParametrization(wrong, psi),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(parametrization.updateParametrizationDerivative(
                        data, wrong_matrix, p, psi),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      parametrization.fromParametrization(
          std::shared_ptr<
              crocoddyl::InertialParametrizationDataAbstractTpl<Scalar> >(),
          psi, p),
      crocoddyl::Exception);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      parametrization.fromParametrization(data, psi, p);
      parametrization.updateParametrizationDerivative(data, dpsi_dp, p, psi);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

template <typename Scalar>
void test_log_cholesky_parametrization() {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  VectorXs p(10);
  p << Scalar(0.2), Scalar(-0.1), Scalar(0.15), Scalar(-0.2), Scalar(0.1),
      Scalar(-0.25), Scalar(0.3), Scalar(0.05), Scalar(-0.08), Scalar(0.12);
  checkParametrization<Scalar,
                       crocoddyl::LogCholeskyParametrizationTpl<Scalar> >(p);

  VectorXs edge = VectorXs::Zero(10);
  edge[0] = Scalar(-8);
  edge.segment(1, 3).setConstant(Scalar(-6));
  checkParametrization<Scalar,
                       crocoddyl::LogCholeskyParametrizationTpl<Scalar> >(edge);
}

template <typename Scalar>
void test_exp_eigenvalue_parametrization() {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  using std::log;
  VectorXs p(10);
  p << log(Scalar(3)), Scalar(0.1), Scalar(-0.2), Scalar(0.3), Scalar(0.25),
      Scalar(-0.3), Scalar(0.2), log(Scalar(0.4)), log(Scalar(0.6)),
      log(Scalar(0.8));
  checkParametrization<Scalar,
                       crocoddyl::ExpEigenValueParametrizationTpl<Scalar> >(p);

  VectorXs repeated(10);
  repeated << log(Scalar(1e-4)), Scalar(1e-7), Scalar(-2e-7), Scalar(1e-7),
      Scalar(1e-8), Scalar(-1e-8), Scalar(2e-8), log(Scalar(0.5)),
      log(Scalar(0.5)), log(Scalar(0.5));
  checkParametrization<Scalar,
                       crocoddyl::ExpEigenValueParametrizationTpl<Scalar> >(
      repeated);
}

template <typename Scalar, typename Parametrization>
void assignRepresentableInertias(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> >& state,
    Parametrization& parametrization,
    const std::vector<std::string>& body_names,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& p) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  BOOST_REQUIRE_EQUAL(static_cast<Eigen::Index>(body_names.size() * 10),
                      p.size());
  const std::shared_ptr<
      typename Parametrization::InertialParametrizationDataAbstract>
      data = parametrization.createData();
  VectorXs psi(10);
  pinocchio::ModelTpl<Scalar>& model = *state->get_pinocchio();
  for (std::size_t j = 0; j < body_names.size(); ++j) {
    const pinocchio::JointIndex joint_id = model.getJointId(body_names[j]);
    parametrization.fromParametrization(data, psi, p.segment(10 * j, 10));
    model.inertias[joint_id] =
        pinocchio::InertiaTpl<Scalar>::FromDynamicParameters(psi);
  }
}

template <typename Scalar>
void test_multibody_layout_update_copy_and_cast() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::LogCholeskyParametrizationTpl<Scalar> Parametrization;
  typedef crocoddyl::MultibodyInertialParamsTpl<Scalar> Model;
  typedef crocoddyl::MultibodyInertialParamsDataTpl<Scalar> Data;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef typename ScalarTraits<Scalar>::OtherScalar OtherScalar;
  const std::shared_ptr<State> state = createState<Scalar>();
  const std::shared_ptr<Parametrization> parametrization =
      std::make_shared<Parametrization>();
  const std::vector<std::string> names = firstBodyNames(state, 3);
  VectorXs seed(30);
  seed << Scalar(0.2), Scalar(-0.1), Scalar(0.15), Scalar(-0.2), Scalar(0.1),
      Scalar(-0.25), Scalar(0.3), Scalar(0.05), Scalar(-0.08), Scalar(0.12),
      Scalar(0.1), Scalar(0.05), Scalar(-0.12), Scalar(0.08), Scalar(-0.2),
      Scalar(0.15), Scalar(-0.1), Scalar(0.04), Scalar(0.09), Scalar(-0.06),
      Scalar(-0.05), Scalar(0.08), Scalar(0.12), Scalar(-0.04), Scalar(0.07),
      Scalar(-0.09), Scalar(0.11), Scalar(-0.03), Scalar(0.06), Scalar(0.1);
  assignRepresentableInertias(state, *parametrization, names, seed);

  Model empty(state, parametrization, std::vector<std::string>());
  BOOST_CHECK_EQUAL(empty.get_np(), 0);
  Model all(state, parametrization);
  BOOST_CHECK_EQUAL(all.get_np(), 10 * (state->get_pinocchio()->njoints - 1));

  Model toggled(state, parametrization, std::vector<std::string>(1, names[0]));
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > one_body =
      toggled.createData();
  toggled.changeBodyStatus(names[0], false);
  BOOST_CHECK_EQUAL(toggled.get_np(), 0);
  BOOST_CHECK(toggled.get_body_names().empty());
  BOOST_CHECK_EQUAL(toggled.get_lb().size(), 0);
  BOOST_CHECK_EQUAL(toggled.get_ub().size(), 0);
  BOOST_CHECK_EQUAL(toggled.zero().size(), 0);
  BOOST_CHECK_EQUAL(toggled.rand().size(), 0);
  BOOST_CHECK(!toggled.checkData(one_body));
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > no_bodies =
      toggled.createData();
  const std::shared_ptr<Data> no_bodies_data =
      std::dynamic_pointer_cast<Data>(no_bodies);
  BOOST_REQUIRE(no_bodies_data != nullptr);
  BOOST_CHECK(no_bodies_data->psi.empty());
  BOOST_CHECK(no_bodies_data->dpsi_dp.empty());
  toggled.update(no_bodies, VectorXs(0));
  toggled.changeBodyStatus(names[0], true);
  BOOST_CHECK_EQUAL(toggled.get_np(), 10);
  BOOST_CHECK_EQUAL(toggled.get_body_names().size(), 1);
  BOOST_CHECK(
      (toggled.get_lb().array() == -std::numeric_limits<Scalar>::max()).all());
  BOOST_CHECK(
      (toggled.get_ub().array() == std::numeric_limits<Scalar>::max()).all());
  BOOST_CHECK(!toggled.checkData(no_bodies));
  BOOST_CHECK(toggled.checkData(toggled.createData()));

  std::vector<std::string> selected;
  selected.push_back(names[0]);
  const pinocchio::JointIndex second_id =
      state->get_pinocchio()->getJointId(names[1]);
  for (std::size_t i = 0; i < state->get_pinocchio()->frames.size(); ++i) {
    const pinocchio::FrameTpl<Scalar>& frame =
        state->get_pinocchio()->frames[i];
    if (frame.parentJoint == second_id && frame.name != names[1]) {
      selected.push_back(frame.name);
      break;
    }
  }
  BOOST_REQUIRE_EQUAL(selected.size(), 2);
  Model model(state, parametrization, selected);
  BOOST_CHECK_EQUAL(model.get_np(), 20);
  BOOST_CHECK_EQUAL(model.get_joint_ids()[0],
                    state->get_pinocchio()->getJointId(names[0]));
  BOOST_CHECK_EQUAL(model.get_joint_ids()[1], second_id);
  BOOST_CHECK_EQUAL(model.get_body_names()[1], names[1]);
  const Scalar max_value = std::numeric_limits<Scalar>::max();
  BOOST_CHECK((model.get_lb().array() == -max_value).all());
  BOOST_CHECK((model.get_ub().array() == max_value).all());

  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > data_base =
      model.createData();
  const std::shared_ptr<Data> data = std::dynamic_pointer_cast<Data>(data_base);
  BOOST_REQUIRE(data != nullptr);
  BOOST_REQUIRE(std::dynamic_pointer_cast<
                    crocoddyl::DynamicsParamsDataAbstractTpl<Scalar> >(
                    data_base) != nullptr);
  BOOST_CHECK(model.checkData(data_base));
  BOOST_CHECK_EQUAL(data->np_action, 0);
  BOOST_CHECK_EQUAL(data->np_dynamics, 20);
  BOOST_CHECK_EQUAL(data->psi.size(), 2);
  BOOST_CHECK_EQUAL(data->dpsi_dp.size(), 2);
  for (std::size_t j = 0; j < data->psi.size(); ++j) {
    BOOST_CHECK_EQUAL(data->psi[j].size(), 10);
    BOOST_CHECK_EQUAL(data->dpsi_dp[j].rows(), 10);
    BOOST_CHECK_EQUAL(data->dpsi_dp[j].cols(), 10);
  }
  BOOST_CHECK_THROW(Data(static_cast<Model*>(nullptr)), crocoddyl::Exception);

  VectorXs p0 = model.zero();
  const auto before = state->get_pinocchio()->inertias;
  VectorXs p = p0;
  p[0] += Scalar(0.05);
  p[11] -= Scalar(0.03);
  model.update(data_base, p);
  BOOST_CHECK(data->p.isApprox(p));
  for (std::size_t j = 0; j < model.get_joint_ids().size(); ++j) {
    BOOST_CHECK(
        data->psi[j].isApprox(state->get_pinocchio()
                                  ->inertias[model.get_joint_ids()[j]]
                                  .toDynamicParameters(),
                              Scalar(ScalarTraits<Scalar>::valueTolerance())));
    BOOST_CHECK_GT(static_cast<double>(data->dpsi_dp[j].norm()), 0.);
  }
  BOOST_CHECK(
      !state->get_pinocchio()
           ->inertias[model.get_joint_ids()[0]]
           .toDynamicParameters()
           .isApprox(before[model.get_joint_ids()[0]].toDynamicParameters()));
  BOOST_CHECK(
      !state->get_pinocchio()
           ->inertias[model.get_joint_ids()[1]]
           .toDynamicParameters()
           .isApprox(before[model.get_joint_ids()[1]].toDynamicParameters()));
  for (pinocchio::JointIndex jid = 1;
       jid <
       static_cast<pinocchio::JointIndex>(state->get_pinocchio()->njoints);
       ++jid) {
    if (std::find(model.get_joint_ids().begin(), model.get_joint_ids().end(),
                  jid) == model.get_joint_ids().end()) {
      BOOST_CHECK(
          state->get_pinocchio()->inertias[jid].toDynamicParameters().isApprox(
              before[jid].toDynamicParameters()));
    }
  }
  BOOST_CHECK(
      model.zero().isApprox(p, Scalar(ScalarTraits<Scalar>::valueTolerance())));

  Data copied(*data);
  typedef crocoddyl::LogCholeskyParametrizationDataTpl<Scalar>
      ParametrizationData;
  const std::shared_ptr<ParametrizationData> original_workspace =
      std::dynamic_pointer_cast<ParametrizationData>(data->parametrization);
  const std::shared_ptr<ParametrizationData> copied_workspace =
      std::dynamic_pointer_cast<ParametrizationData>(copied.parametrization);
  BOOST_REQUIRE(original_workspace != nullptr);
  BOOST_REQUIRE(copied_workspace != nullptr);
  BOOST_CHECK(original_workspace.get() != copied_workspace.get());
  data->p.setZero();
  data->psi[0].setZero();
  copied_workspace->alpha += Scalar(1);
  BOOST_CHECK(copied.p.isApprox(p));
  BOOST_CHECK_GT(static_cast<double>(copied.psi[0].norm()), 0.);
  BOOST_CHECK_NE(original_workspace->alpha, copied_workspace->alpha);

  const VectorXs original_lb = VectorXs::LinSpaced(20, Scalar(-20), Scalar(-1));
  const VectorXs original_ub = VectorXs::LinSpaced(20, Scalar(1), Scalar(20));
  model.set_lb(original_lb);
  model.set_ub(original_ub);

  model.changeBodyStatus(names[0], true);
  model.changeBodyStatus("missing-body", true);
  model.changeBodyStatus("universe", true);
  BOOST_CHECK(model.checkData(data_base));
  BOOST_CHECK_EQUAL(model.get_np(), 20);
  BOOST_CHECK(model.get_lb().isApprox(original_lb));
  BOOST_CHECK(model.get_ub().isApprox(original_ub));

  model.changeBodyStatus(names[2], true);
  BOOST_CHECK_EQUAL(model.get_np(), 30);
  std::vector<std::string> expected_names;
  expected_names.push_back(names[0]);
  expected_names.push_back(names[1]);
  expected_names.push_back(names[2]);
  BOOST_CHECK(model.get_body_names() == expected_names);
  BOOST_CHECK(model.get_lb().head(20).isApprox(original_lb));
  BOOST_CHECK(model.get_ub().head(20).isApprox(original_ub));
  BOOST_CHECK((model.get_lb().tail(10).array() == -max_value).all());
  BOOST_CHECK((model.get_ub().tail(10).array() == max_value).all());
  BOOST_CHECK(!model.checkData(data_base));
  BOOST_CHECK_THROW(model.update(data_base, model.zero()),
                    crocoddyl::Exception);
  BOOST_CHECK_EQUAL(model.zero().size(), 30);
  BOOST_CHECK_EQUAL(model.rand().size(), 30);
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > expanded =
      model.createData();
  const std::shared_ptr<Data> expanded_data =
      std::dynamic_pointer_cast<Data>(expanded);
  BOOST_REQUIRE(expanded_data != nullptr);
  BOOST_CHECK_EQUAL(expanded_data->psi.size(), 3);
  BOOST_CHECK_EQUAL(expanded_data->dpsi_dp.size(), 3);

  model.changeBodyStatus(selected[1], false);
  BOOST_CHECK_EQUAL(model.get_np(), 20);
  expected_names.clear();
  expected_names.push_back(names[0]);
  expected_names.push_back(names[2]);
  BOOST_CHECK(model.get_body_names() == expected_names);
  BOOST_CHECK(model.get_lb().head(10).isApprox(original_lb.head(10)));
  BOOST_CHECK(model.get_ub().head(10).isApprox(original_ub.head(10)));
  BOOST_CHECK((model.get_lb().tail(10).array() == -max_value).all());
  BOOST_CHECK((model.get_ub().tail(10).array() == max_value).all());
  BOOST_CHECK(!model.checkData(expanded));
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > reduced =
      model.createData();
  model.changeBodyStatus(selected[1], false);
  BOOST_CHECK(model.checkData(reduced));

  model.changeBodyStatus(selected[1], true);
  BOOST_CHECK_EQUAL(model.get_np(), 30);
  expected_names.push_back(names[1]);
  BOOST_CHECK(model.get_body_names() == expected_names);
  BOOST_CHECK(!model.checkData(reduced));
  BOOST_CHECK(model.get_lb().head(10).isApprox(original_lb.head(10)));
  BOOST_CHECK(model.get_ub().head(10).isApprox(original_ub.head(10)));
  BOOST_CHECK((model.get_lb().tail(20).array() == -max_value).all());
  BOOST_CHECK((model.get_ub().tail(20).array() == max_value).all());
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > final_data =
      model.createData();
  BOOST_CHECK(model.checkData(final_data));
  const std::shared_ptr<Data> final_inertial_data =
      std::dynamic_pointer_cast<Data>(final_data);
  BOOST_REQUIRE(final_inertial_data != nullptr);
  const VectorXs final_zero = model.zero();
  model.update(final_data, final_zero);
  for (std::size_t j = 0; j < final_inertial_data->dpsi_dp.size(); ++j) {
    BOOST_CHECK_GT(static_cast<double>(final_inertial_data->dpsi_dp[j].norm()),
                   0.);
  }

  Model copied_model(model);
  copied_model.changeBodyStatus(names[0], false);
  BOOST_CHECK_EQUAL(copied_model.get_np(), 20);
  BOOST_CHECK_EQUAL(model.get_np(), 30);
  BOOST_CHECK_EQUAL(model.get_body_names().front(), names[0]);

  crocoddyl::MultibodyInertialParamsTpl<OtherScalar> casted =
      model.template cast<OtherScalar>();
  BOOST_CHECK_EQUAL(casted.get_np(), model.get_np());
  BOOST_CHECK(casted.get_body_names() == model.get_body_names());
  Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> expected_lb =
      model.get_lb().template cast<OtherScalar>();
  Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> expected_ub =
      model.get_ub().template cast<OtherScalar>();
  for (Eigen::Index i = 0; i < model.get_lb().size(); ++i) {
    if (model.get_lb()[i] == -max_value) {
      expected_lb[i] = -std::numeric_limits<OtherScalar>::max();
    }
    if (model.get_ub()[i] == max_value) {
      expected_ub[i] = std::numeric_limits<OtherScalar>::max();
    }
  }
  BOOST_CHECK(casted.get_lb().isApprox(expected_lb));
  BOOST_CHECK(casted.get_ub().isApprox(expected_ub));

  std::vector<std::string> duplicated(2, names[0]);
  std::vector<std::string> universe(1, "universe");
  std::vector<std::string> missing(1, "missing-body");
  BOOST_CHECK_THROW(Model(state, parametrization, duplicated),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(state, parametrization, universe),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(state, parametrization, missing),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(std::shared_ptr<State>(), parametrization),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Model(state, std::shared_ptr<Parametrization>()),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.update(
          final_data,
          VectorXs::Zero(static_cast<Eigen::Index>(model.get_np()) - 1)),
      crocoddyl::Exception);
  BOOST_CHECK(!model.checkData(
      std::make_shared<crocoddyl::DynamicsParamsDataAbstractTpl<Scalar> >(
          model.get_np())));
}

template <typename Scalar>
void test_manager_dynamics_and_no_allocation() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ExpEigenValueParametrizationTpl<Scalar> Parametrization;
  typedef crocoddyl::MultibodyInertialParamsTpl<Scalar> Params;
  typedef crocoddyl::MultibodyInertialParamsDataTpl<Scalar> ParamsData;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Constraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> Forward;
  typedef crocoddyl::DynamicsModelConstrainedInverseTpl<Scalar> Inverse;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> Impulse;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> DynamicsData;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  const std::shared_ptr<State> state = createState<Scalar>();
  const std::vector<std::string> names = firstBodyNames(state, 1);
  const std::shared_ptr<Parametrization> parametrization =
      std::make_shared<Parametrization>();
  using std::log;
  VectorXs seed(10);
  seed << log(Scalar(3)), Scalar(0.1), Scalar(-0.2), Scalar(0.3), Scalar(0.2),
      Scalar(-0.15), Scalar(0.1), log(Scalar(0.4)), log(Scalar(0.6)),
      log(Scalar(0.8));
  assignRepresentableInertias(state, *parametrization, names, seed);
  const std::shared_ptr<Params> params =
      std::make_shared<Params>(state, parametrization, names);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("inertial", params);
  BOOST_CHECK_EQUAL(manager->get_np(), 10);
  BOOST_CHECK_EQUAL(manager->get_np_action(), 0);
  BOOST_CHECK_EQUAL(manager->get_np_dynamics(), 10);
  const std::shared_ptr<crocoddyl::ParameterDataManagerTpl<Scalar> >
      manager_data = manager->createData();
  BOOST_REQUIRE(std::dynamic_pointer_cast<ParamsData>(
                    manager_data->dynamics_params.at("inertial")) != nullptr);

  VectorXs p = manager->zero();
  p[0] += Scalar(0.05);
  p[1] -= Scalar(0.02);
  manager->update(manager_data, p);
  BOOST_CHECK(manager_data->params->p.isApprox(p));
  manager->changeParamStatus("inertial", false);
  BOOST_CHECK_EQUAL(manager->get_np(), 0);
  manager_data->resize(manager.get());
  manager->changeParamStatus("inertial", true);
  manager_data->resize(manager.get());
  manager->update(manager_data, p);

  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::shared_ptr<Constraints> no_constraints =
      std::make_shared<Constraints>(state, actuation->get_nu());
  const std::shared_ptr<Forward> forward =
      std::make_shared<Forward>(state, actuation, no_constraints);
  const std::shared_ptr<DynamicsData> forward_data =
      forward->createData(manager_data);
  forward->set_params(forward_data, manager);
  forward->update_p(forward_data, p);
  const VectorXs x = state->rand();
  const VectorXs u = VectorXs::LinSpaced(
      static_cast<Eigen::Index>(forward->get_nu()), Scalar(-0.1), Scalar(0.2));
  forward->calc(forward_data, x, u);
  forward->calcDiff(forward_data, x, u);
  BOOST_CHECK_EQUAL(forward_data->Fp.rows(), state->get_nv());
  BOOST_CHECK_EQUAL(forward_data->Fp.cols(), 10);

  const Scalar disturbance = ScalarTraits<Scalar>::derivativeDisturbance();
  MatrixXs forward_num = MatrixXs::Zero(state->get_nv(), 10);
  for (Eigen::Index i = 0; i < 10; ++i) {
    VectorXs plus_p = p;
    VectorXs minus_p = p;
    plus_p[i] += disturbance;
    minus_p[i] -= disturbance;
    const std::shared_ptr<DynamicsData> plus = forward->createData();
    const std::shared_ptr<DynamicsData> minus = forward->createData();
    forward->update_p(plus, plus_p);
    forward->calc(plus, x, u);
    forward->update_p(minus, minus_p);
    forward->calc(minus, x, u);
    forward_num.col(i) = (plus->vdot - minus->vdot) / (Scalar(2) * disturbance);
  }
  forward->update_p(forward_data, p);
  forward->calc(forward_data, x, u);
  forward->calcDiff(forward_data, x, u);
  BOOST_CHECK(forward_data->Fp.isApprox(
      forward_num, Scalar(ScalarTraits<Scalar>::derivativeTolerance())));

  const std::shared_ptr<Constraints> inverse_constraints =
      std::make_shared<Constraints>(state, state->get_nv());
  const std::shared_ptr<Inverse> inverse =
      std::make_shared<Inverse>(state, actuation, inverse_constraints, 0,
                                crocoddyl::DynamicsType::ContinuousEstimation);
  inverse->update_tau(
      VectorXs::LinSpaced(static_cast<Eigen::Index>(actuation->get_nu()),
                          Scalar(-0.2), Scalar(0.1)));
  const std::shared_ptr<DynamicsData> inverse_data = inverse->createData();
  inverse->set_params(inverse_data, manager);
  inverse->update_p(inverse_data, p);
  const VectorXs inverse_u =
      VectorXs::LinSpaced(static_cast<Eigen::Index>(inverse->get_nu()),
                          Scalar(-0.05), Scalar(0.08));
  inverse->calc(inverse_data, x, inverse_u);
  inverse->calcDiff(inverse_data, x, inverse_u);
  MatrixXs inverse_num = MatrixXs::Zero(inverse_data->h.size(), 10);
  for (Eigen::Index i = 0; i < 10; ++i) {
    VectorXs plus_p = p;
    VectorXs minus_p = p;
    plus_p[i] += disturbance;
    minus_p[i] -= disturbance;
    const std::shared_ptr<DynamicsData> plus = inverse->createData();
    const std::shared_ptr<DynamicsData> minus = inverse->createData();
    inverse->update_p(plus, plus_p);
    inverse->calc(plus, x, inverse_u);
    inverse->update_p(minus, minus_p);
    inverse->calc(minus, x, inverse_u);
    inverse_num.col(i) = (plus->h - minus->h) / (Scalar(2) * disturbance);
  }
  inverse->update_p(inverse_data, p);
  inverse->calc(inverse_data, x, inverse_u);
  inverse->calcDiff(inverse_data, x, inverse_u);
  BOOST_CHECK(inverse_data->Hp.isApprox(
      inverse_num, Scalar(ScalarTraits<Scalar>::derivativeTolerance())));

  const std::shared_ptr<Constraints> impulse_constraints =
      std::make_shared<Constraints>(state, 0);
  typedef crocoddyl::ContactModelTpl<Scalar> Contact;
  typename Contact::MaskArray mask = {{true, true, true, false, false, false}};
  const pinocchio::FrameIndex frame_id = static_cast<pinocchio::FrameIndex>(
      state->get_pinocchio()->frames.size() - 1);
  const typename Contact::Vector2s gains = Contact::Vector2s::Zero();
  const std::shared_ptr<Contact> contact = std::make_shared<Contact>(
      state, frame_id, state->get_pinocchio()->frames[frame_id].placement,
      pinocchio::LOCAL_WORLD_ALIGNED, 0, gains, mask);
  impulse_constraints->addConstraint("contact", contact);
  const std::shared_ptr<Impulse> impulse =
      std::make_shared<Impulse>(state, impulse_constraints);
  const std::shared_ptr<DynamicsData> impulse_data = impulse->createData();
  impulse->set_params(impulse_data, manager);
  impulse->update_p(impulse_data, p);
  const VectorXs empty_u(0);
  impulse->calc(impulse_data, x, empty_u);
  impulse->calcDiff(impulse_data, x, empty_u);
  MatrixXs impulse_num = MatrixXs::Zero(impulse_data->Fp.rows(), 10);
  for (Eigen::Index i = 0; i < 10; ++i) {
    VectorXs plus_p = p;
    VectorXs minus_p = p;
    plus_p[i] += disturbance;
    minus_p[i] -= disturbance;
    const std::shared_ptr<DynamicsData> plus = impulse->createData();
    const std::shared_ptr<DynamicsData> minus = impulse->createData();
    impulse->update_p(plus, plus_p);
    impulse->calc(plus, x, empty_u);
    impulse->update_p(minus, minus_p);
    impulse->calc(minus, x, empty_u);
    impulse_num.bottomRows(state->get_nv()).col(i) =
        (plus->vdot.tail(state->get_nv()) - minus->vdot.tail(state->get_nv())) /
        (Scalar(2) * disturbance);
  }
  impulse->update_p(impulse_data, p);
  impulse->calc(impulse_data, x, empty_u);
  impulse->calcDiff(impulse_data, x, empty_u);
  BOOST_CHECK_MESSAGE(
      impulse_data->Fp.isApprox(
          impulse_num, Scalar(ScalarTraits<Scalar>::derivativeTolerance())),
      "impulse Fp error=" << (impulse_data->Fp - impulse_num).norm()
                          << ", analytical=" << impulse_data->Fp.norm()
                          << ", numerical=" << impulse_num.norm());

#ifdef NDEBUG
  typedef typename ScalarTraits<Scalar>::OtherScalar OtherScalar;
  typedef crocoddyl::MultibodyInertialParamsTpl<OtherScalar> OtherParams;
  const crocoddyl::DynamicsModelConstrainedForwardTpl<OtherScalar>
      casted_forward = forward->template cast<OtherScalar>();
  const crocoddyl::DynamicsModelConstrainedInverseTpl<OtherScalar>
      casted_inverse = inverse->template cast<OtherScalar>();
  const crocoddyl::DynamicsModelImpulseForwardTpl<OtherScalar> casted_impulse =
      impulse->template cast<OtherScalar>();
  const std::shared_ptr<OtherParams> forward_params =
      std::dynamic_pointer_cast<OtherParams>(casted_forward.get_params()
                                                 ->get_dynamics_params()
                                                 .at("inertial")
                                                 ->get_param());
  const std::shared_ptr<OtherParams> inverse_params =
      std::dynamic_pointer_cast<OtherParams>(casted_inverse.get_params()
                                                 ->get_dynamics_params()
                                                 .at("inertial")
                                                 ->get_param());
  const std::shared_ptr<OtherParams> impulse_params =
      std::dynamic_pointer_cast<OtherParams>(casted_impulse.get_params()
                                                 ->get_dynamics_params()
                                                 .at("inertial")
                                                 ->get_param());
  BOOST_REQUIRE(forward_params != nullptr);
  BOOST_REQUIRE(inverse_params != nullptr);
  BOOST_REQUIRE(impulse_params != nullptr);
  BOOST_CHECK(forward_params->get_state() == casted_forward.get_state());
  BOOST_CHECK(inverse_params->get_state() == casted_inverse.get_state());
  BOOST_CHECK(impulse_params->get_state() == casted_impulse.get_state());
#endif

  const std::shared_ptr<ParamsData> params_data =
      std::dynamic_pointer_cast<ParamsData>(
          manager_data->dynamics_params.at("inertial"));
  params->update(params_data, p);
  forward->calc(forward_data, x, u);
  MatrixXs dtau_dp(state->get_nv(), params->get_np());
  params->computeJointTorqueRegressor(forward_data, params_data, dtau_dp, x, u);
  typedef crocoddyl::DataCollectorMultibodyTpl<Scalar> MultibodyCollector;
  MultibodyCollector* const multibody =
      dynamic_cast<MultibodyCollector*>(forward_data->shared);
  BOOST_REQUIRE(multibody != nullptr);
  BOOST_REQUIRE(multibody->pinocchio != nullptr);
  pinocchio::computeJointTorqueRegressor(
      *state->get_pinocchio(), *multibody->pinocchio, x.head(state->get_nq()),
      x.tail(state->get_nv()), forward_data->vdot);
  const Eigen::Index input_offset = static_cast<Eigen::Index>(
      (params->get_joint_ids()[0] - 1) * Params::kParametersPerBody);
  const MatrixXs expected_regressor =
      multibody->pinocchio->jointTorqueRegressor.middleCols(
          input_offset, Params::kParametersPerBody) *
      params_data->dpsi_dp[0];
  BOOST_CHECK(dtau_dp.isApprox(
      expected_regressor, Scalar(ScalarTraits<Scalar>::derivativeTolerance())));

  crocoddyl::DataCollectorAbstractTpl<Scalar>* const valid_shared =
      forward_data->shared;
  forward_data->shared = nullptr;
  BOOST_CHECK_THROW(params->computeJointTorqueRegressor(
                        forward_data, params_data, dtau_dp, x, u),
                    crocoddyl::Exception);
  crocoddyl::DataCollectorAbstractTpl<Scalar> wrong_collector;
  forward_data->shared = &wrong_collector;
  BOOST_CHECK_THROW(params->computeJointTorqueRegressor(
                        forward_data, params_data, dtau_dp, x, u),
                    crocoddyl::Exception);
  MultibodyCollector null_pinocchio(nullptr);
  forward_data->shared = &null_pinocchio;
  BOOST_CHECK_THROW(params->computeJointTorqueRegressor(
                        forward_data, params_data, dtau_dp, x, u),
                    crocoddyl::Exception);
  forward_data->shared = valid_shared;

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      params->update(params_data, p);
      params->computeJointTorqueRegressor(forward_data, params_data, dtau_dp, x,
                                          u);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  const std::vector<std::string> two_names = firstBodyNames(state, 2);
  params->changeBodyStatus(two_names[1], true);
  BOOST_CHECK_EQUAL(params->get_np(), 20);
  BOOST_CHECK(!params->checkData(params_data));
  BOOST_CHECK_THROW(manager->update(manager_data, p), crocoddyl::Exception);
  BOOST_CHECK_THROW(forward->update_p(forward_data, p), crocoddyl::Exception);
  BOOST_CHECK_THROW(params->update(params_data, VectorXs::Zero(20)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(params->computeJointTorqueRegressor(
                        forward_data, params_data, dtau_dp, x, u),
                    crocoddyl::Exception);

  const std::shared_ptr<Params> reconfigured = std::make_shared<Params>(
      state, parametrization,
      std::vector<std::string>(two_names.begin(), two_names.begin() + 1));
  ParameterManager rebuilt_manager(state);
  rebuilt_manager.addParam("inertial", reconfigured);
  const std::shared_ptr<crocoddyl::ParameterDataManagerTpl<Scalar> > old_data =
      rebuilt_manager.createData();
  rebuilt_manager.removeParam("inertial");
  reconfigured->changeBodyStatus(two_names[1], true);
  rebuilt_manager.addParam("inertial", reconfigured);
  BOOST_CHECK_EQUAL(rebuilt_manager.get_np(), 20);
  BOOST_CHECK_THROW(rebuilt_manager.update(old_data, VectorXs::Zero(20)),
                    crocoddyl::Exception);
  const std::shared_ptr<crocoddyl::ParameterDataManagerTpl<Scalar> >
      rebuilt_data = rebuilt_manager.createData();
  const VectorXs rebuilt_p = VectorXs::Zero(20);
  rebuilt_manager.update(rebuilt_data, rebuilt_p);
  BOOST_CHECK(rebuilt_data->params->p.isApprox(rebuilt_p));
}

}  // namespace

void register_unit_tests() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_log_cholesky_parametrization<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_log_cholesky_parametrization<float>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_exp_eigenvalue_parametrization<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_exp_eigenvalue_parametrization<float>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_multibody_layout_update_copy_and_cast<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_multibody_layout_update_copy_and_cast<float>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_manager_dynamics_and_no_allocation<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_manager_dynamics_and_no_allocation<float>));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
