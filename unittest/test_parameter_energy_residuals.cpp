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

#include "crocoddyl/core/integrator/time.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residuals/parameters.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/actuations/joint-friction.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/params/log-cholesky.hpp"
#include "crocoddyl/multibody/residuals/actuation-parameters.hpp"
#include "crocoddyl/multibody/residuals/inertial-parameters.hpp"
#include "crocoddyl/multibody/residuals/kinetic-energy.hpp"
#include "crocoddyl/multibody/residuals/potential-energy.hpp"
#include "crocoddyl/multibody/residuals/power.hpp"
#include "crocoddyl/multibody/residuals/symmetry-parameters.hpp"
#include "crocoddyl/multibody/residuals/total-mass.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename Scalar>
struct ScalarTraits {
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type OtherScalar;
  static Scalar step() {
    return std::is_same<Scalar, float>::value ? Scalar(2e-3) : Scalar(1e-7);
  }
  static Scalar tolerance() {
    return std::is_same<Scalar, float>::value ? Scalar(3e-2) : Scalar(3e-5);
  }
};

template <typename Scalar>
struct ObserverPayloadTpl {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  ObserverPayloadTpl(const std::size_t nx, const std::size_t ndx,
                     const std::size_t nu, const std::size_t np)
      : xnext(VectorXs::Zero(nx)),
        Fx(MatrixXs::Zero(ndx, ndx)),
        Fu(MatrixXs::Zero(ndx, nu)),
        Fp(MatrixXs::Zero(ndx, np)),
        dissipative_E(VectorXs::Zero(1)),
        Ex(MatrixXs::Zero(1, ndx)),
        Eu(MatrixXs::Zero(1, nu)),
        Ep(MatrixXs::Zero(1, np)) {}

  VectorXs xnext;
  MatrixXs Fx;
  MatrixXs Fu;
  MatrixXs Fp;
  VectorXs dissipative_E;
  MatrixXs Ex;
  MatrixXs Eu;
  MatrixXs Ep;
};

template <typename Scalar>
struct ResidualCollectorTpl : crocoddyl::DataCollectorMultibodyTpl<Scalar>,
                              crocoddyl::DataCollectorParamsTpl<Scalar>,
                              crocoddyl::DataCollectorObserverTpl<Scalar> {
  ResidualCollectorTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      const std::shared_ptr<crocoddyl::ParameterDataManagerTpl<Scalar> >& data)
      : crocoddyl::DataCollectorAbstractTpl<Scalar>(),
        crocoddyl::DataCollectorMultibodyTpl<Scalar>(pinocchio),
        crocoddyl::DataCollectorParamsTpl<Scalar>(data->params, data.get()),
        crocoddyl::DataCollectorObserverTpl<Scalar>() {}
};

template <typename Scalar>
struct ResidualFixtureTpl {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef crocoddyl::ParameterDataManagerTpl<Scalar> ManagerData;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> ActuationParams;
  typedef crocoddyl::MultibodyInertialParamsTpl<Scalar> InertialParams;
  typedef crocoddyl::LogCholeskyParametrizationTpl<Scalar> Parametrization;
  typedef crocoddyl::JointDynamicsModelAbstractTpl<Scalar> JointModel;
  typedef crocoddyl::JointDynamicsModelFrictionTpl<Scalar> Friction;
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;

  ResidualFixtureTpl()
      : model_double(std::make_shared<pinocchio::Model>()),
        state(),
        pinocchio(),
        actuation(),
        actuation_params(),
        inertial_params(),
        inactive_inertial(),
        time(std::make_shared<IntegratorTime>(Scalar(0.02), true)),
        time_params(),
        inactive_time(),
        manager(),
        manager_data(),
        collector(),
        p(),
        x(),
        u() {
    pinocchio::buildModels::humanoidRandom(*model_double, true);
    model_double->lowerPositionLimit.template segment<7>(0).fill(-1.);
    model_double->upperPositionLimit.template segment<7>(0).fill(1.);
    const crocoddyl::StateMultibody state_double(model_double);
    state = std::make_shared<State>(state_double.template cast<Scalar>());
    pinocchio =
        std::make_shared<pinocchio::DataTpl<Scalar> >(*state->get_pinocchio());

    pinocchio::JointIndex joint_id = 1;
    for (; joint_id <
           static_cast<pinocchio::JointIndex>(state->get_pinocchio()->njoints);
         ++joint_id) {
      if (state->get_pinocchio()->joints[joint_id].nv() == 1) {
        break;
      }
    }
    BOOST_REQUIRE_LT(joint_id, static_cast<pinocchio::JointIndex>(
                                   state->get_pinocchio()->njoints));
    VectorXs friction_p(2);
    using std::log;
    friction_p << log(Scalar(0.3)), log(Scalar(4.));
    const std::shared_ptr<Friction> friction = std::make_shared<Friction>(
        joint_id,
        static_cast<std::size_t>(state->get_pinocchio()->joints[joint_id].nq()),
        friction_p, crocoddyl::JointFrictionType::Coulomb);
    std::vector<std::shared_ptr<JointModel> > joints(1, friction);
    actuation = std::make_shared<Actuation>(state, joints);
    actuation_params = std::make_shared<ActuationParams>(actuation);

    const std::shared_ptr<Parametrization> parametrization =
        std::make_shared<Parametrization>();
    std::vector<std::string> body_names;
    body_names.push_back(state->get_pinocchio()->names[1]);
    body_names.push_back(state->get_pinocchio()->names[2]);
    VectorXs inertial_seed(10);
    inertial_seed << Scalar(0.2), Scalar(-0.1), Scalar(0.15), Scalar(-0.2),
        Scalar(0.1), Scalar(-0.25), Scalar(0.3), Scalar(0.05), Scalar(-0.08),
        Scalar(0.12);
    VectorXs physical(10);
    const std::shared_ptr<
        typename Parametrization::InertialParametrizationDataAbstract>
        parametrization_data = parametrization->createData();
    parametrization->fromParametrization(parametrization_data, physical,
                                         inertial_seed);
    state->get_pinocchio()->inertias[1] =
        pinocchio::InertiaTpl<Scalar>::FromDynamicParameters(physical);
    inertial_seed.array() += Scalar(0.07);
    parametrization->fromParametrization(parametrization_data, physical,
                                         inertial_seed);
    state->get_pinocchio()->inertias[2] =
        pinocchio::InertiaTpl<Scalar>::FromDynamicParameters(physical);
    inertial_params =
        std::make_shared<InertialParams>(state, parametrization, body_names);
    inactive_inertial =
        std::make_shared<InertialParams>(state, parametrization, body_names);
    time_params = std::make_shared<TimeParams>(state, time);
    inactive_time = std::make_shared<TimeParams>(state, time);

    manager = std::make_shared<Manager>(state);
    manager->addParam("a_inactive_time", inactive_time, false);
    manager->addParam("z_time", time_params);
    manager->addParam("a_actuation", actuation_params);
    manager->addParam("m_inactive_inertial", inactive_inertial, false);
    manager->addParam("z_inertial", inertial_params);
    BOOST_REQUIRE_EQUAL(manager->get_np_action(), 1);
    BOOST_REQUIRE_EQUAL(manager->get_np_dynamics(), 22);
    BOOST_REQUIRE_EQUAL(manager->get_np(), 23);

    manager_data = manager->createData();
    p = manager->zero();
    p[0] += Scalar(0.04);
    p[1] += Scalar(0.08);
    p[2] -= Scalar(0.03);
    p.tail(20).array() += Scalar(0.01);
    manager->update(manager_data, p);
    collector = std::make_shared<ResidualCollectorTpl<Scalar> >(pinocchio.get(),
                                                                manager_data);

    x = state->zero();
    VectorXs dx =
        VectorXs::LinSpaced(static_cast<Eigen::Index>(state->get_ndx()),
                            Scalar(-0.08), Scalar(0.11));
    state->integrate(x, dx, x);
    u = VectorXs::Constant(static_cast<Eigen::Index>(actuation->get_nu()),
                           Scalar(0.2));
  }

  std::shared_ptr<pinocchio::Model> model_double;
  std::shared_ptr<State> state;
  std::shared_ptr<pinocchio::DataTpl<Scalar> > pinocchio;
  std::shared_ptr<Actuation> actuation;
  std::shared_ptr<ActuationParams> actuation_params;
  std::shared_ptr<InertialParams> inertial_params;
  std::shared_ptr<InertialParams> inactive_inertial;
  std::shared_ptr<IntegratorTime> time;
  std::shared_ptr<TimeParams> time_params;
  std::shared_ptr<TimeParams> inactive_time;
  std::shared_ptr<Manager> manager;
  std::shared_ptr<ManagerData> manager_data;
  std::shared_ptr<ResidualCollectorTpl<Scalar> > collector;
  VectorXs p;
  VectorXs x;
  VectorXs u;
};

template <typename Scalar, typename Model>
typename Model::MatrixXs finiteDifferenceState(
    Model& model,
    const std::shared_ptr<typename Model::ResidualDataAbstract>& data,
    const std::shared_ptr<typename Model::StateAbstract>& state,
    const typename Model::VectorXs& x, const typename Model::VectorXs& u) {
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;
  const Scalar step = ScalarTraits<Scalar>::step();
  MatrixXs numerical(model.get_nr(), state->get_ndx());
  VectorXs dx = VectorXs::Zero(state->get_ndx());
  VectorXs xp(state->get_nx()), xm(state->get_nx());
  for (Eigen::Index i = 0; i < numerical.cols(); ++i) {
    dx.setZero();
    dx[i] = step;
    state->integrate(x, dx, xp);
    model.calc(data, xp, u);
    const VectorXs rp = data->r;
    dx[i] = -step;
    state->integrate(x, dx, xm);
    model.calc(data, xm, u);
    numerical.col(i) = (rp - data->r) / (Scalar(2) * step);
  }
  return numerical;
}

template <typename Scalar, typename Model>
typename Model::MatrixXs finiteDifferenceParameters(
    Model& model,
    const std::shared_ptr<typename Model::ResidualDataAbstract>& data,
    ResidualFixtureTpl<Scalar>& fixture) {
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;
  const Scalar step = ScalarTraits<Scalar>::step();
  MatrixXs numerical(model.get_nr(), fixture.manager->get_np());
  for (Eigen::Index i = 0; i < numerical.cols(); ++i) {
    VectorXs pp = fixture.p;
    VectorXs pm = fixture.p;
    pp[i] += step;
    pm[i] -= step;
    fixture.manager->update(fixture.manager_data, pp);
    model.calc(data, fixture.x, fixture.u);
    const VectorXs rp = data->r;
    fixture.manager->update(fixture.manager_data, pm);
    model.calc(data, fixture.x, fixture.u);
    numerical.col(i) = (rp - data->r) / (Scalar(2) * step);
  }
  fixture.manager->update(fixture.manager_data, fixture.p);
  return numerical;
}

template <typename Scalar>
void test_parameter_residuals() {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
  typedef crocoddyl::ResidualDataActuationParametersTpl<Scalar> ActuationData;
  typedef crocoddyl::ResidualDataInertialParametersTpl<Scalar> InertialData;
  typedef crocoddyl::ResidualDataSymmetryParametersTpl<Scalar> SymmetryData;
  typedef crocoddyl::ResidualDataTotalMassTpl<Scalar> MassData;
  ResidualFixtureTpl<Scalar> fixture;
  const std::size_t np = fixture.manager->get_np();
  const Scalar tol = ScalarTraits<Scalar>::tolerance();

  crocoddyl::ResidualModelParametersTpl<Scalar> parameters(
      fixture.state, fixture.p - VectorXs::Constant(np, Scalar(0.2)),
      fixture.actuation->get_nu());
  const std::shared_ptr<typename decltype(parameters)::ResidualDataAbstract>
      parameter_data = parameters.createData(fixture.collector.get());
  parameters.calc(parameter_data, fixture.x, fixture.u);
  parameters.calcDiff(parameter_data, fixture.x, fixture.u);
  BOOST_CHECK(parameter_data->r.isApprox(VectorXs::Constant(np, Scalar(0.2))));
  BOOST_CHECK(parameter_data->Rp.isApprox(MatrixXs::Identity(np, np)));
  parameters.calc(parameter_data, fixture.x);
  BOOST_CHECK(parameter_data->r.isApprox(VectorXs::Constant(np, Scalar(0.2))));

  const std::shared_ptr<crocoddyl::ActuationMultibodyParamsDataTpl<Scalar> >
      actuation = std::dynamic_pointer_cast<
          crocoddyl::ActuationMultibodyParamsDataTpl<Scalar> >(
          fixture.manager_data->dynamics_params.at("a_actuation"));
  BOOST_REQUIRE(actuation != nullptr);
  const VectorXs gamma_ref =
      actuation->gamma - VectorXs::Constant(actuation->np, Scalar(0.15));
  crocoddyl::ResidualModelActuationParametersTpl<Scalar> actuation_model(
      fixture.state, gamma_ref, fixture.actuation->get_nu(), np, "a_actuation");
  const std::shared_ptr<
      typename decltype(actuation_model)::ResidualDataAbstract>
      actuation_base = actuation_model.createData(fixture.collector.get());
  const std::shared_ptr<ActuationData> actuation_data =
      std::dynamic_pointer_cast<ActuationData>(actuation_base);
  BOOST_REQUIRE(actuation_data != nullptr);
  BOOST_CHECK_EQUAL(actuation_data->np_offset, 1);
  actuation_model.calc(actuation_base, fixture.x, fixture.u);
  actuation_model.calcDiff(actuation_base, fixture.x, fixture.u);
  BOOST_CHECK(actuation_base->r.isApprox(
      VectorXs::Constant(actuation->np, Scalar(0.15)), tol));
  BOOST_CHECK(actuation_base->Rp.block(0, 1, actuation->np, actuation->np)
                  .isApprox(actuation->dgamma_dp, tol));
  actuation_model.calc(actuation_base, fixture.x);
  BOOST_CHECK(actuation_base->r.isApprox(
      VectorXs::Constant(actuation->np, Scalar(0.15)), tol));

  const std::shared_ptr<crocoddyl::MultibodyInertialParamsDataTpl<Scalar> >
      inertial = std::dynamic_pointer_cast<
          crocoddyl::MultibodyInertialParamsDataTpl<Scalar> >(
          fixture.manager_data->dynamics_params.at("z_inertial"));
  BOOST_REQUIRE(inertial != nullptr);
  VectorXs psi(20);
  psi << inertial->psi[0], inertial->psi[1];
  const VectorXs psi_ref = psi - VectorXs::Constant(20, Scalar(0.1));
  crocoddyl::ResidualModelInertialParametersTpl<Scalar> inertial_model(
      fixture.state, psi_ref, fixture.actuation->get_nu(), np, "z_inertial");
  const std::shared_ptr<typename decltype(inertial_model)::ResidualDataAbstract>
      inertial_base = inertial_model.createData(fixture.collector.get());
  const std::shared_ptr<InertialData> inertial_data =
      std::dynamic_pointer_cast<InertialData>(inertial_base);
  BOOST_REQUIRE(inertial_data != nullptr);
  BOOST_CHECK_EQUAL(inertial_data->np_offset, 3);
  inertial_model.calc(inertial_base, fixture.x, fixture.u);
  inertial_model.calcDiff(inertial_base, fixture.x, fixture.u);
  BOOST_CHECK(
      inertial_base->r.isApprox(VectorXs::Constant(20, Scalar(0.1)), tol));
  for (std::size_t i = 0; i < 2; ++i) {
    BOOST_CHECK(inertial_base->Rp.block(10 * i, 3 + 10 * i, 10, 10)
                    .isApprox(inertial->dpsi_dp[i], tol));
  }
  inertial_model.calc(inertial_base, fixture.x);
  BOOST_CHECK(
      inertial_base->r.isApprox(VectorXs::Constant(20, Scalar(0.1)), tol));

  MatrixXs S_generic = MatrixXs::Identity(np, np);
  S_generic.diagonal() = VectorXs::LinSpaced(np, Scalar(0.25), Scalar(1.25));
  crocoddyl::ResidualModelSymmetryParametersTpl<Scalar> generic_symmetry(
      fixture.state, S_generic, fixture.actuation->get_nu(), np);
  std::shared_ptr<typename decltype(generic_symmetry)::ResidualDataAbstract>
      generic_symmetry_data;
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> >
      plain_params =
          std::make_shared<crocoddyl::ParamsDataAbstractTpl<Scalar> >(np, 0);
  plain_params->p = fixture.manager_data->params->p;
  crocoddyl::DataCollectorParamsTpl<Scalar> plain_collector(plain_params);
  BOOST_REQUIRE(plain_collector.parameter_data == nullptr);
  generic_symmetry_data = generic_symmetry.createData(&plain_collector);
  BOOST_CHECK(generic_symmetry_data->Rp.isApprox(S_generic));
  generic_symmetry.calc(generic_symmetry_data, fixture.x, fixture.u);
  generic_symmetry_data->Rp *= Scalar(2);
  const MatrixXs generic_Rp = generic_symmetry_data->Rp;
  generic_symmetry.calcDiff(generic_symmetry_data, fixture.x, fixture.u);
  BOOST_CHECK(generic_symmetry_data->r.isApprox(S_generic * plain_params->p));
  BOOST_CHECK(generic_symmetry_data->Rp.isApprox(generic_Rp));
  const auto generic_symmetry_cast =
      generic_symmetry
          .template cast<typename ScalarTraits<Scalar>::OtherScalar>();
  BOOST_CHECK(generic_symmetry_cast.get_param_name().empty());
  BOOST_CHECK_EQUAL(generic_symmetry_cast.get_np(), np);

  MatrixXs S_actuation = MatrixXs::Identity(actuation->np, actuation->np);
  S_actuation.diagonal() =
      VectorXs::LinSpaced(actuation->np, Scalar(0.4), Scalar(0.8));
  crocoddyl::ResidualModelSymmetryParametersTpl<Scalar> actuation_symmetry(
      fixture.state, S_actuation, fixture.actuation->get_nu(), np,
      "a_actuation");
  const std::shared_ptr<
      typename decltype(actuation_symmetry)::ResidualDataAbstract>
      actuation_symmetry_data =
          actuation_symmetry.createData(fixture.collector.get());
  actuation_symmetry.calc(actuation_symmetry_data, fixture.x, fixture.u);
  actuation_symmetry.calcDiff(actuation_symmetry_data, fixture.x, fixture.u);
  BOOST_CHECK(
      actuation_symmetry_data->r.isApprox(S_actuation * actuation->gamma, tol));
  BOOST_CHECK(
      actuation_symmetry_data->Rp.block(0, 1, actuation->np, actuation->np)
          .isApprox(S_actuation * actuation->dgamma_dp, tol));

  MatrixXs S = MatrixXs::Zero(20, 20);
  S.topLeftCorner(10, 10).diagonal() =
      VectorXs::LinSpaced(10, Scalar(0.5), Scalar(1.4));
  S.topRightCorner(10, 10).diagonal().fill(Scalar(0.15));
  S.bottomLeftCorner(10, 10).diagonal().fill(Scalar(-0.2));
  S.bottomRightCorner(10, 10).diagonal() =
      VectorXs::LinSpaced(10, Scalar(1.5), Scalar(2.4));
  crocoddyl::ResidualModelSymmetryParametersTpl<Scalar> symmetry_model(
      fixture.state, S, fixture.actuation->get_nu(), np, "z_inertial");
  const std::shared_ptr<typename decltype(symmetry_model)::ResidualDataAbstract>
      symmetry_base = symmetry_model.createData(fixture.collector.get());
  const std::shared_ptr<SymmetryData> symmetry_data =
      std::dynamic_pointer_cast<SymmetryData>(symmetry_base);
  BOOST_REQUIRE(symmetry_data != nullptr);
  BOOST_CHECK_EQUAL(symmetry_data->np_offset, 3);
  symmetry_model.calc(symmetry_base, fixture.x, fixture.u);
  symmetry_model.calcDiff(symmetry_base, fixture.x, fixture.u);
  const VectorXs expected_symmetry =
      S.leftCols(10) * inertial->psi[0] + S.rightCols(10) * inertial->psi[1];
  BOOST_CHECK(
      symmetry_base->r.head(10).isApprox(expected_symmetry.head(10), tol));
  BOOST_CHECK(
      symmetry_base->r.tail(10).isApprox(expected_symmetry.tail(10), tol));
  for (std::size_t i = 0; i < 2; ++i) {
    BOOST_CHECK(
        symmetry_base->Rp.block(0, 3 + 10 * i, 20, 10)
            .isApprox(S.middleCols(10 * i, 10) * inertial->dpsi_dp[i], tol));
  }
  symmetry_model.calc(symmetry_base, fixture.x);
  BOOST_CHECK(symmetry_base->r.isApprox(expected_symmetry, tol));

  const Scalar selected_mass = inertial->psi[0][0] + inertial->psi[1][0];
  const Scalar total_mass =
      pinocchio::computeTotalMass(*fixture.state->get_pinocchio());
  BOOST_CHECK_GT(total_mass, selected_mass);
  crocoddyl::ResidualModelTotalMassTpl<Scalar> mass_model(
      fixture.state, total_mass - Scalar(0.25), fixture.actuation->get_nu(), np,
      "z_inertial");
  const std::shared_ptr<typename decltype(mass_model)::ResidualDataAbstract>
      mass_base = mass_model.createData(fixture.collector.get());
  const std::shared_ptr<MassData> mass_data =
      std::dynamic_pointer_cast<MassData>(mass_base);
  BOOST_REQUIRE(mass_data != nullptr);
  BOOST_CHECK_EQUAL(mass_data->np_offset, 3);
  mass_model.calc(mass_base, fixture.x, fixture.u);
  mass_model.calcDiff(mass_base, fixture.x, fixture.u);
  BOOST_CHECK_SMALL(static_cast<double>(mass_base->r[0] - Scalar(0.25)),
                    static_cast<double>(tol));
  for (std::size_t i = 0; i < 2; ++i) {
    BOOST_CHECK(mass_base->Rp.block(0, 3 + 10 * i, 1, 10)
                    .isApprox(inertial->dpsi_dp[i].row(0), tol));
  }
  mass_model.calc(mass_base, fixture.x);
  BOOST_CHECK_SMALL(static_cast<double>(mass_base->r[0] - Scalar(0.25)),
                    static_cast<double>(tol));

  const auto casted =
      inertial_model
          .template cast<typename ScalarTraits<Scalar>::OtherScalar>();
  BOOST_CHECK_EQUAL(casted.get_np(), inertial_model.get_np());
  InertialData copied(*inertial_data);
  copied.Rp.setZero();
  BOOST_CHECK(!copied.Rp.isApprox(inertial_data->Rp));

  fixture.manager->changeParamStatus("a_actuation", false);
  fixture.manager_data->resize(fixture.manager.get());
  BOOST_CHECK_THROW(actuation_model.calc(actuation_base, fixture.x, fixture.u),
                    crocoddyl::Exception);
  fixture.manager->changeParamStatus("a_actuation", true);
  fixture.manager_data->resize(fixture.manager.get());
  fixture.manager->update(fixture.manager_data, fixture.p);
  BOOST_CHECK_NO_THROW(
      actuation_model.calcDiff(actuation_base, fixture.x, fixture.u));

  crocoddyl::DataCollectorAbstractTpl<Scalar> wrong_collector;
  BOOST_CHECK_THROW(actuation_model.createData(&wrong_collector),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(mass_model.createData(&wrong_collector),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(symmetry_model.createData(&wrong_collector),
                    crocoddyl::Exception);
  const std::shared_ptr<MassData> wrong_mass_data =
      std::make_shared<MassData>(&mass_model, &wrong_collector);
  BOOST_CHECK_THROW(mass_model.calc(wrong_mass_data, fixture.x, fixture.u),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(mass_model.calcDiff(wrong_mass_data, fixture.x, fixture.u),
                    crocoddyl::Exception);
  const std::shared_ptr<SymmetryData> wrong_symmetry_data =
      std::make_shared<SymmetryData>(&symmetry_model, &wrong_collector);
  BOOST_CHECK_THROW(
      symmetry_model.calc(wrong_symmetry_data, fixture.x, fixture.u),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      symmetry_model.calcDiff(wrong_symmetry_data, fixture.x, fixture.u),
      crocoddyl::Exception);

  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> >
      saved_params = fixture.manager_data->params;
  fixture.manager_data->params.reset();
  BOOST_CHECK_THROW(mass_model.createData(fixture.collector.get()),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(mass_model.calc(mass_base, fixture.x, fixture.u),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(mass_model.calcDiff(mass_base, fixture.x, fixture.u),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(symmetry_model.createData(fixture.collector.get()),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(symmetry_model.calc(symmetry_base, fixture.x, fixture.u),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      symmetry_model.calcDiff(symmetry_base, fixture.x, fixture.u),
      crocoddyl::Exception);
  fixture.manager_data->params = saved_params;

  const std::shared_ptr<crocoddyl::StateVectorTpl<Scalar> > vector_state =
      std::make_shared<crocoddyl::StateVectorTpl<Scalar> >(4);
  BOOST_CHECK_THROW(crocoddyl::ResidualModelTotalMassTpl<Scalar>(
                        vector_state, Scalar(0), 2, np, "z_inertial"),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(crocoddyl::ResidualModelSymmetryParametersTpl<Scalar>(
                        fixture.state, MatrixXs::Identity(np - 1, np - 1),
                        fixture.actuation->get_nu(), np),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(inertial_model.set_reference(VectorXs::Zero(19)),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_energy_residuals() {
  typedef crocoddyl::ResidualModelPotentialEnergyTpl<Scalar> Potential;
  typedef crocoddyl::ResidualModelKineticEnergyTpl<Scalar> Kinetic;
  ResidualFixtureTpl<Scalar> fixture;
  const Scalar tol = ScalarTraits<Scalar>::tolerance();
  const std::size_t np = fixture.manager->get_np();

  Potential potential(fixture.state, fixture.actuation->get_nu(), np,
                      Scalar(0.3), "z_inertial");
  const std::shared_ptr<typename Potential::ResidualDataAbstract>
      potential_data = potential.createData(fixture.collector.get());
  potential.calc(potential_data, fixture.x, fixture.u);
  potential.calcDiff(potential_data, fixture.x, fixture.u);
  const typename Potential::MatrixXs potential_rx =
      finiteDifferenceState<Scalar>(potential, potential_data, fixture.state,
                                    fixture.x, fixture.u);
  const typename Potential::MatrixXs potential_rp =
      finiteDifferenceParameters<Scalar>(potential, potential_data, fixture);
  potential.calc(potential_data, fixture.x, fixture.u);
  potential.calcDiff(potential_data, fixture.x, fixture.u);
  BOOST_CHECK(potential_data->Rx.isApprox(potential_rx, tol));
  BOOST_CHECK(potential_data->Rp.isApprox(potential_rp, Scalar(2) * tol));
  const typename Potential::VectorXs potential_running = potential_data->r;
  potential.calc(potential_data, fixture.x);
  BOOST_CHECK(potential_data->r.isApprox(potential_running, tol));

  Kinetic kinetic(fixture.state, fixture.actuation->get_nu(), np, Scalar(0.4),
                  "z_inertial");
  const std::shared_ptr<typename Kinetic::ResidualDataAbstract> kinetic_data =
      kinetic.createData(fixture.collector.get());
  kinetic.calc(kinetic_data, fixture.x, fixture.u);
  kinetic.calcDiff(kinetic_data, fixture.x, fixture.u);
  const typename Kinetic::MatrixXs kinetic_rx = finiteDifferenceState<Scalar>(
      kinetic, kinetic_data, fixture.state, fixture.x, fixture.u);
  const typename Kinetic::MatrixXs kinetic_rp =
      finiteDifferenceParameters<Scalar>(kinetic, kinetic_data, fixture);
  kinetic.calc(kinetic_data, fixture.x, fixture.u);
  kinetic.calcDiff(kinetic_data, fixture.x, fixture.u);
  BOOST_CHECK(kinetic_data->Rx.isApprox(kinetic_rx, Scalar(3) * tol));
  BOOST_CHECK(kinetic_data->Rp.isApprox(kinetic_rp, Scalar(2) * tol));
  const typename Kinetic::VectorXs kinetic_running = kinetic_data->r;
  kinetic.calc(kinetic_data, fixture.x);
  BOOST_CHECK(kinetic_data->r.isApprox(kinetic_running, tol));

  Potential potential_copy(potential);
  potential_copy.set_reference(Scalar(-1));
  BOOST_CHECK_NE(potential_copy.get_reference(), potential.get_reference());
  const auto potential_cast =
      potential.template cast<typename ScalarTraits<Scalar>::OtherScalar>();
  BOOST_CHECK_EQUAL(potential_cast.get_np(), potential.get_np());

  const std::shared_ptr<crocoddyl::StateVectorTpl<Scalar> > vector_state =
      std::make_shared<crocoddyl::StateVectorTpl<Scalar> >(4);
  BOOST_CHECK_THROW(Potential(vector_state, 2, 0), crocoddyl::Exception);
  crocoddyl::DataCollectorMultibodyTpl<Scalar> no_params(
      fixture.pinocchio.get());
  BOOST_CHECK_THROW(Potential(fixture.state, fixture.actuation->get_nu(), np,
                              Scalar(0), "z_inertial")
                        .createData(&no_params),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(Potential(fixture.state, fixture.actuation->get_nu(), np,
                              Scalar(0), "a_actuation")
                        .createData(fixture.collector.get()),
                    crocoddyl::Exception);

  fixture.manager->changeParamStatus("z_inertial", false);
  fixture.manager_data->resize(fixture.manager.get());
  BOOST_CHECK_THROW(potential.calc(potential_data, fixture.x, fixture.u),
                    crocoddyl::Exception);
  fixture.manager->changeParamStatus("z_inertial", true);
  fixture.manager_data->resize(fixture.manager.get());
  fixture.manager->update(fixture.manager_data, fixture.p);
  BOOST_CHECK_NO_THROW(potential.calc(potential_data, fixture.x, fixture.u));
}

template <typename Scalar>
void prepareObserver(
    ResidualFixtureTpl<Scalar>& fixture, ObserverPayloadTpl<Scalar>& observer,
    const typename ResidualFixtureTpl<Scalar>::VectorXs& xnext) {
  typedef typename ResidualFixtureTpl<Scalar>::VectorXs VectorXs;
  const std::size_t ndx = fixture.state->get_ndx();
  observer.xnext = xnext;
  observer.Fx.setIdentity();
  observer.Fx.diagonal().array() += Scalar(0.02);
  for (Eigen::Index i = 0; i < observer.Fu.rows(); ++i) {
    for (Eigen::Index j = 0; j < observer.Fu.cols(); ++j) {
      observer.Fu(i, j) = Scalar(0.002) * Scalar(i + j + 1);
    }
  }
  for (Eigen::Index i = 0; i < observer.Fp.rows(); ++i) {
    for (Eigen::Index j = 0; j < observer.Fp.cols(); ++j) {
      observer.Fp(i, j) = Scalar(0.0002) * Scalar(1 + (i + 2 * j) % 7);
    }
  }
  observer.dissipative_E[0] = Scalar(0.07);
  observer.Ex.row(0) =
      VectorXs::LinSpaced(ndx, Scalar(-0.004), Scalar(0.006)).transpose();
  observer.Eu.row(0) =
      VectorXs::LinSpaced(observer.Eu.cols(), Scalar(-0.005), Scalar(0.007))
          .transpose();
  observer.Ep.row(0) = VectorXs::LinSpaced(fixture.manager->get_np(),
                                           Scalar(-0.003), Scalar(0.005))
                           .transpose();
  fixture.collector->shareObserverData(&observer);
  BOOST_REQUIRE_EQUAL(observer.Fx.rows(), static_cast<Eigen::Index>(ndx));
}

template <typename Scalar>
void test_power_and_no_allocation() {
  typedef crocoddyl::ResidualModelPowerTpl<Scalar> Power;
  typedef typename Power::VectorXs VectorXs;
  typedef typename Power::MatrixXs MatrixXs;
  ResidualFixtureTpl<Scalar> fixture;
  const std::size_t ndx = fixture.state->get_ndx();
  const std::size_t np = fixture.manager->get_np();
  const std::size_t nu = fixture.actuation->get_nu();
  const Scalar step = ScalarTraits<Scalar>::step();
  const Scalar tol = ScalarTraits<Scalar>::tolerance();

  VectorXs xnext = fixture.x;
  VectorXs next_dx = VectorXs::LinSpaced(ndx, Scalar(-0.015), Scalar(0.02));
  fixture.state->integrate(fixture.x, next_dx, xnext);
  ObserverPayloadTpl<Scalar> observer(fixture.state->get_nx(), ndx, nu, np);
  prepareObserver(fixture, observer, xnext);

  Power power(fixture.state, nu, np, Scalar(0.025), "z_inertial",
              "a_actuation");
  const std::shared_ptr<typename Power::ResidualDataAbstract> data =
      power.createData(fixture.collector.get());
  power.calc(data, fixture.x, fixture.u);
  power.calcDiff(data, fixture.x, fixture.u);
  const MatrixXs analytical_x = data->Rx;
  const MatrixXs analytical_u = data->Ru;
  const MatrixXs analytical_p = data->Rp;

  const VectorXs base_xnext = observer.xnext;
  const Scalar base_dissipation = observer.dissipative_E[0];
  MatrixXs numerical_x(1, ndx);
  MatrixXs numerical_u(1, nu);
  MatrixXs numerical_p(1, np);
  VectorXs delta = VectorXs::Zero(ndx);
  VectorXs xpert(fixture.state->get_nx());
  VectorXs next_pert(fixture.state->get_nx());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(ndx); ++i) {
    delta.setZero();
    delta[i] = step;
    fixture.state->integrate(fixture.x, delta, xpert);
    fixture.state->integrate(base_xnext, observer.Fx * delta, next_pert);
    observer.xnext = next_pert;
    observer.dissipative_E[0] = base_dissipation + (observer.Ex * delta)[0];
    power.calc(data, xpert, fixture.u);
    const Scalar rp = data->r[0];
    delta[i] = -step;
    fixture.state->integrate(fixture.x, delta, xpert);
    fixture.state->integrate(base_xnext, observer.Fx * delta, next_pert);
    observer.xnext = next_pert;
    observer.dissipative_E[0] = base_dissipation + (observer.Ex * delta)[0];
    power.calc(data, xpert, fixture.u);
    numerical_x(0, i) = (rp - data->r[0]) / (Scalar(2) * step);
  }
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nu); ++i) {
    VectorXs du = VectorXs::Zero(nu);
    du[i] = step;
    fixture.state->integrate(base_xnext, observer.Fu * du, observer.xnext);
    observer.dissipative_E[0] = base_dissipation + (observer.Eu * du)[0];
    power.calc(data, fixture.x, fixture.u + du);
    const Scalar rp = data->r[0];
    du[i] = -step;
    fixture.state->integrate(base_xnext, observer.Fu * du, observer.xnext);
    observer.dissipative_E[0] = base_dissipation + (observer.Eu * du)[0];
    power.calc(data, fixture.x, fixture.u + du);
    numerical_u(0, i) = (rp - data->r[0]) / (Scalar(2) * step);
  }
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(np); ++i) {
    VectorXs dp = VectorXs::Zero(np);
    dp[i] = step;
    fixture.manager->update(fixture.manager_data, fixture.p + dp);
    fixture.state->integrate(base_xnext, observer.Fp * dp, observer.xnext);
    observer.dissipative_E[0] = base_dissipation + (observer.Ep * dp)[0];
    power.calc(data, fixture.x, fixture.u);
    const Scalar rp = data->r[0];
    dp[i] = -step;
    fixture.manager->update(fixture.manager_data, fixture.p + dp);
    fixture.state->integrate(base_xnext, observer.Fp * dp, observer.xnext);
    observer.dissipative_E[0] = base_dissipation + (observer.Ep * dp)[0];
    power.calc(data, fixture.x, fixture.u);
    numerical_p(0, i) = (rp - data->r[0]) / (Scalar(2) * step);
  }
  fixture.manager->update(fixture.manager_data, fixture.p);
  observer.xnext = base_xnext;
  observer.dissipative_E[0] = base_dissipation;
  power.calc(data, fixture.x, fixture.u);
  power.calcDiff(data, fixture.x, fixture.u);
  BOOST_CHECK(analytical_x.isApprox(numerical_x, Scalar(5) * tol));
  BOOST_CHECK(analytical_u.isApprox(numerical_u, Scalar(5) * tol));
  BOOST_CHECK(analytical_p.isApprox(numerical_p, Scalar(8) * tol));

  typedef typename ResidualFixtureTpl<Scalar>::Manager Manager;
  typedef typename ResidualFixtureTpl<Scalar>::ManagerData ManagerData;
  const std::shared_ptr<Manager> action_manager =
      std::make_shared<Manager>(fixture.state);
  action_manager->addParam("time", fixture.time_params);
  const std::shared_ptr<ManagerData> action_manager_data =
      action_manager->createData();
  const VectorXs action_p = action_manager->zero();
  action_manager->update(action_manager_data, action_p);
  ResidualCollectorTpl<Scalar> action_collector(fixture.pinocchio.get(),
                                                action_manager_data);
  ObserverPayloadTpl<Scalar> action_observer(fixture.state->get_nx(), ndx, nu,
                                             action_manager->get_np());
  action_observer.xnext = base_xnext;
  action_observer.Fp.col(0) =
      VectorXs::LinSpaced(ndx, Scalar(-0.006), Scalar(0.009));
  action_collector.shareObserverData(&action_observer);
  Power action_power(fixture.state, nu, action_manager->get_np());
  const std::shared_ptr<typename Power::ResidualDataAbstract> action_data =
      action_power.createData(&action_collector);
  action_power.calc(action_data, fixture.x, fixture.u);
  action_power.calcDiff(action_data, fixture.x, fixture.u);
  const Scalar analytical_action_p = action_data->Rp(0, 0);
  fixture.state->integrate(base_xnext, action_observer.Fp.col(0) * step,
                           action_observer.xnext);
  action_power.calc(action_data, fixture.x, fixture.u);
  const Scalar action_rp = action_data->r[0];
  fixture.state->integrate(base_xnext, action_observer.Fp.col(0) * (-step),
                           action_observer.xnext);
  action_power.calc(action_data, fixture.x, fixture.u);
  const Scalar numerical_action_p =
      (action_rp - action_data->r[0]) / (Scalar(2) * step);
  BOOST_CHECK_GT(std::abs(analytical_action_p), tol);
  BOOST_CHECK_SMALL(
      static_cast<double>(analytical_action_p - numerical_action_p),
      static_cast<double>(Scalar(8) * tol));

  power.calc(data, fixture.x);
  power.calcDiff(data, fixture.x);
  BOOST_CHECK(data->r.isZero());
  BOOST_CHECK(data->Rx.isZero());
  BOOST_CHECK(data->Ru.isZero());
  BOOST_CHECK(data->Rp.isZero());

  crocoddyl::ResidualModelPotentialEnergyTpl<Scalar> potential(
      fixture.state, nu, np, Scalar(0), "z_inertial");
  crocoddyl::ResidualModelKineticEnergyTpl<Scalar> kinetic(
      fixture.state, nu, np, Scalar(0), "z_inertial");
  const std::shared_ptr<typename decltype(potential)::ResidualDataAbstract>
      potential_data = potential.createData(fixture.collector.get());
  const std::shared_ptr<typename decltype(kinetic)::ResidualDataAbstract>
      kinetic_data = kinetic.createData(fixture.collector.get());
  potential.calc(potential_data, fixture.x, fixture.u);
  potential.calcDiff(potential_data, fixture.x, fixture.u);
  kinetic.calc(kinetic_data, fixture.x, fixture.u);
  kinetic.calcDiff(kinetic_data, fixture.x, fixture.u);
  power.calc(data, fixture.x, fixture.u);
  power.calcDiff(data, fixture.x, fixture.u);

  typedef crocoddyl::ResidualModelSymmetryParametersTpl<Scalar> Symmetry;
  typedef crocoddyl::ResidualModelTotalMassTpl<Scalar> TotalMass;
  const std::shared_ptr<crocoddyl::ActuationMultibodyParamsDataTpl<Scalar> >
      actuation = std::dynamic_pointer_cast<
          crocoddyl::ActuationMultibodyParamsDataTpl<Scalar> >(
          fixture.manager_data->dynamics_params.at("a_actuation"));
  const std::shared_ptr<crocoddyl::MultibodyInertialParamsDataTpl<Scalar> >
      inertial = std::dynamic_pointer_cast<
          crocoddyl::MultibodyInertialParamsDataTpl<Scalar> >(
          fixture.manager_data->dynamics_params.at("z_inertial"));
  BOOST_REQUIRE(actuation != nullptr);
  BOOST_REQUIRE(inertial != nullptr);
  Symmetry generic_symmetry(fixture.state, MatrixXs::Identity(np, np), nu, np);
  Symmetry actuation_symmetry(fixture.state,
                              MatrixXs::Identity(actuation->np, actuation->np),
                              nu, np, "a_actuation");
  Symmetry inertial_symmetry(fixture.state,
                             MatrixXs::Identity(inertial->np, inertial->np), nu,
                             np, "z_inertial");
  TotalMass total_mass(
      fixture.state,
      pinocchio::computeTotalMass(*fixture.state->get_pinocchio()), nu, np,
      "z_inertial");
  const std::shared_ptr<typename Symmetry::ResidualDataAbstract> generic_data =
      generic_symmetry.createData(fixture.collector.get());
  const std::shared_ptr<typename Symmetry::ResidualDataAbstract>
      actuation_symmetry_data =
          actuation_symmetry.createData(fixture.collector.get());
  const std::shared_ptr<typename Symmetry::ResidualDataAbstract>
      inertial_symmetry_data =
          inertial_symmetry.createData(fixture.collector.get());
  const std::shared_ptr<typename TotalMass::ResidualDataAbstract>
      total_mass_data = total_mass.createData(fixture.collector.get());
  generic_symmetry.calc(generic_data, fixture.x, fixture.u);
  generic_symmetry.calcDiff(generic_data, fixture.x, fixture.u);
  actuation_symmetry.calc(actuation_symmetry_data, fixture.x, fixture.u);
  actuation_symmetry.calcDiff(actuation_symmetry_data, fixture.x, fixture.u);
  inertial_symmetry.calc(inertial_symmetry_data, fixture.x, fixture.u);
  inertial_symmetry.calcDiff(inertial_symmetry_data, fixture.x, fixture.u);
  total_mass.calc(total_mass_data, fixture.x, fixture.u);
  total_mass.calcDiff(total_mass_data, fixture.x, fixture.u);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      potential.calc(potential_data, fixture.x, fixture.u);
      potential.calcDiff(potential_data, fixture.x, fixture.u);
      kinetic.calc(kinetic_data, fixture.x, fixture.u);
      kinetic.calcDiff(kinetic_data, fixture.x, fixture.u);
      power.calc(data, fixture.x, fixture.u);
      power.calcDiff(data, fixture.x, fixture.u);
      generic_symmetry.calc(generic_data, fixture.x, fixture.u);
      generic_symmetry.calcDiff(generic_data, fixture.x, fixture.u);
      actuation_symmetry.calc(actuation_symmetry_data, fixture.x, fixture.u);
      actuation_symmetry.calcDiff(actuation_symmetry_data, fixture.x,
                                  fixture.u);
      inertial_symmetry.calc(inertial_symmetry_data, fixture.x, fixture.u);
      inertial_symmetry.calcDiff(inertial_symmetry_data, fixture.x, fixture.u);
      total_mass.calc(total_mass_data, fixture.x, fixture.u);
      total_mass.calcDiff(total_mass_data, fixture.x, fixture.u);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  crocoddyl::DataCollectorMultibodyParamsTpl<Scalar> missing_observer(
      fixture.pinocchio.get(), fixture.manager_data->params,
      fixture.manager_data.get());
  const std::shared_ptr<typename Power::ResidualDataAbstract>
      missing_observer_data = power.createData(&missing_observer);
  missing_observer_data->r.setOnes();
  missing_observer_data->Rx.setOnes();
  missing_observer_data->Ru.setOnes();
  missing_observer_data->Rp.setOnes();
  power.calc(missing_observer_data, fixture.x, fixture.u);
  power.calcDiff(missing_observer_data, fixture.x, fixture.u);
  BOOST_CHECK(missing_observer_data->r.isZero());
  BOOST_CHECK(missing_observer_data->Rx.isZero());
  BOOST_CHECK(missing_observer_data->Ru.isZero());
  BOOST_CHECK(missing_observer_data->Rp.isZero());

  ResidualCollectorTpl<Scalar> incomplete_observer(fixture.pinocchio.get(),
                                                   fixture.manager_data);
  const std::shared_ptr<typename Power::ResidualDataAbstract>
      incomplete_observer_data = power.createData(&incomplete_observer);
  incomplete_observer_data->r.setOnes();
  incomplete_observer_data->Rx.setOnes();
  incomplete_observer_data->Ru.setOnes();
  incomplete_observer_data->Rp.setOnes();
  power.calc(incomplete_observer_data, fixture.x, fixture.u);
  power.calcDiff(incomplete_observer_data, fixture.x, fixture.u);
  BOOST_CHECK(incomplete_observer_data->r.isZero());
  BOOST_CHECK(incomplete_observer_data->Rx.isZero());
  BOOST_CHECK(incomplete_observer_data->Ru.isZero());
  BOOST_CHECK(incomplete_observer_data->Rp.isZero());

  incomplete_observer.xnext = &observer.xnext;
  incomplete_observer_data->Rx.setOnes();
  incomplete_observer_data->Ru.setOnes();
  incomplete_observer_data->Rp.setOnes();
  power.calcDiff(incomplete_observer_data, fixture.x, fixture.u);
  BOOST_CHECK(incomplete_observer_data->Rx.isZero());
  BOOST_CHECK(incomplete_observer_data->Ru.isZero());
  BOOST_CHECK(incomplete_observer_data->Rp.isZero());
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_parameter_energy_residuals");
  ts->add(BOOST_TEST_CASE(&test_parameter_residuals<double>));
  ts->add(BOOST_TEST_CASE(&test_parameter_residuals<float>));
  ts->add(BOOST_TEST_CASE(&test_energy_residuals<double>));
  ts->add(BOOST_TEST_CASE(&test_energy_residuals<float>));
  ts->add(BOOST_TEST_CASE(&test_power_and_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_power_and_no_allocation<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
