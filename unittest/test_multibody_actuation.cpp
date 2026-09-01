///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>
#include <pinocchio/multibody/sample-models.hpp>
#include <type_traits>
#include <vector>

#include "crocoddyl/core/numdiff/actuation.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/actuations/joint-friction.hpp"
#include "crocoddyl/multibody/actuations/joint-thruster.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename _Scalar>
class DynamicsDataProbeTpl
    : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase, DynamicsDataProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  DynamicsDataProbeTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t nu)
      : Base(state, crocoddyl::DynamicsType::ContinuousControl, 0, nu) {}

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>&,
            const Eigen::Ref<const VectorXs>&) override {
    data->vdot.setZero();
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setZero();
    data->Fu.setZero();
  }

  template <typename NewScalar>
  DynamicsDataProbeTpl<NewScalar> cast() const {
    return DynamicsDataProbeTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>(), this->get_nu());
  }
};

template <typename Scalar>
std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > createState() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  std::shared_ptr<pinocchio::Model> model =
      std::make_shared<pinocchio::Model>();
  pinocchio::buildModels::humanoidRandom(*model.get(), true);
  model->lowerPositionLimit.template segment<7>(0).fill(-1.);
  model->upperPositionLimit.template segment<7>(0).fill(1.);
  const crocoddyl::StateMultibody state(model);
  return std::make_shared<StateMultibody>(state.template cast<Scalar>());
}

template <typename Scalar>
pinocchio::JointIndex firstActuatedJoint(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> >& state) {
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state->get_pinocchio();
  return pin_model->existJointName("root_joint")
             ? pin_model->getJointId("root_joint") + 1
             : 1;
}

template <typename Scalar>
pinocchio::JointIndex firstSingleDofActuatedJoint(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> >& state) {
  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state->get_pinocchio();
  const pinocchio::JointIndex first_joint = firstActuatedJoint(state);
  for (pinocchio::JointIndex jid = first_joint;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    if (pin_model->joints[jid].nv() == 1) {
      return jid;
    }
  }
  throw_pretty("The test model does not contain any actuated 1-DoF joint");
}

void check_fixed_smoothing_friction_parity(
    const crocoddyl::JointFrictionType base_type,
    const crocoddyl::JointFrictionType fixed_type,
    const Eigen::VectorXd& base_mu, const Eigen::VectorXd& fixed_mu,
    const Eigen::VectorXd& fixed_smoothing,
    const std::vector<Eigen::Index>& regressor_columns) {
  typedef crocoddyl::JointDynamicsModelFriction Model;
  typedef crocoddyl::JointDynamicsDataAbstract Data;

  Model base_model(3, 1, base_mu, base_type);
  Model fixed_model(3, 1, fixed_mu, fixed_type, false, fixed_smoothing);
  std::shared_ptr<Data> base_data = base_model.createData();
  std::shared_ptr<Data> fixed_data = fixed_model.createData();

  BOOST_CHECK_EQUAL(fixed_model.get_np(),
                    static_cast<std::size_t>(fixed_mu.size()));
  BOOST_CHECK(fixed_model.get_parametrization().isApprox(fixed_mu));
  BOOST_CHECK(fixed_model.get_fixed_smoothing_parametrization().isApprox(
      fixed_smoothing));

  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd v = Eigen::VectorXd::Constant(1, 0.3);
  const Eigen::VectorXd u = Eigen::VectorXd::Constant(1, 0.7);

  base_model.calc(base_data, q, v, u);
  fixed_model.calc(fixed_data, q, v, u);
  BOOST_CHECK(base_data->friction.isApprox(fixed_data->friction, 1e-12));
  BOOST_CHECK(base_data->tau.isApprox(fixed_data->tau, 1e-12));

  base_model.calcDiff(base_data, q, v, u);
  fixed_model.calcDiff(fixed_data, q, v, u);
  BOOST_CHECK(base_data->dtau_dv.isApprox(fixed_data->dtau_dv, 1e-12));
  BOOST_CHECK(base_data->dtau_du.isApprox(fixed_data->dtau_du, 1e-12));

  Eigen::MatrixXd base_regressor =
      Eigen::MatrixXd::Zero(1, static_cast<Eigen::Index>(base_model.get_np()));
  Eigen::MatrixXd fixed_regressor =
      Eigen::MatrixXd::Zero(1, static_cast<Eigen::Index>(fixed_model.get_np()));
  base_model.computeJointTorqueRegressor(base_regressor, q, v, u);
  fixed_model.computeJointTorqueRegressor(fixed_regressor, q, v, u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(fixed_regressor.cols()),
                    regressor_columns.size());
  for (std::size_t i = 0; i < regressor_columns.size(); ++i) {
    BOOST_CHECK(fixed_regressor.col(static_cast<Eigen::Index>(i))
                    .isApprox(base_regressor.col(regressor_columns[i]), 1e-12));
  }
}

template <typename Scalar>
std::vector<crocoddyl::ThrusterTpl<Scalar> > createThrusters() {
  typedef crocoddyl::ThrusterTpl<Scalar> Thruster;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  const Scalar d_cog = Scalar(0.1525);
  const Scalar cf = Scalar(6.6e-5);
  const Scalar cm = Scalar(1.e-6);
  const Scalar ctorque = cm / cf;
  std::vector<Thruster> thrusters;
  thrusters.reserve(4);
  thrusters.push_back(
      Thruster(SE3(Eigen::Matrix<Scalar, 3, 3>::Identity(),
                   Eigen::Matrix<Scalar, 3, 1>(d_cog, Scalar(0.), Scalar(0.))),
               ctorque, crocoddyl::CCW, Scalar(0.1), Scalar(5.0)));
  thrusters.push_back(
      Thruster(SE3(Eigen::Matrix<Scalar, 3, 3>::Identity(),
                   Eigen::Matrix<Scalar, 3, 1>(Scalar(0.), d_cog, Scalar(0.))),
               ctorque, crocoddyl::CW, Scalar(0.1), Scalar(5.0)));
  thrusters.push_back(
      Thruster(SE3(Eigen::Matrix<Scalar, 3, 3>::Identity(),
                   Eigen::Matrix<Scalar, 3, 1>(-d_cog, Scalar(0.), Scalar(0.))),
               ctorque, crocoddyl::CCW, Scalar(0.1), Scalar(5.0)));
  thrusters.push_back(
      Thruster(SE3(Eigen::Matrix<Scalar, 3, 3>::Identity(),
                   Eigen::Matrix<Scalar, 3, 1>(Scalar(0.), -d_cog, Scalar(0.))),
               ctorque, crocoddyl::CW, Scalar(0.1), Scalar(5.0)));
  return thrusters;
}

template <typename Scalar>
std::vector<std::shared_ptr<crocoddyl::JointDynamicsModelAbstractTpl<Scalar> > >
createThrusterActuationJoints(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> >& state,
    const std::vector<crocoddyl::ThrusterTpl<Scalar> >& thrusters) {
  typedef crocoddyl::JointDynamicsModelAbstractTpl<Scalar>
      JointDynamicsModelAbstract;
  typedef crocoddyl::JointDynamicsModelIdentityTpl<Scalar>
      JointDynamicsModelIdentity;
  typedef crocoddyl::JointDynamicsModelThrusterTpl<Scalar>
      JointDynamicsModelThruster;

  const std::shared_ptr<pinocchio::ModelTpl<Scalar> >& pin_model =
      state->get_pinocchio();
  const pinocchio::JointIndex root_joint_id =
      pin_model->getJointId("root_joint");
  std::vector<std::shared_ptr<JointDynamicsModelAbstract> > joints;
  joints.reserve(static_cast<std::size_t>(pin_model->njoints) - 1);
  joints.push_back(std::make_shared<JointDynamicsModelThruster>(thrusters));
  for (pinocchio::JointIndex jid = root_joint_id + 1;
       jid < static_cast<pinocchio::JointIndex>(pin_model->njoints); ++jid) {
    const auto& joint = pin_model->joints[jid];
    joints.push_back(std::make_shared<JointDynamicsModelIdentity>(
        jid, static_cast<std::size_t>(joint.nq()),
        static_cast<std::size_t>(joint.nv())));
  }
  return joints;
}

}  // namespace

void test_joint_identity_data() {
  typedef crocoddyl::JointDynamicsModelIdentity Model;
  typedef crocoddyl::JointDynamicsDataAbstract Data;

  Model model(2, 1, 1);
  std::shared_ptr<Data> data = model.createData();

  BOOST_CHECK_EQUAL(model.get_id(), 2);
  BOOST_CHECK_EQUAL(model.get_nq(), 1);
  BOOST_CHECK_EQUAL(model.get_nv(), 1);
  BOOST_CHECK_EQUAL(model.get_nu(), 1);
  BOOST_CHECK(data->dtau_dq.isZero());
  BOOST_CHECK(data->dtau_dv.isZero());
  BOOST_CHECK(data->dtau_du.isApprox(Eigen::MatrixXd::Identity(1, 1)));
  BOOST_CHECK(data->Mtau.isApprox(Eigen::MatrixXd::Identity(1, 1)));
}

void test_default_multibody_against_numdiff() {
  typedef crocoddyl::StateMultibody StateMultibody;
  typedef crocoddyl::ActuationModelMultibody MultibodyModel;
  typedef crocoddyl::ActuationDataMultibody MultibodyData;

  std::shared_ptr<StateMultibody> state = createState<double>();
  std::shared_ptr<MultibodyModel> model =
      std::make_shared<MultibodyModel>(state);
  crocoddyl::ActuationModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract> data =
      model->createData();
  const std::shared_ptr<MultibodyData> multibody_data =
      std::dynamic_pointer_cast<MultibodyData>(data);
  BOOST_REQUIRE(multibody_data != nullptr);
  const std::shared_ptr<crocoddyl::ActuationDataAbstract> data_num_diff =
      model_num_diff.createData();

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u =
      Eigen::VectorXd::Random(static_cast<Eigen::Index>(model->get_nu()));
  const std::shared_ptr<pinocchio::Model> pin_model = state->get_pinocchio();
  const pinocchio::JointIndex root_id = pin_model->getJointId("root_joint");
  const Eigen::Index root_nv = pin_model->joints[root_id].nv();
  BOOST_REQUIRE_EQUAL(static_cast<Eigen::Index>(model->get_nu()),
                      state->get_nv() - root_nv);
  BOOST_CHECK(model->get_u_lb().isApprox(-pin_model->effortLimit.tail(
      static_cast<Eigen::Index>(model->get_nu()))));
  BOOST_CHECK(model->get_u_ub().isApprox(
      pin_model->effortLimit.tail(static_cast<Eigen::Index>(model->get_nu()))));
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(state->get_nv());
       ++i) {
    BOOST_CHECK_EQUAL(data->tau_set[static_cast<std::size_t>(i)], i >= root_nv);
  }

  model->calc(data, x, u);
  Eigen::VectorXd expected_tau = Eigen::VectorXd::Zero(state->get_nv());
  expected_tau.tail(static_cast<Eigen::Index>(model->get_nu())) = u;
  BOOST_CHECK(data->tau.isApprox(expected_tau));
  BOOST_REQUIRE_EQUAL(multibody_data->joint[root_id].size(), 1);
  multibody_data->joint[root_id][0]->tau.setConstant(7.);
  multibody_data->joint[root_id][0]->friction.setConstant(8.);
  data->u.setConstant(42.);
  model->calcDiff(data, x, u);
  BOOST_CHECK(data->u.isConstant(42.));
  BOOST_CHECK(multibody_data->joint[root_id][0]->tau.isConstant(7.));
  BOOST_CHECK(multibody_data->joint[root_id][0]->friction.isConstant(8.));
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  const double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->dtau_dx - data_num_diff->dtau_dx).isZero(tol));
  BOOST_CHECK((data->dtau_du - data_num_diff->dtau_du).isZero(tol));

  const Eigen::VectorXd tau = Eigen::VectorXd::Random(state->get_nv());
  model->commands(data, x, tau);
  BOOST_CHECK(
      data->u.isApprox(tau.tail(static_cast<Eigen::Index>(model->get_nu()))));
  model->torqueTransform(data, x, u);
  Eigen::MatrixXd expected_Mtau =
      Eigen::MatrixXd::Zero(model->get_nu(), state->get_nv());
  expected_Mtau.rightCols(static_cast<Eigen::Index>(model->get_nu()))
      .setIdentity();
  BOOST_CHECK(data->Mtau.isApprox(expected_Mtau));
}

void test_partial_joint_selection() {
  typedef crocoddyl::StateMultibody StateMultibody;
  typedef crocoddyl::ActuationModelMultibody MultibodyModel;
  typedef crocoddyl::JointDynamicsModelAbstract JointDynamicsModelAbstract;
  typedef crocoddyl::JointDynamicsModelIdentity JointIdentityModel;
  typedef crocoddyl::ActuationDataAbstract ActuationDataAbstract;

  std::shared_ptr<StateMultibody> state = createState<double>();
  const std::shared_ptr<pinocchio::Model> pin_model = state->get_pinocchio();
  const pinocchio::JointIndex joint_id = firstActuatedJoint(state);
  const auto& joint = pin_model->joints[joint_id];
  std::vector<std::shared_ptr<JointDynamicsModelAbstract> > joints;
  joints.push_back(std::make_shared<JointIdentityModel>(
      joint_id, static_cast<std::size_t>(joint.nq()),
      static_cast<std::size_t>(joint.nv())));
  MultibodyModel model(state, joints);
  std::shared_ptr<ActuationDataAbstract> data = model.createData();

  BOOST_CHECK_EQUAL(model.get_nu(), static_cast<std::size_t>(joint.nv()));

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u =
      Eigen::VectorXd::Random(static_cast<Eigen::Index>(model.get_nu()));
  model.calc(data, x, u);

  Eigen::VectorXd expected_tau = Eigen::VectorXd::Zero(state->get_nv());
  expected_tau.segment(joint.idx_v(), joint.nv()) = u;
  BOOST_CHECK(data->tau.isApprox(expected_tau));

  model.calcDiff(data, x, u);
  BOOST_CHECK(data->dtau_dx.isZero());
  BOOST_CHECK(data->dtau_du.block(joint.idx_v(), 0, joint.nv(), joint.nv())
                  .isApprox(Eigen::MatrixXd::Identity(joint.nv(), joint.nv())));

  const Eigen::VectorXd tau = Eigen::VectorXd::Random(state->get_nv());
  model.commands(data, x, tau);
  BOOST_CHECK(data->u.isApprox(tau.segment(joint.idx_v(), joint.nv())));

  model.torqueTransform(data, x, u);
  BOOST_CHECK(data->Mtau.block(0, joint.idx_v(), joint.nv(), joint.nv())
                  .isApprox(Eigen::MatrixXd::Identity(joint.nv(), joint.nv())));
}

void test_joint_friction_model() {
  typedef crocoddyl::JointDynamicsModelFriction Model;
  typedef crocoddyl::JointDynamicsDataAbstract Data;

  Eigen::VectorXd mu(3);
  mu << std::log(0.2), std::log(4.), std::log(0.5);
  Model model(3, 1, mu, crocoddyl::JointFrictionType::CoulombViscous);
  std::shared_ptr<Data> data = model.createData();

  BOOST_CHECK_EQUAL(model.get_id(), 3);
  BOOST_CHECK_EQUAL(model.get_nq(), 1);
  BOOST_CHECK_EQUAL(model.get_nv(), 1);
  BOOST_CHECK_EQUAL(model.get_nu(), 1);
  BOOST_CHECK_EQUAL(model.get_np(), 3);
  BOOST_CHECK(model.get_friction_type() ==
              crocoddyl::JointFrictionType::CoulombViscous);
  BOOST_CHECK(!model.get_passive());
  BOOST_CHECK(model.get_parametrization().isApprox(mu));

  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd v = Eigen::VectorXd::Constant(1, 0.3);
  const Eigen::VectorXd u = Eigen::VectorXd::Constant(1, 0.7);
  const double gamma3 = 0.2;
  const double gamma4 = 4.;
  const double gamma5 = 0.5;
  const double tanh4v = std::tanh(gamma4 * v[0]);
  const double expected_friction = gamma3 * tanh4v + gamma5 * v[0];

  model.calc(data, q, v, u);
  BOOST_CHECK_SMALL(data->friction[0] - expected_friction, 1e-12);
  BOOST_CHECK_SMALL(data->tau[0] - (u[0] - expected_friction), 1e-12);

  model.calcDiff(data, q, v, u);
  const double expected_dtau_dv =
      -gamma3 * gamma4 * (1. - tanh4v * tanh4v) - gamma5;
  BOOST_CHECK_SMALL(data->dtau_dv(0, 0) - expected_dtau_dv, 1e-12);
  BOOST_CHECK_SMALL(data->dtau_du(0, 0) - 1., 1e-12);

  Eigen::MatrixXd regressor = Eigen::MatrixXd::Zero(1, 3);
  model.computeJointTorqueRegressor(regressor, q, v, u);
  BOOST_CHECK_SMALL(regressor(0, 0) - gamma3 * tanh4v, 1e-12);
  BOOST_CHECK_SMALL(
      regressor(0, 1) - gamma3 * (1. - tanh4v * tanh4v) * gamma4 * v[0], 1e-12);
  BOOST_CHECK_SMALL(regressor(0, 2) - gamma5 * v[0], 1e-12);

  const Eigen::VectorXd tau = Eigen::VectorXd::Constant(1, -0.1);
  model.commands(data, q, v, tau);
  BOOST_CHECK_SMALL(data->u[0] - (tau[0] + expected_friction), 1e-12);
}

void test_joint_friction_fixed_smoothing_modes() {
  Eigen::VectorXd base_mu(2);
  Eigen::VectorXd fixed_mu(1);
  Eigen::VectorXd fixed_smoothing(1);
  base_mu << std::log(0.2), std::log(4.);
  fixed_mu << std::log(0.2);
  fixed_smoothing << std::log(4.);
  check_fixed_smoothing_friction_parity(
      crocoddyl::JointFrictionType::Coulomb,
      crocoddyl::JointFrictionType::CoulombFixedSmoothing, base_mu, fixed_mu,
      fixed_smoothing, std::vector<Eigen::Index>{0});

  base_mu.resize(2);
  fixed_mu.resize(0);
  fixed_smoothing.resize(2);
  base_mu << 2.0, 5.0;
  fixed_smoothing << 2.0, 5.0;
  check_fixed_smoothing_friction_parity(
      crocoddyl::JointFrictionType::Stribeck,
      crocoddyl::JointFrictionType::StribeckFixedSmoothing, base_mu, fixed_mu,
      fixed_smoothing, std::vector<Eigen::Index>());

  base_mu.resize(3);
  fixed_mu.resize(2);
  fixed_smoothing.resize(1);
  base_mu << std::log(0.2), std::log(4.), std::log(0.5);
  fixed_mu << std::log(0.2), std::log(0.5);
  fixed_smoothing << std::log(4.);
  check_fixed_smoothing_friction_parity(
      crocoddyl::JointFrictionType::CoulombViscous,
      crocoddyl::JointFrictionType::CoulombViscousFixedSmoothing, base_mu,
      fixed_mu, fixed_smoothing, std::vector<Eigen::Index>{0, 2});

  base_mu.resize(4);
  fixed_mu.resize(1);
  fixed_smoothing.resize(3);
  base_mu << 2.0, 5.0, 0.2, 4.0;
  fixed_mu << 0.2;
  fixed_smoothing << 2.0, 5.0, 4.0;
  check_fixed_smoothing_friction_parity(
      crocoddyl::JointFrictionType::CoulombStribeck,
      crocoddyl::JointFrictionType::CoulombStribeckFixedSmoothing, base_mu,
      fixed_mu, fixed_smoothing, std::vector<Eigen::Index>{2});

  base_mu.resize(3);
  fixed_mu.resize(1);
  fixed_smoothing.resize(2);
  base_mu << 2.0, 5.0, std::log(0.5);
  fixed_mu << std::log(0.5);
  fixed_smoothing << 2.0, 5.0;
  check_fixed_smoothing_friction_parity(
      crocoddyl::JointFrictionType::ViscousStribeck,
      crocoddyl::JointFrictionType::ViscousStribeckFixedSmoothing, base_mu,
      fixed_mu, fixed_smoothing, std::vector<Eigen::Index>{2});

  base_mu.resize(6);
  fixed_mu.resize(3);
  fixed_smoothing.resize(3);
  base_mu << 1.3, 2.0, 5.0, 0.2, 4.0, std::log(0.5);
  fixed_mu << 1.3, 0.2, std::log(0.5);
  fixed_smoothing << 2.0, 5.0, 4.0;
  check_fixed_smoothing_friction_parity(
      crocoddyl::JointFrictionType::Full,
      crocoddyl::JointFrictionType::FullFixedSmoothing, base_mu, fixed_mu,
      fixed_smoothing, std::vector<Eigen::Index>{0, 3, 5});
}

void test_passive_joint_friction_model() {
  typedef crocoddyl::JointDynamicsModelFriction Model;
  typedef crocoddyl::JointDynamicsDataAbstract Data;

  Eigen::VectorXd mu(1);
  mu << std::log(0.4);
  Model model(4, 1, mu, crocoddyl::JointFrictionType::Viscous, true);
  std::shared_ptr<Data> data = model.createData();
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd v = Eigen::VectorXd::Constant(1, -0.2);
  const Eigen::VectorXd u(0);

  BOOST_CHECK(model.get_passive());
  BOOST_CHECK_EQUAL(model.get_nu(), 0);
  model.calc(data, q, v, u);
  BOOST_CHECK_SMALL(data->tau[0] - 0.08, 1e-12);
  model.calcDiff(data, q, v, u);
  BOOST_CHECK_SMALL(data->dtau_dv(0, 0) + 0.4, 1e-12);
}

void test_joint_thruster_model() {
  typedef crocoddyl::JointDynamicsModelThruster Model;
  typedef crocoddyl::JointDynamicsDataAbstract Data;

  const std::vector<crocoddyl::Thruster> thrusters = createThrusters<double>();
  Model model(thrusters);
  std::shared_ptr<Data> data = model.createData();

  BOOST_CHECK_EQUAL(model.get_id(), 1);
  BOOST_CHECK_EQUAL(model.get_nq(), 7);
  BOOST_CHECK_EQUAL(model.get_nv(), 6);
  BOOST_CHECK_EQUAL(model.get_nu(), 4);
  BOOST_CHECK_EQUAL(model.get_nthrusters(), 4);
  BOOST_CHECK_EQUAL(model.get_np(), 4);

  const Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  const Eigen::VectorXd v = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd u = Eigen::VectorXd::Ones(4);
  model.calc(data, q, v, u);
  BOOST_CHECK_SMALL(data->tau[0], 1e-12);
  BOOST_CHECK_SMALL(data->tau[1], 1e-12);
  BOOST_CHECK_SMALL(data->tau[2] - 4., 1e-12);
  BOOST_CHECK_SMALL(data->tau[3], 1e-12);
  BOOST_CHECK_SMALL(data->tau[4], 1e-12);
  BOOST_CHECK_SMALL(data->tau[5], 1e-12);

  model.calcDiff(data, q, v, u);
  BOOST_CHECK(data->dtau_dq.isZero());
  BOOST_CHECK(data->dtau_dv.isZero());
  BOOST_CHECK(data->dtau_du.isApprox(model.get_Wthrust()));

  Eigen::MatrixXd regressor = Eigen::MatrixXd::Zero(6, 4);
  const Eigen::VectorXd u_regressor =
      (Eigen::VectorXd(4) << 1., 2., 3., 4.).finished();
  model.computeJointTorqueRegressor(regressor, q, v, u_regressor);
  BOOST_CHECK_SMALL(regressor(5, 0) + 1., 1e-12);
  BOOST_CHECK_SMALL(regressor(5, 1) - 2., 1e-12);
  BOOST_CHECK_SMALL(regressor(5, 2) + 3., 1e-12);
  BOOST_CHECK_SMALL(regressor(5, 3) - 4., 1e-12);

  const Eigen::VectorXd tau =
      (Eigen::VectorXd(6) << 0.1, -0.2, 3.8, 0.3, -0.4, 0.5).finished();
  model.commands(data, q, v, tau);
  BOOST_CHECK(data->u.isApprox(data->Mtau * tau));
}

void test_actuation_multibody_params_friction() {
  typedef crocoddyl::StateMultibody StateMultibody;
  typedef crocoddyl::ActuationModelMultibody ActuationModelMultibody;
  typedef crocoddyl::ActuationMultibodyParams ActuationMultibodyParams;
  typedef crocoddyl::ActuationMultibodyParamsData ActuationMultibodyParamsData;
  typedef crocoddyl::JointDynamicsModelAbstract JointDynamicsModelAbstract;
  typedef crocoddyl::JointDynamicsModelFriction JointFrictionModel;

  std::shared_ptr<StateMultibody> state = createState<double>();
  const std::shared_ptr<pinocchio::Model> pin_model = state->get_pinocchio();
  const pinocchio::JointIndex joint_id = firstSingleDofActuatedJoint(state);
  const auto& joint = pin_model->joints[joint_id];
  Eigen::VectorXd mu(3);
  mu << std::log(0.15), std::log(3.), std::log(0.2);
  std::shared_ptr<JointFrictionModel> friction =
      std::make_shared<JointFrictionModel>(
          joint_id, static_cast<std::size_t>(joint.nq()), mu,
          crocoddyl::JointFrictionType::CoulombViscous);
  std::vector<std::shared_ptr<JointDynamicsModelAbstract> > joints;
  joints.push_back(friction);
  std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state, joints);
  ActuationMultibodyParams params(actuation);
  const std::shared_ptr<crocoddyl::ParamsDataAbstract> data_base =
      params.createData();
  const std::shared_ptr<ActuationMultibodyParamsData> data =
      std::dynamic_pointer_cast<ActuationMultibodyParamsData>(data_base);

  BOOST_CHECK_EQUAL(params.get_np(), 3);
  BOOST_CHECK(params.checkData(data_base));
  BOOST_CHECK(data != nullptr);

  Eigen::VectorXd p(3);
  p << std::log(0.25), std::log(4.5), std::log(0.6);
  params.update(data_base, p);
  BOOST_CHECK(data->p.isApprox(p));
  BOOST_CHECK(data->gamma.isApprox(friction->get_parameters()));
  BOOST_CHECK(data->dgamma_dp.isApprox(
      friction->get_parameters().asDiagonal().toDenseMatrix()));

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(actuation->get_nu());
  Eigen::MatrixXd dtau_dp(state->get_nv(), params.get_np());
  params.computeJointTorqueRegressor(
      std::shared_ptr<crocoddyl::DynamicsDataAbstract>(), data_base, dtau_dp, x,
      u);

  Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(state->get_nv(), 3);
  friction->computeJointTorqueRegressor(
      expected.block(joint.idx_v(), 0, joint.nv(), 3),
      x.segment(joint.idx_q(), joint.nq()),
      x.segment(state->get_nq() + joint.idx_v(), joint.nv()), u);
  BOOST_CHECK(dtau_dp.isApprox(expected));
}

void test_actuation_multibody_params_fixed_smoothing_friction() {
  typedef crocoddyl::StateMultibody StateMultibody;
  typedef crocoddyl::ActuationModelMultibody ActuationModelMultibody;
  typedef crocoddyl::ActuationMultibodyParams ActuationMultibodyParams;
  typedef crocoddyl::ActuationMultibodyParamsData ActuationMultibodyParamsData;
  typedef crocoddyl::JointDynamicsModelAbstract JointDynamicsModelAbstract;
  typedef crocoddyl::JointDynamicsModelFriction JointFrictionModel;
  typedef crocoddyl::ParameterManager ParameterManager;
  typedef crocoddyl::ParameterDataManager ParameterDataManagerData;

  std::shared_ptr<StateMultibody> state = createState<double>();
  const std::shared_ptr<pinocchio::Model> pin_model = state->get_pinocchio();
  const pinocchio::JointIndex joint_id = firstSingleDofActuatedJoint(state);
  const auto& joint = pin_model->joints[joint_id];
  Eigen::VectorXd mu(2);
  Eigen::VectorXd fixed_smoothing(1);
  mu << std::log(0.15), std::log(0.2);
  fixed_smoothing << std::log(3.);
  std::shared_ptr<JointFrictionModel> friction =
      std::make_shared<JointFrictionModel>(
          joint_id, static_cast<std::size_t>(joint.nq()), mu,
          crocoddyl::JointFrictionType::CoulombViscousFixedSmoothing, false,
          fixed_smoothing);
  std::vector<std::shared_ptr<JointDynamicsModelAbstract> > joints;
  joints.push_back(friction);
  std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state, joints);
  std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("actuation",
                    std::make_shared<ActuationMultibodyParams>(actuation));
  const std::shared_ptr<ParameterDataManagerData> data_base =
      manager->createData();
  const std::shared_ptr<ActuationMultibodyParamsData> data =
      std::dynamic_pointer_cast<ActuationMultibodyParamsData>(
          data_base->dynamics_params.at("actuation"));

  BOOST_CHECK_EQUAL(actuation->get_np(), 2);
  BOOST_CHECK_EQUAL(manager->get_np(), 2);
  BOOST_CHECK(data != nullptr);

  const Eigen::VectorXd p = manager->rand();
  manager->update(data_base, p);
  BOOST_CHECK(friction->get_parametrization().isApprox(p));
  BOOST_CHECK(friction->get_fixed_smoothing_parametrization().isApprox(
      fixed_smoothing));
  BOOST_CHECK(data->gamma.isApprox(friction->get_parameters()));

  Eigen::MatrixXd expected_dgamma = Eigen::MatrixXd::Zero(2, 2);
  expected_dgamma.diagonal() = friction->get_parameters();
  BOOST_CHECK(data->dgamma_dp.isApprox(expected_dgamma));

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(actuation->get_nu());
  DynamicsDataProbeTpl<double> dynamics(state, actuation->get_nu());
  const std::shared_ptr<crocoddyl::DynamicsDataAbstract> dynamics_data =
      dynamics.createData();
  Eigen::MatrixXd dtau_dp(state->get_nv(), manager->get_np_dynamics());
  manager->calcDiff_dynamics(data_base, dynamics_data, dtau_dp, x, u);

  Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(state->get_nv(), 2);
  friction->computeJointTorqueRegressor(
      expected.block(joint.idx_v(), 0, joint.nv(), 2),
      x.segment(joint.idx_q(), joint.nq()),
      x.segment(state->get_nq() + joint.idx_v(), joint.nv()), u);
  BOOST_CHECK(dtau_dp.isApprox(expected));
}

void test_actuation_multibody_params_thruster() {
  typedef crocoddyl::StateMultibody StateMultibody;
  typedef crocoddyl::ActuationModelMultibody ActuationModelMultibody;
  typedef crocoddyl::ActuationMultibodyParams ActuationMultibodyParams;
  typedef crocoddyl::ActuationMultibodyParamsData ActuationMultibodyParamsData;
  typedef crocoddyl::JointDynamicsModelAbstract JointDynamicsModelAbstract;
  typedef crocoddyl::JointDynamicsModelThruster JointThrusterModel;

  std::shared_ptr<StateMultibody> state = createState<double>();
  const std::vector<crocoddyl::Thruster> thrusters = createThrusters<double>();
  std::vector<std::shared_ptr<JointDynamicsModelAbstract> > joints =
      createThrusterActuationJoints(state, thrusters);
  std::shared_ptr<JointThrusterModel> thruster =
      std::dynamic_pointer_cast<JointThrusterModel>(joints.front());
  std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state, joints);
  ActuationMultibodyParams params(actuation);
  const std::shared_ptr<crocoddyl::ParamsDataAbstract> data_base =
      params.createData();
  const std::shared_ptr<ActuationMultibodyParamsData> data =
      std::dynamic_pointer_cast<ActuationMultibodyParamsData>(data_base);

  BOOST_CHECK_EQUAL(params.get_np(), thruster->get_np());
  BOOST_CHECK(data != nullptr);

  Eigen::VectorXd p(4);
  p << 0.2, 0.3, 0.4, 0.5;
  params.update(data_base, p);
  BOOST_CHECK(data->p.isApprox(p));
  BOOST_CHECK(data->gamma.isApprox(p));
  BOOST_CHECK(data->dgamma_dp.isApprox(Eigen::MatrixXd::Identity(4, 4)));

  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u =
      Eigen::VectorXd::Random(static_cast<Eigen::Index>(actuation->get_nu()));
  const std::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data =
      actuation->createData();
  actuation->calc(actuation_data, x, u);
  Eigen::VectorXd expected_tau = Eigen::VectorXd::Zero(state->get_nv());
  expected_tau.head(6).noalias() = thruster->get_Wthrust() * u.head(4);
  expected_tau.tail(state->get_nv() - 6) = u.tail(actuation->get_nu() - 4);
  BOOST_CHECK(actuation_data->tau.isApprox(expected_tau));
  actuation->calcDiff(actuation_data, x, u);
  Eigen::MatrixXd expected_dtau_du =
      Eigen::MatrixXd::Zero(state->get_nv(), actuation->get_nu());
  expected_dtau_du.topLeftCorner(6, 4) = thruster->get_Wthrust();
  expected_dtau_du
      .bottomRightCorner(state->get_nv() - 6, actuation->get_nu() - 4)
      .setIdentity();
  BOOST_CHECK(actuation_data->dtau_du.isApprox(expected_dtau_du));
  for (Eigen::Index i = 0; i < 6; ++i) {
    BOOST_CHECK_EQUAL(actuation_data->tau_set[static_cast<std::size_t>(i)],
                      !thruster->get_Wthrust().row(i).isZero());
  }
  for (Eigen::Index i = 6; i < static_cast<Eigen::Index>(state->get_nv());
       ++i) {
    BOOST_CHECK(actuation_data->tau_set[static_cast<std::size_t>(i)]);
  }
  for (Eigen::Index i = 0; i < 4; ++i) {
    BOOST_CHECK_EQUAL(actuation->get_u_lb()[i], thrusters[i].min_thrust);
    BOOST_CHECK_EQUAL(actuation->get_u_ub()[i], thrusters[i].max_thrust);
  }
  const std::shared_ptr<pinocchio::Model> pin_model = state->get_pinocchio();
  BOOST_CHECK(actuation->get_u_lb()
                  .tail(actuation->get_nu() - 4)
                  .isApprox(-pin_model->effortLimit.tail(state->get_nv() - 6)));
  BOOST_CHECK(actuation->get_u_ub()
                  .tail(actuation->get_nu() - 4)
                  .isApprox(pin_model->effortLimit.tail(state->get_nv() - 6)));
  Eigen::MatrixXd dtau_dp(state->get_nv(), params.get_np());
  params.computeJointTorqueRegressor(
      std::shared_ptr<crocoddyl::DynamicsDataAbstract>(), data_base, dtau_dp, x,
      u);

  Eigen::MatrixXd expected_reference =
      Eigen::MatrixXd::Zero(state->get_nv(), 4);
  thruster->computeJointTorqueRegressor(
      expected_reference.topRows(6), x.head(7), x.segment(state->get_nq(), 6),
      u.head(4));
  BOOST_CHECK(dtau_dp.isApprox(expected_reference));
}

template <typename Scalar>
void test_all_friction_modes_and_errors() {
  typedef crocoddyl::JointDynamicsModelFrictionTpl<Scalar> Model;
  typedef crocoddyl::JointDynamicsDataAbstractTpl<Scalar> Data;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  const crocoddyl::JointFrictionType types[] = {
      crocoddyl::JointFrictionType::Coulomb,
      crocoddyl::JointFrictionType::Viscous,
      crocoddyl::JointFrictionType::Stribeck,
      crocoddyl::JointFrictionType::CoulombViscous,
      crocoddyl::JointFrictionType::CoulombStribeck,
      crocoddyl::JointFrictionType::ViscousStribeck,
      crocoddyl::JointFrictionType::Full,
      crocoddyl::JointFrictionType::CoulombFixedSmoothing,
      crocoddyl::JointFrictionType::StribeckFixedSmoothing,
      crocoddyl::JointFrictionType::CoulombViscousFixedSmoothing,
      crocoddyl::JointFrictionType::CoulombStribeckFixedSmoothing,
      crocoddyl::JointFrictionType::ViscousStribeckFixedSmoothing,
      crocoddyl::JointFrictionType::FullFixedSmoothing};
  const std::size_t np[] = {2, 1, 2, 3, 4, 3, 6, 1, 0, 2, 1, 1, 3};
  const std::size_t ns[] = {0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 2, 3};
  const VectorXs q = VectorXs::Zero(1);
  const VectorXs v = VectorXs::Constant(1, Scalar(0.37));
  const VectorXs u = VectorXs::Constant(1, Scalar(0.61));
  const VectorXs no_control(0);
  const VectorXs tau = VectorXs::Constant(1, Scalar(-0.2));
  const Scalar disturbance =
      std::is_same<Scalar, float>::value ? Scalar(1e-3) : Scalar(1e-6);
  const Scalar tolerance =
      std::is_same<Scalar, float>::value ? Scalar(5e-3) : Scalar(5e-6);

  for (std::size_t i = 0; i < 13; ++i) {
    VectorXs mu(static_cast<Eigen::Index>(np[i]));
    for (Eigen::Index j = 0; j < mu.size(); ++j) {
      mu[j] = Scalar(0.15) + Scalar(0.07) * Scalar(j);
    }
    VectorXs smoothing(static_cast<Eigen::Index>(ns[i]));
    for (Eigen::Index j = 0; j < smoothing.size(); ++j) {
      smoothing[j] = Scalar(0.4) + Scalar(0.1) * Scalar(j);
    }
    Model active = ns[i] == 0 ? Model(3, 1, mu, types[i], false)
                              : Model(3, 1, mu, types[i], false, smoothing);
    BOOST_CHECK_EQUAL(active.get_np(), np[i]);
    BOOST_CHECK_EQUAL(active.get_nu(), 1);
    const std::shared_ptr<Data> data = active.createData();
    active.calc(data, q, v, u);
    active.calcDiff(data, q, v, u);

    VectorXs v_plus = v;
    VectorXs v_minus = v;
    v_plus[0] += disturbance;
    v_minus[0] -= disturbance;
    const std::shared_ptr<Data> velocity_plus = active.createData();
    const std::shared_ptr<Data> velocity_minus = active.createData();
    active.calc(velocity_plus, q, v_plus, u);
    active.calc(velocity_minus, q, v_minus, u);
    const Scalar numerical_dtau_dv =
        (velocity_plus->tau[0] - velocity_minus->tau[0]) /
        (Scalar(2) * disturbance);
    BOOST_CHECK_SMALL(
        static_cast<double>(data->dtau_dv(0, 0) - numerical_dtau_dv),
        static_cast<double>(tolerance));

    MatrixXs regressor(1, static_cast<Eigen::Index>(np[i]));
    active.computeJointTorqueRegressor(regressor, q, v, u);
    MatrixXs numerical_regressor(1, static_cast<Eigen::Index>(np[i]));
    MatrixXs numerical_dgamma(static_cast<Eigen::Index>(np[i]),
                              static_cast<Eigen::Index>(np[i]));
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(np[i]); ++j) {
      VectorXs p_plus = mu;
      VectorXs p_minus = mu;
      p_plus[j] += disturbance;
      p_minus[j] -= disturbance;
      Model model_plus(active);
      Model model_minus(active);
      model_plus.set_parameters(p_plus);
      model_minus.set_parameters(p_minus);
      const std::shared_ptr<Data> parameter_plus = model_plus.createData();
      const std::shared_ptr<Data> parameter_minus = model_minus.createData();
      model_plus.calc(parameter_plus, q, v, u);
      model_minus.calc(parameter_minus, q, v, u);
      numerical_regressor.col(j) =
          (parameter_plus->tau - parameter_minus->tau) /
          (Scalar(2) * disturbance);
      numerical_dgamma.col(j) =
          (model_plus.get_parameters() - model_minus.get_parameters()) /
          (Scalar(2) * disturbance);
    }
    BOOST_CHECK_SMALL(
        static_cast<double>((regressor + numerical_regressor).norm()),
        static_cast<double>(tolerance));
    MatrixXs dgamma_dp(static_cast<Eigen::Index>(np[i]),
                       static_cast<Eigen::Index>(np[i]));
    active.updateParametrizationDerivative(dgamma_dp);
    BOOST_CHECK_SMALL(
        static_cast<double>((dgamma_dp - numerical_dgamma).norm()),
        static_cast<double>(tolerance));

    Model passive = ns[i] == 0 ? Model(3, 1, mu, types[i], true)
                               : Model(3, 1, mu, types[i], true, smoothing);
    BOOST_CHECK(passive.get_passive());
    BOOST_CHECK_EQUAL(passive.get_nu(), 0);
    const std::shared_ptr<Data> passive_data = passive.createData();
    passive.calc(passive_data, q, v, no_control);
    passive.calcDiff(passive_data, q, v, no_control);
    const std::shared_ptr<Data> passive_velocity_plus = passive.createData();
    const std::shared_ptr<Data> passive_velocity_minus = passive.createData();
    passive.calc(passive_velocity_plus, q, v_plus, no_control);
    passive.calc(passive_velocity_minus, q, v_minus, no_control);
    const Scalar passive_numerical_dtau_dv =
        (passive_velocity_plus->tau[0] - passive_velocity_minus->tau[0]) /
        (Scalar(2) * disturbance);
    BOOST_CHECK_SMALL(static_cast<double>(passive_data->dtau_dv(0, 0) -
                                          passive_numerical_dtau_dv),
                      static_cast<double>(tolerance));
    MatrixXs passive_regressor(1, static_cast<Eigen::Index>(np[i]));
    MatrixXs passive_numerical_regressor(1, static_cast<Eigen::Index>(np[i]));
    passive.computeJointTorqueRegressor(passive_regressor, q, v, no_control);
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(np[i]); ++j) {
      VectorXs p_plus = mu;
      VectorXs p_minus = mu;
      p_plus[j] += disturbance;
      p_minus[j] -= disturbance;
      Model model_plus(passive);
      Model model_minus(passive);
      model_plus.set_parameters(p_plus);
      model_minus.set_parameters(p_minus);
      const std::shared_ptr<Data> parameter_plus = model_plus.createData();
      const std::shared_ptr<Data> parameter_minus = model_minus.createData();
      model_plus.calc(parameter_plus, q, v, no_control);
      model_minus.calc(parameter_minus, q, v, no_control);
      passive_numerical_regressor.col(j) =
          (parameter_plus->tau - parameter_minus->tau) /
          (Scalar(2) * disturbance);
    }
    BOOST_CHECK_SMALL(
        static_cast<double>(
            (passive_regressor + passive_numerical_regressor).norm()),
        static_cast<double>(tolerance));
    BOOST_CHECK_SMALL(
        static_cast<double>((passive_regressor - regressor).norm()),
        static_cast<double>(tolerance));
    BOOST_CHECK(passive_data->dtau_du.cols() == 0);

    active.commands(data, q, v, tau);
    BOOST_CHECK(data->tau.allFinite());
    BOOST_CHECK(data->friction.allFinite());
    BOOST_CHECK(data->dtau_dv.allFinite());
    BOOST_CHECK(data->u.allFinite());
  }

  BOOST_CHECK_THROW(
      Model(3, 1, VectorXs::Zero(1), crocoddyl::JointFrictionType::Full),
      std::exception);
  Model model(3, 1, VectorXs::Constant(1, Scalar(0.2)),
              crocoddyl::JointFrictionType::Viscous);
  const std::shared_ptr<Data> data = model.createData();
  BOOST_CHECK_THROW(model.calc(data, VectorXs::Zero(2), v, u), std::exception);
  BOOST_CHECK_THROW(model.calc(data, q, VectorXs::Zero(2), u), std::exception);
  BOOST_CHECK_THROW(model.calc(data, q, v, VectorXs::Zero(2)), std::exception);
}

template <typename Scalar>
void test_thruster_two_data_refresh_and_scalar_behavior() {
  typedef crocoddyl::JointDynamicsModelThrusterTpl<Scalar> Model;
  typedef crocoddyl::JointDynamicsDataAbstractTpl<Scalar> Data;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

  Model model(createThrusters<Scalar>());
  const std::shared_ptr<Data> data_a = model.createData();
  const std::shared_ptr<Data> data_b = model.createData();
  const VectorXs q = VectorXs::Zero(7);
  const VectorXs v = VectorXs::Zero(6);
  VectorXs u(4);
  u << Scalar(1), Scalar(2), Scalar(3), Scalar(4);
  model.calc(data_a, q, v, u);
  model.calc(data_b, q, v, u);
  const typename Model::MatrixXs old_mapping = data_a->dtau_du;

  VectorXs p(4);
  p << Scalar(0.2), Scalar(0.3), Scalar(0.4), Scalar(0.5);
  model.set_parameters(p);
  model.calc(data_a, q, v, u);
  model.calc(data_b, q, v, u);
  model.calcDiff(data_a, q, v, u);
  model.calcDiff(data_b, q, v, u);
  BOOST_CHECK(!old_mapping.isApprox(data_a->dtau_du));
  BOOST_CHECK(data_a->dtau_du.isApprox(model.get_Wthrust()));
  BOOST_CHECK(data_b->dtau_du.isApprox(model.get_Wthrust()));
  BOOST_CHECK(data_a->Mtau.isApprox(data_b->Mtau));

  typename Model::MatrixXs regressor(6, 4);
  model.computeJointTorqueRegressor(regressor, q, v, u);
  BOOST_CHECK_SMALL(static_cast<double>(regressor(5, 0) + Scalar(1)), 1e-5);
  BOOST_CHECK_SMALL(static_cast<double>(regressor(5, 1) - Scalar(2)), 1e-5);
  BOOST_CHECK_SMALL(static_cast<double>(regressor(5, 2) + Scalar(3)), 1e-5);
  BOOST_CHECK_SMALL(static_cast<double>(regressor(5, 3) - Scalar(4)), 1e-5);
}

template <typename Scalar>
void test_multibody_parameter_layout_manager_and_copy() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::ActuationDataMultibodyTpl<Scalar> ActuationData;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> Params;
  typedef crocoddyl::ActuationMultibodyParamsDataTpl<Scalar> ParamsData;
  typedef crocoddyl::JointDynamicsModelAbstractTpl<Scalar> JointModel;
  typedef crocoddyl::JointDynamicsModelFrictionTpl<Scalar> Friction;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

  const std::shared_ptr<StateMultibody> state = createState<Scalar>();
  const pinocchio::JointIndex joint_id = firstSingleDofActuatedJoint(state);
  const auto& joint = state->get_pinocchio()->joints[joint_id];
  VectorXs mu(3);
  mu << Scalar(-1.2), Scalar(0.7), Scalar(-1.5);
  const std::shared_ptr<Friction> friction = std::make_shared<Friction>(
      joint_id, static_cast<std::size_t>(joint.nq()), mu,
      crocoddyl::JointFrictionType::CoulombViscous);
  std::vector<std::shared_ptr<JointModel> > joints(1, friction);
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state, joints);
  const std::shared_ptr<ActuationData> actuation_data =
      std::dynamic_pointer_cast<ActuationData>(actuation->createData());
  BOOST_REQUIRE(actuation_data != nullptr);
  BOOST_CHECK_EQUAL(actuation_data->joint[joint_id].size(), 1);

  const std::shared_ptr<Params> params = std::make_shared<Params>(actuation);
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > base_data =
      params->createData();
  const std::shared_ptr<ParamsData> data =
      std::dynamic_pointer_cast<ParamsData>(base_data);
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(std::dynamic_pointer_cast<
                  crocoddyl::DynamicsParamsDataAbstractTpl<Scalar> >(
                  base_data) != nullptr);
  BOOST_CHECK_EQUAL(data->np_action, 0);
  BOOST_CHECK_EQUAL(data->np_dynamics, 3);

  VectorXs p(3);
  p << Scalar(-0.9), Scalar(0.8), Scalar(-1.1);
  params->update(base_data, p);
  BOOST_CHECK(data->p.isApprox(p));
  BOOST_CHECK(data->gamma.isApprox(friction->get_parameters()));

  ParameterManager manager(state);
  manager.addParam("actuation", params);
  const std::shared_ptr<ParameterDataManager> manager_data =
      manager.createData();
  manager.update(manager_data, p);
  BOOST_CHECK_EQUAL(manager.get_np(), 3);
  manager.changeParamStatus("actuation", false);
  BOOST_CHECK_EQUAL(manager.get_np(), 0);
  manager.changeParamStatus("actuation", true);
  BOOST_CHECK_EQUAL(manager.get_np(), 3);
  DynamicsDataProbeTpl<Scalar> dynamics(state, actuation->get_nu());
  const std::shared_ptr<crocoddyl::DynamicsDataAbstractTpl<Scalar> >
      dynamics_data = dynamics.createData();
  const VectorXs x = state->rand();
  const VectorXs u = VectorXs::Constant(actuation->get_nu(), Scalar(0.3));
  typename ParameterManager::MatrixXs dtau_dp(state->get_nv(),
                                              manager.get_np_dynamics());
  manager.calcDiff_dynamics(manager_data, dynamics_data, dtau_dp, x, u);
  BOOST_CHECK(dtau_dp.allFinite());

  ParamsData copied(*data);
  copied.gamma.setZero();
  BOOST_CHECK(!copied.gamma.isApprox(data->gamma));
  BOOST_CHECK_THROW(Params(std::shared_ptr<Actuation>()), std::exception);
  BOOST_CHECK_THROW(params->update(base_data, VectorXs::Zero(2)),
                    std::exception);
}

void test_scalar_casts() {
  crocoddyl::JointDynamicsModelIdentity identity(2, 1, 1);
  const crocoddyl::JointDynamicsModelIdentityTpl<float> identity_f =
      identity.cast<float>();
  BOOST_CHECK_EQUAL(identity_f.get_id(), identity.get_id());
  BOOST_CHECK_EQUAL(identity_f.get_nu(), identity.get_nu());

  Eigen::VectorXd mu(3);
  mu << -1.2, 0.7, -1.5;
  crocoddyl::JointDynamicsModelFriction friction(
      3, 1, mu, crocoddyl::JointFrictionType::CoulombViscous);
  const crocoddyl::JointDynamicsModelFrictionTpl<float> friction_f =
      friction.cast<float>();
  BOOST_CHECK(friction_f.get_parametrization().isApprox(mu.cast<float>()));

  const std::shared_ptr<crocoddyl::StateMultibody> state =
      createState<double>();
  crocoddyl::ActuationModelMultibody actuation(state);
  const crocoddyl::ActuationModelMultibodyTpl<float> actuation_f =
      actuation.cast<float>();
  BOOST_CHECK_EQUAL(actuation_f.get_nu(), actuation.get_nu());
}

template <typename Scalar>
void test_hot_paths_no_allocation() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::ActuationDataAbstractTpl<Scalar> ActuationData;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> Params;
  typedef crocoddyl::JointDynamicsModelAbstractTpl<Scalar> JointModel;
  typedef crocoddyl::JointDynamicsModelFrictionTpl<Scalar> Friction;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

  const std::shared_ptr<StateMultibody> state = createState<Scalar>();
  const pinocchio::JointIndex joint_id = firstSingleDofActuatedJoint(state);
  const auto& joint = state->get_pinocchio()->joints[joint_id];
  VectorXs mu(3);
  mu << Scalar(-1.2), Scalar(0.7), Scalar(-1.5);
  std::vector<std::shared_ptr<JointModel> > joints;
  joints.push_back(std::make_shared<Friction>(
      joint_id, static_cast<std::size_t>(joint.nq()), mu,
      crocoddyl::JointFrictionType::CoulombViscous));
  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state, joints);
  const std::shared_ptr<ActuationData> data = actuation->createData();
  Params params(actuation);
  const std::shared_ptr<crocoddyl::ParamsDataAbstractTpl<Scalar> > params_data =
      params.createData();
  const VectorXs x = state->rand();
  const VectorXs u = VectorXs::Constant(1, Scalar(0.3));
  const VectorXs tau = VectorXs::Random(state->get_nv());
  const VectorXs p = VectorXs::Constant(3, Scalar(-0.4));
  actuation->calc(data, x, u);
  actuation->calcDiff(data, x, u);
  actuation->commands(data, x, tau);
  actuation->torqueTransform(data, x, u);
  params.update(params_data, p);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      actuation->calc(data, x, u);
      actuation->calcDiff(data, x, u);
      actuation->commands(data, x, tau);
      actuation->torqueTransform(data, x, u);
      params.update(params_data, p);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_CHECK(params_data->p.isApprox(p));
}

void register_unit_tests() {
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_joint_identity_data));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_joint_friction_model));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_joint_friction_fixed_smoothing_modes));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_passive_joint_friction_model));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_default_multibody_against_numdiff));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_partial_joint_selection));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_joint_thruster_model));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_actuation_multibody_params_friction));
  boost::unit_test::framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_actuation_multibody_params_fixed_smoothing_friction));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_actuation_multibody_params_thruster));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_all_friction_modes_and_errors<double>));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_all_friction_modes_and_errors<float>));
  boost::unit_test::framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_thruster_two_data_refresh_and_scalar_behavior<double>));
  boost::unit_test::framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_thruster_two_data_refresh_and_scalar_behavior<float>));
  boost::unit_test::framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_multibody_parameter_layout_manager_and_copy<double>));
  boost::unit_test::framework::master_test_suite().add(BOOST_TEST_CASE(
      &test_multibody_parameter_layout_manager_and_copy<float>));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_scalar_casts));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_hot_paths_no_allocation<double>));
  boost::unit_test::framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_hot_paths_no_allocation<float>));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
