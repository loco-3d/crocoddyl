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
#include <limits>
#include <sstream>
#include <type_traits>

#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/controls/poly-one.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/discretized.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/exp-eigenvalue.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

template <typename Scalar>
struct IntegrationScalarTraits {
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type OtherScalar;
};

template <typename _Scalar>
class OrderingDynamicsProbeTpl
    : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase, OrderingDynamicsProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  using Base::calc;
  using Base::calcDiff;

  OrderingDynamicsProbeTpl(std::shared_ptr<StateAbstract> state,
                           const crocoddyl::DynamicsType type,
                           const std::size_t nu)
      : Base(state, type, 0, nu, 2, 1) {}

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>&) override {
    if (this->get_dyn_type() == crocoddyl::DynamicsType::DiscreteTime) {
      data->vdot = x;
    } else {
      data->vdot.setZero();
    }
    data->g << Scalar(101), Scalar(102);
    data->h << Scalar(201);
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setZero();
    if (this->get_dyn_type() == crocoddyl::DynamicsType::DiscreteTime) {
      data->Fx.setIdentity();
    }
    data->Fu.setZero();
    data->Gx.row(0).setConstant(Scalar(301));
    data->Gx.row(1).setConstant(Scalar(302));
    data->Gu.row(0).setConstant(Scalar(401));
    data->Gu.row(1).setConstant(Scalar(402));
    data->Hx.row(0).setConstant(Scalar(501));
    data->Hu.row(0).setConstant(Scalar(601));
  }

  bool checkData(const std::shared_ptr<DynamicsDataAbstract>& data) override {
    return data != nullptr && data->g.size() == 2 && data->h.size() == 1;
  }

  template <typename NewScalar>
  OrderingDynamicsProbeTpl<NewScalar> cast() const {
    return OrderingDynamicsProbeTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>(), this->get_dyn_type(),
        this->get_nu());
  }
};

template <typename _Scalar>
class ResidualDifferentialActionProbeTpl
    : public crocoddyl::DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DifferentialActionModelBase,
                         ResidualDifferentialActionProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef typename Base::DifferentialActionDataAbstract
      DifferentialActionDataAbstract;
  typedef typename Base::VectorXs VectorXs;
  using Base::calc;
  using Base::calcDiff;

  ResidualDifferentialActionProbeTpl()
      : Base(std::make_shared<crocoddyl::StateVectorTpl<Scalar>>(2), 1, 2) {}

  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    data->xout[0] = x[1] + u[0];
    data->r << x[0] + u[0], x[1] - u[0];
    data->cost = Scalar(0.5) * data->r.squaredNorm();
  }

  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    data->r << x[0], x[1];
    data->cost = Scalar(0.5) * data->r.squaredNorm();
  }

  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setZero();
    data->Fu.setZero();
    data->Lx.setZero();
    data->Lu.setZero();
    data->Lxx.setZero();
    data->Lxu.setZero();
    data->Luu.setZero();
  }

  bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data) override {
    return data != nullptr;
  }

  template <typename NewScalar>
  ResidualDifferentialActionProbeTpl<NewScalar> cast() const {
    return ResidualDifferentialActionProbeTpl<NewScalar>();
  }
};

template <typename _Scalar>
class IntegrationParameterResidualTpl
    : public crocoddyl::ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ResidualModelBase,
                         IntegrationParameterResidualTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ResidualModelAbstractTpl<Scalar> Base;
  typedef typename Base::ResidualDataAbstract ResidualDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  IntegrationParameterResidualTpl(std::shared_ptr<StateAbstract> state,
                                  const std::size_t nu, const std::size_t np)
      : Base(state, 1, nu, true, true, nu != 0, np) {}

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    const crocoddyl::DataCollectorParamsTpl<Scalar>* params =
        dynamic_cast<const crocoddyl::DataCollectorParamsTpl<Scalar>*>(
            data->shared);
    if (params == nullptr || params->params == nullptr ||
        params->params->p.size() != static_cast<Eigen::Index>(this->get_np())) {
      throw_pretty(
          "Invalid argument: parameter residual has no compatible "
          "shared parameter payload");
    }
    data->r[0] = params->params->p[0] +
                 Scalar(2) * params->params->p[this->get_np() - 1] +
                 Scalar(0.2) * x[this->get_state()->get_nq()];
    if (this->get_nu() != 0) {
      data->r[0] += Scalar(0.3) * u[0];
    }
  }

  void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    data->Rx.setZero();
    data->Rx(0, this->get_state()->get_nv()) = Scalar(0.2);
    data->Ru.setZero();
    if (this->get_nu() != 0) {
      data->Ru(0, 0) = Scalar(0.3);
    }
    data->Rp.setZero();
    data->Rp(0, 0) = Scalar(1);
    data->Rp(0, this->get_np() - 1) = Scalar(2);
  }

  template <typename NewScalar>
  IntegrationParameterResidualTpl<NewScalar> cast() const {
    return IntegrationParameterResidualTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>(), this->get_nu(),
        this->get_np());
  }
};

template <typename Scalar>
struct IntegrationFixtureTpl {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> Actuation;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraints;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>
      ContinuousDynamics;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> DiscreteDynamics;
  typedef crocoddyl::ContactModelTpl<Scalar> Contact;
  typedef crocoddyl::CostModelSumTpl<Scalar> Costs;
  typedef crocoddyl::CostModelResidualTpl<Scalar> ResidualCost;
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> Constraints;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> ResidualConstraint;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> StateResidual;
  typedef crocoddyl::ResidualModelControlTpl<Scalar> ControlResidual;
  typedef IntegrationParameterResidualTpl<Scalar> ParameterResidual;
  typedef crocoddyl::ExpEigenValueParametrizationTpl<Scalar>
      InertialParametrization;
  typedef crocoddyl::MultibodyInertialParamsTpl<Scalar> InertialParams;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  IntegrationFixtureTpl()
      : state(create_state()),
        actuation(std::make_shared<Actuation>(state)),
        implicit_constraints(
            std::make_shared<ImplicitConstraints>(state, actuation->get_nu())),
        dynamics(std::make_shared<ContinuousDynamics>(state, actuation,
                                                      implicit_constraints)),
        costs(create_costs(actuation->get_nu())),
        constraints(create_constraints(actuation->get_nu())),
        time(std::make_shared<crocoddyl::IntegratorTimeTpl<Scalar>>(
            Scalar(0.02), true)) {}

  static std::shared_ptr<State> create_state() {
    const std::shared_ptr<crocoddyl::StateMultibody> state64 =
        std::static_pointer_cast<crocoddyl::StateMultibody>(
            StateModelFactory().create(
                StateModelTypes::StateMultibody_TalosArm));
    return std::make_shared<State>(state64->template cast<Scalar>());
  }

  std::shared_ptr<Costs> create_costs(const std::size_t nu) const {
    const std::shared_ptr<Costs> result = std::make_shared<Costs>(state, nu);
    if (nu != 0) {
      const std::shared_ptr<ControlResidual> residual =
          std::make_shared<ControlResidual>(state, nu);
      result->addCost("control",
                      std::make_shared<ResidualCost>(state, residual),
                      Scalar(0.7));
    } else {
      const std::shared_ptr<StateResidual> residual =
          std::make_shared<StateResidual>(state, nu);
      result->addCost("state", std::make_shared<ResidualCost>(state, residual),
                      Scalar(0.7));
    }
    return result;
  }

  std::shared_ptr<Constraints> create_constraints(const std::size_t nu) const {
    const std::shared_ptr<Constraints> result =
        std::make_shared<Constraints>(state, nu);
    if (nu != 0) {
      const std::shared_ptr<ControlResidual> control =
          std::make_shared<ControlResidual>(state, nu);
      result->addConstraint(
          "running",
          std::make_shared<ResidualConstraint>(
              state, control,
              VectorXs::Constant(control->get_nr(), Scalar(-0.4)),
              VectorXs::Constant(control->get_nr(), Scalar(0.6)), false));
    }
    const std::shared_ptr<StateResidual> terminal =
        std::make_shared<StateResidual>(state, nu);
    result->addConstraint("terminal", std::make_shared<ResidualConstraint>(
                                          state, terminal, true));
    return result;
  }

  std::shared_ptr<Costs> create_parameter_costs(const std::size_t nu,
                                                const std::size_t np) const {
    const std::shared_ptr<Costs> result =
        std::make_shared<Costs>(state, nu, np);
    const std::shared_ptr<ParameterResidual> residual =
        std::make_shared<ParameterResidual>(state, nu, np);
    result->addCost("parameters",
                    std::make_shared<ResidualCost>(state, residual),
                    Scalar(0.7));
    return result;
  }

  std::shared_ptr<Constraints> create_parameter_constraints(
      const std::size_t nu, const std::size_t np) const {
    const std::shared_ptr<Constraints> result =
        std::make_shared<Constraints>(state, nu, np);
    const std::shared_ptr<ParameterResidual> residual =
        std::make_shared<ParameterResidual>(state, nu, np);
    result->addConstraint(
        "parameter_inequality",
        std::make_shared<ResidualConstraint>(
            state, residual, VectorXs::Constant(1, Scalar(-10)),
            VectorXs::Constant(1, Scalar(10)), false));
    result->addConstraint(
        "parameter_equality",
        std::make_shared<ResidualConstraint>(state, residual, false));
    result->addConstraint(
        "parameter_terminal",
        std::make_shared<ResidualConstraint>(state, residual, true));
    return result;
  }

  std::shared_ptr<InertialParams> create_inertial_params() const {
    const std::vector<std::string> names(1, state->get_pinocchio()->names[1]);
    return std::make_shared<InertialParams>(
        state, std::make_shared<InertialParametrization>(), names);
  }

  std::shared_ptr<DiscreteDynamics> create_discrete_dynamics() const {
    const std::shared_ptr<ImplicitConstraints> contacts =
        std::make_shared<ImplicitConstraints>(state, 0);
    typename Contact::MaskArray mask = {
        {true, true, true, false, false, false}};
    const typename Contact::Vector2s gains = Contact::Vector2s::Zero();
    const pinocchio::FrameIndex frame_id = static_cast<pinocchio::FrameIndex>(
        state->get_pinocchio()->frames.size() - 1);
    contacts->addConstraint(
        "contact", std::make_shared<Contact>(
                       state, frame_id, pinocchio::SE3Tpl<Scalar>::Identity(),
                       pinocchio::LOCAL_WORLD_ALIGNED, 0, gains, mask));
    return std::make_shared<DiscreteDynamics>(state, contacts);
  }

  VectorXs state_point() const {
    const VectorXs x0 = state->zero();
    const VectorXs dx =
        VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.08), Scalar(0.12));
    VectorXs x(state->get_nx());
    state->integrate(x0, dx, x);
    return x;
  }

  std::shared_ptr<State> state;
  std::shared_ptr<Actuation> actuation;
  std::shared_ptr<ImplicitConstraints> implicit_constraints;
  std::shared_ptr<ContinuousDynamics> dynamics;
  std::shared_ptr<Costs> costs;
  std::shared_ptr<Constraints> constraints;
  std::shared_ptr<crocoddyl::IntegratorTimeTpl<Scalar>> time;
};

template <typename Scalar>
Scalar derivative_tolerance() {
  return std::is_same<Scalar, float>::value ? Scalar(4e-2) : Scalar(2e-4);
}

template <typename Scalar>
Scalar parameter_step() {
  return std::is_same<Scalar, float>::value ? Scalar(2e-3) : Scalar(1e-6);
}

template <typename Scalar, typename Action>
void check_action_derivatives(const std::shared_ptr<Action>& model,
                              const typename Action::VectorXs& x,
                              const typename Action::VectorXs& u) {
  typedef typename Action::VectorXs VectorXs;
  const std::shared_ptr<typename Action::ActionDataAbstract> data =
      model->createData();
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  crocoddyl::ActionModelNumDiffTpl<Scalar> numdiff(model, true);
  numdiff.set_disturbance(parameter_step<Scalar>());
  const std::shared_ptr<typename Action::ActionDataAbstract> numerical =
      numdiff.createData();
  numdiff.calc(numerical, x, u);
  numdiff.calcDiff(numerical, x, u);
  const Scalar tol = derivative_tolerance<Scalar>();
  BOOST_CHECK_MESSAGE(
      data->Fx.isApprox(numerical->Fx, tol),
      "Fx error " << (data->Fx - numerical->Fx).cwiseAbs().maxCoeff());
  if (u.size() != 0) {
    BOOST_CHECK_MESSAGE(
        data->Fu.isApprox(numerical->Fu, tol),
        "Fu error " << (data->Fu - numerical->Fu).cwiseAbs().maxCoeff());
  } else {
    BOOST_CHECK_EQUAL(data->Fu.cols(), 0);
  }
  BOOST_CHECK_MESSAGE(
      data->Lx.isApprox(numerical->Lx, tol),
      "Lx error " << (data->Lx - numerical->Lx).cwiseAbs().maxCoeff());
  if (u.size() != 0) {
    BOOST_CHECK_MESSAGE(
        data->Lu.isApprox(numerical->Lu, tol),
        "Lu error " << (data->Lu - numerical->Lu).cwiseAbs().maxCoeff());
  } else {
    BOOST_CHECK_EQUAL(data->Lu.size(), 0);
  }
  BOOST_CHECK(data->Lxx.allFinite());
  BOOST_CHECK(data->Lxu.allFinite());
  BOOST_CHECK(data->Luu.allFinite());
  BOOST_CHECK(data->Gx.isApprox(numerical->Gx, tol));
  BOOST_CHECK(data->Gu.isApprox(numerical->Gu, tol));
  BOOST_CHECK(data->Hx.isApprox(numerical->Hx, tol));
  BOOST_CHECK(data->Hu.isApprox(numerical->Hu, tol));

  data->Fu.setConstant(Scalar(11));
  data->Lu.setConstant(Scalar(12));
  data->Lxu.setConstant(Scalar(13));
  data->Luu.setConstant(Scalar(14));
  data->Lpu.setConstant(Scalar(15));
  data->Gu.setConstant(Scalar(16));
  data->Hu.setConstant(Scalar(17));
  model->calc(data, x);
  model->calcDiff(data, x);
  BOOST_CHECK((data->Fu.array() == Scalar(11)).all());
  BOOST_CHECK((data->Lu.array() == Scalar(12)).all());
  BOOST_CHECK((data->Lxu.array() == Scalar(13)).all());
  BOOST_CHECK((data->Luu.array() == Scalar(14)).all());
  BOOST_CHECK((data->Lpu.array() == Scalar(15)).all());
  BOOST_CHECK((data->Gu.array() == Scalar(16)).all());
  BOOST_CHECK((data->Hu.array() == Scalar(17)).all());
  numdiff.calc(numerical, x);
  numdiff.calcDiff(numerical, x);
  BOOST_CHECK(data->Lx.isApprox(numerical->Lx, tol));
  BOOST_CHECK(data->Lxx.allFinite());
  BOOST_CHECK(data->Gx.isApprox(numerical->Gx, tol));
  BOOST_CHECK(data->Hx.isApprox(numerical->Hx, tol));
  BOOST_CHECK_EQUAL(data->g.size(), model->get_ng_T());
  BOOST_CHECK_EQUAL(data->h.size(), model->get_nh_T());
  BOOST_CHECK(data->xnext.isApprox(x));
  BOOST_CHECK_THROW(model->calc(data, x, VectorXs::Zero(u.size() + 1)),
                    crocoddyl::Exception);
}

template <typename Scalar, typename Action>
void check_control_cost_hessian(const std::shared_ptr<Action>& model,
                                const typename Action::VectorXs& x,
                                const typename Action::VectorXs& u) {
  typedef typename Action::VectorXs VectorXs;
  typedef typename Action::MatrixXs MatrixXs;
  const std::shared_ptr<typename Action::ActionDataAbstract> data =
      model->createData();
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  const MatrixXs analytical = data->Luu;
  MatrixXs numerical(model->get_nu(), model->get_nu());
  const Scalar eps = parameter_step<Scalar>();
  for (Eigen::Index iu = 0; iu < numerical.cols(); ++iu) {
    VectorXs um = u;
    VectorXs up = u;
    um[iu] -= eps;
    up[iu] += eps;
    model->calc(data, x, um);
    model->calcDiff(data, x, um);
    const VectorXs lm = data->Lu;
    model->calc(data, x, up);
    model->calcDiff(data, x, up);
    numerical.col(iu) = (data->Lu - lm) / (Scalar(2) * eps);
  }
  const Scalar tol = Scalar(5) * derivative_tolerance<Scalar>();
  BOOST_CHECK(analytical.isApprox(numerical, tol));
  BOOST_CHECK(analytical.isApprox(analytical.transpose(), tol));
  BOOST_CHECK(data->Lxx.isZero(tol));
  BOOST_CHECK(data->Lxu.isZero(tol));
}

template <typename Scalar, typename Action>
void check_time_parameter(const std::shared_ptr<Action>& model,
                          const typename Action::VectorXs& x,
                          const typename Action::VectorXs& u) {
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef typename Action::Data Data;
  typedef typename Action::VectorXs VectorXs;
  const std::shared_ptr<TimeParams> time_params = std::make_shared<TimeParams>(
      model->get_state(), model->get_integrator_time());
  const std::shared_ptr<Manager> manager =
      std::make_shared<Manager>(model->get_state());
  manager->addParam("time", time_params);
  const std::shared_ptr<typename Manager::ParameterDataManager> payload =
      manager->createData();
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model->createData());
  const std::shared_ptr<Data> data2 =
      std::dynamic_pointer_cast<Data>(model->createData(payload));
  BOOST_REQUIRE(data != nullptr);
  BOOST_REQUIRE(data2 != nullptr);
  BOOST_CHECK(data2->params == payload);
  BOOST_CHECK_THROW(model->update_p(data, VectorXs::Zero(0)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model->set_params(data, std::shared_ptr<Manager>()),
                    crocoddyl::Exception);
  const std::shared_ptr<Manager> wrong_manager = std::make_shared<Manager>(
      std::make_shared<crocoddyl::StateVectorTpl<Scalar>>(3));
  BOOST_CHECK_THROW(model->set_params(data, wrong_manager),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model->set_params(std::shared_ptr<typename Action::ActionDataAbstract>(),
                        manager),
      crocoddyl::Exception);

  const std::shared_ptr<Manager> inactive_manager =
      std::make_shared<Manager>(model->get_state());
  inactive_manager->addParam("inactive_lqr",
                             std::make_shared<LQRParams>(model->get_state(), 1),
                             false);
  const std::shared_ptr<typename Action::ActionDataAbstract> inactive_data =
      model->createData();
  BOOST_CHECK_NO_THROW(model->set_params(inactive_data, inactive_manager));

  const std::shared_ptr<Manager> different_time_manager =
      std::make_shared<Manager>(model->get_state());
  different_time_manager->addParam(
      "time", std::make_shared<TimeParams>(
                  model->get_state(),
                  std::make_shared<IntegratorTime>(model->get_dt(), true)));
  BOOST_CHECK_THROW(model->set_params(data, different_time_manager),
                    crocoddyl::Exception);

  const std::shared_ptr<Manager> incompatible_manager =
      std::make_shared<Manager>(model->get_state());
  incompatible_manager->addParam(
      "lqr", std::make_shared<LQRParams>(model->get_state(), 1));
  BOOST_CHECK_THROW(model->set_params(data, incompatible_manager),
                    crocoddyl::Exception);

  const std::shared_ptr<Manager> duplicate_time_manager =
      std::make_shared<Manager>(model->get_state());
  duplicate_time_manager->addParam("time_a", time_params);
  duplicate_time_manager->addParam(
      "time_b", std::make_shared<TimeParams>(model->get_state(),
                                             model->get_integrator_time()));
  BOOST_CHECK_THROW(model->set_params(data, duplicate_time_manager),
                    crocoddyl::Exception);

  model->set_params(data, manager);
  model->set_params(data2, manager);
  VectorXs p(1);
  using std::log;
  p[0] = log(Scalar(0.025));
  model->update_p(data, p);
  model->update_p(data2, p);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model->calc(data2, x, u);
  model->calcDiff(data2, x, u);
  BOOST_CHECK(data->xnext.isApprox(data2->xnext));
  BOOST_CHECK_EQUAL(data->Fp.cols(), 1);
  const VectorXs analytical_Fp = data->Fp.col(0);
  const Scalar analytical_Lp = data->Lp[0];
  const Scalar analytical_Lpp = data->Lpp(0, 0);
  const typename Action::MatrixXs analytical_Lpx = data->Lpx;
  const typename Action::MatrixXs analytical_Lpu = data->Lpu;

  const Scalar eps = parameter_step<Scalar>();
  VectorXs pm = p;
  VectorXs pp = p;
  pm[0] -= eps;
  pp[0] += eps;
  model->update_p(data2, pm);
  model->calc(data2, x, u);
  const VectorXs xm = data2->xnext;
  const Scalar cost_m = data2->cost;
  model->calcDiff(data2, x, u);
  const Scalar lp_m = data2->Lp[0];
  model->update_p(data2, pp);
  model->calc(data2, x, u);
  const VectorXs xp = data2->xnext;
  const Scalar cost_p = data2->cost;
  model->calcDiff(data2, x, u);
  const Scalar lp_p = data2->Lp[0];
  VectorXs numerical(model->get_state()->get_ndx());
  model->get_state()->diff(xm, xp, numerical);
  numerical /= Scalar(2) * eps;
  const Scalar tol = Scalar(3) * derivative_tolerance<Scalar>();
  BOOST_CHECK(analytical_Fp.isApprox(numerical, tol));
  BOOST_CHECK_SMALL(analytical_Lp - (cost_p - cost_m) / (Scalar(2) * eps), tol);
  BOOST_CHECK_SMALL(analytical_Lpp - (lp_p - lp_m) / (Scalar(2) * eps),
                    Scalar(5) * tol);

  typename Action::MatrixXs numerical_Lpx(1, model->get_state()->get_ndx());
  for (Eigen::Index ix = 0; ix < numerical_Lpx.cols(); ++ix) {
    VectorXs dx = VectorXs::Zero(model->get_state()->get_ndx());
    dx[ix] = eps;
    VectorXs x_minus(model->get_state()->get_nx());
    VectorXs x_plus(model->get_state()->get_nx());
    model->get_state()->integrate(x, -dx, x_minus);
    model->get_state()->integrate(x, dx, x_plus);
    model->update_p(data2, p);
    model->calc(data2, x_minus, u);
    model->calcDiff(data2, x_minus, u);
    const Scalar minus = data2->Lp[0];
    model->calc(data2, x_plus, u);
    model->calcDiff(data2, x_plus, u);
    numerical_Lpx(0, ix) = (data2->Lp[0] - minus) / (Scalar(2) * eps);
  }
  BOOST_CHECK(analytical_Lpx.isApprox(numerical_Lpx, Scalar(8) * tol));

  typename Action::MatrixXs numerical_Lpu(1, model->get_nu());
  for (Eigen::Index iu = 0; iu < numerical_Lpu.cols(); ++iu) {
    VectorXs u_minus = u;
    VectorXs u_plus = u;
    u_minus[iu] -= eps;
    u_plus[iu] += eps;
    model->calc(data2, x, u_minus);
    model->calcDiff(data2, x, u_minus);
    const Scalar minus = data2->Lp[0];
    model->calc(data2, x, u_plus);
    model->calcDiff(data2, x, u_plus);
    numerical_Lpu(0, iu) = (data2->Lp[0] - minus) / (Scalar(2) * eps);
  }
  BOOST_CHECK(analytical_Lpu.isApprox(numerical_Lpu, Scalar(8) * tol));
  BOOST_CHECK(data->Gp.isZero(tol));
  BOOST_CHECK(data->Hp.isZero(tol));

  manager->changeParamStatus("time", false);
  BOOST_CHECK_THROW(model->update_p(data, VectorXs::Zero(0)),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_constraint_row_ordering() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef OrderingDynamicsProbeTpl<Scalar> Dynamics;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Discretized;
  typedef typename Fixture::VectorXs VectorXs;
  Fixture fixture;
  const VectorXs x = fixture.state_point();
  const VectorXs u = VectorXs::LinSpaced(fixture.actuation->get_nu(),
                                         Scalar(-0.3), Scalar(0.4));
  const std::shared_ptr<Dynamics> continuous = std::make_shared<Dynamics>(
      fixture.state, crocoddyl::DynamicsType::ContinuousControl, u.size());

  const std::shared_ptr<Euler> euler = std::make_shared<Euler>(
      continuous, fixture.costs, fixture.constraints, nullptr, fixture.time);
  const std::shared_ptr<typename Euler::Data> euler_data =
      std::dynamic_pointer_cast<typename Euler::Data>(euler->createData());
  BOOST_REQUIRE(euler_data != nullptr);
  euler->calc(euler_data, x, u);
  euler->calcDiff(euler_data, x, u);
  BOOST_CHECK(euler_data->g.head(2).isApprox(
      (VectorXs(2) << Scalar(101), Scalar(102)).finished()));
  BOOST_CHECK(euler_data->g.tail(fixture.constraints->get_ng())
                  .isApprox(euler_data->constraints->g));
  BOOST_CHECK_EQUAL(euler_data->h[0], Scalar(201));
  BOOST_CHECK(euler_data->Gx.topRows(2).row(0).isConstant(Scalar(301)));
  BOOST_CHECK(euler_data->Gx.topRows(2).row(1).isConstant(Scalar(302)));
  BOOST_CHECK(euler_data->Gx.bottomRows(fixture.constraints->get_ng())
                  .isApprox(euler_data->constraints->Gx));
  BOOST_CHECK(euler_data->Gu.topRows(2).row(0).isConstant(Scalar(401)));
  BOOST_CHECK(euler_data->Gu.topRows(2).row(1).isConstant(Scalar(402)));
  BOOST_CHECK(euler_data->Gu.bottomRows(fixture.constraints->get_ng())
                  .isApprox(euler_data->constraints->Gu));
  BOOST_CHECK((euler->get_g_lb().head(2).array() ==
               -std::numeric_limits<Scalar>::infinity())
                  .all());
  BOOST_CHECK(euler->get_g_ub().head(2).isZero());
  BOOST_CHECK(euler->get_g_lb()
                  .tail(fixture.constraints->get_ng())
                  .isApprox(fixture.constraints->get_lb()));
  euler->calc(euler_data, x);
  euler->calcDiff(euler_data, x);
  BOOST_CHECK(euler_data->h.isApprox(euler_data->constraints->h));

  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::shared_ptr<RK> rk =
        std::make_shared<RK>(continuous, fixture.costs, fixture.constraints,
                             nullptr, fixture.time, types[i]);
    const std::shared_ptr<typename RK::Data> data =
        std::dynamic_pointer_cast<typename RK::Data>(rk->createData());
    BOOST_REQUIRE(data != nullptr);
    rk->calc(data, x, u);
    rk->calcDiff(data, x, u);
    BOOST_CHECK(data->g.head(2).isApprox(
        (VectorXs(2) << Scalar(101), Scalar(102)).finished()));
    BOOST_CHECK(data->g.tail(fixture.constraints->get_ng())
                    .isApprox(data->constraints->g));
    BOOST_CHECK_EQUAL(data->h[0], Scalar(201));
    BOOST_CHECK(data->Gx.topRows(2).row(0).isConstant(Scalar(301)));
    BOOST_CHECK(data->Gx.bottomRows(fixture.constraints->get_ng())
                    .isApprox(data->constraints->Gx));
    BOOST_CHECK(data->Gu.topRows(2).row(1).isConstant(Scalar(402)));
    BOOST_CHECK(data->Gu.bottomRows(fixture.constraints->get_ng())
                    .isApprox(data->constraints->Gu));
    BOOST_CHECK((rk->get_g_lb().head(2).array() ==
                 -std::numeric_limits<Scalar>::infinity())
                    .all());
    BOOST_CHECK(rk->get_g_ub().head(2).isZero());
    BOOST_CHECK(rk->get_g_lb()
                    .tail(fixture.constraints->get_ng())
                    .isApprox(fixture.constraints->get_lb()));
  }

  const std::shared_ptr<Dynamics> discrete_dynamics =
      std::make_shared<Dynamics>(fixture.state,
                                 crocoddyl::DynamicsType::DiscreteTime, 0);
  const std::shared_ptr<typename Fixture::Constraints> discrete_constraints =
      std::make_shared<typename Fixture::Constraints>(fixture.state, 0);
  const std::shared_ptr<typename Fixture::StateResidual> running_residual =
      std::make_shared<typename Fixture::StateResidual>(fixture.state, 0);
  discrete_constraints->addConstraint(
      "running",
      std::make_shared<typename Fixture::ResidualConstraint>(
          fixture.state, running_residual,
          VectorXs::Constant(running_residual->get_nr(), Scalar(-3)),
          VectorXs::Constant(running_residual->get_nr(), Scalar(4)), false));
  discrete_constraints->addConstraint(
      "terminal", std::make_shared<typename Fixture::ResidualConstraint>(
                      fixture.state, running_residual, true));
  const std::shared_ptr<Discretized> discretized =
      std::make_shared<Discretized>(discrete_dynamics, fixture.create_costs(0),
                                    discrete_constraints);
  const std::shared_ptr<typename Discretized::Data> discrete_data =
      std::dynamic_pointer_cast<typename Discretized::Data>(
          discretized->createData());
  BOOST_REQUIRE(discrete_data != nullptr);
  const VectorXs u0(0);
  discretized->calc(discrete_data, x, u0);
  discretized->calcDiff(discrete_data, x, u0);
  BOOST_CHECK(discrete_data->g.head(2).isApprox(
      (VectorXs(2) << Scalar(101), Scalar(102)).finished()));
  BOOST_CHECK(discrete_data->g.tail(discrete_constraints->get_ng())
                  .isApprox(discrete_data->constraints->g));
  BOOST_CHECK_EQUAL(discrete_data->h[0], Scalar(201));
  BOOST_CHECK(discrete_data->Gx.topRows(2).row(0).isConstant(Scalar(301)));
  BOOST_CHECK(discrete_data->Gx.bottomRows(discrete_constraints->get_ng())
                  .isApprox(discrete_data->constraints->Gx));
  BOOST_CHECK((discretized->get_g_lb().head(2).array() ==
               -std::numeric_limits<Scalar>::infinity())
                  .all());
  BOOST_CHECK(discretized->get_g_ub().head(2).isZero());
  BOOST_CHECK(discretized->get_g_lb()
                  .tail(discrete_constraints->get_ng())
                  .isApprox(discrete_constraints->get_lb()));
  discrete_data->dynamics->g.setZero();
  discretized->calc(discrete_data, x);
  discretized->calcDiff(discrete_data, x);
  BOOST_CHECK(discrete_data->dynamics->g.isApprox(
      (VectorXs(2) << Scalar(101), Scalar(102)).finished()));
  BOOST_CHECK(discrete_data->h.isApprox(discrete_data->constraints->h));
}

template <typename Scalar>
void test_live_constraint_activation() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef OrderingDynamicsProbeTpl<Scalar> Dynamics;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Discretized;
  typedef typename Fixture::Constraints Constraints;
  typedef typename Fixture::ResidualConstraint ResidualConstraint;
  typedef typename Fixture::StateResidual StateResidual;
  typedef typename Fixture::VectorXs VectorXs;

  Fixture fixture;
  const VectorXs x = fixture.state_point();
  const VectorXs u = VectorXs::LinSpaced(fixture.actuation->get_nu(),
                                         Scalar(-0.2), Scalar(0.3));
  const std::shared_ptr<Dynamics> continuous = std::make_shared<Dynamics>(
      fixture.state, crocoddyl::DynamicsType::ContinuousControl, u.size());
  const std::size_t nr = fixture.state->get_ndx();
  const auto create_constraints = [&](const std::size_t nu) {
    const std::shared_ptr<Constraints> constraints =
        fixture.create_constraints(nu);
    const std::shared_ptr<StateResidual> residual =
        std::make_shared<StateResidual>(fixture.state, nu);
    constraints->addConstraint(
        "z_live_inequality",
        std::make_shared<ResidualConstraint>(fixture.state, residual,
                                             VectorXs::Constant(nr, Scalar(-5)),
                                             VectorXs::Constant(nr, Scalar(6))),
        false);
    constraints->addConstraint(
        "z_live_equality",
        std::make_shared<ResidualConstraint>(fixture.state, residual), false);
    return constraints;
  };

  const std::shared_ptr<Constraints> euler_constraints =
      create_constraints(u.size());
  const std::shared_ptr<Euler> euler = std::make_shared<Euler>(
      continuous, fixture.costs, euler_constraints, nullptr, fixture.time);
  const std::shared_ptr<typename Euler::Data> euler_data =
      std::dynamic_pointer_cast<typename Euler::Data>(euler->createData());
  BOOST_REQUIRE(euler_data != nullptr);
  euler->calc(euler_data, x, u);
  const Eigen::Index euler_ng_before =
      euler_data->constraints->g_internal.size();
  const Eigen::Index euler_nh_before =
      euler_data->constraints->h_internal.size();
  euler_constraints->changeConstraintStatus("z_live_inequality", true);
  euler_constraints->changeConstraintStatus("z_live_equality", true);
  euler->calc(euler_data, x, u);
  euler->calcDiff(euler_data, x, u);
  BOOST_CHECK_GT(euler_data->constraints->g_internal.size(), euler_ng_before);
  BOOST_CHECK_GT(euler_data->constraints->h_internal.size(), euler_nh_before);
  const std::size_t euler_ng_d = continuous->get_ng();
  const std::size_t euler_nh_d = continuous->get_nh();
  BOOST_CHECK(euler_data->g.segment(euler_ng_d, euler_constraints->get_ng())
                  .isApprox(euler_data->constraints->g));
  BOOST_CHECK(euler_data->h.segment(euler_nh_d, euler_constraints->get_nh())
                  .isApprox(euler_data->constraints->h));
  BOOST_CHECK(euler_data->Gx.middleRows(euler_ng_d, euler_constraints->get_ng())
                  .isApprox(euler_data->constraints->Gx));
  BOOST_CHECK(euler_data->Gu.middleRows(euler_ng_d, euler_constraints->get_ng())
                  .isApprox(euler_data->constraints->Gu));
  BOOST_CHECK(euler_data->Hx.middleRows(euler_nh_d, euler_constraints->get_nh())
                  .isApprox(euler_data->constraints->Hx));
  BOOST_CHECK(euler_data->Hu.middleRows(euler_nh_d, euler_constraints->get_nh())
                  .isApprox(euler_data->constraints->Hu));
  BOOST_CHECK(euler->get_g_lb()
                  .segment(euler_ng_d, euler_constraints->get_ng())
                  .isApprox(euler_constraints->get_lb()));
  BOOST_CHECK(euler->get_g_ub()
                  .segment(euler_ng_d, euler_constraints->get_ng())
                  .isApprox(euler_constraints->get_ub()));
  euler->calc(euler_data, x);
  euler->calcDiff(euler_data, x);
  BOOST_CHECK(euler_data->g.isApprox(euler_data->constraints->g));
  BOOST_CHECK(euler_data->h.isApprox(euler_data->constraints->h));
  BOOST_CHECK(euler_data->Gx.isApprox(euler_data->constraints->Gx));
  BOOST_CHECK(euler_data->Hx.isApprox(euler_data->constraints->Hx));

  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};
  std::vector<std::shared_ptr<RK>> rks;
  std::vector<std::shared_ptr<typename RK::Data>> rk_data;
  for (std::size_t i = 0; i < 3; ++i) {
    const std::shared_ptr<Constraints> constraints =
        create_constraints(u.size());
    rks.push_back(std::make_shared<RK>(continuous, fixture.costs, constraints,
                                       nullptr, fixture.time, types[i]));
    rk_data.push_back(
        std::dynamic_pointer_cast<typename RK::Data>(rks.back()->createData()));
    BOOST_REQUIRE(rk_data.back() != nullptr);
    rks.back()->calc(rk_data.back(), x, u);
    const Eigen::Index ng_before =
        rk_data.back()->constraints->g_internal.size();
    const Eigen::Index nh_before =
        rk_data.back()->constraints->h_internal.size();
    constraints->changeConstraintStatus("z_live_inequality", true);
    constraints->changeConstraintStatus("z_live_equality", true);
    rks.back()->calc(rk_data.back(), x, u);
    rks.back()->calcDiff(rk_data.back(), x, u);
    BOOST_CHECK_GT(rk_data.back()->constraints->g_internal.size(), ng_before);
    BOOST_CHECK_GT(rk_data.back()->constraints->h_internal.size(), nh_before);
    BOOST_CHECK(rk_data.back()
                    ->g.segment(continuous->get_ng(), constraints->get_ng())
                    .isApprox(rk_data.back()->constraints->g));
    BOOST_CHECK(rk_data.back()
                    ->Gx.middleRows(continuous->get_ng(), constraints->get_ng())
                    .isApprox(rk_data.back()->constraints->Gx));
    BOOST_CHECK(rk_data.back()
                    ->Hx.middleRows(continuous->get_nh(), constraints->get_nh())
                    .isApprox(rk_data.back()->constraints->Hx));
    BOOST_CHECK(rks.back()
                    ->get_g_lb()
                    .segment(continuous->get_ng(), constraints->get_ng())
                    .isApprox(constraints->get_lb()));
    rks.back()->calc(rk_data.back(), x);
    rks.back()->calcDiff(rk_data.back(), x);
    BOOST_CHECK(rk_data.back()->g.isApprox(rk_data.back()->constraints->g));
    BOOST_CHECK(rk_data.back()->h.isApprox(rk_data.back()->constraints->h));
  }

  const std::shared_ptr<Dynamics> discrete_dynamics =
      std::make_shared<Dynamics>(fixture.state,
                                 crocoddyl::DynamicsType::DiscreteTime, 0);
  const std::shared_ptr<Constraints> discrete_constraints =
      create_constraints(0);
  const std::shared_ptr<Discretized> discretized =
      std::make_shared<Discretized>(discrete_dynamics, fixture.create_costs(0),
                                    discrete_constraints);
  const std::shared_ptr<typename Discretized::Data> discrete_data =
      std::dynamic_pointer_cast<typename Discretized::Data>(
          discretized->createData());
  BOOST_REQUIRE(discrete_data != nullptr);
  const VectorXs u0(0);
  discretized->calc(discrete_data, x, u0);
  const Eigen::Index discrete_ng_before =
      discrete_data->constraints->g_internal.size();
  const Eigen::Index discrete_nh_before =
      discrete_data->constraints->h_internal.size();
  discrete_constraints->changeConstraintStatus("z_live_inequality", true);
  discrete_constraints->changeConstraintStatus("z_live_equality", true);
  discretized->calc(discrete_data, x, u0);
  discretized->calcDiff(discrete_data, x, u0);
  BOOST_CHECK_GT(discrete_data->constraints->g_internal.size(),
                 discrete_ng_before);
  BOOST_CHECK_GT(discrete_data->constraints->h_internal.size(),
                 discrete_nh_before);
  BOOST_CHECK(
      discrete_data->g
          .segment(discrete_dynamics->get_ng(), discrete_constraints->get_ng())
          .isApprox(discrete_data->constraints->g));
  BOOST_CHECK(discrete_data->Gx
                  .middleRows(discrete_dynamics->get_ng(),
                              discrete_constraints->get_ng())
                  .isApprox(discrete_data->constraints->Gx));
  BOOST_CHECK(discrete_data->Hx
                  .middleRows(discrete_dynamics->get_nh(),
                              discrete_constraints->get_nh())
                  .isApprox(discrete_data->constraints->Hx));
  BOOST_CHECK(
      discretized->get_g_lb()
          .segment(discrete_dynamics->get_ng(), discrete_constraints->get_ng())
          .isApprox(discrete_constraints->get_lb()));
  BOOST_CHECK(
      discretized->get_g_ub()
          .segment(discrete_dynamics->get_ng(), discrete_constraints->get_ng())
          .isApprox(discrete_constraints->get_ub()));
  discretized->calc(discrete_data, x);
  discretized->calcDiff(discrete_data, x);
  BOOST_CHECK(discrete_data->g.isApprox(discrete_data->constraints->g));
  BOOST_CHECK(discrete_data->h.isApprox(discrete_data->constraints->h));
  BOOST_CHECK(discrete_data->Gx.isApprox(discrete_data->constraints->Gx));
  BOOST_CHECK(discrete_data->Hx.isApprox(discrete_data->constraints->Hx));

  euler->calc(euler_data, x, u);
  euler->calcDiff(euler_data, x, u);
  for (std::size_t i = 0; i < rks.size(); ++i) {
    rks[i]->calc(rk_data[i], x, u);
    rks[i]->calcDiff(rk_data[i], x, u);
  }
  discretized->calc(discrete_data, x, u0);
  discretized->calcDiff(discrete_data, x, u0);
  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      euler->calc(euler_data, x, u);
      euler->calcDiff(euler_data, x, u);
      for (std::size_t j = 0; j < rks.size(); ++j) {
        rks[j]->calc(rk_data[j], x, u);
        rks[j]->calcDiff(rk_data[j], x, u);
      }
      discretized->calc(discrete_data, x, u0);
      discretized->calcDiff(discrete_data, x, u0);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

template <typename Scalar, typename Action>
void check_nonzero_parameter_derivatives(
    const std::shared_ptr<Action>& model,
    const std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar>>& manager,
    const typename Action::VectorXs& p, const typename Action::VectorXs& x,
    const typename Action::VectorXs& u) {
  typedef typename Action::VectorXs VectorXs;
  typedef typename Action::MatrixXs MatrixXs;
  const std::shared_ptr<typename Action::ActionDataAbstract> data =
      model->createData();
  model->set_params(data, manager);
  model->update_p(data, p);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  const MatrixXs Fp = data->Fp;
  const VectorXs Lp = data->Lp;
  const MatrixXs Lpp = data->Lpp;
  const MatrixXs Lpx = data->Lpx;
  const MatrixXs Lpu = data->Lpu;
  const MatrixXs Gp = data->Gp;
  const MatrixXs Hp = data->Hp;
  BOOST_CHECK_GT(Fp.norm(), Scalar(0));
  BOOST_CHECK_GT(Lp.norm(), Scalar(0));
  BOOST_CHECK_GT(Lpp.norm(), Scalar(0));
  BOOST_CHECK_GT(Lpx.norm(), Scalar(0));
  if (u.size() != 0) {
    BOOST_CHECK_GT(Lpu.norm(), Scalar(0));
  }
  BOOST_CHECK_GT(Gp.norm(), Scalar(0));
  BOOST_CHECK_GT(Hp.norm(), Scalar(0));

  const Scalar eps = parameter_step<Scalar>();
  MatrixXs Fp_num = MatrixXs::Zero(Fp.rows(), Fp.cols());
  VectorXs Lp_num = VectorXs::Zero(Lp.size());
  MatrixXs Lpp_num = MatrixXs::Zero(Lpp.rows(), Lpp.cols());
  MatrixXs Gp_num = MatrixXs::Zero(Gp.rows(), Gp.cols());
  MatrixXs Hp_num = MatrixXs::Zero(Hp.rows(), Hp.cols());
  for (Eigen::Index ip = 0; ip < p.size(); ++ip) {
    VectorXs pm = p;
    VectorXs pp = p;
    pm[ip] -= eps;
    pp[ip] += eps;
    model->update_p(data, pm);
    model->calc(data, x, u);
    const VectorXs xm = data->xnext;
    const Scalar cm = data->cost;
    const VectorXs gm = data->g;
    const VectorXs hm = data->h;
    model->calcDiff(data, x, u);
    const VectorXs lpm = data->Lp;
    model->update_p(data, pp);
    model->calc(data, x, u);
    const VectorXs xp = data->xnext;
    const Scalar cp = data->cost;
    const VectorXs gp = data->g;
    const VectorXs hp = data->h;
    model->calcDiff(data, x, u);
    VectorXs dx(model->get_state()->get_ndx());
    model->get_state()->diff(xm, xp, dx);
    Fp_num.col(ip) = dx / (Scalar(2) * eps);
    Lp_num[ip] = (cp - cm) / (Scalar(2) * eps);
    Lpp_num.col(ip) = (data->Lp - lpm) / (Scalar(2) * eps);
    Gp_num.col(ip) = (gp - gm) / (Scalar(2) * eps);
    Hp_num.col(ip) = (hp - hm) / (Scalar(2) * eps);
  }

  model->update_p(data, p);
  MatrixXs Lpx_num = MatrixXs::Zero(Lpx.rows(), Lpx.cols());
  for (Eigen::Index ix = 0; ix < Lpx.cols(); ++ix) {
    VectorXs dx = VectorXs::Zero(model->get_state()->get_ndx());
    dx[ix] = eps;
    VectorXs xm(model->get_state()->get_nx());
    VectorXs xp(model->get_state()->get_nx());
    model->get_state()->integrate(x, -dx, xm);
    model->get_state()->integrate(x, dx, xp);
    model->calc(data, xm, u);
    model->calcDiff(data, xm, u);
    const VectorXs lpm = data->Lp;
    model->calc(data, xp, u);
    model->calcDiff(data, xp, u);
    Lpx_num.col(ix) = (data->Lp - lpm) / (Scalar(2) * eps);
  }
  MatrixXs Lpu_num = MatrixXs::Zero(Lpu.rows(), Lpu.cols());
  for (Eigen::Index iu = 0; iu < Lpu.cols(); ++iu) {
    VectorXs um = u;
    VectorXs up = u;
    um[iu] -= eps;
    up[iu] += eps;
    model->calc(data, x, um);
    model->calcDiff(data, x, um);
    const VectorXs lpm = data->Lp;
    model->calc(data, x, up);
    model->calcDiff(data, x, up);
    Lpu_num.col(iu) = (data->Lp - lpm) / (Scalar(2) * eps);
  }

  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(5e-2) : Scalar(5e-5);
  BOOST_CHECK_MESSAGE((Fp - Fp_num).isZero(tol),
                      "Fp error " << (Fp - Fp_num).norm());
  BOOST_CHECK_MESSAGE((Lp - Lp_num).isZero(tol),
                      "Lp error " << (Lp - Lp_num).norm());
  BOOST_CHECK_MESSAGE((Lpp - Lpp_num).isZero(Scalar(4) * tol),
                      "Lpp error " << (Lpp - Lpp_num).norm());
  BOOST_CHECK_MESSAGE((Lpx - Lpx_num).isZero(Scalar(4) * tol),
                      "Lpx error " << (Lpx - Lpx_num).norm());
  BOOST_CHECK_MESSAGE((Lpu - Lpu_num).isZero(Scalar(4) * tol),
                      "Lpu error " << (Lpu - Lpu_num).norm());
  BOOST_CHECK_MESSAGE((Gp - Gp_num).isZero(tol),
                      "Gp error " << (Gp - Gp_num).norm());
  BOOST_CHECK_MESSAGE((Hp - Hp_num).isZero(tol),
                      "Hp error " << (Hp - Hp_num).norm());
  model->update_p(data, p);
}

template <typename Scalar>
void test_nonzero_parameter_derivatives() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Discretized;
  typedef typename Fixture::VectorXs VectorXs;
  Fixture fixture;
  const VectorXs x = fixture.state_point();
  const VectorXs u = VectorXs::LinSpaced(fixture.actuation->get_nu(),
                                         Scalar(-0.12), Scalar(0.18));

  const std::shared_ptr<Manager> mixed =
      std::make_shared<Manager>(fixture.state);
  mixed->addParam("time",
                  std::make_shared<TimeParams>(fixture.state, fixture.time));
  mixed->addParam("inertial", fixture.create_inertial_params());
  BOOST_REQUIRE_EQUAL(mixed->get_np_action(), 1);
  BOOST_REQUIRE_EQUAL(mixed->get_np_dynamics(), 10);
  VectorXs p = mixed->zero();
  using std::log;
  p[0] = log(Scalar(5e-4));
  p[p.size() - 1] += Scalar(0.08);
  const std::shared_ptr<typename Fixture::Costs> costs =
      fixture.create_parameter_costs(u.size(), mixed->get_np());
  const std::shared_ptr<typename Fixture::Constraints> constraints =
      fixture.create_parameter_constraints(u.size(), mixed->get_np());
  const std::shared_ptr<Euler> euler = std::make_shared<Euler>(
      fixture.dynamics, costs, constraints, nullptr, fixture.time);
  BOOST_TEST_CONTEXT("Euler mixed parameters") {
    check_nonzero_parameter_derivatives<Scalar>(euler, mixed, p, x, u);
  }
  const std::shared_ptr<RK> rk =
      std::make_shared<RK>(fixture.dynamics, costs, constraints, nullptr,
                           fixture.time, crocoddyl::four);
  BOOST_TEST_CONTEXT("RK4 mixed parameters") {
    check_nonzero_parameter_derivatives<Scalar>(rk, mixed, p, x, u);
  }

  const std::shared_ptr<Manager> inertial =
      std::make_shared<Manager>(fixture.state);
  inertial->addParam("inertial", fixture.create_inertial_params());
  VectorXs p_inertial = inertial->zero();
  p_inertial[0] += Scalar(0.04);
  p_inertial[p_inertial.size() - 1] += Scalar(0.08);
  const std::shared_ptr<typename Fixture::DiscreteDynamics> dynamics =
      fixture.create_discrete_dynamics();
  const std::shared_ptr<Discretized> discretized =
      std::make_shared<Discretized>(
          dynamics, fixture.create_parameter_costs(0, inertial->get_np()),
          fixture.create_parameter_constraints(0, inertial->get_np()));
  const VectorXs u0(0);
  BOOST_TEST_CONTEXT("Discretized inertial parameters") {
    check_nonzero_parameter_derivatives<Scalar>(discretized, inertial,
                                                p_inertial, x, u0);
  }

  const std::shared_ptr<typename Discretized::ActionDataAbstract> bad_data =
      discretized->createData();
  BOOST_CHECK_THROW(discretized->set_params(bad_data, mixed),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_euler_and_rk_dynamics_backends() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::IntegratedActionDataAbstractTpl<Scalar> IntegratedData;
  typedef typename IntegrationScalarTraits<Scalar>::OtherScalar OtherScalar;
  typedef typename Fixture::VectorXs VectorXs;
  Fixture fixture;
  const VectorXs x = fixture.state_point();
  const VectorXs u = VectorXs::LinSpaced(fixture.actuation->get_nu(),
                                         Scalar(-0.15), Scalar(0.2));

  const std::shared_ptr<Euler> euler =
      std::make_shared<Euler>(fixture.dynamics, fixture.costs,
                              fixture.constraints, nullptr, fixture.time);
  BOOST_CHECK(euler->get_differential() == nullptr);
  BOOST_CHECK(euler->get_dynamics() == fixture.dynamics);
  BOOST_CHECK(euler->get_integrator_time() == fixture.time);
  check_action_derivatives<Scalar>(euler, x, u);
  check_control_cost_hessian<Scalar>(euler, x, u);

  const std::shared_ptr<typename Euler::ActionDataAbstract> data =
      euler->createData();
  fixture.constraints->changeConstraintStatus("running", false);
  euler->calc(data, x, u);
  BOOST_CHECK_EQUAL(data->g.size(), 0);
  BOOST_CHECK_EQUAL(data->h.size(), fixture.state->get_ndx());
  fixture.constraints->changeConstraintStatus("running", true);
  euler->calc(data, x, u);
  BOOST_CHECK_EQUAL(data->g.size(), fixture.actuation->get_nu());
  BOOST_CHECK(euler->get_g_lb()
                  .head(data->g.size())
                  .isApprox(fixture.constraints->get_lb()));

  check_time_parameter<Scalar>(euler, x, u);

  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::shared_ptr<RK> rk = std::make_shared<RK>(
        fixture.dynamics, fixture.costs, fixture.constraints, nullptr,
        fixture.time, types[i]);
    BOOST_CHECK_EQUAL(rk->get_ni(), static_cast<std::size_t>(types[i]));
    BOOST_TEST_CONTEXT("RK" << static_cast<std::size_t>(types[i])) {
      check_action_derivatives<Scalar>(rk, x, u);
      check_control_cost_hessian<Scalar>(rk, x, u);
      check_time_parameter<Scalar>(rk, x, u);
    }
  }

  const std::shared_ptr<RK> rk2 =
      std::make_shared<RK>(fixture.dynamics, fixture.costs, fixture.constraints,
                           nullptr, fixture.time, crocoddyl::two);
  const std::shared_ptr<RK> rk4 =
      std::make_shared<RK>(fixture.dynamics, fixture.costs, fixture.constraints,
                           nullptr, fixture.time, crocoddyl::four);
  const std::shared_ptr<typename RK::ActionDataAbstract> rk2_data =
      rk2->createData();
  BOOST_CHECK_THROW(rk4->calc(rk2_data, x, u), crocoddyl::Exception);
  BOOST_CHECK_THROW(rk4->calcDiff(rk2_data, x, u), crocoddyl::Exception);
  BOOST_CHECK_THROW(rk4->calc(rk2_data, x), crocoddyl::Exception);
  BOOST_CHECK_THROW(rk4->calcDiff(rk2_data, x), crocoddyl::Exception);

  Fixture live_fixture;
  live_fixture.time->set_time_step(Scalar(0.037));
  const std::shared_ptr<Euler> live_euler = std::make_shared<Euler>(
      live_fixture.dynamics, live_fixture.costs, live_fixture.constraints,
      nullptr, live_fixture.time);
  const std::shared_ptr<RK> live_rk = std::make_shared<RK>(
      live_fixture.dynamics, live_fixture.costs, live_fixture.constraints,
      nullptr, live_fixture.time, crocoddyl::four);
  BOOST_CHECK_EQUAL(live_euler->get_dt(), Scalar(0.037));
  std::ostringstream euler_stream;
  euler_stream << *live_euler;
  BOOST_CHECK(euler_stream.str().find("0.037") != std::string::npos);
  const crocoddyl::IntegratedActionModelEulerTpl<OtherScalar> euler_cast =
      live_euler->template cast<OtherScalar>();
  BOOST_CHECK_CLOSE(static_cast<double>(euler_cast.get_dt()), 0.037, 1e-3);
  std::ostringstream rk_stream;
  rk_stream << *live_rk;
  BOOST_CHECK(rk_stream.str().find("0.037") != std::string::npos);
  const crocoddyl::IntegratedActionModelRKTpl<OtherScalar> rk_cast =
      live_rk->template cast<OtherScalar>();
  BOOST_CHECK_CLOSE(static_cast<double>(rk_cast.get_dt()), 0.037, 1e-3);

  const std::shared_ptr<crocoddyl::DifferentialActionModelLQRTpl<Scalar>>
      differential =
          std::make_shared<crocoddyl::DifferentialActionModelLQRTpl<Scalar>>(4,
                                                                             2);
  Euler legacy_euler(differential, Scalar(0.02));
  legacy_euler.get_integrator_time()->set_time_step(Scalar(0.041));
  const crocoddyl::IntegratedActionModelEulerTpl<OtherScalar>
      legacy_euler_cast = legacy_euler.template cast<OtherScalar>();
  BOOST_CHECK_CLOSE(static_cast<double>(legacy_euler_cast.get_dt()), 0.041,
                    1e-3);
  RK legacy_rk(differential, crocoddyl::four, Scalar(0.02));
  legacy_rk.get_integrator_time()->set_time_step(Scalar(0.043));
  const crocoddyl::IntegratedActionModelRKTpl<OtherScalar> legacy_rk_cast =
      legacy_rk.template cast<OtherScalar>();
  BOOST_CHECK_CLOSE(static_cast<double>(legacy_rk_cast.get_dt()), 0.043, 1e-3);

  VectorXs quasi = VectorXs::Zero(euler->get_nu());
  const VectorXs x_static = fixture.state->zero();
  BOOST_CHECK_NO_THROW(
      euler->quasiStatic(data, quasi, x_static, 20, Scalar(1e-7)));
  BOOST_CHECK_THROW(
      Euler(std::shared_ptr<typename Euler::DynamicsModelAbstract>(),
            fixture.costs),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      Euler(fixture.create_discrete_dynamics(), fixture.create_costs(0)),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      std::make_shared<IntegratedData>(
          static_cast<crocoddyl::IntegratedActionModelAbstractTpl<Scalar>*>(
              nullptr)),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      std::make_shared<typename Euler::Data>(static_cast<Euler*>(nullptr)),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      std::make_shared<typename RK::Data>(static_cast<RK*>(nullptr)),
      crocoddyl::Exception);
}

template <typename Scalar>
void test_rk_legacy_residual_behavior() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef ResidualDifferentialActionProbeTpl<Scalar> Differential;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef typename Fixture::VectorXs VectorXs;

  const std::shared_ptr<Differential> differential =
      std::make_shared<Differential>();
  VectorXs x(2);
  x << Scalar(0.4), Scalar(-0.3);
  VectorXs u(1);
  u << Scalar(0.2);
  const VectorXs running_expected =
      (VectorXs(2) << Scalar(0.6), Scalar(-0.5)).finished();
  const VectorXs terminal_expected = x;
  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};

  for (std::size_t i = 0; i < 3; ++i) {
    RK model(differential, types[i], Scalar(0.02), true);
    const std::shared_ptr<typename RK::Data> data =
        std::dynamic_pointer_cast<typename RK::Data>(model.createData());
    BOOST_REQUIRE(data != nullptr);
    BOOST_REQUIRE(!data->differential.empty());
    BOOST_TEST_CONTEXT("legacy RK" << static_cast<std::size_t>(types[i])) {
      data->r.setConstant(Scalar(-1));
      model.calc(data, x, u);
      BOOST_CHECK(data->r.isApprox(data->differential[0]->r));
      BOOST_CHECK(data->r.isApprox(running_expected));

      data->r.setConstant(Scalar(-1));
      model.calc(data, x);
      BOOST_CHECK(data->r.isApprox(data->differential[0]->r));
      BOOST_CHECK(data->r.isApprox(terminal_expected));
    }
  }

  Fixture fixture;
  const VectorXs dynamics_x = fixture.state_point();
  const VectorXs dynamics_u = VectorXs::LinSpaced(fixture.actuation->get_nu(),
                                                  Scalar(-0.15), Scalar(0.2));
  for (std::size_t i = 0; i < 3; ++i) {
    const std::shared_ptr<RK> model = std::make_shared<RK>(
        fixture.dynamics, fixture.costs, fixture.constraints, nullptr,
        fixture.time, types[i]);
    const std::shared_ptr<typename RK::Data> data =
        std::dynamic_pointer_cast<typename RK::Data>(model->createData());
    BOOST_REQUIRE(data != nullptr);
    BOOST_TEST_CONTEXT("dynamics RK" << static_cast<std::size_t>(types[i])) {
      data->r.setConstant(Scalar(-1));
      model->calc(data, dynamics_x, dynamics_u);
      BOOST_REQUIRE(!data->costs[0]->costs.empty());
      BOOST_CHECK(!data->costs[0]->costs.begin()->second->residual->r.isZero(
          derivative_tolerance<Scalar>()));
      BOOST_CHECK(data->r.isZero(derivative_tolerance<Scalar>()));

      data->r.setConstant(Scalar(-1));
      model->calc(data, dynamics_x);
      BOOST_CHECK(data->r.isZero(derivative_tolerance<Scalar>()));
    }
  }
}

template <typename Scalar>
void test_discretized_dynamics_backend() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Action;
  typedef crocoddyl::DiscretizedActionDataTpl<Scalar> Data;
  typedef typename Fixture::VectorXs VectorXs;
  Fixture fixture;
  const std::shared_ptr<typename Fixture::DiscreteDynamics> dynamics =
      fixture.create_discrete_dynamics();
  const std::shared_ptr<typename Fixture::Costs> costs =
      fixture.create_costs(0);
  const std::shared_ptr<typename Fixture::Constraints> constraints =
      fixture.create_constraints(0);
  const std::shared_ptr<Action> model =
      std::make_shared<Action>(dynamics, costs, constraints);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model->createData());
  BOOST_REQUIRE(data != nullptr);
  const VectorXs x = fixture.state_point();
  const VectorXs u(0);
  model->calc(data, x, u);
  const VectorXs xnext = data->xnext;
  const Scalar cost = data->cost;
  BOOST_CHECK(data->xnext.isApprox(data->dynamics->vdot));
  BOOST_CHECK_EQUAL(cost, data->costs->cost);
  model->calcDiff(data, x, u);
  BOOST_CHECK(data->Fx.isApprox(data->dynamics->Fx));
  BOOST_CHECK(data->Fu.isApprox(data->dynamics->Fu));
  BOOST_CHECK(data->Lx.isApprox(data->costs->Lx));
  BOOST_CHECK(data->Lxx.isApprox(data->costs->Lxx));
  BOOST_CHECK(data->Lxu.isApprox(data->costs->Lxu));
  BOOST_CHECK(data->Luu.isApprox(data->costs->Luu));
  BOOST_CHECK(data->xnext.isApprox(xnext));

  check_action_derivatives<Scalar>(model, x, u);
  model->calc(data, x);
  model->calcDiff(data, x);
  BOOST_CHECK(data->xnext.isApprox(x));
  BOOST_CHECK(data->Fx.isIdentity(derivative_tolerance<Scalar>()));
  BOOST_CHECK_EQUAL(data->g.size(), model->get_ng_T());
  BOOST_CHECK_EQUAL(data->h.size(), model->get_nh_T());

  const std::shared_ptr<typename crocoddyl::ParameterManagerTpl<Scalar>>
      manager = std::make_shared<crocoddyl::ParameterManagerTpl<Scalar>>(
          fixture.state);
  const std::shared_ptr<typename Action::ParameterDataManager> payload =
      manager->createData();
  const std::shared_ptr<Data> shared =
      std::dynamic_pointer_cast<Data>(model->createData(payload));
  BOOST_REQUIRE(shared != nullptr);
  BOOST_CHECK(shared->params == payload);
  const std::shared_ptr<crocoddyl::DynamicsDataImpulseForwardTpl<Scalar>>
      dynamics_data = std::dynamic_pointer_cast<
          crocoddyl::DynamicsDataImpulseForwardTpl<Scalar>>(shared->dynamics);
  BOOST_REQUIRE(dynamics_data != nullptr);
  BOOST_CHECK(dynamics_data->params == payload);
  model->set_params(shared, manager);
  model->update_p(shared, VectorXs::Zero(0));
  BOOST_CHECK_THROW(
      model->set_params(std::shared_ptr<typename Action::ActionDataAbstract>(),
                        manager),
      crocoddyl::Exception);

  Action copied(*model);
  const std::shared_ptr<typename Action::ActionDataAbstract> copied_data =
      copied.createData();
  copied.calc(copied_data, x, u);
  BOOST_CHECK(copied_data->xnext.isApprox(xnext));
  BOOST_CHECK_THROW(Action(fixture.dynamics, fixture.costs),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      Action(std::shared_ptr<typename Action::DynamicsModelAbstract>(), costs),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(std::make_shared<Data>(static_cast<Action*>(nullptr)),
                    crocoddyl::Exception);
}

template <typename Scalar>
void test_control_parametrization_chain_rules() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef crocoddyl::ControlParametrizationModelPolyOneTpl<Scalar> Control;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef typename Fixture::VectorXs VectorXs;
  Fixture fixture;
  const std::shared_ptr<Control> control =
      std::make_shared<Control>(fixture.actuation->get_nu());
  const VectorXs x = fixture.state_point();

  const std::shared_ptr<Euler> euler =
      std::make_shared<Euler>(fixture.dynamics, fixture.costs,
                              fixture.constraints, control, fixture.time);
  const VectorXs u =
      VectorXs::LinSpaced(euler->get_nu(), Scalar(-0.2), Scalar(0.25));
  BOOST_CHECK_EQUAL(euler->get_nu(), 2 * fixture.actuation->get_nu());
  const std::shared_ptr<Euler> euler_zero =
      std::make_shared<Euler>(fixture.dynamics, fixture.costs,
                              fixture.constraints, nullptr, fixture.time);
  const std::shared_ptr<typename Euler::ActionDataAbstract> euler_data =
      euler->createData();
  const std::shared_ptr<typename Euler::ActionDataAbstract> zero_data =
      euler_zero->createData();
  const Eigen::Index nw =
      static_cast<Eigen::Index>(fixture.actuation->get_nu());
  euler->calc(euler_data, x, u);
  euler->calcDiff(euler_data, x, u);
  euler_zero->calc(zero_data, x, u.head(nw));
  euler_zero->calcDiff(zero_data, x, u.head(nw));
  const Scalar tol = derivative_tolerance<Scalar>();
  BOOST_CHECK(euler_data->Fx.isApprox(zero_data->Fx, tol));
  BOOST_CHECK(euler_data->Fu.leftCols(nw).isApprox(zero_data->Fu, tol));
  BOOST_CHECK(euler_data->Fu.rightCols(nw).isZero(tol));
  BOOST_CHECK(euler_data->Lu.head(nw).isApprox(zero_data->Lu, tol));
  BOOST_CHECK(euler_data->Lu.tail(nw).isZero(tol));
  BOOST_CHECK(
      euler_data->Luu.topLeftCorner(nw, nw).isApprox(zero_data->Luu, tol));
  BOOST_CHECK(euler_data->Luu.topRightCorner(nw, nw).isZero(tol));
  BOOST_CHECK(euler_data->Luu.bottomRows(nw).isZero(tol));
  BOOST_CHECK(euler_data->Gu.leftCols(nw).isApprox(zero_data->Gu, tol));
  BOOST_CHECK(euler_data->Gu.rightCols(nw).isZero(tol));
  BOOST_CHECK(euler_data->Hu.leftCols(nw).isApprox(zero_data->Hu, tol));
  BOOST_CHECK(euler_data->Hu.rightCols(nw).isZero(tol));

  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::shared_ptr<RK> rk = std::make_shared<RK>(
        fixture.dynamics, fixture.costs, fixture.constraints, control,
        fixture.time, types[i]);
    BOOST_TEST_CONTEXT("RK" << static_cast<std::size_t>(types[i])) {
      check_action_derivatives<Scalar>(rk, x, u);
      check_control_cost_hessian<Scalar>(rk, x, u);
    }
  }
}

template <typename Scalar>
void test_runtime_no_allocation() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Discretized;
  typedef typename Fixture::VectorXs VectorXs;
  Fixture fixture;
  const VectorXs x = fixture.state_point();
  const VectorXs u = VectorXs::Zero(fixture.actuation->get_nu());
  const std::shared_ptr<Euler> euler = std::make_shared<Euler>(
      fixture.dynamics, fixture.costs, nullptr, nullptr, fixture.time);
  const std::shared_ptr<typename Euler::ActionDataAbstract> euler_data =
      euler->createData();
  euler->calc(euler_data, x, u);
  euler->calcDiff(euler_data, x, u);

  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};
  std::vector<std::shared_ptr<RK>> rks;
  std::vector<std::shared_ptr<typename RK::ActionDataAbstract>> rk_data;
  for (std::size_t i = 0; i < 3; ++i) {
    rks.push_back(std::make_shared<RK>(fixture.dynamics, fixture.costs, nullptr,
                                       nullptr, fixture.time, types[i]));
    rk_data.push_back(rks.back()->createData());
    rks.back()->calc(rk_data.back(), x, u);
    rks.back()->calcDiff(rk_data.back(), x, u);
  }

  const std::shared_ptr<typename Fixture::DiscreteDynamics> impulse =
      fixture.create_discrete_dynamics();
  const std::shared_ptr<Discretized> discretized =
      std::make_shared<Discretized>(impulse, fixture.create_costs(0));
  const std::shared_ptr<typename Discretized::ActionDataAbstract>
      discrete_data = discretized->createData();
  const VectorXs u0(0);
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  const std::shared_ptr<Manager> manager =
      std::make_shared<Manager>(fixture.state);
  manager->addParam("time",
                    std::make_shared<TimeParams>(fixture.state, fixture.time));
  manager->addParam("inertial", fixture.create_inertial_params());
  VectorXs p = manager->zero();
  using std::log;
  p[0] = log(Scalar(0.025));
  p[p.size() - 1] += Scalar(0.08);
  euler->set_params(euler_data, manager);
  euler->update_p(euler_data, p);
  for (std::size_t i = 0; i < rks.size(); ++i) {
    rks[i]->set_params(rk_data[i], manager);
    rks[i]->update_p(rk_data[i], p);
  }
  const std::shared_ptr<Manager> discrete_manager =
      std::make_shared<Manager>(fixture.state);
  discrete_manager->addParam("inertial", fixture.create_inertial_params());
  VectorXs discrete_p = discrete_manager->zero();
  discrete_p[discrete_p.size() - 1] += Scalar(0.08);
  discretized->set_params(discrete_data, discrete_manager);
  discretized->update_p(discrete_data, discrete_p);
  euler->calc(euler_data, x, u);
  euler->calcDiff(euler_data, x, u);
  for (std::size_t i = 0; i < rks.size(); ++i) {
    rks[i]->calc(rk_data[i], x, u);
    rks[i]->calcDiff(rk_data[i], x, u);
  }
  discretized->calc(discrete_data, x, u0);
  discretized->calcDiff(discrete_data, x, u0);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t repeat = 0; repeat < 100; ++repeat) {
      euler->update_p(euler_data, p);
      euler->calc(euler_data, x, u);
      euler->calcDiff(euler_data, x, u);
      for (std::size_t i = 0; i < rks.size(); ++i) {
        rks[i]->update_p(rk_data[i], p);
        rks[i]->calc(rk_data[i], x, u);
        rks[i]->calcDiff(rk_data[i], x, u);
      }
      discretized->update_p(discrete_data, discrete_p);
      discretized->calc(discrete_data, x, u0);
      discretized->calcDiff(discrete_data, x, u0);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_CHECK(true);
}

template <typename Scalar>
void test_scalar_casts() {
  typedef IntegrationFixtureTpl<Scalar> Fixture;
  typedef typename IntegrationScalarTraits<Scalar>::OtherScalar OtherScalar;
  typedef crocoddyl::ParameterManagerTpl<Scalar> Manager;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<OtherScalar> TimeParamsOther;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelEulerTpl<OtherScalar> EulerOther;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::IntegratedActionModelRKTpl<OtherScalar> RKOther;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Discretized;
  typedef crocoddyl::DiscretizedActionModelTpl<OtherScalar> DiscretizedOther;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<OtherScalar>
      DynamicsOther;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<OtherScalar> ImpulseOther;
  typedef crocoddyl::MultibodyInertialParamsTpl<OtherScalar> InertialOther;
  typedef typename Fixture::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<OtherScalar>::VectorXs VectorXsOther;

  Fixture fixture;
  const std::shared_ptr<Manager> manager =
      std::make_shared<Manager>(fixture.state);
  manager->addParam(
      "a_inactive",
      std::make_shared<LQRParams>(fixture.state, static_cast<std::size_t>(1)),
      false);
  manager->addParam("b_time",
                    std::make_shared<TimeParams>(fixture.state, fixture.time));
  manager->addParam("c_inertial", fixture.create_inertial_params());
  VectorXs p = manager->zero();
  using std::log;
  p[0] = log(Scalar(0.017));
  p[p.size() - 1] += Scalar(0.06);
  const VectorXs x = fixture.state_point();
  const VectorXs u = VectorXs::LinSpaced(fixture.actuation->get_nu(),
                                         Scalar(-0.1), Scalar(0.15));

  const std::shared_ptr<Euler> euler = std::make_shared<Euler>(
      fixture.dynamics,
      fixture.create_parameter_costs(u.size(), manager->get_np()),
      fixture.create_parameter_constraints(u.size(), manager->get_np()),
      nullptr, fixture.time);
  const std::shared_ptr<typename Euler::ActionDataAbstract> euler_data =
      euler->createData();
  euler->set_params(euler_data, manager);
  euler->update_p(euler_data, p);
  EulerOther euler_cast = euler->template cast<OtherScalar>();
  const std::shared_ptr<typename EulerOther::Data> euler_cast_data =
      std::dynamic_pointer_cast<typename EulerOther::Data>(
          euler_cast.createData());
  BOOST_REQUIRE(euler_cast_data != nullptr);
  BOOST_REQUIRE(euler_cast_data->params != nullptr);
  BOOST_CHECK_EQUAL(euler_cast_data->params->parameter_data,
                    euler_cast_data->params.get());
  BOOST_CHECK_EQUAL(euler_cast_data->params->params->np_action, 1u);
  BOOST_CHECK_EQUAL(euler_cast_data->params->params->np_dynamics, 10u);
  const std::shared_ptr<DynamicsOther> euler_dynamics =
      std::dynamic_pointer_cast<DynamicsOther>(euler_cast.get_dynamics());
  BOOST_REQUIRE(euler_dynamics != nullptr);
  BOOST_CHECK(!euler_dynamics->get_params()->getParamStatus("a_inactive"));
  BOOST_CHECK(euler_dynamics->get_params()->getParamStatus("b_time"));
  const std::shared_ptr<TimeParamsOther> euler_time =
      std::dynamic_pointer_cast<TimeParamsOther>(euler_dynamics->get_params()
                                                     ->get_action_params()
                                                     .at("b_time")
                                                     ->get_param());
  BOOST_REQUIRE(euler_time != nullptr);
  BOOST_CHECK(euler_time->get_integrator_time() ==
              euler_cast.get_integrator_time());
  const std::shared_ptr<InertialOther> euler_inertial =
      std::dynamic_pointer_cast<InertialOther>(euler_dynamics->get_params()
                                                   ->get_dynamics_params()
                                                   .at("c_inertial")
                                                   ->get_param());
  BOOST_REQUIRE(euler_inertial != nullptr);
  BOOST_CHECK(euler_inertial->get_state() == euler_dynamics->get_state());
  BOOST_CHECK_CLOSE(static_cast<double>(euler_cast.get_dt()), 0.017, 1e-3);
  const VectorXsOther p_other = p.template cast<OtherScalar>();
  const VectorXsOther x_other = x.template cast<OtherScalar>();
  const VectorXsOther u_other = u.template cast<OtherScalar>();
  euler_cast.update_p(euler_cast_data, p_other);
  euler_cast.calc(euler_cast_data, x_other, u_other);
  euler_cast.calcDiff(euler_cast_data, x_other, u_other);
  BOOST_CHECK_CLOSE(static_cast<double>(euler_cast.get_dt()), 0.017, 1e-3);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(euler_cast_data->Fp.cols()), 11u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(euler_cast_data->Gp.cols()), 11u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(euler_cast_data->Hp.cols()), 11u);

  const crocoddyl::RKType types[] = {crocoddyl::two, crocoddyl::three,
                                     crocoddyl::four};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::shared_ptr<RK> rk = std::make_shared<RK>(
        fixture.dynamics,
        fixture.create_parameter_costs(u.size(), manager->get_np()),
        fixture.create_parameter_constraints(u.size(), manager->get_np()),
        nullptr, fixture.time, types[i]);
    const std::shared_ptr<typename RK::ActionDataAbstract> rk_data =
        rk->createData();
    rk->set_params(rk_data, manager);
    rk->update_p(rk_data, p);
    RKOther rk_cast = rk->template cast<OtherScalar>();
    const std::shared_ptr<typename RKOther::Data> rk_cast_data =
        std::dynamic_pointer_cast<typename RKOther::Data>(rk_cast.createData());
    BOOST_REQUIRE(rk_cast_data != nullptr);
    BOOST_REQUIRE(rk_cast_data->params != nullptr);
    BOOST_CHECK_EQUAL(rk_cast_data->params->parameter_data,
                      rk_cast_data->params.get());
    BOOST_CHECK_EQUAL(rk_cast_data->params->params->np_action, 1u);
    BOOST_CHECK_EQUAL(rk_cast_data->params->params->np_dynamics, 10u);
    const std::shared_ptr<DynamicsOther> rk_dynamics =
        std::dynamic_pointer_cast<DynamicsOther>(rk_cast.get_dynamics());
    BOOST_REQUIRE(rk_dynamics != nullptr);
    BOOST_CHECK(!rk_dynamics->get_params()->getParamStatus("a_inactive"));
    BOOST_CHECK(rk_dynamics->get_params()->getParamStatus("b_time"));
    const std::shared_ptr<TimeParamsOther> rk_time =
        std::dynamic_pointer_cast<TimeParamsOther>(rk_dynamics->get_params()
                                                       ->get_action_params()
                                                       .at("b_time")
                                                       ->get_param());
    BOOST_REQUIRE(rk_time != nullptr);
    BOOST_CHECK(rk_time->get_integrator_time() ==
                rk_cast.get_integrator_time());
    const std::shared_ptr<InertialOther> rk_inertial =
        std::dynamic_pointer_cast<InertialOther>(rk_dynamics->get_params()
                                                     ->get_dynamics_params()
                                                     .at("c_inertial")
                                                     ->get_param());
    BOOST_REQUIRE(rk_inertial != nullptr);
    BOOST_CHECK(rk_inertial->get_state() == rk_dynamics->get_state());
    BOOST_CHECK_CLOSE(static_cast<double>(rk_cast.get_dt()), 0.017, 1e-3);
    rk_cast.update_p(rk_cast_data, p_other);
    rk_cast.calc(rk_cast_data, x_other, u_other);
    rk_cast.calcDiff(rk_cast_data, x_other, u_other);
    BOOST_CHECK_EQUAL(static_cast<std::size_t>(rk_cast_data->Fp.cols()), 11u);
    BOOST_CHECK_EQUAL(static_cast<std::size_t>(rk_cast_data->Gp.cols()), 11u);
    BOOST_CHECK_EQUAL(static_cast<std::size_t>(rk_cast_data->Hp.cols()), 11u);
  }

  const std::shared_ptr<Manager> discrete_manager =
      std::make_shared<Manager>(fixture.state);
  discrete_manager->addParam(
      "a_inactive",
      std::make_shared<LQRParams>(fixture.state, static_cast<std::size_t>(1)),
      false);
  discrete_manager->addParam("c_inertial", fixture.create_inertial_params());
  VectorXs p_discrete = discrete_manager->zero();
  p_discrete[p_discrete.size() - 1] += Scalar(0.06);
  const std::shared_ptr<Discretized> discrete = std::make_shared<Discretized>(
      fixture.create_discrete_dynamics(),
      fixture.create_parameter_costs(0, discrete_manager->get_np()),
      fixture.create_parameter_constraints(0, discrete_manager->get_np()));
  const std::shared_ptr<typename Discretized::ActionDataAbstract>
      discrete_data = discrete->createData();
  discrete->set_params(discrete_data, discrete_manager);
  discrete->update_p(discrete_data, p_discrete);
  DiscretizedOther discrete_cast = discrete->template cast<OtherScalar>();
  const std::shared_ptr<typename DiscretizedOther::Data> discrete_cast_data =
      std::dynamic_pointer_cast<typename DiscretizedOther::Data>(
          discrete_cast.createData());
  BOOST_REQUIRE(discrete_cast_data != nullptr);
  BOOST_REQUIRE(discrete_cast_data->params != nullptr);
  BOOST_CHECK_EQUAL(discrete_cast_data->params->parameter_data,
                    discrete_cast_data->params.get());
  BOOST_CHECK_EQUAL(discrete_cast_data->params->params->np_action, 0u);
  BOOST_CHECK_EQUAL(discrete_cast_data->params->params->np_dynamics, 10u);
  const std::shared_ptr<ImpulseOther> discrete_dynamics =
      std::dynamic_pointer_cast<ImpulseOther>(discrete_cast.get_dynamics());
  BOOST_REQUIRE(discrete_dynamics != nullptr);
  BOOST_CHECK(!discrete_dynamics->get_params()->getParamStatus("a_inactive"));
  const std::shared_ptr<InertialOther> discrete_inertial =
      std::dynamic_pointer_cast<InertialOther>(discrete_dynamics->get_params()
                                                   ->get_dynamics_params()
                                                   .at("c_inertial")
                                                   ->get_param());
  BOOST_REQUIRE(discrete_inertial != nullptr);
  BOOST_CHECK(discrete_inertial->get_state() == discrete_dynamics->get_state());
  const VectorXsOther p_discrete_other =
      p_discrete.template cast<OtherScalar>();
  const VectorXsOther u0(0);
  discrete_cast.update_p(discrete_cast_data, p_discrete_other);
  discrete_cast.calc(discrete_cast_data, x_other, u0);
  discrete_cast.calcDiff(discrete_cast_data, x_other, u0);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(discrete_cast_data->Fp.cols()),
                    10u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(discrete_cast_data->Gp.cols()),
                    10u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(discrete_cast_data->Hp.cols()),
                    10u);
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_action_integration");
  ts->add(BOOST_TEST_CASE(&test_euler_and_rk_dynamics_backends<double>));
  ts->add(BOOST_TEST_CASE(&test_euler_and_rk_dynamics_backends<float>));
  ts->add(BOOST_TEST_CASE(&test_rk_legacy_residual_behavior<double>));
  ts->add(BOOST_TEST_CASE(&test_rk_legacy_residual_behavior<float>));
  ts->add(BOOST_TEST_CASE(&test_discretized_dynamics_backend<double>));
  ts->add(BOOST_TEST_CASE(&test_discretized_dynamics_backend<float>));
  ts->add(BOOST_TEST_CASE(&test_control_parametrization_chain_rules<double>));
  ts->add(BOOST_TEST_CASE(&test_control_parametrization_chain_rules<float>));
  ts->add(BOOST_TEST_CASE(&test_constraint_row_ordering<double>));
  ts->add(BOOST_TEST_CASE(&test_constraint_row_ordering<float>));
  ts->add(BOOST_TEST_CASE(&test_live_constraint_activation<double>));
  ts->add(BOOST_TEST_CASE(&test_live_constraint_activation<float>));
  ts->add(BOOST_TEST_CASE(&test_nonzero_parameter_derivatives<double>));
  ts->add(BOOST_TEST_CASE(&test_nonzero_parameter_derivatives<float>));
  ts->add(BOOST_TEST_CASE(&test_runtime_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_runtime_no_allocation<float>));
  ts->add(BOOST_TEST_CASE(&test_scalar_casts<double>));
  ts->add(BOOST_TEST_CASE(&test_scalar_casts<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
