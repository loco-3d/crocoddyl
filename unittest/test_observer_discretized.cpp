///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC
#endif

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <type_traits>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/numdiff/observer.hpp"
#include "crocoddyl/core/observer/discretized.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/exp-eigenvalue.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

template <typename _Scalar>
class RestorationObserverTpl
    : public crocoddyl::ObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_CAST(crocoddyl::ActionModelBase,
                              RestorationObserverTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ObserverModelAbstractTpl<Scalar> Base;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::ParameterManager ParameterManager;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  explicit RestorationObserverTpl(std::shared_ptr<StateAbstract> state)
      : Base(state, 0, 2, 0, 0, 0, 0, 0, 1),
        current(VectorXs::Zero(1)),
        nominal(VectorXs::Zero(1)),
        reject_perturbed(false) {}

  void calc(const std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>&) override {
    if (reject_perturbed && !current.isApprox(nominal, Scalar(0))) {
      throw_pretty("Deliberate perturbed-parameter failure");
    }
    data->xnext = x;
    data->cost = current.squaredNorm();
  }

  void calcDiff(const std::shared_ptr<ActionDataAbstract>&,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {}

  void set_params(const std::shared_ptr<ActionDataAbstract>&,
                  std::shared_ptr<ParameterManager>) override {}

  void update_p(const std::shared_ptr<ActionDataAbstract>&,
                const Eigen::Ref<const VectorXs>& p) override {
    current = p;
  }

  VectorXs current;
  VectorXs nominal;
  bool reject_perturbed;
};

template <typename Scalar>
void test_discretized_observer_scalar() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Constraints;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> Dynamics;
  typedef crocoddyl::CostModelSumTpl<Scalar> Costs;
  typedef crocoddyl::DiscretizedObserverModelTpl<Scalar> Model;
  typedef crocoddyl::DiscretizedObserverDataTpl<Scalar> Data;
  typedef crocoddyl::ObserverModelNumDiffTpl<Scalar> NumDiff;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ExpEigenValueParametrizationTpl<Scalar>
      InertialParametrization;
  typedef crocoddyl::MultibodyInertialParamsTpl<Scalar> InertialParams;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  const std::vector<std::string> body_names(1,
                                            state->get_pinocchio()->names[1]);
  params->addParam(
      "inertia",
      std::make_shared<InertialParams>(
          state, std::make_shared<InertialParametrization>(), body_names));
  const std::shared_ptr<Constraints> constraints =
      std::make_shared<Constraints>(state, 0);
  const std::shared_ptr<Dynamics> dynamics =
      std::make_shared<Dynamics>(state, constraints, params->get_np());
  const std::shared_ptr<Costs> costs =
      std::make_shared<Costs>(state, state->get_ndx(), params->get_np());
  const std::shared_ptr<Model> model =
      std::make_shared<Model>(dynamics, costs, 0);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(model->createData(params->createData()));
  BOOST_REQUIRE(data != nullptr);
  model->set_params(data, params);

  const VectorXs x = state->rand();
  const VectorXs w =
      VectorXs::LinSpaced(model->get_nu(), Scalar(-0.1), Scalar(0.1));
  const VectorXs p = params->rand();
  model->update_p(data, p);
  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  BOOST_CHECK(data->xnext.isApprox(data->dynamics->vdot));
  BOOST_CHECK(data->Fx.isApprox(data->dynamics->Fx));
  BOOST_CHECK(data->Fu.isZero());
  const std::shared_ptr<Data> second =
      std::dynamic_pointer_cast<Data>(model->createData(params->createData()));
  BOOST_REQUIRE(second != nullptr);
  model->set_params(second, params);
  model->update_p(second, p);
  model->calc(second, x, -w);
  const VectorXs second_xnext = second->xnext;
  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  BOOST_CHECK(second->xnext.isApprox(second_xnext));
  BOOST_CHECK(second->dynamics != data->dynamics);
  BOOST_CHECK(second->costs != data->costs);

  NumDiff numerical(model, params, false);
  numerical.set_disturbance(std::is_same<Scalar, float>::value ? Scalar(2e-3)
                                                               : Scalar(2e-6));
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > data_nd =
      numerical.createData(params->createData());
  numerical.update_p(data_nd, p);
  numerical.calc(data_nd, x, w);
  numerical.calcDiff(data_nd, x, w);
  const Scalar tolerance =
      std::is_same<Scalar, float>::value ? Scalar(5e-2) : Scalar(2e-3);
  BOOST_CHECK(data->Fx.isApprox(data_nd->Fx, tolerance));
  BOOST_CHECK(data->Fu.isApprox(data_nd->Fu, tolerance));

  data->Fu.setConstant(Scalar(11));
  data->Lu.setConstant(Scalar(12));
  data->Lxu.setConstant(Scalar(13));
  data->Luu.setConstant(Scalar(14));
  data->Lpu.setConstant(Scalar(15));
  data->Gu.setConstant(Scalar(16));
  data->Hu.setConstant(Scalar(17));
  data->dissipative_E.setConstant(Scalar(18));
  data->Ex.setConstant(Scalar(19));
  data->Eu.setConstant(Scalar(20));
  data->Ep.setConstant(Scalar(21));
  model->calc(data, x);
  model->calcDiff(data, x);
  BOOST_CHECK(data->Fu.isConstant(Scalar(11)));
  BOOST_CHECK(data->Lu.isConstant(Scalar(12)));
  BOOST_CHECK(data->Lxu.isConstant(Scalar(13)));
  BOOST_CHECK(data->Luu.isConstant(Scalar(14)));
  BOOST_CHECK(data->Lpu.isConstant(Scalar(15)));
  BOOST_CHECK(data->Gu.isConstant(Scalar(16)));
  BOOST_CHECK(data->Hu.isConstant(Scalar(17)));
  BOOST_CHECK(data->dissipative_E.isZero());
  BOOST_CHECK(data->Ex.isZero());
  BOOST_CHECK(data->Eu.isZero());
  BOOST_CHECK(data->Ep.isZero());

  typedef typename std::conditional<std::is_same<Scalar, float>::value, double,
                                    float>::type OtherScalar;
  crocoddyl::DiscretizedObserverModelTpl<OtherScalar> casted =
      model->template cast<OtherScalar>();
  BOOST_REQUIRE(casted.get_params() != nullptr);
  const std::shared_ptr<crocoddyl::DynamicsModelImpulseForwardTpl<OtherScalar> >
      casted_dynamics = std::dynamic_pointer_cast<
          crocoddyl::DynamicsModelImpulseForwardTpl<OtherScalar> >(
          casted.get_dynamics());
  const std::shared_ptr<crocoddyl::MultibodyInertialParamsTpl<OtherScalar> >
      casted_inertia = std::dynamic_pointer_cast<
          crocoddyl::MultibodyInertialParamsTpl<OtherScalar> >(
          casted.get_params()
              ->get_dynamics_params()
              .at("inertia")
              ->get_param());
  BOOST_REQUIRE(casted_dynamics != nullptr);
  BOOST_REQUIRE(casted_inertia != nullptr);
  BOOST_CHECK(casted_inertia->get_state() == casted_dynamics->get_state());
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<OtherScalar> >
      casted_data = casted.createData();
  casted.update_p(casted_data, p.template cast<OtherScalar>());
  casted.calc(casted_data, x.template cast<OtherScalar>(),
              w.template cast<OtherScalar>());
  BOOST_CHECK(casted_data->xnext.allFinite());

  model->calc(data, x, w);
  model->calcDiff(data, x, w);
  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model->update_p(data, p);
      model->calc(data, x, w);
      model->calcDiff(data, x, w);
      model->calc(data, x);
      model->calcDiff(data, x);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }

  BOOST_CHECK_THROW(Data(static_cast<Model*>(nullptr)), crocoddyl::Exception);
}

template <typename Scalar>
void test_observer_numdiff_exception_restoration() {
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef RestorationObserverTpl<Scalar> Observer;
  typedef crocoddyl::ObserverModelAbstractTpl<Scalar> ObserverBase;
  typedef crocoddyl::ObserverModelNumDiffTpl<Scalar> NumDiff;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  const std::shared_ptr<Observer> observer = std::make_shared<Observer>(state);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(state);
  params->addParam("parameter", std::make_shared<LQRParams>(state, 1));
  NumDiff numerical(std::static_pointer_cast<ObserverBase>(observer), params,
                    false);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > data =
      numerical.createData(params->createData());
  const VectorXs p = VectorXs::Constant(1, Scalar(0.3));
  observer->nominal = p;
  numerical.update_p(data, p);
  const VectorXs x =
      VectorXs::LinSpaced(state->get_nx(), Scalar(-0.2), Scalar(0.1));
  const VectorXs w =
      VectorXs::LinSpaced(numerical.get_nu(), Scalar(-0.1), Scalar(0.2));
  numerical.calc(data, x, w);
  observer->reject_perturbed = true;
  BOOST_CHECK_THROW(numerical.calcDiff(data, x, w), crocoddyl::Exception);
  BOOST_CHECK(observer->current.isApprox(p));
  observer->reject_perturbed = false;
  numerical.calc(data, x, w);
  numerical.calcDiff(data, x, w);
  BOOST_CHECK(data->Fp.allFinite());
}

}  // namespace

void register_unit_tests() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_discretized_observer_scalar<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_discretized_observer_scalar<float>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_numdiff_exception_restoration<double>));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_numdiff_exception_restoration<float>));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
