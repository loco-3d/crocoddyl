///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <pinocchio/multibody/sample-models.hpp>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/numdiff/actuation.hpp"
#include "crocoddyl/core/numdiff/dynamics.hpp"
#include "crocoddyl/core/numdiff/residual.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename Scalar>
Scalar tolerance() {
  return std::is_same<Scalar, float>::value ? Scalar(2e-2) : Scalar(2e-5);
}

template <typename _Scalar>
class SharedDynamicsParamsTpl
    : public crocoddyl::DynamicsParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ParamsModelBase, SharedDynamicsParamsTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsParamsAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  SharedDynamicsParamsTpl(std::shared_ptr<StateAbstract> state,
                          std::shared_ptr<VectorXs> value)
      : Base(state, static_cast<std::size_t>(value->size())), value_(value) {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (data == nullptr ||
        static_cast<std::size_t>(p.size()) != this->get_np()) {
      throw_pretty("Invalid argument: parameter update is inconsistent");
    }
    data->p = p;
    *value_ = p;
  }

  void computeJointTorqueRegressor(
      const std::shared_ptr<DynamicsDataAbstract>&,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>&,
      const Eigen::Ref<const VectorXs>&) override {
    params->dtau_dp.setZero();
  }

  template <typename NewScalar>
  SharedDynamicsParamsTpl<NewScalar> cast() const {
    typedef typename crocoddyl::MathBaseTpl<NewScalar>::VectorXs VectorXsNew;
    return SharedDynamicsParamsTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>(),
        std::make_shared<VectorXsNew>(value_->template cast<NewScalar>()));
  }

 private:
  std::shared_ptr<VectorXs> value_;
};

template <typename _Scalar>
struct LinearDynamicsDataTpl
    : public crocoddyl::DynamicsDataAbstractTpl<_Scalar> {
  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsDataAbstractTpl<Scalar> Base;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit LinearDynamicsDataTpl(Model<Scalar>* const model)
      : Base(model), p(model->get_np()) {
    p.setZero();
  }
  VectorXs p;
};

template <typename _Scalar>
class LinearDynamicsModelTpl
    : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase, LinearDynamicsModelTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef LinearDynamicsDataTpl<Scalar> Data;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::ParameterDataManager ParameterDataManager;
  typedef typename Base::ParameterManager ParameterManager;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  LinearDynamicsModelTpl(std::shared_ptr<StateAbstract> state,
                         const std::size_t np)
      : Base(state, crocoddyl::DynamicsType::ContinuousControl, np, 2, 1, 1),
        A(2, 4),
        B(2, 2),
        P(2, np) {
    A << Scalar(0.4), Scalar(-0.2), Scalar(0.1), Scalar(0.5), Scalar(0.3),
        Scalar(0.7), Scalar(-0.4), Scalar(0.2);
    B << Scalar(0.6), Scalar(0.1), Scalar(-0.4), Scalar(0.8);
    P << Scalar(0.5), Scalar(-0.3), Scalar(0.2), Scalar(0.9);
  }

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());
    data->vdot.noalias() = A * x + B * u + P * d->p;
    data->dissipative_P(0) = x.tail(2).squaredNorm() + d->p.squaredNorm();
    data->g(0) = x(0) + Scalar(2) * u(0) + Scalar(3) * d->p(0);
    data->h(0) = x(3) - u(1) + Scalar(4) * d->p(1);
    if (throw_on_perturbed && !d->p.isApprox(nominal)) {
      throw_pretty("deliberate dynamics perturbation failure");
    }
  }

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    calc(data, x, VectorXs::Zero(this->get_nu()));
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>&,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {}

  std::shared_ptr<DynamicsDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  std::shared_ptr<DynamicsDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>&) override {
    return createData();
  }

  void set_params(const std::shared_ptr<DynamicsDataAbstract>&,
                  std::shared_ptr<ParameterManager>) override {}

  void update_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override {
    if (static_cast<std::size_t>(p.size()) != this->get_np()) {
      throw_pretty("Invalid argument: p has wrong dimension");
    }
    static_cast<Data*>(data.get())->p = p;
  }

  template <typename NewScalar>
  LinearDynamicsModelTpl<NewScalar> cast() const {
    LinearDynamicsModelTpl<NewScalar> out(
        this->get_state()->template cast<NewScalar>(), this->get_np());
    out.A = A.template cast<NewScalar>();
    out.B = B.template cast<NewScalar>();
    out.P = P.template cast<NewScalar>();
    return out;
  }

  MatrixXs A, B, P;
  VectorXs nominal;
  bool throw_on_perturbed = false;
};

template <typename _Scalar>
class ManifoldDiscreteDynamicsTpl
    : public crocoddyl::DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::DynamicsModelBase,
                         ManifoldDiscreteDynamicsTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::DynamicsModelAbstractTpl<Scalar> Base;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  explicit ManifoldDiscreteDynamicsTpl(std::shared_ptr<StateAbstract> state)
      : Base(state, crocoddyl::DynamicsType::DiscreteTime, 0, 2, 0, 0),
        B(state->get_ndx(), 2) {
    B.col(0) = VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.2), Scalar(0.3));
    B.col(1) = VectorXs::LinSpaced(state->get_ndx(), Scalar(0.4), Scalar(-0.1));
  }

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    const VectorXs dx = B * u;
    this->get_state()->integrate(x, dx, data->vdot);
  }

  void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    const VectorXs dx = VectorXs::Zero(this->get_state()->get_ndx());
    this->get_state()->integrate(x, dx, data->vdot);
  }

  void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>&,
                   const Eigen::Ref<const VectorXs>&,
                   const Eigen::Ref<const VectorXs>&) override {}

  template <typename NewScalar>
  ManifoldDiscreteDynamicsTpl<NewScalar> cast() const {
    ManifoldDiscreteDynamicsTpl<NewScalar> out(
        this->get_state()->template cast<NewScalar>());
    out.B = B.template cast<NewScalar>();
    return out;
  }

  MatrixXs B;
};

template <typename _Scalar>
class LinearActuationModelTpl
    : public crocoddyl::ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ActuationModelBase, LinearActuationModelTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ActuationModelAbstractTpl<Scalar> Base;
  typedef typename Base::ActuationDataAbstract ActuationDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  LinearActuationModelTpl(std::shared_ptr<StateAbstract> state,
                          std::shared_ptr<VectorXs> p)
      : Base(state, 2), p_(p), A(2, 4), B(2, 2), P(2, 2) {
    A << Scalar(0.2), Scalar(0.4), Scalar(-0.1), Scalar(0.3), Scalar(0.5),
        Scalar(-0.2), Scalar(0.6), Scalar(0.1);
    B << Scalar(0.7), Scalar(-0.2), Scalar(0.5), Scalar(0.6);
    P << Scalar(0.8), Scalar(0.1), Scalar(-0.3), Scalar(0.9);
  }

  void calc(const std::shared_ptr<ActuationDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    data->tau.noalias() = A * x + B * u + P * (*p_);
    if (throw_on_perturbed && !p_->isApprox(nominal)) {
      throw_pretty("deliberate actuation perturbation failure");
    }
  }

  void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    data->dtau_dx = A;
    data->dtau_du = B;
  }

  void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>& tau) override {
    data->u = tau;
  }

  template <typename NewScalar>
  LinearActuationModelTpl<NewScalar> cast() const {
    typedef typename crocoddyl::MathBaseTpl<NewScalar>::VectorXs VectorXsNew;
    LinearActuationModelTpl<NewScalar> out(
        this->get_state()->template cast<NewScalar>(),
        std::make_shared<VectorXsNew>(p_->template cast<NewScalar>()));
    out.A = A.template cast<NewScalar>();
    out.B = B.template cast<NewScalar>();
    out.P = P.template cast<NewScalar>();
    return out;
  }

  std::shared_ptr<VectorXs> p_;
  MatrixXs A, B, P;
  VectorXs nominal;
  bool throw_on_perturbed = false;
};

template <typename _Scalar>
class LinearResidualModelTpl
    : public crocoddyl::ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ResidualModelBase, LinearResidualModelTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ResidualModelAbstractTpl<Scalar> Base;
  typedef typename Base::DataCollectorAbstract DataCollectorAbstract;
  typedef typename Base::ResidualDataAbstract ResidualDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  LinearResidualModelTpl(std::shared_ptr<StateAbstract> state,
                         const std::size_t np)
      : Base(state, 2, 2, true, true, true, np), X(2, 4), U(2, 2), P(2, np) {
    X << Scalar(0.3), Scalar(-0.1), Scalar(0.2), Scalar(0.5), Scalar(0.6),
        Scalar(0.2), Scalar(-0.4), Scalar(0.7);
    U << Scalar(0.4), Scalar(0.7), Scalar(-0.2), Scalar(0.5);
    P << Scalar(0.9), Scalar(-0.3), Scalar(0.1), Scalar(0.8);
  }

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    crocoddyl::DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<crocoddyl::DataCollectorParamsTpl<Scalar>*>(data->shared);
    if (collector == nullptr || collector->params == nullptr) {
      throw_pretty("Invalid argument: parameter collector is missing");
    }
    data->r.noalias() = X * x + U * u + P * collector->params->p;
    if (throw_on_perturbed && !collector->params->p.isApprox(nominal)) {
      throw_pretty("deliberate residual perturbation failure");
    }
  }

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    calc(data, x, VectorXs::Zero(this->get_nu()));
  }

  template <typename NewScalar>
  LinearResidualModelTpl<NewScalar> cast() const {
    LinearResidualModelTpl<NewScalar> out(
        this->get_state()->template cast<NewScalar>(), this->get_np());
    out.X = X.template cast<NewScalar>();
    out.U = U.template cast<NewScalar>();
    out.P = P.template cast<NewScalar>();
    return out;
  }

  MatrixXs X, U, P;
  VectorXs nominal;
  bool throw_on_perturbed = false;
};

template <typename _Scalar>
struct CallbackParamsCollectorTpl
    : public crocoddyl::DataCollectorParamsTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::DataCollectorParamsTpl<Scalar> Base;
  typedef crocoddyl::ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  explicit CallbackParamsCollectorTpl(
      const std::shared_ptr<ParameterDataManager>& parameter_data)
      : Base(parameter_data->params, parameter_data.get()),
        observed(parameter_data->params->np) {
    observed.setZero();
  }

  VectorXs observed;
};

template <typename _Scalar>
class CallbackResidualModelTpl
    : public crocoddyl::ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ResidualModelBase, CallbackResidualModelTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ResidualModelAbstractTpl<Scalar> Base;
  typedef typename Base::DataCollectorAbstract DataCollectorAbstract;
  typedef typename Base::ResidualDataAbstract ResidualDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  CallbackResidualModelTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t np)
      : Base(state, 2, 2, false, false, false, np), P(2, np), nominal(np) {
    P << Scalar(0.7), Scalar(-0.2), Scalar(0.1), Scalar(0.9);
    nominal.setZero();
  }

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>&,
            const Eigen::Ref<const VectorXs>&) override {
    CallbackParamsCollectorTpl<Scalar>* collector =
        dynamic_cast<CallbackParamsCollectorTpl<Scalar>*>(data->shared);
    if (collector == nullptr) {
      throw_pretty("Invalid argument: callback collector is missing");
    }
    data->r.noalias() = P * collector->observed;
    if (throw_on_perturbed && !collector->observed.isApprox(nominal)) {
      throw_pretty("deliberate callback residual perturbation failure");
    }
  }

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    calc(data, x, VectorXs::Zero(this->get_nu()));
  }

  template <typename NewScalar>
  CallbackResidualModelTpl<NewScalar> cast() const {
    CallbackResidualModelTpl<NewScalar> out(
        this->get_state()->template cast<NewScalar>(), this->get_np());
    out.P = P.template cast<NewScalar>();
    out.nominal = nominal.template cast<NewScalar>();
    return out;
  }

  MatrixXs P;
  VectorXs nominal;
  bool throw_on_perturbed = false;
};

template <typename _Scalar>
struct NodeConstraintActionDataTpl
    : public crocoddyl::ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionDataAbstractTpl<Scalar> Base;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit NodeConstraintActionDataTpl(Model<Scalar>* const model)
      : Base(model), p(model->get_np()) {
    p.setZero();
  }

  VectorXs p;
};

template <typename _Scalar>
class NodeConstraintActionModelTpl
    : public crocoddyl::ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ActionModelBase,
                         NodeConstraintActionModelTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionModelAbstractTpl<Scalar> Base;
  typedef NodeConstraintActionDataTpl<Scalar> Data;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::ParameterDataManager ParameterDataManager;
  typedef typename Base::ParameterManager ParameterManager;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  explicit NodeConstraintActionModelTpl(std::shared_ptr<StateAbstract> state)
      : Base(state, 2, 0, 1, 2, 3, 4, 2) {}

  void calc(const std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());
    d->resize(this, true);
    data->xnext = x;
    data->cost = u.squaredNorm() + d->p.squaredNorm();
    data->g(0) = d->p(0) + Scalar(2) * d->p(1);
    data->h(0) = Scalar(3) * d->p(0) - d->p(1);
    data->h(1) = Scalar(-2) * d->p(0) + Scalar(4) * d->p(1);
  }

  void calc(const std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    Data* d = static_cast<Data*>(data.get());
    d->resize(this, false);
    data->xnext = x;
    data->cost = d->p.squaredNorm();
    data->g << d->p(0), d->p(1), d->p.sum();
    data->h << Scalar(2) * d->p(0), Scalar(3) * d->p(1), d->p(0) - d->p(1),
        Scalar(-4) * d->p(0) + Scalar(5) * d->p(1);
  }

  void calcDiff(const std::shared_ptr<ActionDataAbstract>&,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {}

  void calcDiff(const std::shared_ptr<ActionDataAbstract>&,
                const Eigen::Ref<const VectorXs>&) override {}

  std::shared_ptr<ActionDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>&) override {
    return createData();
  }

  void set_params(const std::shared_ptr<ActionDataAbstract>&,
                  std::shared_ptr<ParameterManager>) override {}

  void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override {
    if (static_cast<std::size_t>(p.size()) != this->get_np()) {
      throw_pretty("Invalid argument: p has wrong dimension");
    }
    static_cast<Data*>(data.get())->p = p;
  }

  template <typename NewScalar>
  NodeConstraintActionModelTpl<NewScalar> cast() const {
    return NodeConstraintActionModelTpl<NewScalar>(
        this->get_state()->template cast<NewScalar>());
  }
};

template <typename Scalar>
void test_action_numdiff_parameter_blocks() {
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef crocoddyl::ActionModelNumDiffTpl<Scalar> NumDiff;
  typedef crocoddyl::LQRParamsTpl<Scalar> Params;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename Model::MatrixXs MatrixXs;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 3, nu = 2, np = 2, ng = 1, nh = 1;
  const std::shared_ptr<Model> model =
      std::make_shared<Model>(nx, nu, np, ng, nh, false);
  MatrixXs A(nx, nx), B(nx, nu), P(nx, np), Q(nx, nx), R(nu, nu);
  MatrixXs N(nx, nu), W(np, np), Y(nx, np), V(nu, np);
  A << Scalar(0.6), Scalar(0.1), Scalar(-0.2), Scalar(0.2), Scalar(0.8),
      Scalar(0.3), Scalar(-0.1), Scalar(0.4), Scalar(0.7);
  B << Scalar(0.5), Scalar(-0.3), Scalar(0.2), Scalar(0.6), Scalar(-0.4),
      Scalar(0.1);
  P << Scalar(0.7), Scalar(-0.2), Scalar(0.1), Scalar(0.5), Scalar(-0.3),
      Scalar(0.8);
  Q.setIdentity();
  Q *= Scalar(2);
  R.setIdentity();
  R *= Scalar(3);
  N.setConstant(Scalar(0.07));
  W << Scalar(1.4), Scalar(0.2), Scalar(0.2), Scalar(1.8);
  Y.setConstant(Scalar(-0.11));
  V.setConstant(Scalar(0.13));
  model->set_A(A);
  model->set_B(B);
  model->set_P(P);
  model->set_Q(Q);
  model->set_R(R);
  model->set_N(N);
  model->set_W(W);
  model->set_Y(Y);
  model->set_V(V);
  MatrixXs G(ng, nx + nu + np), H(nh, nx + nu + np);
  G << Scalar(-0.4), Scalar(-0.25), Scalar(-0.1), Scalar(0.05), Scalar(0.2),
      Scalar(0.35), Scalar(0.5);
  H << Scalar(0.6), Scalar(0.45), Scalar(0.3), Scalar(0.15), Scalar(0),
      Scalar(-0.1), Scalar(-0.2);
  model->set_G(G);
  model->set_H(H);

  NumDiff legacy_numdiff(model, true);
  BOOST_CHECK_EQUAL(legacy_numdiff.get_np(), 0);
  const std::shared_ptr<typename Model::ActionDataAbstract> legacy_data =
      legacy_numdiff.createData();
  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(-0.4), Scalar(0.6));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.2), Scalar(0.7));
  BOOST_CHECK_NO_THROW(legacy_numdiff.calc(legacy_data, x, u));
  BOOST_CHECK_NO_THROW(legacy_numdiff.calcDiff(legacy_data, x, u));
  BOOST_CHECK_EQUAL(legacy_numdiff.template cast<float>().get_np(), 0);
  BOOST_CHECK_THROW(NumDiff(model, std::shared_ptr<ParameterManager>(), true),
                    crocoddyl::Exception);

  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(model->get_state());
  manager->addParam("lqr", std::make_shared<Params>(model->get_state(), np));
  manager->addParam("inactive", std::make_shared<Params>(model->get_state(), 1),
                    false);
  NumDiff numdiff(model, manager);
  const std::shared_ptr<typename Model::ActionDataAbstract> data =
      model->createData(manager->createData());
  const std::shared_ptr<typename Model::ActionDataAbstract> numerical =
      numdiff.createData();
  model->set_params(data, manager);
  const VectorXs p = VectorXs::LinSpaced(np, Scalar(-0.3), Scalar(0.5));
  model->update_p(data, p);
  numdiff.update_p(numerical, p);
  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  numdiff.calc(numerical, x, u);
  numdiff.calcDiff(numerical, x, u);
  const Scalar tol = tolerance<Scalar>();
  BOOST_CHECK(numerical->Fx.isApprox(data->Fx, tol));
  BOOST_CHECK(numerical->Fu.isApprox(data->Fu, tol));
  BOOST_CHECK_SMALL((numerical->Fp - data->Fp).norm(), tol);
  BOOST_CHECK(numerical->Lx.isApprox(data->Lx, tol));
  BOOST_CHECK(numerical->Lu.isApprox(data->Lu, tol));
  BOOST_CHECK(numerical->Lp.isApprox(data->Lp, tol));
  BOOST_CHECK(numerical->Lxx.isApprox(data->Lxx, tol));
  BOOST_CHECK(numerical->Lxu.isApprox(data->Lxu, tol));
  BOOST_CHECK(numerical->Luu.isApprox(data->Luu, tol));
  BOOST_CHECK(numerical->Lpp.isApprox(data->Lpp, tol));
  BOOST_CHECK(numerical->Lpx.isApprox(data->Lpx, tol));
  BOOST_CHECK(numerical->Lpu.isApprox(data->Lpu, tol));
  BOOST_CHECK(numerical->Gx.isApprox(data->Gx, tol));
  BOOST_CHECK(numerical->Gu.isApprox(data->Gu, tol));
  BOOST_CHECK(numerical->Gp.isApprox(data->Gp, tol));
  BOOST_CHECK(numerical->Hx.isApprox(data->Hx, tol));
  BOOST_CHECK(numerical->Hu.isApprox(data->Hu, tol));
  BOOST_CHECK(numerical->Hp.isApprox(data->Hp, tol));

  NumDiff gauss_numdiff(model, manager, true);
  const std::shared_ptr<typename Model::ActionDataAbstract> gauss_data =
      gauss_numdiff.createData(manager->createData());
  gauss_numdiff.update_p(gauss_data, p);
  gauss_numdiff.calc(gauss_data, x, u);
  gauss_numdiff.calcDiff(gauss_data, x, u);
  BOOST_CHECK(gauss_data->Lpp.isApprox(data->Lpp, tol));
  BOOST_CHECK(gauss_data->Lpx.isApprox(data->Lpx, tol));
  BOOST_CHECK(gauss_data->Lpu.isApprox(data->Lpu, tol));

  model->calc(data, x);
  model->calcDiff(data, x);
  numdiff.calc(numerical, x);
  numdiff.calcDiff(numerical, x);
  BOOST_CHECK_SMALL((numerical->Fp - data->Fp).norm(), tol);
  BOOST_CHECK(numerical->Lp.isApprox(data->Lp, tol));
  BOOST_CHECK(numerical->Lpp.isApprox(data->Lpp, tol));
  BOOST_CHECK(numerical->Lpx.isApprox(data->Lpx, tol));
  BOOST_CHECK_THROW(numdiff.set_disturbance(Scalar(-1)), std::exception);
  BOOST_CHECK_THROW(numdiff.update_p(numerical, VectorXs::Zero(np + 1)),
                    std::exception);
  const std::shared_ptr<ParameterManager> wrong_manager =
      std::make_shared<ParameterManager>(model->get_state());
  wrong_manager->addParam("wrong",
                          std::make_shared<Params>(model->get_state(), np + 1));
  BOOST_CHECK_THROW(numdiff.createData(wrong_manager->createData()),
                    std::exception);

  numdiff.calc(numerical, x, u);
  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      numdiff.calcDiff(numerical, x, u);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);

  const NumDiff copied(numdiff);
  BOOST_CHECK(copied.get_model() == model);
  const crocoddyl::ActionModelNumDiffTpl<float> casted =
      numdiff.template cast<float>();
  BOOST_CHECK_EQUAL(casted.get_np(), np);
}

template <typename Scalar>
void test_action_terminal_constraint_resize() {
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef NodeConstraintActionModelTpl<Scalar> Model;
  typedef crocoddyl::ActionModelNumDiffTpl<Scalar> NumDiff;
  typedef crocoddyl::LQRParamsTpl<Scalar> Params;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename Model::MatrixXs MatrixXs;
  typedef typename Model::VectorXs VectorXs;

  const std::shared_ptr<State> state = std::make_shared<State>(4);
  const std::shared_ptr<Model> model = std::make_shared<Model>(state);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("node", std::make_shared<Params>(state, 2));
  NumDiff numdiff(model, manager);
  const std::shared_ptr<typename Model::ActionDataAbstract> data =
      numdiff.createData(manager->createData());
  const VectorXs x =
      (VectorXs(4) << Scalar(0.2), Scalar(-0.1), Scalar(0.4), Scalar(0.3))
          .finished();
  const VectorXs u = (VectorXs(2) << Scalar(-0.2), Scalar(0.5)).finished();
  const VectorXs p = (VectorXs(2) << Scalar(0.3), Scalar(-0.4)).finished();
  numdiff.update_p(data, p);
  numdiff.calc(data, x, u);
  numdiff.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(data->Gp.rows(), 1);
  BOOST_CHECK_EQUAL(data->Hp.rows(), 2);
  MatrixXs expected_Gp(1, 2), expected_Hp(2, 2);
  expected_Gp << Scalar(1), Scalar(2);
  expected_Hp << Scalar(3), Scalar(-1), Scalar(-2), Scalar(4);
  BOOST_CHECK(data->Gp.isApprox(expected_Gp, tolerance<Scalar>()));
  BOOST_CHECK(data->Hp.isApprox(expected_Hp, tolerance<Scalar>()));

  data->Fp.setConstant(Scalar(6));
  data->Fu.setConstant(Scalar(7));
  data->Lu.setConstant(Scalar(8));
  data->Lxu.setConstant(Scalar(9));
  data->Luu.setConstant(Scalar(10));
  data->Lpu.setConstant(Scalar(11));
  data->Gu.setConstant(Scalar(12));
  data->Hu.setConstant(Scalar(13));
  const MatrixXs Fp = data->Fp;
  const MatrixXs Fu = data->Fu;
  const VectorXs Lu = data->Lu;
  const MatrixXs Lxu = data->Lxu;
  const MatrixXs Luu = data->Luu;
  const MatrixXs Lpu = data->Lpu;
  const MatrixXs Gu = data->Gu;
  const MatrixXs Hu = data->Hu;

  numdiff.calc(data, x);
  numdiff.calcDiff(data, x);
  BOOST_CHECK_EQUAL(data->Gp.rows(), 3);
  BOOST_CHECK_EQUAL(data->Hp.rows(), 4);
  MatrixXs expected_Gp_T(3, 2), expected_Hp_T(4, 2);
  expected_Gp_T << Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(1),
      Scalar(1);
  expected_Hp_T << Scalar(2), Scalar(0), Scalar(0), Scalar(3), Scalar(1),
      Scalar(-1), Scalar(-4), Scalar(5);
  BOOST_CHECK(data->Gp.isApprox(expected_Gp_T, tolerance<Scalar>()));
  BOOST_CHECK(data->Hp.isApprox(expected_Hp_T, tolerance<Scalar>()));
  BOOST_CHECK(data->Fp.isApprox(Fp));
  BOOST_CHECK(data->Fu.isApprox(Fu));
  BOOST_CHECK(data->Lu.isApprox(Lu));
  BOOST_CHECK(data->Lxu.isApprox(Lxu));
  BOOST_CHECK(data->Luu.isApprox(Luu));
  BOOST_CHECK(data->Lpu.isApprox(Lpu));
  BOOST_CHECK(data->Gu.isApprox(Gu));
  BOOST_CHECK(data->Hu.isApprox(Hu));

  numdiff.calc(data, x, u);
  numdiff.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(data->Gp.rows(), 1);
  BOOST_CHECK_EQUAL(data->Hp.rows(), 2);

  numdiff.calc(data, x);
  numdiff.calcDiff(data, x);
  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      numdiff.calc(data, x);
      numdiff.calcDiff(data, x);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
}

template <typename Scalar>
void test_discrete_manifold_dynamics() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef ManifoldDiscreteDynamicsTpl<Scalar> Model;
  typedef crocoddyl::DynamicsModelNumDiffTpl<Scalar> NumDiff;
  typedef typename Model::VectorXs VectorXs;

  const std::shared_ptr<pinocchio::Model> model_double =
      std::make_shared<pinocchio::Model>();
  pinocchio::buildModels::humanoidRandom(*model_double, true);
  const crocoddyl::StateMultibody state_double(model_double);
  const std::shared_ptr<State> state =
      std::make_shared<State>(state_double.template cast<Scalar>());
  BOOST_REQUIRE_NE(state->get_nx(), state->get_ndx());

  const std::shared_ptr<Model> model = std::make_shared<Model>(state);
  NumDiff numdiff(model);
  numdiff.set_disturbance(std::is_same<Scalar, float>::value ? Scalar(2e-3)
                                                             : Scalar(1e-7));
  const std::shared_ptr<typename Model::DynamicsDataAbstract> data =
      numdiff.createData();
  VectorXs dx =
      VectorXs::LinSpaced(state->get_ndx(), Scalar(-0.1), Scalar(0.15));
  VectorXs x(state->get_nx());
  state->integrate(state->zero(), dx, x);
  const VectorXs u = VectorXs::Zero(model->get_nu());
  numdiff.calc(data, x, u);
  numdiff.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(data->Fu.rows(), state->get_ndx());
  BOOST_CHECK(data->Fu.isApprox(model->B, tolerance<Scalar>()));
}

template <typename Scalar>
void test_dynamics_numdiff_and_restoration() {
  typedef LinearDynamicsModelTpl<Scalar> Model;
  typedef crocoddyl::DynamicsModelNumDiffTpl<Scalar> NumDiff;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::LQRParamsTpl<Scalar> Params;
  typedef typename Model::VectorXs VectorXs;

  const std::shared_ptr<crocoddyl::StateVectorTpl<Scalar> > state =
      std::make_shared<crocoddyl::StateVectorTpl<Scalar> >(4);
  const std::shared_ptr<Model> model = std::make_shared<Model>(state, 2);
  BOOST_CHECK_THROW(NumDiff(model, std::shared_ptr<ParameterManager>()),
                    crocoddyl::Exception);
  NumDiff legacy_numdiff(model);
  BOOST_CHECK_EQUAL(legacy_numdiff.template cast<float>().get_np(), 2);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("parameters", std::make_shared<Params>(state, 2));
  NumDiff numdiff(model, manager);
  const std::shared_ptr<typename Model::DynamicsDataAbstract> data =
      numdiff.createData();
  const VectorXs x =
      (VectorXs(4) << Scalar(0.2), Scalar(-0.4), Scalar(0.3), Scalar(-0.1))
          .finished();
  const VectorXs u = (VectorXs(2) << Scalar(0.6), Scalar(0.1)).finished();
  const VectorXs p = (VectorXs(2) << Scalar(-0.3), Scalar(0.7)).finished();
  model->nominal = p;
  numdiff.update_p(data, p);
  numdiff.calc(data, x, u);
  numdiff.calcDiff(data, x, u);
  const Scalar tol = tolerance<Scalar>();
  BOOST_CHECK(data->Fx.isApprox(model->A, tol));
  BOOST_CHECK(data->Fu.isApprox(model->B, tol));
  BOOST_CHECK(data->Fp.isApprox(model->P, tol));
  BOOST_CHECK_CLOSE(data->dP_dv(0, 0), Scalar(2) * x(2), Scalar(100) * tol);
  BOOST_CHECK_CLOSE(data->dP_dv(0, 1), Scalar(2) * x(3), Scalar(100) * tol);
  BOOST_CHECK(data->dP_dp.isApprox(Scalar(2) * p.transpose(), tol));
  BOOST_CHECK_CLOSE(data->Gx(0, 0), Scalar(1), Scalar(100) * tol);
  BOOST_CHECK_CLOSE(data->Gu(0, 0), Scalar(2), Scalar(100) * tol);
  BOOST_CHECK_CLOSE(data->Gp(0, 0), Scalar(3), Scalar(100) * tol);
  BOOST_CHECK_CLOSE(data->Hx(0, 3), Scalar(1), Scalar(100) * tol);
  BOOST_CHECK_CLOSE(data->Hu(0, 1), Scalar(-1), Scalar(100) * tol);
  BOOST_CHECK_CLOSE(data->Hp(0, 1), Scalar(4), Scalar(100) * tol);

  numdiff.calc(data, x);
  numdiff.calcDiff(data, x);
  BOOST_CHECK(data->Fx.isApprox(model->A, tol));
  BOOST_CHECK(data->Fu.isZero());
  model->throw_on_perturbed = true;
  numdiff.calc(data, x, u);
  BOOST_CHECK_THROW(numdiff.calcDiff(data, x, u), std::exception);
  const typename NumDiff::Data* nd =
      static_cast<const typename NumDiff::Data*>(data.get());
  for (std::size_t i = 0; i < nd->data_p.size(); ++i) {
    BOOST_CHECK(static_cast<LinearDynamicsDataTpl<Scalar>*>(nd->data_p[i].get())
                    ->p.isApprox(p));
  }
  model->throw_on_perturbed = false;
  numdiff.calc(data, x, u);
  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      numdiff.calcDiff(data, x, u);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  BOOST_CHECK_EQUAL(numdiff.template cast<float>().get_np(), 2);
  BOOST_CHECK_THROW(NumDiff(std::shared_ptr<typename Model::Base>()),
                    std::exception);
}

template <typename Scalar>
void test_actuation_and_residual_numdiff() {
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::shared_ptr<crocoddyl::StateVectorTpl<Scalar> > state =
      std::make_shared<crocoddyl::StateVectorTpl<Scalar> >(4);
  const std::shared_ptr<VectorXs> shared_p =
      std::make_shared<VectorXs>(VectorXs::Zero(2));
  const std::shared_ptr<SharedDynamicsParamsTpl<Scalar> > params =
      std::make_shared<SharedDynamicsParamsTpl<Scalar> >(state, shared_p);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("shared", params);
  const VectorXs x =
      (VectorXs(4) << Scalar(0.3), Scalar(-0.5), Scalar(0.2), Scalar(0.6))
          .finished();
  const VectorXs u = (VectorXs(2) << Scalar(0.2), Scalar(0.8)).finished();
  const VectorXs p = (VectorXs(2) << Scalar(-0.4), Scalar(0.6)).finished();

  const std::shared_ptr<LinearActuationModelTpl<Scalar> > actuation =
      std::make_shared<LinearActuationModelTpl<Scalar> >(state, shared_p);
  crocoddyl::ActuationModelNumDiffTpl<Scalar> act_numdiff(actuation, manager);
  const std::shared_ptr<
      typename LinearActuationModelTpl<Scalar>::ActuationDataAbstract>
      adata = act_numdiff.createData();
  manager->update(
      static_cast<typename crocoddyl::ActuationModelNumDiffTpl<Scalar>::Data*>(
          adata.get())
          ->parameter_data,
      p);
  actuation->nominal = p;
  act_numdiff.calc(adata, x, u);
  act_numdiff.calcDiff(adata, x, u);
  const typename crocoddyl::ActuationModelNumDiffTpl<Scalar>::Data* andata =
      static_cast<
          const typename crocoddyl::ActuationModelNumDiffTpl<Scalar>::Data*>(
          adata.get());
  const Scalar tol = tolerance<Scalar>();
  BOOST_CHECK(adata->dtau_dx.isApprox(actuation->A, tol));
  BOOST_CHECK(adata->dtau_du.isApprox(actuation->B, tol));
  BOOST_CHECK(andata->dtau_dp.isApprox(actuation->P, tol));
  const MatrixXs control_sentinel = MatrixXs::Constant(
      adata->dtau_du.rows(), adata->dtau_du.cols(), Scalar(7));
  adata->dtau_du = control_sentinel;
  act_numdiff.calc(adata, x);
  act_numdiff.calcDiff(adata, x);
  BOOST_CHECK(adata->dtau_du.isApprox(control_sentinel));
  actuation->throw_on_perturbed = true;
  act_numdiff.calc(adata, x, u);
  BOOST_CHECK_THROW(act_numdiff.calcDiff(adata, x, u), std::exception);
  BOOST_CHECK(shared_p->isApprox(p));
  actuation->throw_on_perturbed = false;

  act_numdiff.calc(adata, x, u);
  bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      act_numdiff.calcDiff(adata, x, u);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);

  const std::shared_ptr<LinearResidualModelTpl<Scalar> > residual =
      std::make_shared<LinearResidualModelTpl<Scalar> >(state, 2);
  crocoddyl::ResidualModelNumDiffTpl<Scalar> res_numdiff(residual, manager);
  const std::shared_ptr<typename ParameterManager::ParameterDataManager>
      manager_data = manager->createData();
  crocoddyl::DataCollectorParamsTpl<Scalar> collector(manager_data->params,
                                                      manager_data.get());
  const std::shared_ptr<
      typename LinearResidualModelTpl<Scalar>::ResidualDataAbstract>
      rdata = res_numdiff.createData(&collector, manager_data);
  typedef typename crocoddyl::ResidualModelNumDiffTpl<Scalar>::Data
      ResidualNumDiffData;
  const std::shared_ptr<
      typename LinearResidualModelTpl<Scalar>::ResidualDataAbstract>
      inferred_rdata = res_numdiff.createData(&collector);
  const std::shared_ptr<
      typename LinearResidualModelTpl<Scalar>::ResidualDataAbstract>
      second_inferred_rdata = res_numdiff.createData(&collector);
  BOOST_CHECK_EQUAL(static_cast<ResidualNumDiffData*>(inferred_rdata.get())
                        ->parameter_data.get(),
                    manager_data.get());
  BOOST_CHECK_EQUAL(
      static_cast<ResidualNumDiffData*>(second_inferred_rdata.get())
          ->parameter_data.get(),
      manager_data.get());
  BOOST_CHECK_EQUAL(collector.params, manager_data->params);
  BOOST_CHECK_EQUAL(collector.parameter_data, manager_data.get());
  BOOST_CHECK_EQUAL(manager_data->parameter_data, manager_data.get());
  manager->update(manager_data, p);
  BOOST_CHECK(manager_data->params->p.isApprox(p));
  res_numdiff.update_p(inferred_rdata, -p);
  BOOST_CHECK(manager_data->params->p.isApprox(-p));
  res_numdiff.update_p(second_inferred_rdata, p);
  BOOST_CHECK(manager_data->params->p.isApprox(p));
  BOOST_CHECK_EQUAL(collector.params, manager_data->params);
  BOOST_CHECK_EQUAL(collector.parameter_data, manager_data.get());
  BOOST_CHECK_EQUAL(manager_data->parameter_data, manager_data.get());
  const std::shared_ptr<typename ParameterManager::ParameterDataManager>
      other_manager_data = manager->createData();
  BOOST_CHECK_THROW(res_numdiff.createData(&collector, other_manager_data),
                    std::exception);
  crocoddyl::DataCollectorParamsTpl<Scalar> null_payload(nullptr, nullptr);
  BOOST_CHECK_THROW(res_numdiff.createData(&null_payload), std::exception);
  crocoddyl::DataCollectorParamsTpl<Scalar> null_parameter_data(
      manager_data->params, nullptr);
  BOOST_CHECK_THROW(res_numdiff.createData(&null_parameter_data),
                    std::exception);
  res_numdiff.update_p(rdata, p);
  residual->nominal = p;
  VectorXs reevaluated_x = VectorXs::Zero(4);
  VectorXs reevaluated_u = VectorXs::Zero(2);
  std::vector<
      typename crocoddyl::ResidualModelNumDiffTpl<Scalar>::ReevaluationFunction>
      reevals;
  reevals.push_back([&](const VectorXs& xr, const VectorXs& ur) {
    reevaluated_x = xr;
    reevaluated_u = ur;
  });
  res_numdiff.set_reevals(reevals);
  res_numdiff.calc(rdata, x, u);
  res_numdiff.calcDiff(rdata, x, u);
  BOOST_CHECK(rdata->Rx.isApprox(residual->X, tol));
  BOOST_CHECK(rdata->Ru.isApprox(residual->U, tol));
  BOOST_CHECK(rdata->Rp.isApprox(residual->P, tol));
  BOOST_CHECK(reevaluated_x.isApprox(x));
  BOOST_CHECK(reevaluated_u.isApprox(u));
  const MatrixXs residual_sentinel =
      MatrixXs::Constant(rdata->Ru.rows(), rdata->Ru.cols(), Scalar(-5));
  rdata->Ru = residual_sentinel;
  res_numdiff.calc(rdata, x);
  res_numdiff.calcDiff(rdata, x);
  BOOST_CHECK(rdata->Rx.isApprox(residual->X, tol));
  BOOST_CHECK(rdata->Rp.isApprox(residual->P, tol));
  BOOST_CHECK(rdata->Ru.isApprox(residual_sentinel));
  residual->throw_on_perturbed = true;
  res_numdiff.calc(rdata, x, u);
  BOOST_CHECK_THROW(res_numdiff.calcDiff(rdata, x, u), std::exception);
  BOOST_CHECK(manager_data->params->p.isApprox(p));
  BOOST_CHECK(reevaluated_x.isApprox(x));
  BOOST_CHECK(reevaluated_u.isApprox(u));
  residual->throw_on_perturbed = false;

  res_numdiff.calc(rdata, x, u);
  malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      res_numdiff.calcDiff(rdata, x, u);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);

  const std::shared_ptr<typename ParameterManager::ParameterDataManager>
      callback_manager_data = manager->createData();
  CallbackParamsCollectorTpl<Scalar> callback_collector(callback_manager_data);
  const std::shared_ptr<CallbackResidualModelTpl<Scalar> > callback_residual =
      std::make_shared<CallbackResidualModelTpl<Scalar> >(state, 2);
  crocoddyl::ResidualModelNumDiffTpl<Scalar> callback_numdiff(callback_residual,
                                                              manager);
  const std::shared_ptr<
      typename CallbackResidualModelTpl<Scalar>::ResidualDataAbstract>
      callback_data = callback_numdiff.createData(&callback_collector,
                                                  callback_manager_data);
  VectorXs callback_x = VectorXs::Zero(4);
  VectorXs callback_u = VectorXs::Zero(2);
  std::vector<
      typename crocoddyl::ResidualModelNumDiffTpl<Scalar>::ReevaluationFunction>
      callback_reevals;
  callback_reevals.push_back([&](const VectorXs& xr, const VectorXs& ur) {
    callback_collector.observed = callback_manager_data->params->p;
    callback_x = xr;
    callback_u = ur;
  });
  callback_numdiff.set_reevals(callback_reevals);
  callback_numdiff.update_p(callback_data, p);
  callback_residual->nominal = p;
  callback_reevals[0](x, u);
  callback_numdiff.calc(callback_data, x, u);
  callback_numdiff.calcDiff(callback_data, x, u);
  BOOST_CHECK(callback_data->Rx.isZero(tol));
  BOOST_CHECK(callback_data->Ru.isZero(tol));
  BOOST_CHECK(callback_data->Rp.isApprox(callback_residual->P, tol));
  BOOST_CHECK(callback_manager_data->params->p.isApprox(p));
  BOOST_CHECK(callback_collector.observed.isApprox(p));
  BOOST_CHECK(callback_x.isApprox(x));
  BOOST_CHECK(callback_u.isApprox(u));

  callback_reevals[0](x, VectorXs::Zero(2));
  callback_numdiff.calc(callback_data, x);
  callback_numdiff.calcDiff(callback_data, x);
  BOOST_CHECK(callback_data->Rp.isApprox(callback_residual->P, tol));
  BOOST_CHECK(callback_manager_data->params->p.isApprox(p));
  BOOST_CHECK(callback_collector.observed.isApprox(p));
  BOOST_CHECK(callback_x.isApprox(x));
  BOOST_CHECK(callback_u.isZero());

  callback_residual->throw_on_perturbed = true;
  callback_reevals[0](x, u);
  callback_numdiff.calc(callback_data, x, u);
  BOOST_CHECK_THROW(callback_numdiff.calcDiff(callback_data, x, u),
                    std::exception);
  BOOST_CHECK(callback_manager_data->params->p.isApprox(p));
  BOOST_CHECK(callback_collector.observed.isApprox(p));
  BOOST_CHECK(callback_x.isApprox(x));
  BOOST_CHECK(callback_u.isApprox(u));
  callback_residual->throw_on_perturbed = false;

  callback_reevals[0](x, u);
  callback_numdiff.calc(callback_data, x, u);
  malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      callback_numdiff.calcDiff(callback_data, x, u);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  BOOST_CHECK_THROW(
      crocoddyl::ResidualModelNumDiffTpl<Scalar>(
          std::shared_ptr<typename LinearResidualModelTpl<Scalar>::Base>()),
      std::exception);
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_numdiff");
  ts->add(BOOST_TEST_CASE(&test_action_numdiff_parameter_blocks<double>));
  ts->add(BOOST_TEST_CASE(&test_action_numdiff_parameter_blocks<float>));
  ts->add(BOOST_TEST_CASE(&test_action_terminal_constraint_resize<double>));
  ts->add(BOOST_TEST_CASE(&test_action_terminal_constraint_resize<float>));
  ts->add(BOOST_TEST_CASE(&test_discrete_manifold_dynamics<double>));
  ts->add(BOOST_TEST_CASE(&test_discrete_manifold_dynamics<float>));
  ts->add(BOOST_TEST_CASE(&test_dynamics_numdiff_and_restoration<double>));
  ts->add(BOOST_TEST_CASE(&test_dynamics_numdiff_and_restoration<float>));
  ts->add(BOOST_TEST_CASE(&test_actuation_and_residual_numdiff<double>));
  ts->add(BOOST_TEST_CASE(&test_actuation_and_residual_numdiff<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
