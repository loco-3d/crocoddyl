///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OBSERVER_EULER_HPP_
#define CROCODDYL_CORE_OBSERVER_EULER_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-observer-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"

namespace crocoddyl {

/**
 * @brief Euler integrated observer model
 *
 * This model discretizes a continuous-time constrained dynamics model for
 * optimal estimation. The observer control vector is split as
 * \f$w = [\eta, u]\f$, where \f$\eta\in\mathbb{R}^{ndx}\f$ is process noise in
 * tangent space and \f$u\f$ is the dynamics control.
 */
template <typename _Scalar>
class IntegratedObserverModelEulerTpl
    : public IntegratedObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, IntegratedObserverModelEulerTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedObserverModelAbstractTpl<Scalar> Base;
  typedef IntegratedObserverDataEulerTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedObserverModelEulerTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr,
      const Scalar time_step = Scalar(1e-3));
  virtual ~IntegratedObserverModelEulerTpl() = default;

  using Base::createData;

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  virtual void print(std::ostream& os) const override;

  template <typename NewScalar>
  IntegratedObserverModelEulerTpl<NewScalar> cast() const;

 protected:
  using Base::constraints_;
  using Base::costs_;
  using Base::dynamics_;
  using Base::np_;
  using Base::nr_;
  using Base::params_;
  using Base::state_;
  using Base::time_step2_;
  using Base::time_step_;
  using Base::u_zero_;

  std::shared_ptr<Data> cast_data(
      const std::shared_ptr<ActionDataAbstract>& data) const;
};

template <typename _Scalar>
struct IntegratedObserverDataEulerTpl
    : public IntegratedObserverDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedObserverDataAbstractTpl<Scalar> Base;
  typedef DataCollectorObserverTpl<Scalar> DataCollectorObserver;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedObserverDataEulerTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(model, params_data),
        Jfirst(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Jsecond(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        ddx_du(model->get_state()->get_ndx(), model->get_dynamics()->get_nu()) {
    DataCollectorObserver* const shared =
        dynamic_cast<DataCollectorObserver*>(Base::dynamics->shared);
    if (shared == nullptr) {
      throw_pretty(
          "Invalid argument: dynamics shared collector does not expose "
          "observer data");
    }
    shared->shareObserverData(this);
    Jfirst.setZero();
    Jsecond.setZero();
    ddx_du.setZero();
  }
  virtual ~IntegratedObserverDataEulerTpl() = default;

  virtual void setZero() override {
    Base::setZero();
    Jfirst.setZero();
    Jsecond.setZero();
    ddx_du.setZero();
  }

  MatrixXs Jfirst;
  MatrixXs Jsecond;
  MatrixXs ddx_du;
};

}  // namespace crocoddyl

#include "crocoddyl/core/observer/euler.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::IntegratedObserverModelEulerTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratedObserverDataEulerTpl)

#endif  // CROCODDYL_CORE_OBSERVER_EULER_HPP_
