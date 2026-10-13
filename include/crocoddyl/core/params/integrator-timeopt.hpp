///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_PARAMS_INTEGRATOR_TIMEOPT_HPP_
#define CROCODDYL_CORE_PARAMS_INTEGRATOR_TIMEOPT_HPP_

#include "crocoddyl/core/integrator/time.hpp"
#include "crocoddyl/core/params-base.hpp"

namespace crocoddyl {

/**
 * @brief Action parameter that optimizes a shared integration time
 *
 * The single parameter is \f$p=\log(dt)\f$, so `update()` enforces
 * \f$dt=\exp(p)>0\f$. The model and each integrated action retain shared
 * ownership of the same `IntegratorTimeTpl`; updates are therefore visible to
 * every action using that handle.
 */
template <typename _Scalar>
class IntegratorTimeoptParamsTpl
    : public ActionModelParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelParamsAbstractTpl<Scalar> Base;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef IntegratorTimeoptParamsDataTpl<Scalar> IntegratorTimeoptParamsData;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  CROCODDYL_DERIVED_CAST(ParamsModelBase, IntegratorTimeoptParamsTpl)

  explicit IntegratorTimeoptParamsTpl(
      std::shared_ptr<StateAbstract> state,
      std::shared_ptr<IntegratorTime> integrator_time);
  virtual ~IntegratorTimeoptParamsTpl() = default;

  virtual void update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) override;

  virtual void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      Eigen::Ref<MatrixXs> dx_dp, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  virtual std::shared_ptr<ParamsDataAbstract> createData() override;
  virtual bool checkData(
      const std::shared_ptr<ParamsDataAbstract>& data) const override;
  virtual VectorXs rand() const override;

  const std::shared_ptr<IntegratorTime>& get_integrator_time() const;

  template <typename NewScalar>
  IntegratorTimeoptParamsTpl<NewScalar> cast() const;

  virtual void print(std::ostream& os) const override;

 protected:
  std::shared_ptr<IntegratorTimeoptParamsData> castData(
      const std::shared_ptr<ParamsDataAbstract>& data) const;

  std::shared_ptr<IntegratorTime> integrator_time_;
};

/**
 * @brief Data for the logarithmic integration-time parameter
 *
 * The inherited parameter data has one active action parameter and no
 * dynamics partition. `dt` and `dt_dp` are scalar cached values; resizing and
 * zeroing preserve the inherited active state.
 */
template <typename _Scalar>
struct IntegratorTimeoptParamsDataTpl
    : public ActionModelParamsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelParamsDataAbstractTpl<Scalar> Base;

 private:
  template <template <typename Scalar> class Model>
  static Model<Scalar>* checkModel(Model<Scalar>* const model) {
    if (model == nullptr) {
      throw_pretty(
          "Invalid argument: integration-time parameter model is null");
    }
    return model;
  }

 public:
  template <template <typename Scalar> class Model>
  explicit IntegratorTimeoptParamsDataTpl(Model<Scalar>* const model)
      : Base(checkModel(model)->get_np()), dt(Scalar(0.)), dt_dp(Scalar(0.)) {}
  virtual ~IntegratorTimeoptParamsDataTpl() = default;

  virtual void resize(const std::size_t np_action,
                      const std::size_t np_dynamics) override {
    Base::resize(np_action, np_dynamics);
    dt = Scalar(0.);
    dt_dp = Scalar(0.);
  }

  virtual void setZero() override {
    Base::setZero();
    dt = Scalar(0.);
    dt_dp = Scalar(0.);
  }

  Scalar dt;
  Scalar dt_dp;
};

}  // namespace crocoddyl

#include "crocoddyl/core/params/integrator-timeopt.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::IntegratorTimeoptParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratorTimeoptParamsDataTpl)

#endif  // CROCODDYL_CORE_PARAMS_INTEGRATOR_TIMEOPT_HPP_
