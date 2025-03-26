///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, IRI: CSIC-UPC,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_SQUASHING_HPP_
#define CROCODDYL_CORE_ACTUATION_SQUASHING_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActuationSquashingModelTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase, ActuationSquashingModelTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationSquashingDataTpl<Scalar> Data;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef SquashingModelAbstractTpl<Scalar> SquashingModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActuationSquashingModelTpl(std::shared_ptr<Base> actuation,
                             std::shared_ptr<SquashingModelAbstract> squashing,
                             const std::size_t nu)
      : Base(actuation->get_state(), nu),
        squashing_(squashing),
        actuation_(actuation) {};

  virtual ~ActuationSquashingModelTpl() = default;

  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());

    squashing_->calc(d->squashing, u);
    actuation_->calc(d->actuation, x, d->squashing->u);
    data->tau = d->actuation->tau;
    data->tau_set = d->actuation->tau_set;
  };

  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());

    squashing_->calcDiff(d->squashing, u);
    actuation_->calcDiff(d->actuation, x, d->squashing->u);
    data->dtau_du.noalias() = d->actuation->dtau_du * d->squashing->du_ds;
  };

  virtual void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& tau) override {
    if (static_cast<std::size_t>(tau.size()) != this->state_->get_nv()) {
      throw_pretty("Invalid argument: "
                   << "tau has wrong dimension (it should be " +
                          std::to_string(this->state_->get_nv()) + ")");
    }
    torqueTransform(data, x, tau);
    data->u.noalias() = data->Mtau * tau;
  }

  std::shared_ptr<ActuationDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  template <typename NewScalar>
  ActuationSquashingModelTpl<NewScalar> cast() const {
    typedef ActuationSquashingModelTpl<NewScalar> ReturnType;
    ReturnType ret(actuation_->template cast<NewScalar>(),
                   squashing_->template cast<NewScalar>(), nu_);
    return ret;
  }

  const std::shared_ptr<SquashingModelAbstract>& get_squashing() const {
    return squashing_;
  };
  const std::shared_ptr<Base>& get_actuation() const { return actuation_; };

 protected:
  std::shared_ptr<SquashingModelAbstract> squashing_;
  std::shared_ptr<Base> actuation_;
  using Base::nu_;
  using Base::torqueTransform;
};

template <typename _Scalar>
struct ActuationSquashingDataTpl : public ActuationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActuationDataAbstractTpl<Scalar> Base;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActuationSquashingDataTpl(Model<Scalar>* const model)
      : Base(model),
        squashing(model->get_squashing()->createData()),
        actuation(model->get_actuation()->createData()) {}

  virtual ~ActuationSquashingDataTpl() = default;

  std::shared_ptr<SquashingDataAbstract> squashing;
  std::shared_ptr<Base> actuation;

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::Mtau;
  using Base::tau;
  using Base::tau_set;
  using Base::u;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActuationSquashingModelTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActuationSquashingDataTpl)

#endif  // CROCODDYL_CORE_ACTIVATION_SQUASH_BASE_HPP_
