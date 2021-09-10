///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_SQUASHING_HPP_
#define CROCODDYL_CORE_ACTUATION_SQUASHING_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/actuation/squashing-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActuationSquashingModelTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationSquashingDataTpl<Scalar> Data;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef SquashingModelAbstractTpl<Scalar> SquashingModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActuationSquashingModelTpl(boost::shared_ptr<ActuationModelAbstract> actuation,
                             boost::shared_ptr<SquashingModelAbstract> squashing, const std::size_t nu)
      : Base(actuation->get_state(), nu), squashing_(squashing), actuation_(actuation){};

  virtual ~ActuationSquashingModelTpl(){};

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) {
    Data* d = static_cast<Data*>(data.get());

    squashing_->calc(d->squashing, u);
    actuation_->calc(d->actuation, x, d->squashing->u);
    data->tau = d->actuation->tau;
  };

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>&) {
    data->tau.setZero();
  };

  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) {
    Data* d = static_cast<Data*>(data.get());

    squashing_->calcDiff(d->squashing, u);
    actuation_->calcDiff(d->actuation, x, d->squashing->u);
    data->dtau_du.noalias() = d->actuation->dtau_du * d->squashing->du_ds;
  };

#ifndef NDEBUG
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>&) {
#else
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>&, const Eigen::Ref<const VectorXs>&) {
#endif
    assert_pretty(MatrixXs(data->dtau_du).isZero(), "dtau_du has wrong value");
  };

  boost::shared_ptr<ActuationDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  const boost::shared_ptr<SquashingModelAbstract>& get_squashing() const { return squashing_; };
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const { return actuation_; };

 protected:
  boost::shared_ptr<SquashingModelAbstract> squashing_;
  boost::shared_ptr<ActuationModelAbstract> actuation_;
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

  ~ActuationSquashingDataTpl() {}

  boost::shared_ptr<SquashingDataAbstract> squashing;
  boost::shared_ptr<ActuationDataAbstract> actuation;

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::tau;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_SQUASH_BASE_HPP_
