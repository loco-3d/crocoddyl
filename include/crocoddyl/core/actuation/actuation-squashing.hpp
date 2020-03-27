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
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef ActuationSquashingDataTpl<Scalar> ActuationSquashingData;
  typedef SquashingModelAbstractTpl<Scalar> SquashingModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActuationSquashingModelTpl(boost::shared_ptr<ActuationModelAbstract> actuation, boost::shared_ptr<SquashingModelAbstract> squashing, const std::size_t& nu) : Base(actuation->get_state(), nu), squashing_(squashing), actuation_(actuation) {};
  
  ~ActuationSquashingModelTpl(){};

  virtual void calc(const boost::shared_ptr<ActuationSquashingData>& data, const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u)
  {
    squashing_->calc(data->squashing, u);
    actuation_->calc(data->actuation, data->squashing.u);
    data->tau = data->actuation.tau;
  };
  
  virtual void calcDiff(const boost::shared_ptr<ActuationSquashingData>& data, const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u)
  {
    squashing_->calcDiff(data->squashing, u);
    squashing_->calcDiff(data->actuation, data->squashing.u);
    data->dtau_du = data->actuation.dtau_du * data->squashing.du_ds;
  };

  boost::shared_ptr<ActuationSquashingData> createData() {
    return boost::make_shared<ActuationSquashingData>(this);
  };

  const boost::shared_ptr<SquashingModelAbstract>& get_squashing() const { return squashing_; };
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const { return actuation_; };

 protected:
  boost::shared_ptr<SquashingModelAbstract> squashing_;
  boost::shared_ptr<ActuationModelAbstract> actuation_;
};

template <typename _Scalar>
struct ActuationSquashingDataTpl : public ActuationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActuationSquashingDataTpl(Model<Scalar>* const model)
      : ActuationDataAbstract(model), squashing(model->get_squashing()->createData()), actuation(model->get_actuation()->createData()) {}
  
  ~ActuationSquashingDataTpl() {}
  
  ActuationDataAbstract actuation;
  SquashingDataAbstract squashing;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_SQUASH_BASE_HPP_