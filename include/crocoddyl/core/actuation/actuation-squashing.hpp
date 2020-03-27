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
#include "crocoddyl/core/data/squashing.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
template <typename _Scalar>
class ActuationModelSquashingTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef ActuationDataSquashingTpl<Scalar> ActuationDataSquashing;
  typedef SquashingModelAbstractTpl<Scalar> SquashingModelAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  
  explicit ActuationModelSquashingTpl(boost::shared_ptr<StateAbstract> state, boost::shared_ptr<ActuationModelAbstract> actuation, boost::shared_ptr<SquashingModelAbstract> squashing, const std::size_t& nu) : Base(state, nu), squashing_(squashing), actuation_(actuation) {};
  
  ~ActuationModelSquashingTpl(){};

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u)
  {

  };
  
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u)
  {

  };

  boost::shared_ptr<ActuationDataSquashing> createData() {
    boost::shared_ptr<ActuationDataSquashing> data = boost::make_shared<ActuationDataSquashing>(this);
  };

  const boost::shared_ptr<SquashingModelAbstract>& get_squashing() const;
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

 protected:
  boost::shared_ptr<SquashingModelAbstract> squashing_;
  boost::shared_ptr<ActuationModelAbstract> actuation_;
};

template <typename _Scalar>
struct ActuationDataSquashingTpl : public ActuationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename <typename Scalar> class Model>
  explicit ActuationDataSquashingTpl(Model<Scalar>* const model)
      : ActuationDataAbstract(model), squashing(model->get_squashing()->createData()), actuation(model->get_actuation()->createData()) {}
  
  ~ActuationDataSquashing() {}
  
  ActuationDataAbstract actuation;
  SquashingDataAbstract squashing;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_SQUASH_BASE_HPP_