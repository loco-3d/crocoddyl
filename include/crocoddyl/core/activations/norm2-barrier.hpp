///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, Airbus
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_COLLISIONS_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_COLLISIONS_HPP_

#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelNorm2BarrierTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataCollisionTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;

  // TODO: Magic number, check with Teguh/Nicolas/Crocoddyl Team 
  explicit ActivationModelNorm2BarrierTpl(const std::size_t nr, const Scalar& threshold ) : 
    Base(nr), threshold_(threshold) {};
  virtual ~ActivationModelNorm2BarrierTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    d->distance_ = r.norm();
    if(d->distance_ < threshold_) {
      data->a_value = Scalar(0.5) * (d->distance_ - threshold_) * (d->distance_ - threshold_);
    }
    else {
      data->a_value = Scalar(0.0);
    }
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
    
    if(d->distance_ < threshold_) {
      data->Ar = (d->distance_ - threshold_) / d->distance_ * r;
      data->Arr.diagonal() = threshold_ * r.array().square() / std::pow(d->distance_, 3);
      data->Arr.diagonal().array() += (d->distance_ - threshold_) / d->distance_;
      //data->Arr.diagonal() = (MatrixXs::Identity(nr_, nr_) * (d->distance_ - threshold_) / d->distance_ + threshold_ * r * r.transpose() / std::pow(d->distance_, 3)).diagonal();
    }
    else {
      data->Ar.setZero();
      data->Arr.setZero();
    }
  };

  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };
  
  const Scalar& get_threshold() const { return threshold_; };
  void set_threshold(const Scalar& threshold) { threshold_ = threshold; };

 protected:
  using Base::nr_;
  Scalar threshold_;
};

template <typename _Scalar>
struct ActivationDataCollisionTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename Activation>
  explicit ActivationDataCollisionTpl(Activation* const activation)
    : Base(activation),
      distance_(Scalar(0))
  {}

  Scalar distance_;
  
  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_COLLISIONS_HPP_
