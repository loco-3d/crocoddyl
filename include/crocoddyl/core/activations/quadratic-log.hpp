///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_LOG_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_LOG_HPP_

#include <stdexcept>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include <iostream>

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelQuadLogTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelQuadLogTpl(const std::size_t& nr,const Scalar& sigma2)
     : Base(nr), sigma2_(sigma2){};
  virtual ~ActivationModelQuadLogTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    data->a_value = log(Scalar(1.0) + (r.transpose() * r)[0] / sigma2_);
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    a0_ = Scalar(2.0) / (Scalar(1.0) + (r.transpose() * r)[0] / sigma2_) / sigma2_;
    data->Ar = a0_ * r;
    data->Arr.diagonal() = - a0_*a0_  * (r * r.transpose()).diagonal();
    data->Arr.diagonal().array() += a0_;
  };

  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    boost::shared_ptr<ActivationDataAbstract> data =
        boost::allocate_shared<ActivationDataAbstract>(Eigen::aligned_allocator<ActivationDataAbstract>(), this);
    data->Arr.diagonal().fill((Scalar)1.);
    return data;
  };
  
  const Scalar& get_sigma2() const { return sigma2_; };
  void set_sigma2(const Scalar& sigma2) { sigma2_ = sigma2; };
  
 protected:
  using Base::nr_;
  
 private:
  Scalar sigma2_;
  Scalar a0_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_HPP_
