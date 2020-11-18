///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_HPP_

#include <stdexcept>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include <iostream>

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelQuadFlatTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelQuadFlatTpl(const std::size_t& nr,const Scalar& sigma2)
     : Base(nr), sigma2_(sigma2){};
  virtual ~ActivationModelQuadFlatTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    data->a_value = Scalar(1.0) - exp((-r.transpose() * r)[0] / sigma2_);
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }

    data->Ar = Scalar(2.0) / sigma2_ * exp((-r.transpose() * r)[0] / sigma2_) * r;
    //data->Arr = - Scalar(4.0) / (sigma2_ * sigma2_) * exp((-r.transpose() * r)[0] / sigma2_) * (r * r.transpose()).resize(nr_,nr_);
    data->Arr.diagonal().array() = Scalar(2.0) / sigma2_ * exp((-r.transpose() * r)[0] / sigma2_);
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
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_HPP_
