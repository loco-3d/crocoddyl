///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include <stdexcept>
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template<typename _Scalar>
class ActivationModelQuadTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  explicit ActivationModelQuadTpl(const std::size_t& nr) : Base(nr) {};
  ~ActivationModelQuadTpl() {};

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    data->a_value = 0.5 * r.transpose() * r;
  };
  
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    
    data->Ar = r;
    // The Hessian has constant values which were set in createData.
    assert_pretty(data->Arr == MatrixXs::Identity(nr_, nr_), "Arr has wrong value");
  };

  boost::shared_ptr<ActivationDataAbstract> createData() {
    boost::shared_ptr<ActivationDataAbstract> data = boost::make_shared<ActivationDataAbstract>(this);
    data->Arr.diagonal().fill(1.);
    return data;
  };
  
protected:
  using Base::nr_;
  
};

typedef ActivationModelQuadTpl<double> ActivationModelQuad;
  
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
