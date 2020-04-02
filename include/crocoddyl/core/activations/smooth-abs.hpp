///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_

#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelSmoothAbsTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataSmoothAbsTpl<Scalar> ActivationDataSmoothAbs;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelSmoothAbsTpl(const std::size_t& nr) : Base(nr){};
  virtual ~ActivationModelSmoothAbsTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<ActivationDataSmoothAbs> d = boost::static_pointer_cast<ActivationDataSmoothAbs>(data);

    d->a = (r.array().cwiseAbs2().array() + Scalar(1)).array().cwiseSqrt();
    data->a_value = d->a.sum();
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }

    boost::shared_ptr<ActivationDataSmoothAbs> d = boost::static_pointer_cast<ActivationDataSmoothAbs>(data);
    data->Ar = r.cwiseProduct(d->a.cwiseInverse());
    data->Arr.diagonal() = d->a.cwiseProduct(d->a).cwiseProduct(d->a).cwiseInverse();
  };

  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::make_shared<ActivationDataSmoothAbs>(this);
  };

 protected:
  using Base::nr_;
};

template <typename _Scalar>
struct ActivationDataSmoothAbsTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename Activation>
  explicit ActivationDataSmoothAbsTpl(Activation* const activation)
      : Base(activation), a(VectorXs::Zero(activation->get_nr())) {
    Arr.diagonal().array() = Scalar(2);
  }

  VectorXs a;
  using Base::Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
