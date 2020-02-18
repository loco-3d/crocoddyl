///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include <stdexcept>

namespace crocoddyl {

template<typename _Scalar> struct ActivationDataWeightedQuadTpl;
  
template<typename _Scalar>
class ActivationModelWeightedQuadTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataWeightedQuadTpl<Scalar> ActivationDataWeightedQuad;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  explicit ActivationModelWeightedQuadTpl(const VectorXs& weights)
    : Base(weights.size()), weights_(weights) {};
  ~ActivationModelWeightedQuadTpl() {};

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<ActivationDataWeightedQuad> d = boost::static_pointer_cast<ActivationDataWeightedQuad>(data);
    
    d->Wr = weights_.cwiseProduct(r);
    data->a_value = 0.5 * r.dot(d->Wr);
  };
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    
    boost::shared_ptr<ActivationDataWeightedQuad> d = boost::static_pointer_cast<ActivationDataWeightedQuad>(data);
  data->Ar = d->Wr;
  // The Hessian has constant values which were set in createData.
#ifndef NDEBUG
  assert_pretty(data->Arr == Arr_, "Arr has wrong value");
#endif
  };
  boost::shared_ptr<ActivationDataAbstract> createData() {
    boost::shared_ptr<ActivationDataWeightedQuad> data =
      boost::make_shared<ActivationDataWeightedQuad>(this);
    data->Arr.diagonal() = weights_;

#ifndef NDEBUG
    Arr_ = data->Arr;
#endif
    
    return data;
  };

  const VectorXs& get_weights() const  { return weights_; };
  void set_weights(const VectorXs& weights) {
  if (weights.size() != weights_.size()) {
    throw_pretty("Invalid argument: "
                 << "weight vector has wrong dimension (it should be " + std::to_string(weights_.size()) + ")");
  }

  weights_ = weights;
};

protected:
  using Base::nr_;
  
 private:
  VectorXs weights_;

#ifndef NDEBUG
  MatrixXs Arr_;
#endif
};

template<typename _Scalar>
struct ActivationDataWeightedQuadTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  
  template <typename Activation>
  explicit ActivationDataWeightedQuadTpl(Activation* const activation)
      : Base(activation), Wr(VectorXs::Zero(activation->get_nr())) {}

  VectorXs Wr;
};

typedef ActivationModelWeightedQuadTpl<double> ActivationModelWeightedQuad;
typedef ActivationDataWeightedQuadTpl<double> ActivationDataWeightedQuad;
  
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
