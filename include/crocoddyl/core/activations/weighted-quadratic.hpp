///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_

#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelWeightedQuadTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataWeightedQuadTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelWeightedQuadTpl(const VectorXs& weights)
      : Base(weights.size()), weights_(weights), new_weights_(false){};
  virtual ~ActivationModelWeightedQuadTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    d->Wr = weights_.cwiseProduct(r);
    data->a_value = Scalar(0.5) * r.dot(d->Wr);
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }

    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
    data->Ar = d->Wr;
    if (new_weights_) {
      data->Arr.diagonal() = weights_;
      new_weights_ = false;
    }
    // The Hessian has constant values which were set in createData.
#ifndef NDEBUG
    assert_pretty(MatrixXs(data->Arr).isApprox(Arr_), "Arr has wrong value");
#endif
  };

  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    boost::shared_ptr<Data> data = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    data->Arr.diagonal() = weights_;

#ifndef NDEBUG
    Arr_ = data->Arr;
#endif

    return data;
  };

  const VectorXs& get_weights() const { return weights_; };
  void set_weights(const VectorXs& weights) {
    if (weights.size() != weights_.size()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " + std::to_string(weights_.size()) + ")");
    }

    weights_ = weights;
    new_weights_ = true;
  };

 protected:
  /**
   * @brief Print relevant information of the quadratic-weighted model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const { os << "ActivationModelQuad {nr=" << nr_ << "}"; }

  using Base::nr_;

 private:
  VectorXs weights_;
  bool new_weights_;

#ifndef NDEBUG
  MatrixXs Arr_;
#endif
};

template <typename _Scalar>
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

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
