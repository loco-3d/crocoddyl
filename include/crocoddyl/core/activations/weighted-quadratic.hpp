///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelWeightedQuadTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase, ActivationModelWeightedQuadTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataWeightedQuadTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelWeightedQuadTpl(const VectorXs& weights)
      : Base(weights.size()), weights_(weights), new_weights_(false) {};
  virtual ~ActivationModelWeightedQuadTpl() = default;

  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

    d->Wr = weights_.cwiseProduct(r);
    data->a_value = Scalar(0.5) * r.dot(d->Wr);
  };

  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }

    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
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

  virtual std::shared_ptr<ActivationDataAbstract> createData() override {
    std::shared_ptr<Data> data =
        std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    data->Arr.diagonal() = weights_;

#ifndef NDEBUG
    Arr_ = data->Arr;
#endif

    return data;
  };

  template <typename NewScalar>
  ActivationModelWeightedQuadTpl<NewScalar> cast() const {
    typedef ActivationModelWeightedQuadTpl<NewScalar> ReturnType;
    ReturnType res(weights_.template cast<NewScalar>());
    return res;
  }

  const VectorXs& get_weights() const { return weights_; };
  void set_weights(const VectorXs& weights) {
    if (weights.size() != weights_.size()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " +
                          std::to_string(weights_.size()) + ")");
    }

    weights_ = weights;
    new_weights_ = true;
  };

  /**
   * @brief Print relevant information of the quadratic-weighted model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModelQuad {nr=" << nr_ << "}";
  }

 protected:
  using Base::nr_;

 private:
  VectorXs weights_;
  bool new_weights_;

#ifndef NDEBUG
  MatrixXs Arr_;
#endif
};

template <typename _Scalar>
struct ActivationDataWeightedQuadTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  template <typename Activation>
  explicit ActivationDataWeightedQuadTpl(Activation* const activation)
      : Base(activation), Wr(VectorXs::Zero(activation->get_nr())) {}
  virtual ~ActivationDataWeightedQuadTpl() = default;

  VectorXs Wr;

  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActivationModelWeightedQuadTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActivationDataWeightedQuadTpl)

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
