///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelQuadTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase, ActivationModelQuadTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelQuadTpl(const std::size_t nr) : Base(nr) {};
  virtual ~ActivationModelQuadTpl() = default;

  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    data->a_value = Scalar(0.5) * r.dot(r);
  };

  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }

    data->Ar = r;
    // The Hessian has constant values which were set in createData.
    assert_pretty(MatrixXs(data->Arr).isApprox(MatrixXs::Identity(nr_, nr_)),
                  "Arr has wrong value");
  };

  virtual std::shared_ptr<ActivationDataAbstract> createData() override {
    std::shared_ptr<ActivationDataAbstract> data =
        std::allocate_shared<ActivationDataAbstract>(
            Eigen::aligned_allocator<ActivationDataAbstract>(), this);
    data->Arr.diagonal().setOnes();
    return data;
  };

  template <typename NewScalar>
  ActivationModelQuadTpl<NewScalar> cast() const {
    typedef ActivationModelQuadTpl<NewScalar> ReturnType;
    ReturnType res(nr_);
    return res;
  }

  /**
   * @brief Print relevant information of the quadratic model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModelQuad {nr=" << nr_ << "}";
  }

 protected:
  using Base::nr_;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActivationModelQuadTpl)

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
