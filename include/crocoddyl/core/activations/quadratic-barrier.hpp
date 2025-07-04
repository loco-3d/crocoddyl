///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files. All
// rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ActivationBoundsTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActivationBoundsTpl(const VectorXs& lower, const VectorXs& upper,
                      const Scalar b = Scalar(1.))
      : lb(lower), ub(upper), beta(b) {
    if (lb.size() != ub.size()) {
      throw_pretty("Invalid argument: "
                   << "The lower and upper bounds don't have the same "
                      "dimension (lb,ub dimensions equal to " +
                          std::to_string(lb.size()) + "," +
                          std::to_string(ub.size()) + ", respectively)");
    }
    if (beta < Scalar(0) || beta > Scalar(1.)) {
      throw_pretty(
          "Invalid argument: " << "The range of beta is between 0 and 1");
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(lb.size()); ++i) {
      if (isfinite(lb(i)) && isfinite(ub(i))) {
        if (lb(i) - ub(i) > Scalar(0)) {
          throw_pretty("Invalid argument: "
                       << "The lower and upper bounds are badly defined; ub "
                          "has to be bigger / equals to lb");
        }
      }
      // Assign the maximum value for infinity/nan values
      if (!isfinite(lb(i))) {
        lb(i) = -std::numeric_limits<Scalar>::max();
      }
      if (!isfinite(ub(i))) {
        ub(i) = std::numeric_limits<Scalar>::max();
      }
    }

    if (beta >= Scalar(0) && beta <= Scalar(1.)) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(lb.size()); ++i) {
        // do not use beta when one of the bounds is inf
        if (lb(i) != (-std::numeric_limits<Scalar>::max()) &&
            ub(i) != (std::numeric_limits<Scalar>::max())) {
          Scalar m = Scalar(0.5) * (lb(i) + ub(i));
          Scalar d = Scalar(0.5) * (ub(i) - lb(i));
          lb(i) = m - beta * d;
          ub(i) = m + beta * d;
        }
      }
    } else {
      beta = Scalar(1.);
    }
  }
  ActivationBoundsTpl(const ActivationBoundsTpl& other)
      : lb(other.lb), ub(other.ub), beta(other.beta) {}
  ActivationBoundsTpl() : beta(Scalar(1.)) {}

  template <typename NewScalar>
  ActivationBoundsTpl<NewScalar> cast() const {
    typedef ActivationBoundsTpl<NewScalar> ReturnType;
    ReturnType res(lb.template cast<NewScalar>(), ub.template cast<NewScalar>(),
                   scalar_cast<NewScalar>(beta));
    return res;
  }

  ActivationBoundsTpl& operator=(const ActivationBoundsTpl& other) {
    if (this != &other) {
      lb = other.lb;
      ub = other.ub;
      beta = other.beta;
    }
    return *this;
  }

  /**
   * @brief Print information on the activation bounds
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const ActivationBoundsTpl& bounds) {
    bounds.print(os);
    return os;
  }

  void print(std::ostream& os) const {
    os << "ActivationBounds {lb=" << lb.transpose() << ", ub=" << ub.transpose()
       << ", beta=" << beta << "}";
  }

  VectorXs lb;
  VectorXs ub;
  Scalar beta;
};

template <typename _Scalar>
class ActivationModelQuadraticBarrierTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase,
                         ActivationModelQuadraticBarrierTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataQuadraticBarrierTpl<Scalar> Data;
  typedef ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelQuadraticBarrierTpl(const ActivationBounds& bounds)
      : Base(bounds.lb.size()), bounds_(bounds) {};
  virtual ~ActivationModelQuadraticBarrierTpl() = default;

  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }

    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

    d->rlb_min_ = (r - bounds_.lb).array().min(Scalar(0.));
    d->rub_max_ = (r - bounds_.ub).array().max(Scalar(0.));
    data->a_value = Scalar(0.5) * d->rlb_min_.matrix().squaredNorm() +
                    Scalar(0.5) * d->rub_max_.matrix().squaredNorm();
  };

  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }

    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
    data->Ar = (d->rlb_min_ + d->rub_max_).matrix();

    using pinocchio::internal::if_then_else;
    for (Eigen::Index i = 0; i < data->Arr.cols(); i++) {
      data->Arr.diagonal()[i] = if_then_else(
          pinocchio::internal::LE, r[i] - bounds_.lb[i], Scalar(0.), Scalar(1.),
          if_then_else(pinocchio::internal::GE, r[i] - bounds_.ub[i],
                       Scalar(0.), Scalar(1.), Scalar(0.)));
    }
  };

  virtual std::shared_ptr<ActivationDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  template <typename NewScalar>
  ActivationModelQuadraticBarrierTpl<NewScalar> cast() const {
    typedef ActivationModelQuadraticBarrierTpl<NewScalar> ReturnType;
    ReturnType res(bounds_.template cast<NewScalar>());
    return res;
  }

  const ActivationBounds& get_bounds() const { return bounds_; };
  void set_bounds(const ActivationBounds& bounds) { bounds_ = bounds; };

  /**
   * @brief Print relevant information of the quadratic barrier model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModelQuadraticBarrier {nr=" << nr_ << "}";
  }

 protected:
  using Base::nr_;

 private:
  ActivationBounds bounds_;
};

template <typename _Scalar>
struct ActivationDataQuadraticBarrierTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::ArrayXs ArrayXs;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  template <typename Activation>
  explicit ActivationDataQuadraticBarrierTpl(Activation* const activation)
      : Base(activation),
        rlb_min_(activation->get_nr()),
        rub_max_(activation->get_nr()) {
    rlb_min_.setZero();
    rub_max_.setZero();
  }

  ArrayXs rlb_min_;
  ArrayXs rub_max_;

  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActivationBoundsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActivationModelQuadraticBarrierTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActivationDataQuadraticBarrierTpl)

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_
