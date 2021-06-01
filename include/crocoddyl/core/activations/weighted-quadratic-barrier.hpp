///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_

#include <pinocchio/utils/static-if.hpp>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelWeightedQuadraticBarrierTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataQuadraticBarrierTpl<Scalar> Data;
  typedef ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelWeightedQuadraticBarrierTpl(const ActivationBounds& bounds, const VectorXs& weights)
      : Base(bounds.lb.size()), bounds_(bounds), weights_(weights){};
  virtual ~ActivationModelWeightedQuadraticBarrierTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    d->rlb_min_ = (r - bounds_.lb).array().min(Scalar(0.));
    d->rub_max_ = (r - bounds_.ub).array().max(Scalar(0.));
    d->rlb_min_.array() *= weights_.array();
    d->rub_max_.array() *= weights_.array();
    data->a_value =
        Scalar(0.5) * d->rlb_min_.matrix().squaredNorm() + Scalar(0.5) * d->rub_max_.matrix().squaredNorm();
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
    data->Ar = (d->rlb_min_ + d->rub_max_).matrix();
    data->Ar.array() *= weights_.array();

    using pinocchio::internal::if_then_else;
    for (Eigen::Index i = 0; i < data->Arr.cols(); i++) {
      data->Arr.diagonal()[i] = if_then_else(
          pinocchio::internal::LE, r[i] - bounds_.lb[i], Scalar(0.), Scalar(1.),
          if_then_else(pinocchio::internal::GE, r[i] - bounds_.ub[i], Scalar(0.), Scalar(1.), Scalar(0.)));
    }

    data->Arr.diagonal().array() *= weights_.array();
  };

  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  const ActivationBounds& get_bounds() const { return bounds_; };
  const VectorXs& get_weights() const { return weights_; };
  void set_bounds(const ActivationBounds& bounds) { bounds_ = bounds; };
  void set_weights(const VectorXs& weights) {
    if (weights.size() != weights_.size()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " + std::to_string(weights_.size()) + ")");
    }
    weights_ = weights;
  };

  /**
   * @brief Print relevant information of the quadratic barrier model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const { os << "ActivationModelWeightedQuadraticBarrier {nr=" << nr_ << "}"; }

 protected:
  using Base::nr_;

 private:
  ActivationBounds bounds_;
  VectorXs weights_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_
