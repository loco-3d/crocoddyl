///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, IRI: CSIC-UPC,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SQUASHING_BASE_HPP_
#define CROCODDYL_CORE_SQUASHING_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

class SquashingModelBase {
 public:
  virtual ~SquashingModelBase() = default;

  CROCODDYL_BASE_CAST(SquashingModelBase, SquashingModelAbstractTpl)
};

template <typename _Scalar>
class SquashingModelAbstractTpl : public SquashingModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  SquashingModelAbstractTpl(const std::size_t ns) : ns_(ns) {
    if (ns_ == 0) {
      throw_pretty("Invalid argument: " << "ns cannot be zero");
    }
  };
  virtual ~SquashingModelAbstractTpl() = default;

  virtual void calc(const std::shared_ptr<SquashingDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& s) = 0;
  virtual void calcDiff(const std::shared_ptr<SquashingDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& s) = 0;
  virtual std::shared_ptr<SquashingDataAbstract> createData() {
    return std::allocate_shared<SquashingDataAbstract>(
        Eigen::aligned_allocator<SquashingDataAbstract>(), this);
  }

  /**
   * @brief Print information on the actuation model
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const SquashingModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the squashing model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const {
    os << boost::core::demangle(typeid(*this).name());
  }

  std::size_t get_ns() const { return ns_; };
  const VectorXs& get_s_lb() const { return s_lb_; };
  const VectorXs& get_s_ub() const { return s_ub_; };

  void set_s_lb(const VectorXs& s_lb) { s_lb_ = s_lb; };
  void set_s_ub(const VectorXs& s_ub) { s_ub_ = s_ub; };

 protected:
  std::size_t ns_;
  VectorXs u_ub_;  // Squashing function upper bound
  VectorXs u_lb_;  // Squashing function lower bound
  VectorXs
      s_ub_;  // Bound for the s variable (to apply using the Quadratic barrier)
  VectorXs
      s_lb_;  // Bound for the s variable (to apply using the Quadratic barrier)
  SquashingModelAbstractTpl() : ns_(0) {};
};

template <typename _Scalar>
struct SquashingDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit SquashingDataAbstractTpl(Model<Scalar>* const model)
      : u(model->get_ns()), du_ds(model->get_ns(), model->get_ns()) {
    u.setZero();
    du_ds.setZero();
  }
  SquashingDataAbstractTpl() {}
  virtual ~SquashingDataAbstractTpl() = default;

  VectorXs u;
  MatrixXs du_ds;
};

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const SquashingModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SquashingModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::SquashingDataAbstractTpl)

#endif  // CROCODDYL_CORE_SQUASHING_BASE_HPP_
