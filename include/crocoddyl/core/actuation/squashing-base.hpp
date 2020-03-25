///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SQUASHING_BASE_HPP_
#define CROCODDYL_CORE_SQUASHING_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SquashingModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  SquashingModelAbstractTpl(const std::size_t& ns) : ns_(ns) {
    if (ns_ == 0) {
      throw_pretty("Invalid argument: "
                   << "ns cannot be zero");
    }
  };
  virtual ~SquashingModelAbstractTpl(){};

  virtual void calc(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& s) = 0;
  virtual void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& s) = 0;
  virtual boost::shared_ptr<SquashingDataAbstract> createData() {
    return boost::make_shared<SquashingDataAbstract>(this);
  }

  const std::size_t& get_ns() const { return ns_; };
  const VectorXs& get_s_lb() const { return s_lb_; };
  const VectorXs& get_s_ub() const { return s_ub_; };

  void set_s_lb(const VectorXs& s_lb) { s_lb_ = s_lb; };
  void set_s_ub(const VectorXs& s_ub) { s_ub_ = s_ub; };

 protected:
  std::size_t ns_;
  VectorXs u_ub_;  // Squashing function upper bound
  VectorXs u_lb_;  // Squashing function lower bound
  VectorXs s_ub_;  // Bound for the s variable (to apply using the Quadratic barrier)
  VectorXs s_lb_;  // Bound for the s variable (to apply using the Quadratic barrier)

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<SquashingDataAbstract>& data, const VectorXs& s) { calc(data, s); }

  void calcDiff_wrap(const boost::shared_ptr<SquashingDataAbstract>& data, const VectorXs& s) { calcDiff(data, s); }

#endif
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
  virtual ~SquashingDataAbstractTpl() {}

  VectorXs u;
  MatrixXs du_ds;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SQUASHING_BASE_HPP_
