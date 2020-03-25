///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, Universitat Politècinca de Catalunya
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

  virtual void calc(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& u) = 0;
  virtual boost::shared_ptr<SquashingDataAbstract> createData() {
    return boost::make_shared<SquashingDataAbstract>(this);
  }

  const std::size_t& get_ns() const;
  const VectorXs& get_u_lb() const { return u_lb_; };
  const VectorXs& get_u_ub() const { return u_ub_; };

  void set_u_lb(const VectorXs& u_lb) { u_lb_ = u_lb; };
  void set_u_ub(const VectorXs& u_ub) { u_ub_ = u_ub; };

 protected:
  std::size_t ns_;
  VectorXs s_ub_;  // Squashing function upper bound
  VectorXs s_lb_;  // Squashing function lower bound
  VectorXs u_ub_;  // Bound for the u variable (to apply using the Quadratic barrier)
  VectorXs u_lb_;  // Bound for the u variable (to apply using the Quadratic barrier)

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<SquashingDataAbstract>& data, const VectorXs& u) { calc(data, u); }

  void calcDiff_wrap(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::VectorXs& u) {
    calcDiff(data, u);
  }

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
      : s(model->get_ns()), ds_du(model->get_ns(), model->get_ns()) {
    s.setZero();
    ds_du.setZero();
  }
  virtual ~SquashingDataAbstractTpl() {}

  VectorXs s;
  MatrixXs ds_du;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SQUASHING_BASE_HPP_