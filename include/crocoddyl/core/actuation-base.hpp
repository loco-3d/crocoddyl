///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActuationModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActuationModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t& nu) : nu_(nu), state_(state) {
    if (nu_ == 0) {
      throw_pretty("Invalid argument: "
                   << "nu cannot be zero");
    }
  };
  virtual ~ActuationModelAbstractTpl(){};

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;
  virtual boost::shared_ptr<ActuationDataAbstract> createData() {
    return boost::make_shared<ActuationDataAbstract>(this);
  };

  const std::size_t& get_nu() const { return nu_; };
  const boost::shared_ptr<StateAbstract>& get_state() const { return state_; };

 protected:
  std::size_t nu_;
  boost::shared_ptr<StateAbstract> state_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ActuationDataAbstract>& data, const VectorXs& x, const VectorXs& u) {
    calc(data, x, u);
  }

  void calcDiff_wrap(const boost::shared_ptr<ActuationDataAbstract>& data, const VectorXs& x, const VectorXs& u) {
    calcDiff(data, x, u);
  }

#endif
};

template <typename _Scalar>
struct ActuationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActuationDataAbstractTpl(Model<Scalar>* const model)
      : tau(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dtau_du(model->get_state()->get_nv(), model->get_nu()) {
    tau.setZero();
    dtau_dx.setZero();
    dtau_du.setZero();
  }
  virtual ~ActuationDataAbstractTpl() {}

  VectorXs tau;
  MatrixXs dtau_dx;
  MatrixXs dtau_du;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTUATION_BASE_HPP_
