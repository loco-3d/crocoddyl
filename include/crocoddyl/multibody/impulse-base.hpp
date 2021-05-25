///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/force-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ImpulseModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImpulseDataAbstractTpl<Scalar> ImpulseDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ImpulseModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nc);
  virtual ~ImpulseModelAbstractTpl();

  virtual void calc(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;

  virtual void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& force) = 0;
  void updateForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const MatrixXs& df_dx) const;
  void setZeroForce(const boost::shared_ptr<ImpulseDataAbstract>& data) const;
  void setZeroForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data) const;

  virtual boost::shared_ptr<ImpulseDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  std::size_t get_nc() const;
  DEPRECATED("Use get_nc().", std::size_t get_ni() const;)
  std::size_t get_nu() const;

  /**
   * @brief Print information on the impulse model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const ImpulseModelAbstractTpl<Scalar>& model);

 protected:
  /**
   * @brief Print relevant information of the impulse model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

  boost::shared_ptr<StateMultibody> state_;
  std::size_t nc_;
};

template <typename _Scalar>
struct ImpulseDataAbstractTpl : public ForceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ForceDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ImpulseDataAbstractTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data), dv0_dq(model->get_nc(), model->get_state()->get_nv()) {
    dv0_dq.setZero();
  }
  virtual ~ImpulseDataAbstractTpl() {}

  using Base::df_dx;
  using Base::f;
  using Base::frame;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;
  MatrixXs dv0_dq;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulse-base.hxx"

#endif  // CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
