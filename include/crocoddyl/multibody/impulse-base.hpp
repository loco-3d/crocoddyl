///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/multibody/force-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

class ImpulseModelBase {
 public:
  virtual ~ImpulseModelBase() = default;

  CROCODDYL_BASE_CAST(ImpulseModelBase, ImpulseModelAbstractTpl)
};

template <typename _Scalar>
class ImpulseModelAbstractTpl : public ImpulseModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImpulseDataAbstractTpl<Scalar> ImpulseDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ImpulseModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                          const pinocchio::ReferenceFrame type,
                          const std::size_t nc);

  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ImpulseModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                              const std::size_t nc);)
  virtual ~ImpulseModelAbstractTpl() = default;

  virtual void calc(const std::shared_ptr<ImpulseDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) = 0;
  virtual void calcDiff(const std::shared_ptr<ImpulseDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) = 0;

  virtual void updateForce(const std::shared_ptr<ImpulseDataAbstract>& data,
                           const VectorXs& force) = 0;
  void updateForceDiff(const std::shared_ptr<ImpulseDataAbstract>& data,
                       const MatrixXs& df_dx) const;
  void setZeroForce(const std::shared_ptr<ImpulseDataAbstract>& data) const;
  void setZeroForceDiff(const std::shared_ptr<ImpulseDataAbstract>& data) const;

  virtual std::shared_ptr<ImpulseDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data);

  const std::shared_ptr<StateMultibody>& get_state() const;
  std::size_t get_nc() const;
  DEPRECATED("Use get_nc().", std::size_t get_ni() const;)
  std::size_t get_nu() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the type of contact
   */
  void set_type(const pinocchio::ReferenceFrame type);

  /**
   * @brief Return the type of contact
   */
  pinocchio::ReferenceFrame get_type() const;

  /**
   * @brief Print information on the impulse model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ImpulseModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the impulse model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::shared_ptr<StateMultibody> state_;
  std::size_t nc_;
  pinocchio::FrameIndex id_;        //!< Reference frame id of the contact
  pinocchio::ReferenceFrame type_;  //!< Type of contact
  ImpulseModelAbstractTpl() : state_(nullptr), nc_(0), id_(0) {};
};

template <typename _Scalar>
struct ImpulseDataAbstractTpl : public ForceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ForceDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;

  template <template <typename Scalar> class Model>
  ImpulseDataAbstractTpl(Model<Scalar>* const model,
                         pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        dv0_dq(model->get_nc(), model->get_state()->get_nv()),
        dtau_dq(model->get_state()->get_nv(), model->get_state()->get_nv()) {
    dv0_dq.setZero();
    dtau_dq.setZero();
  }
  virtual ~ImpulseDataAbstractTpl() = default;

  using Base::df_dx;
  using Base::f;
  using Base::frame;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;

  typename SE3::ActionMatrixType fXj;
  MatrixXs dv0_dq;
  MatrixXs dtau_dq;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulse-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ImpulseModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ImpulseDataAbstractTpl)

#endif  // CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
