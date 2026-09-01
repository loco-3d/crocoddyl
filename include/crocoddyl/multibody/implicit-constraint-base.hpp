///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINT_BASE_HPP_
#define CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINT_BASE_HPP_

#include "crocoddyl/multibody/force-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

class ImplicitConstraintModelBase {
 public:
  virtual ~ImplicitConstraintModelBase() = default;

  CROCODDYL_BASE_CAST(ImplicitConstraintModelBase,
                      ImplicitConstraintModelAbstractTpl)
};

template <typename _Scalar>
class ImplicitConstraintModelAbstractTpl : public ImplicitConstraintModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImplicitConstraintDataAbstractTpl<Scalar>
      ImplicitConstraintDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the implicit-constraint abstraction
   *
   * @param[in] state  State of the multibody system
   * @param[in] type   Type of implicit constraint
   * @param[in] nc     Dimension of the implicit constraint
   * @param[in] nu     Dimension of the control vector
   */
  ImplicitConstraintModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                                     const pinocchio::ReferenceFrame type,
                                     const std::size_t nc,
                                     const std::size_t nu);

  /**
   * @brief Initialize the implicit-constraint abstraction
   *
   * The default `nu` value is obtained from `StateMultibodyTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] type   Type of implicit constraint
   * @param[in] nc     Dimension of the implicit constraint
   */
  ImplicitConstraintModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                                     const pinocchio::ReferenceFrame type,
                                     const std::size_t nc);
  virtual ~ImplicitConstraintModelAbstractTpl() = default;

  /**
   * @brief Compute the implicit-constraint Jacobian and acceleration drift
   *
   * @param[in] data  Implicit-constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) = 0;

  /**
   * @brief Compute the derivatives of the implicit constraint
   *
   * @param[in] data  Implicit-constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) = 0;

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   Implicit-constraint data
   * @param[in] force  Implicit-constraint force
   */
  virtual void updateForce(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const VectorXs& force) = 0;

  /**
   * @brief Update the stack of spatial-force Jacobians
   *
   * @param[in] data   Implicit-constraint data
   * @param[in] df_dx  Jacobian of force with respect to state
   * @param[in] df_du  Jacobian of force with respect to control
   */
  virtual void updateForceDiff(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& df_dx,
      const Eigen::Ref<const MatrixXs>& df_du) const;

  /**
   * @brief Set the stack of spatial forces to zero
   *
   * @param[in] data  Implicit-constraint data
   */
  void setZeroForce(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data) const;

  /**
   * @brief Set the stack of spatial-force Jacobians to zero
   *
   * @param[in] data  Implicit-constraint data
   */
  void setZeroForceDiff(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data) const;

  /**
   * @brief Create the implicit-constraint data
   */
  virtual std::shared_ptr<ImplicitConstraintDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateMultibody>& get_state() const;

  /**
   * @brief Return the dimension of the implicit constraint
   */
  std::size_t get_nc() const;

  /**
   * @brief Return the dimension of the control vector
   */
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
   * @brief Modify the type of implicit constraint
   */
  void set_type(const pinocchio::ReferenceFrame type);

  /**
   * @brief Return the type of implicit constraint
   */
  pinocchio::ReferenceFrame get_type() const;

  /**
   * @brief Print information on the implicit-constraint model
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os,
      const ImplicitConstraintModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the implicit-constraint model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::shared_ptr<StateMultibody> state_;
  std::size_t nc_;
  std::size_t nu_;
  pinocchio::FrameIndex id_;        //!< Reference frame id of the constraint
  pinocchio::ReferenceFrame type_;  //!< Type of implicit constraint
  ImplicitConstraintModelAbstractTpl()
      : state_(nullptr),
        nc_(0),
        nu_(0),
        id_(0),
        type_(pinocchio::ReferenceFrame::LOCAL) {}
};

template <typename _Scalar>
struct ImplicitConstraintDataAbstractTpl
    : public ForceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ForceDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;

 private:
  template <template <typename Scalar> class Model>
  static Model<Scalar>* checkModel(Model<Scalar>* const model) {
    if (model == nullptr) {
      throw_pretty("Invalid argument: implicit-constraint model is null");
    }
    return model;
  }

 public:
  /**
   * @brief Initialize the implicit-constraint data
   *
   * @param[in] model  Implicit-constraint model
   * @param[in] data   Pinocchio data
   */
  template <template <typename Scalar> class Model>
  ImplicitConstraintDataAbstractTpl(Model<Scalar>* const model,
                                    pinocchio::DataTpl<Scalar>* const data)
      : Base(checkModel(model), data),
        fXj(this->jMf.inverse().toActionMatrix()),
        a0(model->get_nc()),
        da0_dx(model->get_nc(), model->get_state()->get_ndx()),
        dv0_dq(model->get_nc(), model->get_state()->get_nv()),
        dtau_dq(model->get_state()->get_nv(), model->get_state()->get_nv()) {
    if (data == nullptr) {
      throw_pretty("Invalid argument: Pinocchio data is null");
    }
    a0.setZero();
    da0_dx.setZero();
    dv0_dq.setZero();
    dtau_dq.setZero();
  }
  virtual ~ImplicitConstraintDataAbstractTpl() = default;

  using Base::df_du;
  using Base::df_dx;
  using Base::f;
  using Base::fext;
  using Base::frame;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;
  using Base::type;

  typename SE3::ActionMatrixType fXj;
  VectorXs a0;
  MatrixXs da0_dx;
  MatrixXs dv0_dq;
  MatrixXs dtau_dq;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/implicit-constraint-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ImplicitConstraintModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ImplicitConstraintDataAbstractTpl)

#endif  // CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINT_BASE_HPP_
