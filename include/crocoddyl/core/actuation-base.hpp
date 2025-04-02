///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

class ActuationModelBase {
 public:
  virtual ~ActuationModelBase() = default;

  CROCODDYL_BASE_CAST(ActuationModelBase, ActuationModelAbstractTpl)
};

/**
 * @brief Abstract class for the actuation-mapping model
 *
 * The generalized torques \f$\boldsymbol{\tau}\in\mathbb{R}^{nv}\f$ can by any
 * nonlinear function of the joint-torque inputs
 * \f$\mathbf{u}\in\mathbb{R}^{nu}\f$, and state point
 * \f$\mathbf{x}\in\mathbb{R}^{nx}\f$, where `nv`, `nu`, and `ndx` are the
 * number of joints, dimension of the joint torque input and state manifold,
 * respectively. Additionally, the generalized torques are also named as the
 * actuation signals of our system.
 *
 * The main computations are carried out in `calc()`, and `calcDiff()`, where
 * the former computes actuation signal, and the latter computes the Jacobians
 * of the actuation-mapping function, i.e.,
 * \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{x}}\f$ and
 * \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{u}}\f$. Note that
 * `calcDiff()` requires to run `calc()` first.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelAbstractTpl : public ActuationModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the actuation model
   *
   * @param[in] state  State description
   * @param[in] nu     Dimension of joint-torque input
   */
  ActuationModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                            const std::size_t nu);
  virtual ~ActuationModelAbstractTpl() = default;

  /**
   * @brief Compute the actuation signal from the state point
   * \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$ and joint torque inputs
   * \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Ignore the computation of the actuation signal
   *
   * It does not update the actuation signal as this function is used in the
   * terminal nodes of an optimal control problem.
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const std::shared_ptr<ActuationDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobians of the actuation function
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Ignore the computation of the Jacobians of the actuation function
   *
   * It does not update the Jacobians of the actuation function as this function
   * is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the joint torque input from the generalized torques
   *
   * It stores the results in `ActuationDataAbstractTpl::u`.
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] tau   Generalized torques \f$\mathbf{u}\in\mathbb{R}^{nv}\f$
   */
  virtual void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& tau) = 0;

  /**
   * @brief Compute the torque transform from generalized torques to joint
   * torque inputs
   *
   * It stores the results in `ActuationDataAbstractTpl::Mtau`.
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] tau   Joint-torque inputs \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void torqueTransform(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  /**
   * @brief Create the actuation data
   *
   * @return the actuation data
   */
  virtual std::shared_ptr<ActuationDataAbstract> createData();

  /**
   * @brief Return the dimension of the joint-torque input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Print information on the actuation model
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const ActuationModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the residual model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::size_t nu_;                        //!< Dimension of joint torque inputs
  std::shared_ptr<StateAbstract> state_;  //!< Model of the state
  ActuationModelAbstractTpl() : nu_(0), state_(nullptr) {};
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
        u(model->get_nu()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dtau_du(model->get_state()->get_nv(), model->get_nu()),
        Mtau(model->get_nu(), model->get_state()->get_nv()),
        tau_set(model->get_state()->get_nv(), true) {
    tau.setZero();
    u.setZero();
    dtau_dx.setZero();
    dtau_du.setZero();
    Mtau.setZero();
  }
  virtual ~ActuationDataAbstractTpl() = default;

  VectorXs tau;      //!< Generalized torques
  VectorXs u;        //!< Joint torques
  MatrixXs dtau_dx;  //!< Partial derivatives of the actuation model w.r.t. the
                     //!< state point
  MatrixXs dtau_du;  //!< Partial derivatives of the actuation model w.r.t. the
                     //!< joint torque input
  MatrixXs Mtau;  //!< Torque transform from generalized torques to joint torque
                  //!< inputs
  std::vector<bool> tau_set;  //!< True for joints that are actuacted
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/actuation-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActuationModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActuationDataAbstractTpl)

#endif  // CROCODDYL_CORE_ACTUATION_BASE_HPP_
