///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUAL_BASE_HPP_
#define CROCODDYL_CORE_RESIDUAL_BASE_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

class ResidualModelBase {
 public:
  virtual ~ResidualModelBase() = default;

  CROCODDYL_BASE_CAST(ResidualModelBase, ResidualModelAbstractTpl)
};

/**
 * @brief Abstract class for residual models
 *
 * A residual model defines a vector function \f$\mathbf{r}(\mathbf{x},
 * \mathbf{u})\mathbb{R}^{nr}\f$ where `nr` describes its dimension in the
 * Euclidean space. This function depends on the state point
 * \f$\mathbf{x}\in\mathcal{X}\f$, which lies in the state manifold described
 * with a `nq`-tuple, its velocity \f$\dot{\mathbf{x}}\in
 * T_{\mathbf{x}}\mathcal{X}\f$ that belongs to the tangent space with `nv`
 * dimension, and the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$. The
 * residual function can used across cost and constraint models.
 *
 * The main computations are carring out in `calc` and `calcDiff` routines.
 * `calc` computes the residual vector and `calcDiff` computes the Jacobians of
 * the residual function. Additionally, it is important to note that
 * `calcDiff()` computes the Jacobians using the latest stored values by
 * `calc()`. Thus, we need to first run `calc()`.
 *
 * \sa `StateAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelAbstractTpl : public ResidualModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_CAST(ResidualModelBase, ResidualModelAbstractTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;

  /**
   * @brief Initialize the residual model
   *
   * @param[in] state        State of the system
   * @param[in] nr           Dimension of residual vector
   * @param[in] nu           Dimension of control vector
   * @param[in] q_dependent  Define if the residual function depends on q
   * (default true)
   * @param[in] v_dependent  Define if the residual function depends on v
   * (default true)
   * @param[in] u_dependent  Define if the residual function depends on u
   * (default true)
   */
  ResidualModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t nr, const std::size_t nu,
                           const bool q_dependent = true,
                           const bool v_dependent = true,
                           const bool u_dependent = true);

  /**
   * @copybrief ResidualModelAbstractTpl()
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state        State of the system
   * @param[in] nr           Dimension of residual vector
   * @param[in] q_dependent  Define if the residual function depends on q
   * (default true)
   * @param[in] v_dependent  Define if the residual function depends on v
   * (default true)
   * @param[in] u_dependent  Define if the residual function depends on u
   * (default true)
   */
  ResidualModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t nr, const bool q_dependent = true,
                           const bool v_dependent = true,
                           const bool u_dependent = true);
  virtual ~ResidualModelAbstractTpl() = default;

  /**
   * @brief Compute the residual vector
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the residual vector for nodes that depends only on the state
   *
   * It updates the residual vector based on the state only. This function is
   * used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobian of the residual vector
   *
   * It computes the Jacobian the residual function. It assumes that `calc()`
   * has been run first.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobian of the residual functions with respect to the
   * state only
   *
   * It updates the Jacobian of the residual function based on the state only.
   * This function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the residual data
   *
   * The default data contains objects to store the values of the residual
   * vector and their Jacobians. However, it is possible to specialize this
   * function if we need to create additional data, for instance, to avoid
   * dynamic memory allocation.
   *
   * @param data  Data collector
   * @return the residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Compute the derivative of the cost function
   *
   * This function assumes that the derivatives of the activation and residual
   * are computed via calcDiff functions.
   *
   * @param cdata     Cost data
   * @param rdata     Residual data
   * @param adata     Activation data
   * @param update_u  Update the derivative of the cost function w.r.t. to the
   * control if True.
   */
  virtual void calcCostDiff(
      const std::shared_ptr<CostDataAbstract>& cdata,
      const std::shared_ptr<ResidualDataAbstract>& rdata,
      const std::shared_ptr<ActivationDataAbstract>& adata,
      const bool update_u = true);

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the dimension of the residual vector
   */
  std::size_t get_nr() const;

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return true if the residual function depends on q
   */
  bool get_q_dependent() const;

  /**
   * @brief Return true if the residual function depends on v
   */
  bool get_v_dependent() const;

  /**
   * @brief Return true if the residual function depends on u
   */
  bool get_u_dependent() const;

  /**
   * @brief Print information on the residual model
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const ResidualModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the residual model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::shared_ptr<StateAbstract> state_;  //!< State description
  std::size_t nr_;                        //!< Residual vector dimension
  std::size_t nu_;                        //!< Control dimension
  VectorXs unone_;                        //!< No control vector
  bool q_dependent_;  //!< Label that indicates if the residual function depends
                      //!< on q
  bool v_dependent_;  //!< Label that indicates if the residual function depends
                      //!< on v
  bool u_dependent_;  //!< Label that indicates if the residual function depends
                      //!< on u
  ResidualModelAbstractTpl()
      : state_(nullptr),
        nr_(0),
        nu_(0),
        q_dependent_(false),
        v_dependent_(false),
        u_dependent_(false) {};
};

template <typename _Scalar>
struct ResidualDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ResidualDataAbstractTpl(Model<Scalar>* const model,
                          DataCollectorAbstract* const data)
      : shared(data),
        r(model->get_nr()),
        Rx(model->get_nr(), model->get_state()->get_ndx()),
        Ru(model->get_nr(), model->get_nu()),
        Arr_Rx(model->get_nr(), model->get_state()->get_ndx()),
        Arr_Ru(model->get_nr(), model->get_nu()) {
    r.setZero();
    Rx.setZero();
    Ru.setZero();
    Arr_Rx.setZero();
    Arr_Ru.setZero();
  }
  virtual ~ResidualDataAbstractTpl() = default;

  DataCollectorAbstract* shared;  //!< Shared data allocated by the action model
  VectorXs r;                     //!< Residual vector
  MatrixXs Rx;  //!< Jacobian of the residual vector with respect the state
  MatrixXs Ru;  //!< Jacobian of the residual vector with respect the control
  MatrixXs Arr_Rx;
  MatrixXs Arr_Ru;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residual-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataAbstractTpl)

#endif  // CROCODDYL_CORE_RESIDUAL_BASE_HPP_
