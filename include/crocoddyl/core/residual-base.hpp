///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, University of Edinburgh, Heriot-Watt University
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
 * \mathbf{u},\mathbf{p})\in\mathbb{R}^{nr}\f$ where `nr` describes its
 * dimension in the Euclidean space. This function depends on the state point
 * \f$\mathbf{x}\in\mathcal{X}\f$, which lies in the state manifold described
 * with a `nq`-tuple, its velocity \f$\dot{\mathbf{x}}\in
 * T_{\mathbf{x}}\mathcal{X}\f$ that belongs to the tangent space with `nv`
 * dimension, the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$, and the
 * parameter vector \f$\mathbf{p}\in\mathbb{R}^{np}\f$. The residual function
 * can be used across cost and constraint models.
 *
 * The main computations are carring out in `calc` and `calcDiff` routines.
 * `calc` computes the residual vector and `calcDiff` computes the Jacobians of
 * the residual function, including
 * \f$\mathbf{R_p}=\partial\mathbf{r}/\partial\mathbf{p}
 * \in\mathbb{R}^{nr\times np}\f$. The parameters have to be updated before
 * calling `calc()` and `calcDiff()`. Additionally, it is important to note that
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
   * This overload explicitly sets the parameter-vector dimension.
   *
   * @param[in] state        State of the system
   * @param[in] nr           Dimension of residual vector
   * @param[in] nu           Dimension of control vector
   * @param[in] q_dependent  Define if the residual function depends on q
   * @param[in] v_dependent  Define if the residual function depends on v
   * @param[in] u_dependent  Define if the residual function depends on u
   * @param[in] np           Dimension of parameter vector
   */
  ResidualModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t nr, const std::size_t nu,
                           const bool q_dependent, const bool v_dependent,
                           const bool u_dependent, const std::size_t np);

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
   * @brief Return the dimension of the parameter vector
   */
  std::size_t get_np() const;

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
  std::size_t np_;                        //!< Parameter vector dimension
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
        np_(0),
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
        Rp(model->get_nr(), model->get_np()),
        Arr_Rx(model->get_nr(), model->get_state()->get_ndx()),
        Arr_Ru(model->get_nr(), model->get_nu()),
        Arr_Rp(model->get_nr(), model->get_np()) {
    r.setZero();
    Rx.setZero();
    Ru.setZero();
    Rp.setZero();
    Arr_Rx.setZero();
    Arr_Ru.setZero();
    Arr_Rp.setZero();
  }
  virtual ~ResidualDataAbstractTpl() = default;

  DataCollectorAbstract* shared;  //!< Shared data allocated by the action model
  VectorXs r;                     //!< Residual vector
  MatrixXs Rx;  //!< Jacobian of the residual vector with respect the state
  MatrixXs Ru;  //!< Jacobian of the residual vector with respect the control
  MatrixXs Rp;  //!< Jacobian of the residual vector with respect the parameters
  MatrixXs Arr_Rx;
  MatrixXs Arr_Ru;
  MatrixXs Arr_Rp;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residual-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataAbstractTpl)

#endif  // CROCODDYL_CORE_RESIDUAL_BASE_HPP_
