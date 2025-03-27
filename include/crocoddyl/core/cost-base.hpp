///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COST_BASE_HPP_
#define CROCODDYL_CORE_COST_BASE_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

class CostModelBase {
 public:
  virtual ~CostModelBase() = default;

  CROCODDYL_BASE_CAST(CostModelBase, CostModelAbstractTpl)
};

/**
 * @brief Abstract class for cost models
 *
 * A cost model is defined by the scalar activation function \f$a(\cdot)\f$ and
 * by the residual function \f$\mathbf{r}(\cdot)\f$ as follows: \f[
 * \ell(\mathbf{x},\mathbf{u}) = a(\mathbf{r}(\mathbf{x}, \mathbf{u})), \f]
 * where the residual function depends on the state point
 * \f$\mathbf{x}\in\mathcal{X}\f$, which lies in the state manifold described
 * with a `nx`-tuple, its velocity \f$\dot{\mathbf{x}}\in
 * T_{\mathbf{x}}\mathcal{X}\f$ that belongs to the tangent space with `ndx`
 * dimension, and the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$. The
 * residual vector is defined by \f$\mathbf{r}\in\mathbb{R}^{nr}\f$ where `nr`
 * describes its dimension in the Euclidean space. On the other hand, the
 * activation function builds a cost value based on the definition of the
 * residual vector. The residual vector has to be specialized in a derived
 * classes.
 *
 * The main computations are carring out in `calc()` and `calcDiff()` routines.
 * `calc()` computes the cost (and its residual) and `calcDiff()` computes the
 * derivatives of the cost function (and its residual). Concretely speaking,
 * `calcDiff()` builds a linear-quadratic approximation of the cost function
 * with the form: \f$\mathbf{l_x}\in\mathbb{R}^{ndx}\f$,
 * \f$\mathbf{l_u}\in\mathbb{R}^{nu}\f$,
 * \f$\mathbf{l_{xx}}\in\mathbb{R}^{ndx\times ndx}\f$,
 * \f$\mathbf{l_{xu}}\in\mathbb{R}^{ndx\times nu}\f$,
 * \f$\mathbf{l_{uu}}\in\mathbb{R}^{nu\times nu}\f$ are the Jacobians and
 * Hessians, respectively. Additionally, it is important to note that
 * `calcDiff()` computes the derivatives using the latest stored values by
 * `calc()`. Thus, we need to first run `calc()`.
 *
 * \sa `ActivationModelAbstractTpl`, `ResidualModelAbstractTpl` `calc()`,
 * `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelAbstractTpl : public CostModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelAbstractTpl<Scalar> ResidualModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the cost model
   *
   * @param[in] state       State of the dynamical system
   * @param[in] activation  Activation model
   * @param[in] residual    Residual model
   */
  CostModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       std::shared_ptr<ActivationModelAbstract> activation,
                       std::shared_ptr<ResidualModelAbstract> residual);

  /**
   * @brief Initialize the cost model
   *
   * @param[in] state       State of the dynamical system
   * @param[in] activation  Activation model
   * @param[in] nu          Dimension of control vector
   */
  CostModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       std::shared_ptr<ActivationModelAbstract> activation,
                       const std::size_t nu);

  /**
   * @copybrief CostModelAbstractTpl()
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the dynamical system
   * @param[in] activation  Activation model
   */
  CostModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       std::shared_ptr<ActivationModelAbstract> activation);

  /**
   * @copybrief CostModelAbstractTpl()
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e.,
   * \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$)
   *
   * @param[in] state     State of the dynamical system
   * @param[in] residual  Residual model
   */
  CostModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       std::shared_ptr<ResidualModelAbstract> residual);

  /**
   * @copybrief CostModelAbstractTpl()
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e.,
   * \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$)
   *
   * @param[in] state  State of the system
   * @param[in] nr     Dimension of residual vector
   * @param[in] nu     Dimension of control vector
   */
  CostModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t nr, const std::size_t nu);

  /**
   * @copybrief CostModelAbstractTpl()
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e.,
   * \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$). Furthermore, the default `nu` value
   * is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the dynamical system
   * @param[in] nr     Dimension of residual vector
   * @param[in] nu     Dimension of control vector
   */
  CostModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t nr);
  virtual ~CostModelAbstractTpl() = default;

  /**
   * @brief Compute the cost value and its residual vector
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<CostDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the total cost value for nodes that depends only on the
   * state
   *
   * It updates the total cost based on the state only. This function is used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<CostDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobian and Hessian of cost and its residual vector
   *
   * It computes the Jacobian and Hessian of the cost function. It assumes that
   * `calc()` has been run first.
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the Jacobian and Hessian of the cost functions with respect
   * to the state only
   *
   * It updates the Jacobian and Hessian of the cost function based on the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the cost data
   *
   * The default data contains objects to store the values of the cost, residual
   * vector and their derivatives (first and second order derivatives). However,
   * it is possible to specialize this function if we need to create additional
   * data, for instance, to avoid dynamic memory allocation.
   *
   * @param data  Data collector
   * @return the cost data
   */
  virtual std::shared_ptr<CostDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the activation model
   */
  const std::shared_ptr<ActivationModelAbstract>& get_activation() const;

  /**
   * @brief Return the residual model
   */
  const std::shared_ptr<ResidualModelAbstract>& get_residual() const;

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Print information on the cost model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const CostModelAbstractTpl<Scalar>& model);

  /**
   * @brief Modify the cost reference
   */
  template <class ReferenceType>
  void set_reference(ReferenceType ref);

  /**
   * @brief Return the cost reference
   */
  template <class ReferenceType>
  ReferenceType get_reference();

  /**
   * @brief Print relevant information of the cost model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  /**
   * @copybrief set_reference()
   */
  virtual void set_referenceImpl(const std::type_info&, const void*);

  /**
   * @copybrief get_reference()
   */
  virtual void get_referenceImpl(const std::type_info&, void*);

  std::shared_ptr<StateAbstract> state_;                 //!< State description
  std::shared_ptr<ActivationModelAbstract> activation_;  //!< Activation model
  std::shared_ptr<ResidualModelAbstract> residual_;      //!< Residual model
  std::size_t nu_;                                       //!< Control dimension
  VectorXs unone_;                                       //!< No control vector
  CostModelAbstractTpl()
      : state_(nullptr), activation_(nullptr), residual_(nullptr), nu_(0) {}
};

template <typename _Scalar>
struct CostDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataAbstractTpl(Model<Scalar>* const model,
                      DataCollectorAbstract* const data)
      : shared(data),
        activation(model->get_activation()->createData()),
        residual(model->get_residual()->createData(data)),
        cost(Scalar(0.)),
        Lx(model->get_state()->get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()) {
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
  }
  virtual ~CostDataAbstractTpl() = default;

  DEPRECATED(
      "Use residual.r", const VectorXs& get_r() const { return residual->r; };)
  DEPRECATED(
      "Use residual.Rx",
      const MatrixXs& get_Rx() const { return residual->Rx; };)
  DEPRECATED(
      "Use residual.Ru",
      const MatrixXs& get_Ru() const { return residual->Ru; };)
  DEPRECATED(
      "Use residual.r", void set_r(const VectorXs& r) { residual->r = r; };)
  DEPRECATED(
      "Use residual.Rx",
      void set_Rx(const MatrixXs& Rx) { residual->Rx = Rx; };)
  DEPRECATED(
      "Use residual.Ru",
      void set_Ru(const MatrixXs& Ru) { residual->Ru = Ru; };)

  DataCollectorAbstract* shared;
  std::shared_ptr<ActivationDataAbstract> activation;
  std::shared_ptr<ResidualDataAbstract> residual;
  Scalar cost;
  VectorXs Lx;
  VectorXs Lu;
  MatrixXs Lxx;
  MatrixXs Lxu;
  MatrixXs Luu;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/cost-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::CostModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::CostDataAbstractTpl)

#endif  // CROCODDYL_CORE_COST_BASE_HPP_
