///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
#define CROCODDYL_CORE_CONSTRAINT_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
//
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

enum ConstraintType { Inequality = 0, Equality, Both };

class ConstraintModelBase {
 public:
  virtual ~ConstraintModelBase() = default;

  CROCODDYL_BASE_CAST(ConstraintModelBase, ConstraintModelAbstractTpl)
};

/**
 * @brief Abstract class for constraint models
 *
 * A constraint model defines both: inequality \f$\mathbf{g}(\mathbf{x},
 * \mathbf{u})\in\mathbb{R}^{ng}\f$ and equality \f$\mathbf{h}(\mathbf{x},
 * \mathbf{u})\in\mathbb{R}^{nh}\f$ constraints. The constraint function depends
 * on the state point \f$\mathbf{x}\in\mathcal{X}\f$, which lies in the state
 * manifold described with a `nx`-tuple, its velocity \f$\dot{\mathbf{x}}\in
 * T_{\mathbf{x}}\mathcal{X}\f$ that belongs to the tangent space with `ndx`
 * dimension, and the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$.
 *
 * The main computations are carried out in `calc()` and `calcDiff()` routines.
 * `calc()` computes the constraint residual and `calcDiff()` computes the
 * Jacobians of the constraint function. Concretely speaking, `calcDiff()`
 * builds a linear approximation of the constraint function with the form:
 * \f$\mathbf{g_x}\in\mathbb{R}^{ng\times ndx}\f$,
 * \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$,
 * \f$\mathbf{h_x}\in\mathbb{R}^{nh\times ndx}\f$
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$. Additionally, it is important
 * to note that `calcDiff()` computes the derivatives using the latest stored
 * values by `calc()`. Thus, we need to first run `calc()`.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ConstraintModelAbstractTpl : public ConstraintModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ResidualModelAbstractTpl<Scalar> ResidualModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the constraint model
   *
   * @param[in] state     State of the multibody system
   * @param[in] residual  Residual model
   * @param[in] ng        Number of inequality constraints
   * @param[in] nh        Number of equality constraints
   */
  ConstraintModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                             std::shared_ptr<ResidualModelAbstract> residual,
                             const std::size_t ng, const std::size_t nh);

  /**
   * @copybrief Initialize the constraint model
   *
   * @param[in] state    State of the multibody system
   * @param[in] nu       Dimension of control vector
   * @param[in] ng       Number of inequality constraints
   * @param[in] nh       Number of equality constraints
   * @param[in] T_const  True if this is a constraint in both running and
   * terminal nodes. False if it is a constraint on running nodes only (default
   * true)
   */
  ConstraintModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                             const std::size_t nu, const std::size_t ng,
                             const std::size_t nh, const bool T_const = true);

  /**
   * @copybrief ConstraintModelAbstractTpl()
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state    State of the multibody system
   * @param[in] ng       Number of inequality constraints
   * @param[in] nh       Number of equality constraints
   * @param[in] T_const  True if this is a constraint in both running and
   * terminal nodes. False if it is a constraint on running nodes only (default
   * true)
   */
  ConstraintModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                             const std::size_t ng, const std::size_t nh,
                             const bool T_const = true);
  virtual ~ConstraintModelAbstractTpl() = default;

  /**
   * @brief Compute the constraint value
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the constraint value for nodes that depends only on the
   * state
   *
   * It updates the constraint based on the state only. This function is
   * commonly used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobian of the constraint
   *
   * It computes the Jacobian of the constraint function. It assumes that
   * `calc()` has been run first.
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ConstraintDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the Jacobian of the constraint with respect to the state
   * only
   *
   * It computes the Jacobian of the constraint function based on the state
   * only. This function is commonly used in the terminal nodes of an optimal
   * control problem.
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ConstraintDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the constraint data
   *
   * The default data contains objects to store the values of the constraint,
   * residual vector and their first derivatives. However, it is possible to
   * specialize this function is we need to create additional data, for
   * instance, to avoid dynamic memory allocation.
   *
   * @param data  Data collector
   * @return the constraint data
   */
  virtual std::shared_ptr<ConstraintDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Update the lower and upper bounds the upper bound of constraint
   */
  void update_bounds(const VectorXs& lower, const VectorXs& upper);

  /**
   * @brief Remove the bounds of the constraint
   */
  void remove_bounds();

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the residual model
   */
  const std::shared_ptr<ResidualModelAbstract>& get_residual() const;

  /**
   * @brief Return the type of constraint
   */
  ConstraintType get_type() const;

  /**
   * @brief Return the lower bound of the constraint
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the upper bound of the constraint
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the number of inequality constraints
   */
  std::size_t get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  std::size_t get_nh() const;

  /**
   * @brief Return true if the constraint is imposed in terminal nodes as well.
   */
  bool get_T_constraint() const;

  /**
   * @brief Print information on the constraint model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const CostModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the constraint model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 private:
  std::size_t ng_internal_;  //!< Number of inequality constraints defined at
                             //!< construction time
  std::size_t nh_internal_;  //!< Number of equality constraints defined at
                             //!< construction time

 protected:
  std::shared_ptr<StateAbstract> state_;             //!< State description
  std::shared_ptr<ResidualModelAbstract> residual_;  //!< Residual model
  ConstraintType
      type_;           //!< Type of constraint: inequality=0, equality=1, both=2
  VectorXs lb_;        //!< Lower bound of the constraint
  VectorXs ub_;        //!< Upper bound of the constraint
  std::size_t nu_;     //!< Control dimension
  std::size_t ng_;     //!< Number of inequality constraints
  std::size_t nh_;     //!< Number of equality constraints
  bool T_constraint_;  //!< Label that indicates if the constraint is imposed in
                       //!< terminal nodes as well
  VectorXs unone_;     //!< No control vector
  ConstraintModelAbstractTpl()
      : state_(nullptr), residual_(nullptr), nu_(0), ng_(0), nh_(0) {}
};

template <typename _Scalar>
struct ConstraintDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ConstraintDataAbstractTpl(Model<Scalar>* const model,
                            DataCollectorAbstract* const data)
      : shared(data),
        residual(model->get_residual()->createData(data)),
        g(model->get_ng()),
        Gx(model->get_ng(), model->get_state()->get_ndx()),
        Gu(model->get_ng(), model->get_nu()),
        h(model->get_nh()),
        Hx(model->get_nh(), model->get_state()->get_ndx()),
        Hu(model->get_nh(), model->get_nu()) {
    if (model->get_ng() == 0 && model->get_nh() == 0) {
      throw_pretty("Invalid argument: " << "ng and nh cannot be equals to 0");
    }
    g.setZero();
    Gx.setZero();
    Gu.setZero();
    h.setZero();
    Hx.setZero();
    Hu.setZero();
  }
  virtual ~ConstraintDataAbstractTpl() = default;

  DataCollectorAbstract* shared;                   //!< Shared data
  std::shared_ptr<ResidualDataAbstract> residual;  //!< Residual data
  VectorXs g;   //!< Inequality constraint values
  MatrixXs Gx;  //!< Jacobian of the inequality constraint
  MatrixXs Gu;  //!< Jacobian of the inequality constraint
  VectorXs h;   //!< Equality constraint values
  MatrixXs Hx;  //!< Jacobian of the equality constraint
  MatrixXs Hu;  //!< Jacobian of the equality constraint
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/constraint-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ConstraintModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ConstraintDataAbstractTpl)

#endif  // CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
