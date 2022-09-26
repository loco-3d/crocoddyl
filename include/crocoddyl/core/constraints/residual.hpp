///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONSTRAINTS_RESIDUAL_CONSTRAINT_HPP_
#define CROCODDYL_CORE_CONSTRAINTS_RESIDUAL_CONSTRAINT_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/constraint-base.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Residual-based constraint
 *
 * This constraint function uses a residual model to define equality / inequality constraint as \f[
 * \mathbf{\underline{r}} \leq \mathbf{r}(\mathbf{x}, \mathbf{u}) \leq \mathbf{\bar{r}} \f] where
 * \f$\mathbf{r}(\cdot)\f$ describes the residual function, and \f$\mathbf{\underline{r}}\f$, \f$\mathbf{\bar{r}}\f$
 * are the lower and upper bounds, respectively. We can define element-wise equality constraints by defining the same
 * value for both: lower and upper values. Additionally, if we do not define the bounds, then it is assumed that
 * \f$\mathbf{\underline{r}}=\mathbf{\bar{r}}=\mathbf{0}\f$.
 *
 * The main computations are carring out in `calc` and `calcDiff` routines. `calc` computes the constraint residual and
 * `calcDiff` computes the Jacobians of the constraint function. Concretely speaking, `calcDiff` builds
 * a linear approximation of the constraint function with the form: \f$\mathbf{g_x}\in\mathbb{R}^{ng\times ndx}\f$,
 * \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$, \f$\mathbf{h_x}\in\mathbb{R}^{nh\times ndx}\f$
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$.
 * Additionally, it is important remark that `calcDiff()` computes the derivatives using the latest stored values by
 * `calc()`. Thus, we need to run first `calc()`.
 *
 * \sa `ConstraintModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ConstraintModelResidualTpl : public ConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintModelAbstractTpl<Scalar> Base;
  typedef ConstraintDataResidualTpl<Scalar> Data;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef ResidualModelAbstractTpl<Scalar> ResidualModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBaseTpl<std::size_t>::ArrayXs ArrayXi;

  /**
   * @brief Initialize the residual constraint model
   *
   * @param[in] state       State of the multibody system
   * @param[in] residual    Residual model
   * @param[in] lower       Lower bound (dimension of the residual vector)
   * @param[in] upper       Upper bound (dimension of the residual vector)
   */
  ConstraintModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                             boost::shared_ptr<ResidualModelAbstract> residual, const VectorXs& lower,
                             const VectorXs& upper);

  /**
   * @brief Initialize the residual constraint model as equality one
   *
   * @param[in] state       State of the multibody system
   * @param[in] residual    Residual model
   */
  ConstraintModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                             boost::shared_ptr<ResidualModelAbstract> residual);
  virtual ~ConstraintModelResidualTpl();

  /**
   * @brief Compute the residual constraint
   *
   * @param[in] data  Residual constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the residual constraint based on state only
   *
   * It updates the constraint based on the state only. This function is commonly used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the residual constraint
   *
   * @param[in] data  Residual constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the residual constraint with respect to the state only
   *
   * It updates the Jacobian of the constraint function based on the state only. This function is commonly used
   * in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the residual constraint data
   */
  virtual boost::shared_ptr<ConstraintDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the upper bound of the friction cone
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Return the lower bound of the friction cone
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Update the lower and upper bounds the upper bound of the friction cone
   */
  void update_bounds(const VectorXs& lower, const VectorXs& upper);

  /**
   * @brief Print relevant information of the cost-residual model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 private:
  void updateConstraintType();

 protected:
  using Base::ng_;
  using Base::nh_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;
  VectorXs lb_;              //!< Lower bound of the constraint
  VectorXs ub_;              //!< Upper bound of the constraint
  ArrayXi constraint_type_;  //!< Define the type of contraint: equality=0, inequality=1, lower_inequality=2,
                             //!< upper_inequality=3, no_constraint=4
};

template <typename _Scalar>
struct ConstraintDataResidualTpl : public ConstraintDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  ConstraintDataResidualTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {}

  using Base::g;
  using Base::Gu;
  using Base::Gx;
  using Base::h;
  using Base::Hu;
  using Base::Hx;
  using Base::residual;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/constraints/residual.hxx"

#endif  // CROCODDYL_CORE_CONSTRAINTS_RESIDUAL_CONSTRAINT_HPP_
