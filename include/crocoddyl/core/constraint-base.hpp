///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
#define CROCODDYL_CORE_CONSTRAINT_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/data-collector-base.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for constraint models
 *
 * In Crocoddyl, a constraint model defines both: inequality \f$\mathbf{g}(\mathbf{x}, \mathbf{u})\in\mathbb{R}^{ng}\f$
 * and equality \f$\mathbf{h}(\mathbf{x}, \mathbf{u})\in\mathbb{R}^{nh}\f$ constraints.
 * The constraint function depends on the state point \f$\mathbf{x}\in\mathcal{X}\f$, which lies in the state manifold
 * described with a `nq`-tuple, its velocity \f$\dot{\mathbf{x}}\in T_{\mathbf{x}}\mathcal{X}\f$ that belongs to
 * the tangent space with `nv` dimension, and the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$.
 *
 * The main computations are carring out in `calc` and `calcDiff` routines. `calc` computes the constraint residual and
 * `calcDiff` computes the Jacobians of the constraint function. Concretely speaking, `calcDiff` builds
 * a linear approximation of the constraint function with the form: \f$\mathbf{g_x}\in\mathbb{R}^{ng\times ndx}\f$,
 * \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$, \f$\mathbf{h_x}\in\mathbb{R}^{nh\times ndx}\f$
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$.
 *
 * \sa `StateAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ConstraintModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the constraint model
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of control vector
   * @param[in] ng     Number of inequality constraints
   * @param[in] nh     Number of equality constraints
   */
  ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t& nu, const std::size_t& ng,
                             const std::size_t& nh);

  /**
   * @copybrief ConstraintModelAbstractTpl()
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] ng     Number of inequality constraints
   * @param[in] nh     Number of equality constraints
   */
  ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t& ng, const std::size_t& nh);
  virtual ~ConstraintModelAbstractTpl();

  /**
   * @brief Compute the constraint value and its residual vector
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the Jacobian of the constraint
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Create the constraint data
   *
   * The default data contains objects to store the values of the constraint, residual vector and their first
   * derivatives. However, it is possible to specialized this function is we need to create additional data, for
   * instance, to avoid dynamic memory allocation.
   *
   * @param data  Data collector
   * @return the constraint data
   */
  virtual boost::shared_ptr<ConstraintDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @copybrief calc()
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point
   */
  void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @copybrief calcDiff()
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point
   */
  void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Return the state
   */
  const boost::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the dimension of the control input
   */
  const std::size_t& get_nu() const;

  /**
   * @brief Return the number of inequality constraints
   */
  const std::size_t& get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  const std::size_t& get_nh() const;

  /**
   * @brief Modify the constraint reference
   */
  template <class ReferenceType>
  void set_reference(ReferenceType ref);

  /**
   * @brief Return the constraint reference
   */
  template <class ReferenceType>
  ReferenceType get_reference() const;

 protected:
  /**
   * @copybrief set_reference()
   */
  virtual void set_referenceImpl(const std::type_info&, const void*);

  /**
   * @copybrief get_reference()
   */
  virtual void get_referenceImpl(const std::type_info&, void*) const;

  boost::shared_ptr<StateAbstract> state_;  //!< State description
  std::size_t nu_;                          //!< Control dimension
  std::size_t ng_;                          //!< Number of inequality constraints
  std::size_t nh_;                          //!< Number of equality constraints
  VectorXs unone_;                          //!< No control vector
};

template <typename _Scalar>
struct ConstraintDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ConstraintDataAbstractTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : shared(data),
        g(model->get_ng()),
        Gx(model->get_ng(), model->get_state()->get_ndx()),
        Gu(model->get_ng(), model->get_nu()),
        h(model->get_nh()),
        Hx(model->get_nh(), model->get_state()->get_ndx()),
        Hu(model->get_nh(), model->get_nu()) {
    g.setZero();
    Gx.setZero();
    Gu.setZero();
    h.setZero();
    Hx.setZero();
    Hu.setZero();
  }
  virtual ~ConstraintDataAbstractTpl() {}

  DataCollectorAbstract* shared;  //!< Shared data
  VectorXs g;                     //!< Inequality constraint values
  MatrixXs Gx;                    //!< Jacobian of the inequality constraint
  MatrixXs Gu;                    //!< Jacobian of the inequality constraint
  VectorXs h;                     //!< Equality constraint values
  MatrixXs Hx;                    //!< Jacobian of the equality constraint
  MatrixXs Hu;                    //!< Jacobian of the equality constraint
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/constraint-base.hxx"

#endif  // CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
