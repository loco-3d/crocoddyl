///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_CONSTRAINT_HPP_
#define CROCODDYL_CORE_NUMDIFF_CONSTRAINT_HPP_

#include <boost/function.hpp>

#include "crocoddyl/core/constraint-base.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of a constraint
 * model.
 *
 * It computes the Jacobian of the constraint model via numerical
 * differentiation, i.e., \f$\mathbf{g_x}\f$, \f$\mathbf{g_u}\f$ and
 * \f$\mathbf{h_x}\f$, \f$\mathbf{h_u}\f$, which denote the Jacobians of the
 * inequality and equality constraints, respectively.
 *
 * \sa `ConstraintModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class ConstraintModelNumDiffTpl : public ConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ConstraintModelBase, ConstraintModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef ConstraintModelAbstractTpl<Scalar> Base;
  typedef ConstraintDataNumDiffTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef boost::function<void(const VectorXs&, const VectorXs&)>
      ReevaluationFunction;

  /**
   * @brief Initialize the numdiff constraint model
   *
   * @param model
   */
  explicit ConstraintModelNumDiffTpl(const std::shared_ptr<Base>& model);

  /**
   * @brief Initialize the numdiff constraint model
   */
  virtual ~ConstraintModelNumDiffTpl() = default;

  /**
   * @brief @copydoc ConstraintModelAbstract::calc()
   */
  virtual void calc(const std::shared_ptr<ConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc ConstraintModelAbstract::calc(const
   * std::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc ConstraintModelAbstract::calcDiff()
   */
  virtual void calcDiff(const std::shared_ptr<ConstraintDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc ConstraintModelAbstract::calcDiff(const
   * std::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calcDiff(const std::shared_ptr<ConstraintDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::createData()
   */
  virtual std::shared_ptr<ConstraintDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the constraint numdiff model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ConstraintModelNumDiffTpl<NewScalar> A constraint model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ConstraintModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Return the original constraint model
   */
  const std::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance constant used by the numerical
   * differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance constant used by the numerical
   * differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

  /**
   * @brief Register functions that updates the shared data computed for a
   * system rollout The updated data is used to evaluate of the gradient and
   * Hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  /**
   * @brief Make sure that when we finite difference the constraint model, the
   * user does not face unknown behaviour because of the finite differencing of
   * a quaternion around pi. For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/);

  std::shared_ptr<Base> model_;  //!< Constraint model hat we want to apply
                                 //!< the numerical differentiation
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation
  std::vector<ReevaluationFunction>
      reevals_;  //!< Functions that needs execution before calc or calcDiff
};

template <typename _Scalar>
struct ConstraintDataNumDiffTpl : public ConstraintDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit ConstraintDataNumDiffTpl(Model<Scalar>* const model,
                                    DataCollectorAbstract* const shared_data)
      : Base(model, shared_data),
        dx(model->get_state()->get_ndx()),
        xp(model->get_state()->get_nx()),
        du(model->get_nu()),
        up(model->get_nu()) {
    dx.setZero();
    xp.setZero();
    du.setZero();
    up.setZero();

    const std::size_t& ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t& nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData(shared_data);
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData(shared_data));
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData(shared_data));
    }
  }

  virtual ~ConstraintDataNumDiffTpl() {}

  using Base::Gu;
  using Base::Gx;
  using Base::Hu;
  using Base::Hx;
  using Base::shared;

  Scalar x_norm;  //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  Scalar
      uh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{u} \f$
  VectorXs dx;  //!< State disturbance.
  VectorXs xp;  //!< The integrated state from the disturbance on one DoF "\f$
                //!< \int x dx_i \f$".
  VectorXs du;  //!< Control disturbance.
  VectorXs up;  //!< The integrated control from the disturbance on one DoF "\f$
                //!< \int u du_i = u + du \f$".
  std::shared_ptr<Base> data_0;  //!< The data at the approximation point.
  std::vector<std::shared_ptr<Base> >
      data_x;  //!< The temporary data associated with the state variation.
  std::vector<std::shared_ptr<Base> >
      data_u;  //!< The temporary data associated with the control variation.
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/constraint.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_CONSTRAINT_HPP_
