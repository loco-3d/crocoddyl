///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_CONSTRAINT_HPP_
#define CROCODDYL_CORE_NUMDIFF_CONSTRAINT_HPP_

#include <boost/function.hpp>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/constraint-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ConstraintModelNumDiffTpl : public ConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef ConstraintModelAbstractTpl<Scalar> Base;
  typedef ConstraintDataNumDiffTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  //   typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef boost::function<void(const VectorXs&)> ReevaluationFunction;

  /**
   * @brief Initialize the numdiff constraint model
   *
   * @param model
   */
  explicit ConstraintModelNumDiffTpl(const boost::shared_ptr<Base>& model);

  /**
   * @brief Initialize the numdiff constraint model
   */
  virtual ~ConstraintModelNumDiffTpl();

  /**
   * @brief @copydoc ConstraintModelAbstract::calc()
   */
  virtual void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc ConstraintModelAbstract::calcDiff()
   */
  virtual void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create a Data object
   *
   * @param data  Data collector used for the original model
   * @return the constraint data
   */
  virtual boost::shared_ptr<ConstraintDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the original model
   */
  const boost::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance value used by the numdiff routine
   */
  const Scalar& get_disturbance() const;

  /**
   * @brief Modify the disturbance value used by the numdiff routine
   */
  void set_disturbance(const Scalar& disturbance);

  /**
   * @brief Register functions that updates the shared data computed for a system rollout
   * The updated data is used to evaluate of the gradient and hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

  /** @brief Model of the cost. */
  boost::shared_ptr<Base> model_;

  /** @brief Numerical disturbance used in the numerical differentiation. */
  Scalar disturbance_;

  /** @brief Functions that needs execution before calc or calcDiff. */
  std::vector<ReevaluationFunction> reevals_;

 private:
  /**
   * @brief Make sure that when we finite difference the constraint model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/);
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
  explicit ConstraintDataNumDiffTpl(Model<Scalar>* const model, DataCollectorAbstract* const shared_data)
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

  VectorXs dx;  //!< State disturbance.
  VectorXs xp;  //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$".
  VectorXs du;  //!< Control disturbance.
  VectorXs up;  //!< The integrated control from the disturbance on one DoF "\f$ \int u du_i = u + du \f$".
  boost::shared_ptr<Base> data_0;                //!< The data at the approximation point.
  std::vector<boost::shared_ptr<Base> > data_x;  //!< The temporary data associated with the state variation.
  std::vector<boost::shared_ptr<Base> > data_u;  //!< The temporary data associated with the control variation.
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/constraint.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_CONSTRAINT_HPP_
