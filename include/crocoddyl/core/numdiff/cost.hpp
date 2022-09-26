///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, New York University,
//                          Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_COST_HPP_
#define CROCODDYL_CORE_NUMDIFF_COST_HPP_

#include <boost/function.hpp>
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of a cost model.
 *
 * It computes the Jacobian and Hessian of the cost model via numerical differentiation, i.e., \f$\mathbf{\ell_x}\f$,
 * \f$\mathbf{\ell_u}\f$, \f$\mathbf{\ell_{xx}}\f$, \f$\mathbf{\ell_{uu}}\f$, and \f$\mathbf{\ell_{xu}}\f$ which denote
 * the Jacobians and Hessians of the cost function, respectively.
 *
 * \sa `CostModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class CostModelNumDiffTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataNumDiffTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef boost::function<void(const VectorXs&, const VectorXs&)> ReevaluationFunction;

  /**
   * @brief Initialize the numdiff cost model
   *
   * @param model  Cost model that we want to apply the numerical differentiation
   */
  explicit CostModelNumDiffTpl(const boost::shared_ptr<Base>& model);

  /**
   * @brief Initialize the numdiff cost model
   */
  virtual ~CostModelNumDiffTpl();

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>&
   * x)
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create a numdiff cost data
   *
   * @param data  Data collector used by the original model
   * @return the numdiff cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the original cost model
   */
  const boost::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance value used by the numdiff routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance value used by the numdiff routine
   */
  void set_disturbance(const Scalar disturbance);

  /**
   * @brief Identify if the Gauss approximation is going to be used or not.
   *
   * @return true
   * @return false
   */
  bool get_with_gauss_approx();

  /**
   * @brief Register functions that updates the shared data computed for a system rollout
   * The updated data is used to evaluate of the gradient and Hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  /**
   * @brief Make sure that when we finite difference the Cost Model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. This behaviour might occur if CostModelState and
   * FloatingInContact differential model are used together.
   *
   * For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/);

  boost::shared_ptr<Base> model_;              //!< Cost model hat we want to apply the numerical differentiation
  Scalar disturbance_;                         //!< Disturbance used in the numerical differentiation routine
  std::vector<ReevaluationFunction> reevals_;  //!< Functions that needs execution before calc or calcDiff
};

template <typename _Scalar>
struct CostDataNumDiffTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  /**
   * @brief Initialize the numdiff cost data
   *
   * @tparam Model is the type of the `CostModelAbstractTpl`.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit CostDataNumDiffTpl(Model<Scalar>* const model, DataCollectorAbstract* const shared_data)
      : Base(model, shared_data),
        dx(model->get_state()->get_ndx()),
        xp(model->get_state()->get_nx()),
        du(model->get_nu()),
        up(model->get_nu()) {
    dx.setZero();
    xp.setZero();
    du.setZero();
    up.setZero();

    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData(shared_data);
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData(shared_data));
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData(shared_data));
    }
  }

  virtual ~CostDataNumDiffTpl() {}

  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::residual;
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
#include "crocoddyl/core/numdiff/cost.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_COST_HPP_
