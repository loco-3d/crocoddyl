///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTION_HPP_

#include "crocoddyl/core/action-base.hpp"
#include <vector>
#include <iostream>

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of an ActionModel.
 *
 * It computes the same quantity as a normal model would do but using numerical
 * differentiation.
 * The subtility is in the computation of the Hessian of the cost. Let us
 * consider that the ActionModel owns a cost residual. This means that the cost
 * is the square of a residual \f$ l(x,u) = .5 r(x,u)**2 \f$, with
 * \f$ r(x,u) \f$ being the residual vector. Therefore the derivatives of the
 * cost \f$ l \f$ can be expressed in function of the derivatives of the
 * residuals (Jacobians), denoted by \f$ R_x \f$ and \f$ R_u \f$. Which would be:
 * \f{eqnarray*}{
 *     L_x    &=& R_x^T r \\
 *     L_u    &=& R_u^T r \\
 *     L_{xx} &=& R_x^T R_x + R_{xx} r
 * \f}
 * with \f$ R_{xx} \f$ the derivatives of the Jacobian (i.e. not a matrix, but a
 * dim-3 tensor). The Gauss approximation consists in neglecting these.
 * So \f$ L_{xx} \sim R_x^T R_x \f$. Similarly for \f$ L_{xu} \sim R_x^T R_u \f$
 * and \f$ L_{uu} \sim R_u^T R_u \f$. The above set of equations becomes:
 * \f{eqnarray*}{
 *     L_x    &=& R_x^T r \\
 *     L_u    &=& R_u^T r \\
 *     L_{xx} &\sim& R_x^T R_x \\
 *     L_{xu} &\sim& R_x^T R_u \\
 *     L_{uu} &\sim& R_u^T R_u
 * \f}
 * In the case that the cost does not have a residual we set the Hessian to
 * \f$ 0 \f$, i.e. \f$ L_{xx} = L_{xu} = L_{uu} = 0 \f$.
 */
class ActionModelNumDiff : public ActionModelAbstract {
 public:
  /**
   * @brief Construct a new ActionModelNumDiff object
   *
   * @param model
   */
  explicit ActionModelNumDiff(boost::shared_ptr<ActionModelAbstract> model);

  /**
   * @brief Destroy the ActionModelNumDiff object
   */
  ~ActionModelNumDiff();

  /**
   * @brief @copydoc ActionModelAbstract::calc()
   */
  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);

  /**
   * @brief @copydoc ActionModelAbstract::calcDiff()
   */
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

  /**
   * @brief Create a Data object from the given model.
   *
   * @return boost::shared_ptr<ActionDataAbstract>
   */
  boost::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Get the model_ object
   *
   * @return ActionModelAbstract&
   */
  const boost::shared_ptr<ActionModelAbstract>& get_model() const;

  /**
   * @brief Get the disturbance_ object
   *
   * @return const double&
   */
  const double& get_disturbance() const;

  /**
   * @brief Set the disturbance_ object
   *
   * @param disturbance is the value used to find the numerical derivative
   */
  void set_disturbance(const double& disturbance);

  /**
   * @brief Identify if the Gauss approximation is going to be used or not.
   *
   * @return true
   * @return false
   */
  bool get_with_gauss_approx();

 private:
  /**
   * @brief Make sure that when we finite difference the Action Model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. This behaviour might occur if CostModelState and
   * FloatingInContact differential model are used together.
   *
   * For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
   * @brief This is the model to compute the finite differenciation from
   */
  boost::shared_ptr<ActionModelAbstract> model_;

  /**
   * @brief This is the numerical disturbance value used during the numerical
   * differenciations
   */
  double disturbance_;
};

struct ActionDataNumDiff : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a new ActionDataNumDiff object
   *
   * @tparam Model is the type of the ActionModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <typename Model>
  explicit ActionDataNumDiff(Model* const model)
      : ActionDataAbstract(model),
        Rx(model->get_model()->get_nr(), model->get_model()->get_state()->get_ndx()),
        Ru(model->get_model()->get_nr(), model->get_model()->get_nu()),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    du.setZero();
    xp.setZero();

    const std::size_t& ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t& nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

  Eigen::MatrixXd Rx;  //!< Cost residual jacobian: \f$ \frac{d r(x,u)}{dx} \f$
  Eigen::MatrixXd Ru;  //!< Cost residual jacobian: \f$ \frac{d r(x,u)}{du} \f$
  Eigen::VectorXd dx;  //!< State disturbance
  Eigen::VectorXd du;  //!< Control disturbance
  Eigen::VectorXd xp;  //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$"
  boost::shared_ptr<ActionDataAbstract> data_0;  //!< The data that contains the final results
  std::vector<boost::shared_ptr<ActionDataAbstract> >
      data_x;  //!< The temporary data associated with the state variation
  std::vector<boost::shared_ptr<ActionDataAbstract> >
      data_u;  //!< The temporary data associated with the control variation
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
