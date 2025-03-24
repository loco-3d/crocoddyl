///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_CONTROL_GRAVITY_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_CONTROL_GRAVITY_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Control gravity residual under contact
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{u}-(\mathbf{g}(\mathbf{q}) - \sum
 * \mathbf{J}_c(\mathbf{q})^{\top} \mathbf{f}_c)\f$, where
 * \f$\mathbf{u}\in~\mathbb{R}^{nu}\f$ is the current control input,
 * \f$\mathbf{J}_c(\mathbf{q})\f$ is the contact Jacobians, \f$\mathbf{f}_c\f$
 * contains the contact forces, \f$\mathbf{g}(\mathbf{q})\f$ is the gravity
 * torque corresponding to the current configuration,
 * \f$\mathbf{q}\in~\mathbb{R}^{nq}\f$ is the current position joints input.
 * Note that the dimension of the residual vector is obtained from
 * `state->get_nv()`.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * Jacobians are calculated by `calc()` and `calcDiff()`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelContactControlGravTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelContactControlGravTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactControlGravTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact control gravity contact residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelContactControlGravTpl(std::shared_ptr<StateMultibody> state,
                                     const std::size_t nu);

  /**
   * @brief Initialize the contact control gravity contact residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   */
  explicit ResidualModelContactControlGravTpl(
      std::shared_ptr<StateMultibody> state);
  virtual ~ResidualModelContactControlGravTpl() = default;

  /**
   * @brief Compute the contact control gravity contact residual
   *
   * @param[in] data  Contact control gravity residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the residual vector for nodes that depends only on the state
   *
   * It updates the residual vector based on the state only (i.e., it ignores
   * the contact forces). This function is used in the terminal nodes of an
   * optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the Jacobians of the contact control gravity contact
   * residual
   *
   * @param[in] data  Contact control gravity residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the Jacobian of the residual functions with respect to the
   * state only
   *
   * It updates the Jacobian of the residual function based on the state only
   * (i.e., it ignores the contact forces). This function is used in the
   * terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the contact-control-gravity residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the contact-control-gravity residual model to a different
   * scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelContactControlGravTpl<NewScalar> A residual model with
   * the new scalar type.
   */
  template <typename NewScalar>
  ResidualModelContactControlGravTpl<NewScalar> cast() const;

  /**
   * @brief Print relevant information of the contact-control-grav residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::v_dependent_;

 private:
  typename StateMultibody::PinocchioModel pin_model_;
};

template <typename _Scalar>
struct ResidualDataContactControlGravTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::DataTpl<Scalar> PinocchioData;

  template <template <typename Scalar> class Model>
  ResidualDataContactControlGravTpl(Model<Scalar>* const model,
                                    DataCollectorAbstract* const data)
      : Base(model, data) {
    StateMultibody* sm = static_cast<StateMultibody*>(model->get_state().get());
    pinocchio = PinocchioData(*(sm->get_pinocchio().get()));

    // Check that proper shared data has been passed
    DataCollectorActMultibodyInContactTpl<Scalar>* d =
        dynamic_cast<DataCollectorActMultibodyInContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorActMultibodyInContactTpl");
    }
    // Avoids data casting at runtime
    // pinocchio = d->pinocchio;
    fext = d->contacts->fext;
    actuation = d->actuation;
  }
  virtual ~ResidualDataContactControlGravTpl() = default;

  PinocchioData pinocchio;  //!< Pinocchio data
  std::shared_ptr<ActuationDataAbstractTpl<Scalar> >
      actuation;  //!< Actuation data
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >
      fext;  //!< External spatial forces
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-control-gravity.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelContactControlGravTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataContactControlGravTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_CONTROL_GRAVITY_HPP_
