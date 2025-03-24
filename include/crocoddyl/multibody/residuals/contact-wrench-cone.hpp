///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_WRENCH_CONE_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"

namespace crocoddyl {

/**
 * @brief Contact wrench cone residual function
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\f$, where \f$\mathbf{A}\f$ is
 * the inequality matrix defined by the contact wrench cone, and
 * \f$\boldsymbol{\lambda}\f$ is the current spatial forces. The current spatial
 * forces \f$\boldsymbol{\lambda}\in\mathbb{R}^{nc}\f$ is computed by
 * `DifferentialActionModelContactFwdDynamicsTpl` or
 * `ActionModelImpulseFwdDynamicTpl`, with `nc` as the dimension of the contact.
 *
 * Both residual and residual Jacobians are computed analytically, where the
 * force vector \f$\boldsymbol{\lambda}\f$ and its Jacobians
 * \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are
 * computed by `DifferentialActionModelContactFwdDynamicsTpl` or
 * `ActionModelImpulseFwdDynamicTpl`. These values are stored in a shared data
 * (i.e., `DataCollectorContactTpl` or `DataCollectorImpulseTpl`). Note that
 * this residual function cannot be used with other action models.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`,
 * `DifferentialActionModelContactFwdDynamicsTpl`,
 * `ActionModelImpulseFwdDynamicTpl`, `DataCollectorForceTpl`
 */
template <typename _Scalar>
class ResidualModelContactWrenchConeTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelContactWrenchConeTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactWrenchConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef WrenchConeTpl<Scalar> WrenchCone;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX6s MatrixX6s;

  /**
   * @brief Initialize the contact wrench cone residual model
   *
   * Note that for the inverse-dynamic cases, the control vector contains the
   * generalized accelerations, torques, and all the contact forces.
   *
   * @param[in] state   Multibody state
   * @param[in] id      Reference frame id
   * @param[in] fref    Reference contact wrench cone
   * @param[in] nu      Dimension of control vector
   * @param[in] fwddyn  Indicates that we have a forward dynamics problem (true)
   * or inverse dynamics (false)
   */
  ResidualModelContactWrenchConeTpl(std::shared_ptr<StateMultibody> state,
                                    const pinocchio::FrameIndex id,
                                    const WrenchCone& fref,
                                    const std::size_t nu,
                                    const bool fwddyn = true);

  /**
   * @brief Initialize the contact wrench cone residual model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`. Note that
   * this constructor can be used for forward-dynamics cases only.
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference contact wrench cone
   */
  ResidualModelContactWrenchConeTpl(std::shared_ptr<StateMultibody> state,
                                    const pinocchio::FrameIndex id,
                                    const WrenchCone& fref);
  virtual ~ResidualModelContactWrenchConeTpl() = default;

  /**
   * @brief Compute the contact wrench cone residual
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the
   * force vector is computed by `DifferentialActionModelContactFwdDynamicsTpl`
   * or `ActionModelImpulseFwdDynamicTpl` which is stored in
   * `DataCollectorContactTpl` or `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Contact force data
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
   * @brief Compute the derivatives of the contact wrench cone residual
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the
   * force vector is computed by `DifferentialActionModelContactFwdDynamicsTpl`
   * or `ActionModelImpulseFwdDynamicTpl` which is stored in
   * `DataCollectorContactTpl` or `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Contact force data
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
   * @brief Create the contact wrench cone residual data
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the residual data.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Update the Jacobians of the contact friction cone residual
   *
   * @param[in] data  Contact friction cone residual data
   */
  void updateJacobians(const std::shared_ptr<ResidualDataAbstract>& data);

  /**
   * @brief Cast the contact-wrench-cone residual model to a different scalar
   * type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelContactWrenchConeTpl<NewScalar> A residual model with
   * the new scalar type.
   */
  template <typename NewScalar>
  ResidualModelContactWrenchConeTpl<NewScalar> cast() const;

  /**
   * @brief Indicates if we are using the forward-dynamics (true) or
   * inverse-dynamics (false)
   */
  bool is_fwddyn() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference contact wrench cone
   */
  const WrenchCone& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  DEPRECATED("Do not use set_id, instead create a new model",
             void set_id(const pinocchio::FrameIndex id);)

  /**
   * @brief Modify the reference contact wrench cone
   */
  void set_reference(const WrenchCone& reference);

  /**
   * @brief Print relevant information of the contact-wrench-cone residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;

 private:
  bool fwddyn_;  //!< Indicates if we are using this function for forward
                 //!< dynamics
  bool update_jacobians_;     //!< Indicates if we need to update the Jacobians
                              //!< (used for inverse dynamics case)
  pinocchio::FrameIndex id_;  //!< Reference frame id
  WrenchCone fref_;           //!< Reference contact wrench cone
};

template <typename _Scalar>
struct ResidualDataContactWrenchConeTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ResidualDataContactWrenchConeTpl(Model<Scalar>* const model,
                                   DataCollectorAbstract* const data)
      : Base(model, data) {
    // Check that proper shared data has been passed
    bool is_contact = true;
    DataCollectorContactTpl<Scalar>* d1 =
        dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    DataCollectorImpulseTpl<Scalar>* d2 =
        dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d1 == NULL && d2 == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorContact or DataCollectorImpulse");
    }
    if (d2 != NULL) {
      is_contact = false;
    }

    // Avoids data casting at runtime
    const pinocchio::FrameIndex id = model->get_id();
    const std::shared_ptr<StateMultibody>& state =
        std::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[id].name;
    bool found_contact = false;
    if (is_contact) {
      for (typename ContactModelMultiple::ContactDataContainer::iterator it =
               d1->contacts->contacts.begin();
           it != d1->contacts->contacts.end(); ++it) {
        if (it->second->frame == id) {
          ContactData3DTpl<Scalar>* d3d =
              dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
          if (d3d != NULL) {
            found_contact = true;
            contact = it->second;
            throw_pretty(
                "Domain error: there isn't defined at least a 6d contact for " +
                frame_name);
            break;
          }
          ContactData6DTpl<Scalar>* d6d =
              dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
          if (d6d != NULL) {
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty(
              "Domain error: there isn't defined at least a 6d contact for " +
              frame_name);
          break;
        }
      }
    } else {
      for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it =
               d2->impulses->impulses.begin();
           it != d2->impulses->impulses.end(); ++it) {
        if (it->second->frame == id) {
          ImpulseData3DTpl<Scalar>* d3d =
              dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
          if (d3d != NULL) {
            found_contact = true;
            contact = it->second;
            throw_pretty(
                "Domain error: there isn't defined at least a 6d contact for " +
                frame_name);
            break;
          }
          ImpulseData6DTpl<Scalar>* d6d =
              dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
          if (d6d != NULL) {
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty(
              "Domain error: there isn't defined at least a 6d contact for " +
              frame_name);
          break;
        }
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " +
                   frame_name);
    }
  }
  virtual ~ResidualDataContactWrenchConeTpl() = default;

  std::shared_ptr<ForceDataAbstractTpl<Scalar> >
      contact;  //!< Contact force data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelContactWrenchConeTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataContactWrenchConeTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_WRENCH_CONE_HPP_
