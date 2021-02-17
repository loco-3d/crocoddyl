///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_FORCE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_FORCE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Define a contact force residual function
 *
 * This residual function is defined as \f$\mathbf{r}=\boldsymbol{\lambda}-\boldsymbol{\lambda}^*\f$,
 * where \f$\boldsymbol{\lambda}, \boldsymbol{\lambda}^*\f$ are the current and reference spatial forces, respectively.
 * The current spatial forces \f$\boldsymbol{\lambda}\in\mathbb{R}^{nc}\f$ is computed by
 * `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl`, with `nc` as the dimension of
 * the contact.
 *
 * Both residual and residual Jacobians are computed analytically, where the force vector \f$\boldsymbol{\lambda}\f$
 * and its Jacobians \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl`. These values are stored in a
 * shared data (i.e., `DataCollectorContactTpl` or `DataCollectorImpulseTpl`). Note that this residual function cannot
 * be used with other action models.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`,
 * `DifferentialActionModelContactFwdDynamicsTpl`, `ActionModelImpulseFwdDynamicTpl`, `DataCollectorContactTpl`,
 * `DataCollectorImpulseTpl`
 */
template <typename _Scalar>
class ResidualModelContactForceTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactForceTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact force residual model
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference spatial contact force in the contact coordinates
   * @param[in] nu     Dimension of control vector
   */
  ResidualModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                               const Force& fref, const std::size_t nu);

  /**
   * @brief Initialize the contact force residual model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference spatial contact force in the contact coordinates
   */
  ResidualModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                               const Force& fref);
  virtual ~ResidualModelContactForceTpl();

  /**
   * @brief Compute the contact force residual
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the force vector is computed by
   * `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl` which is stored in
   * `DataCollectorContactTpl` or `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Contact force data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact force residual
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the force vector is computed by
   * `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl` which is stored in
   * `DataCollectorContactTpl` or `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Contact force data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact force residual data
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the residual data.
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference spatial contact force in the contact coordinates
   */
  const Force& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference spatial contact force in the contact coordinates
   */
  void set_reference(const Force& reference);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  pinocchio::FrameIndex id_;  //!< Reference frame id
  Force fref_;                //!< Reference spatial contact force in the contact coordinates
};

template <typename _Scalar>
struct ResidualDataContactForceTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ResidualDataContactForceTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
    contact_type = ContactUndefined;

    // Check that proper shared data has been passed
    bool is_contact = true;
    DataCollectorContactTpl<Scalar>* d1 = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    DataCollectorImpulseTpl<Scalar>* d2 = dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d1 == NULL && d2 == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from DataCollectorContact or DataCollectorImpulse");
    }
    if (d2 != NULL) {
      is_contact = false;
    }

    // Avoids data casting at runtime
    const pinocchio::FrameIndex id = model->get_id();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[id].name;
    bool found_contact = false;
    if (is_contact) {
      for (typename ContactModelMultiple::ContactDataContainer::iterator it = d1->contacts->contacts.begin();
           it != d1->contacts->contacts.end(); ++it) {
        if (it->second->frame == id) {
          ContactData3DTpl<Scalar>* d3d = dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
          if (d3d != NULL) {
            contact_type = Contact3D;
            model->set_nr(3);
            r.resize(3);
            Rx.resize(3, model->get_state()->get_ndx());
            Ru.resize(3, model->get_nu());
            found_contact = true;
            contact = it->second;
            break;
          }
          ContactData6DTpl<Scalar>* d6d = dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
          if (d6d != NULL) {
            contact_type = Contact6D;
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty("Domain error: there isn't defined at least a 3d contact for " + frame_name);
          break;
        }
      }
    } else {
      for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d2->impulses->impulses.begin();
           it != d2->impulses->impulses.end(); ++it) {
        if (it->second->frame == id) {
          ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
          if (d3d != NULL) {
            contact_type = Contact3D;
            model->set_nr(3);
            r.resize(3);
            Rx.resize(3, model->get_state()->get_ndx());
            Ru.resize(3, model->get_nu());
            found_contact = true;
            contact = it->second;
            break;
          }
          ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
          if (d6d != NULL) {
            contact_type = Contact6D;
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty("Domain error: there isn't defined at least a 3d impulse for " + frame_name);
          break;
        }
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact/impulse data for " + frame_name);
    }
  }

  boost::shared_ptr<ForceDataAbstractTpl<Scalar> > contact;  //!< Contact force data
  ContactType contact_type;                                  //!< Type of contact (3D / 6D)
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-force.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_FORCE_HPP_
