///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_COP_POSITION_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_COP_POSITION_HPP_

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
#include "crocoddyl/multibody/cop-support.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Center of pressure residual function
 *
 * It builds a residual function that bounds the center of pressure (CoP) for one contact surface to
 * lie inside a certain geometric area defined around the reference contact frame. The residual residual
 * vector is described as \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\geq\mathbf{0}\f$, where
 * \f[
 *   \mathbf{A}=
 *     \begin{bmatrix} 0 & 0 & X/2 & 0 & -1 & 0 \\ 0 & 0 & X/2 & 0 & 1 & 0 \\ 0 & 0 & Y/2 & 1 & 0 & 0 \\
 *      0 & 0 & Y/2 & -1 & 0 & 0
 *     \end{bmatrix}
 * \f]
 * is the inequality matrix and \f$\boldsymbol{\lambda}\f$ is the reference spatial contact force in the frame
 * coordinate. The CoP lies inside the convex hull of the foot, see eq.(18-19) of
 * https://hal.archives-ouvertes.fr/hal-02108449/document is we have:
 * \f[
 *  \begin{align}\begin{split}\tau^x &\leq
 * Yf^z \\-\tau^x &\leq Yf^z \\\tau^y &\leq Yf^z \\-\tau^y &\leq Yf^z
 *  \end{split}\end{align}
 * \f]
 * with \f$\boldsymbol{\lambda}= \begin{bmatrix}f^x & f^y & f^z & \tau^x & \tau^y & \tau^z \end{bmatrix}^T\f$.
 *
 * The residual is computed, from the residual vector \f$\mathbf{r}\f$, through an user defined activation model.
 * Additionally, the contact frame id, the desired support region for the CoP and the inequality matrix
 * are handled within `CoPSupportTpl`. The force vector \f$\boldsymbol{\lambda}\f$ and its derivatives are
 * computed by `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl`. These values are
 * stored in a shared data (i.e., `DataCollectorContactTpl` or `DataCollectorImpulseTpl`). Note that this residual
 * function cannot be used with other action models.
 *
 * \sa `ResidualModelAbstractTpl`, calc(), calcDiff(), createData(),
 * `DifferentialActionModelContactFwdDynamicsTpl`, `ActionModelImpulseFwdDynamicTpl`, `DataCollectorForceTpl`
 */
template <typename _Scalar>
class ResidualModelContactCoPPositionTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactCoPPositionTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef CoPSupportTpl<Scalar> CoPSupport;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix46s Matrix46;

  /**
   * @brief Initialize the contact CoP residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] cref   Reference support region of the CoP
   * @param[in] nu     Dimension of control vector
   */
  ResidualModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                     const CoPSupport& cref, const std::size_t nu);

  /**
   * @brief Initialize the contact CoP residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] cref   Reference support region of the CoP
   */
  ResidualModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                     const CoPSupport& cref);
  virtual ~ResidualModelContactCoPPositionTpl();

  /**
   * @brief Compute the contact CoP residual
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the force vector is computed by
   * `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl` which is stored in
   * `DataCollectorContactTpl` or `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Contact CoP data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobians of the contact CoP residual
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the force vector is computed by
   * `DifferentialActionModelContactFwdDynamicsTpl` or `ActionModelImpulseFwdDynamicTpl` which is stored in
   * `DataCollectorContactTpl` or `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Contact CoP data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact CoP residual data
   *
   * Each residual model has its own data that needs to be allocated.
   * This function returns the allocated data for a predefined residual.
   *
   * @param[in] data  Shared data (it should be of type `DataCollectorContactTpl`)
   * @return the residual data.
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference support region of CoP
   */
  const CoPSupport& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference support region of CoP
   */
  void set_reference(const CoPSupport& reference);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  pinocchio::FrameIndex id_;  //!< Reference frame id
  CoPSupport cref_;           //!< Reference support region of CoP
};

template <typename _Scalar>
struct ResidualDataContactCoPPositionTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameCoPSupportTpl<Scalar> FrameCoPSupport;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ResidualDataContactCoPPositionTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data) {
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
            found_contact = true;
            contact = it->second;
            throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
            break;
          }
          ContactData6DTpl<Scalar>* d6d = dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
          if (d6d != NULL) {
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
          break;
        }
      }
    } else {
      for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d2->impulses->impulses.begin();
           it != d2->impulses->impulses.end(); ++it) {
        if (it->second->frame == id) {
          ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
          if (d3d != NULL) {
            found_contact = true;
            contact = it->second;
            throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
            break;
          }
          ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
          if (d6d != NULL) {
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
          break;
        }
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " + frame_name);
    }
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  boost::shared_ptr<ForceDataAbstractTpl<Scalar> > contact;  //!< contact force
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-cop-position.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_COP_POSITION_HPP_
